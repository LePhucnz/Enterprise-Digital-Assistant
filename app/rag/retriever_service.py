"""
Retriever Service — Cấp cao hơn VectorStoreService.get_retriever().

Cung cấp 3 chiến lược tìm kiếm và trả về RetrievalResult có cấu trúc,
bao gồm context string và source citations đã được format.

Chiến lược search (từ config):
    - similarity:                 Cosine similarity, top-k kết quả.
    - mmr:                        Max Marginal Relevance — giảm trùng lặp.
    - similarity_score_threshold: Chỉ lấy kết quả >= score threshold.

Log:
    - Retriever time
    - Document count retrieved
    - Source files

Thiết kế:
    RetrieverService không biết về RAGChain hay ChatService.
    Chỉ nhận query → trả về RetrievalResult.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from app.config.settings import get_settings
from app.core.exceptions import CitationException, EmptyRetrievalError, RetrieverException
from app.rag.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


# ===========================================================
# Result Data Class
# ===========================================================


@dataclass
class SourceDocument:
    """
    Thông tin nguồn tài liệu cho một chunk đã retrieve.

    Attributes:
        document:  Tên file gốc (e.g., "annual_report.pdf").
        page:      Số trang (0-indexed từ loader, hiển thị +1).
        chunk_id:  Index của chunk trong document.
        content:   Preview nội dung chunk (100 ký tự đầu).
        doc_id:    UUID của document cha.
    """

    document: str
    page: int
    chunk_id: int
    content: str
    doc_id: str = ""


@dataclass
class RetrievalResult:
    """
    Kết quả trả về từ RetrieverService.retrieve().

    Attributes:
        documents:        List Document chunks từ ChromaDB.
        sources:          List SourceDocument (citations) đã format.
        context:          Chuỗi context đã format để inject vào LLM prompt.
        retrieval_time:   Thời gian retrieve (giây).
        is_empty:         True nếu không tìm được document nào.
        expanded_queries: Danh sách các truy vấn mở rộng được sinh ra (nếu có).
    """

    documents: list[Document] = field(default_factory=list)
    sources: list[SourceDocument] = field(default_factory=list)
    context: str = ""
    retrieval_time: float = 0.0
    is_empty: bool = True
    expanded_queries: list[str] = field(default_factory=list)


# ===========================================================
# Retriever Service
# ===========================================================


class RetrieverService:
    """
    Service cấp cao để retrieve documents từ ChromaDB.

    Wrap VectorStoreService với:
    - Hỗ trợ 3 search strategies (similarity, mmr, similarity_score_threshold).
    - Hỗ trợ kỹ thuật Query Expansion (Multi-Query + Reciprocal Rank Fusion - RRF).
    - Format context string cho LLM.
    - Extract source citations từ metadata.
    - Logging timing và document count.

    Attributes:
        _vector_store:            VectorStoreService instance (injected).
        _query_expander:          QueryExpander instance (injected, optional).
        _search_type:             Chiến lược tìm kiếm từ config.
        _top_k:                   Số documents tối đa trả về (single query).
        _score_threshold:         Ngưỡng score (chỉ dùng với threshold mode).
        _collection_name:         Tên ChromaDB collection.
        _enable_query_expansion:  Bật/tắt Query Expansion từ config.
        _expansion_num_queries:   Số lượng query mở rộng cần sinh.
        _expansion_top_k:         Số document chunks tối đa sau khi fuse.
        _rrf_k:                   Hằng số k của Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        vector_store_service: VectorStoreService,
        query_expander: Optional[Any] = None,
    ) -> None:
        """
        Khởi tạo RetrieverService.

        Args:
            vector_store_service: VectorStoreService instance (injected).
            query_expander:       QueryExpander instance (injected, optional).
        """
        settings = get_settings()
        self._vector_store = vector_store_service
        self._search_type = settings.rag_search_type
        self._top_k = settings.retriever_top_k
        self._score_threshold = settings.rag_score_threshold
        self._relevance_distance_threshold = settings.rag_relevance_distance_threshold
        self._collection_name = settings.collection_name
        self._enable_query_expansion = settings.rag_enable_query_expansion
        self._expansion_num_queries = settings.rag_expansion_num_queries
        self._expansion_top_k = settings.rag_expansion_top_k
        self._rrf_k = settings.rag_expansion_rrf_k

        if query_expander is not None:
            self._query_expander = query_expander
        elif self._enable_query_expansion:
            try:
                from app.rag.query_expander import QueryExpander
                self._query_expander = QueryExpander()
            except Exception as e:
                logger.warning(
                    "RetrieverService: Could not auto-initialize QueryExpander: %s",
                    str(e),
                )
                self._query_expander = None
        else:
            self._query_expander = None

        logger.info(
            "RetrieverService initialized (search_type=%s, top_k=%d, threshold=%.2f, distance_threshold=%.2f, query_expansion=%s).",
            self._search_type,
            self._top_k,
            self._score_threshold,
            self._relevance_distance_threshold,
            "enabled" if self._query_expander is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_retriever(self):
        """
        Tạo LangChain retriever với search config từ settings.

        Returns:
            VectorStoreRetriever: Configured retriever.
        """
        vs = self._vector_store._get_vector_store()

        if self._search_type == "mmr":
            return vs.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self._top_k,
                    "fetch_k": self._top_k * 3,   # Fetch nhiều hơn rồi re-rank
                    "lambda_mult": 0.5,            # Cân bằng relevance/diversity
                },
            )
        elif self._search_type == "similarity_score_threshold":
            return vs.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self._top_k,
                    "score_threshold": self._score_threshold,
                },
            )
        else:
            # Default: similarity
            return vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self._top_k},
            )

    def _retrieve_with_score_filter(self, query: str) -> list[Document]:
        """
        Truy vấn ChromaDB với bộ lọc khoảng cách L2 và xếp hạng lai (Hybrid Ranking).

        Chiến lược:
            1. Fetch tối thiểu 25 candidates (hoặc top_k * 4) để không bỏ sót các chunk
               đặc thù (như điều khoản pháp lý, điều kiện ủy quyền) có vector distance
               ở mức trung bình nhưng khớp từ khóa quan trọng.
            2. Lọc bỏ các chunk có L2 distance vượt quá threshold.
            3. Xếp hạng lai (Hybrid re-ranking) kết hợp độ tương đồng vector và mức độ
               trùng khớp từ khóa (keyword overlap) để đưa các đoạn tài liệu chứa chính xác
               từ khóa nghiệp vụ lên hàng đầu.

        Args:
            query: Câu hỏi từ người dùng.

        Returns:
            list[Document]: Danh sách chunks có độ liên quan cao nhất, tối đa top_k.
        """
        import re

        vs = self._vector_store._get_vector_store()
        fetch_k = max(25, self._top_k * 4)
        candidates = vs.similarity_search_with_score(query, k=fetch_k)

        # Lọc: chỉ giữ chunks có L2 distance <= threshold
        filtered = [
            (doc, score)
            for doc, score in candidates
            if score <= self._relevance_distance_threshold
        ]

        logger.info(
            "RetrieverService._retrieve_with_score_filter: %d/%d candidates passed threshold %.2f.",
            len(filtered),
            len(candidates),
            self._relevance_distance_threshold,
        )

        if not filtered:
            return []

        # Tách từ khóa quan trọng và cụm từ từ query
        stopwords = {
            "cho", "các", "những", "được", "trong", "theo", "này", "khi",
            "của", "với", "hoặc", "và", "là", "thì", "mà", "có", "gì", "nào",
            "phải", "cần", "không", "như", "thế", "đến", "từ"
        }
        words = [w.lower() for w in re.findall(r"[\w]+", query) if len(w) > 2]
        meaningful_words = [w for w in words if w not in stopwords]

        # Trích xuất các cụm 2 và 3 từ
        phrases = []
        for n in [3, 2]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if any(w in meaningful_words for w in phrase.split()):
                    phrases.append(phrase)

        # Hybrid re-ranking: Kết hợp vector similarity và IDF phrase weighting
        scored_docs = []
        for doc, dist in filtered:
            content_lower = doc.page_content.lower()
            # Điểm vector chuẩn hóa (0.0 -> 1.0)
            vec_sim = max(0.0, 1.0 - (dist / self._relevance_distance_threshold))

            # Điểm cụm từ: Cụm từ chuyên môn/hiếm (ủy quyền, điều kiện, thẩm quyền) có trọng số cao
            phrase_score = 0.0
            for p in phrases:
                if p in content_lower:
                    p_len = len(p.split())
                    if "ủy quyền" in p or "uy quyen" in p:
                        phrase_score += 1.5 * p_len
                    elif "điều kiện" in p or "tiêu chuẩn" in p or "thẩm quyền" in p:
                        phrase_score += 1.0 * p_len
                    else:
                        phrase_score += 0.2 * p_len

            total_score = vec_sim + phrase_score
            scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[: self._top_k]]

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[SourceDocument]:
        """
        Extract source citations từ document metadata.

        Args:
            documents: List chunks từ ChromaDB.

        Returns:
            list[SourceDocument]: Citations đã deduplicate theo file+page.
        """
        sources = []
        seen: set[tuple[str, int]] = set()  # (file_name, page)

        for doc in documents:
            meta = doc.metadata
            file_name = meta.get("file_name", meta.get("source", "Unknown"))
            page = meta.get("page", 0)
            chunk_id = meta.get("chunk_id", 0)
            doc_id = meta.get("doc_id", "")

            # Content preview — 150 ký tự đầu
            content_preview = doc.page_content[:150].replace("\n", " ").strip()
            if len(doc.page_content) > 150:
                content_preview += "..."

            key = (file_name, page)
            if key not in seen:
                seen.add(key)
                sources.append(SourceDocument(
                    document=file_name,
                    page=page + 1,      # Convert 0-indexed → 1-indexed cho user
                    chunk_id=chunk_id,
                    content=content_preview,
                    doc_id=doc_id,
                ))

        return sources

    @staticmethod
    def _reciprocal_rank_fusion(
        results_by_query: list[list[Document]],
        k: int = 60,
        top_k: int = 6,
        query_weights: Optional[list[float]] = None,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion (RRF) có trọng số để xếp hạng lại tài liệu.

        Câu hỏi gốc luôn có trọng số cao hơn các câu hỏi mở rộng để đảm bảo
        ngữ cảnh trọng tâm không bị loãng bởi các biến thể truy vấn.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        if not query_weights or len(query_weights) != len(results_by_query):
            query_weights = [1.0] * len(results_by_query)
            if query_weights:
                query_weights[0] = 3.0  # Trọng số ưu tiên câu hỏi gốc

        for q_idx, doc_list in enumerate(results_by_query):
            weight = query_weights[q_idx]
            for rank, doc in enumerate(doc_list, start=1):
                meta = doc.metadata
                file_name = meta.get("file_name", meta.get("source", "Unknown"))
                chunk_id = meta.get("chunk_id", 0)
                page = meta.get("page", 0)
                doc_id = meta.get("doc_id", "")
                content_hash = hash(doc.page_content.strip())
                doc_key = f"{doc_id}:{file_name}:{page}:{chunk_id}:{content_hash}"

                if doc_key not in doc_map:
                    doc_map[doc_key] = doc
                    scores[doc_key] = 0.0

                scores[doc_key] += weight / (k + rank)

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_k]]

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """
        Format list documents thành context string cho LLM prompt.

        Mỗi chunk được format với metadata header để LLM biết nguồn:
            [Nguồn: file.pdf | Trang: 3 | Chunk: 5]
            Nội dung chunk...

        Args:
            documents: List chunks đã retrieve.

        Returns:
            str: Formatted context string.
        """
        if not documents:
            return "Không có tài liệu liên quan được tìm thấy."

        parts = []
        for i, doc in enumerate(documents, start=1):
            meta = doc.metadata
            file_name = meta.get("file_name", meta.get("source", "Unknown"))
            page = meta.get("page", 0) + 1  # 1-indexed
            chunk_id = meta.get("chunk_id", 0)

            header = f"--- Trích dẫn từ: {file_name} (Trang {page}) ---"
            parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        fallback_on_empty: bool = True,
        enable_query_expansion: Optional[bool] = None,
    ) -> RetrievalResult:
        """
        Retrieve documents liên quan đến query từ ChromaDB.

        Hỗ trợ kỹ thuật Query Expansion:
            - Nếu enable_query_expansion=True (hoặc theo config):
              Sinh các biến thể câu hỏi bằng LLM, truy vấn song song cho từng biến thể,
              kết hợp và re-rank tài liệu bằng Reciprocal Rank Fusion (RRF).
            - Nếu False hoặc không có QueryExpander:
              Truy vấn đơn trực tiếp từ ChromaDB.

        Args:
            query:                  Câu hỏi từ người dùng.
            fallback_on_empty:      Nếu True, trả về RetrievalResult rỗng thay vì raise error.
                                    Nếu False, raise EmptyRetrievalError khi không có kết quả.
            enable_query_expansion: Bật/tắt Query Expansion (None = dùng cài đặt hệ thống).

        Returns:
            RetrievalResult: Kết quả với documents, sources, context và expanded_queries.

        Raises:
            EmptyRetrievalError: Nếu fallback_on_empty=False và không có kết quả.
            RetrieverException:  Nếu ChromaDB connection lỗi.
        """
        start_time = time.monotonic()

        should_expand = (
            enable_query_expansion
            if enable_query_expansion is not None
            else self._enable_query_expansion
        ) and (self._query_expander is not None)

        logger.info(
            "RetrieverService.retrieve: query='%s...' (search_type=%s, top_k=%d, expansion=%s).",
            query[:60],
            self._search_type,
            self._top_k,
            should_expand,
        )

        try:
            expanded_queries: list[str] = []

            if should_expand:
                # 1. Sinh danh sách các truy vấn mở rộng
                all_queries = self._query_expander.expand(
                    query=query,
                    num_queries=self._expansion_num_queries,
                )
                raw_expanded = [
                    q for q in all_queries if q.strip().lower() != query.strip().lower()
                ]
                # Lọc bỏ query mang tính hội thoại (như "Cảm ơn...", "Xin lỗi...", "Lưu ý...")
                expanded_queries = [
                    eq for eq in raw_expanded
                    if not eq.lower().startswith(("cảm ơn", "xin lỗi", "chào", "note:", "lưu ý:"))
                    and len(eq.strip()) > 10
                ]
                final_queries = [query] + expanded_queries

                logger.info(
                    "RetrieverService: executing retrieval for %d queries (original + %d expansions)...",
                    len(final_queries),
                    len(expanded_queries),
                )

                # 2. Truy vấn tài liệu cho từng query với bộ lọc ngưỡng L2
                results_by_query = []
                query_weights = []
                for i, q in enumerate(final_queries):
                    try:
                        docs = self._retrieve_with_score_filter(q)
                        if docs:
                            results_by_query.append(docs)
                            query_weights.append(3.0 if i == 0 else 1.0)
                    except Exception as e:
                        logger.warning("Query retrieval failed for '%s': %s", q[:40], str(e))

                # 3. Fuse kết quả qua Reciprocal Rank Fusion (RRF) có trọng số
                if results_by_query:
                    documents = self._reciprocal_rank_fusion(
                        results_by_query=results_by_query,
                        k=self._rrf_k,
                        top_k=self._expansion_top_k,
                        query_weights=query_weights,
                    )
                else:
                    documents = []
            else:
                # Truy vấn đơn với bộ lọc ngưỡng L2 (tránh hallucination)
                documents = self._retrieve_with_score_filter(query)

            retrieval_time = time.monotonic() - start_time

            logger.info(
                "RetrieverService: retrieved %d documents in %.3fs.",
                len(documents),
                retrieval_time,
            )

            if not documents:
                if not fallback_on_empty:
                    raise EmptyRetrievalError(
                        query=query,
                        collection_name=self._collection_name,
                    )
                logger.warning(
                    "RetrieverService: no documents found for query '%s...' — returning empty context.",
                    query[:40],
                )
                return RetrievalResult(
                    documents=[],
                    sources=[],
                    context="Không có tài liệu liên quan được tìm thấy.",
                    retrieval_time=retrieval_time,
                    is_empty=True,
                    expanded_queries=expanded_queries,
                )

            # Extract sources và format context
            try:
                sources = self._extract_sources(documents)
            except Exception as e:
                logger.warning("RetrieverService: citation extraction failed: %s", str(e))
                sources = []

            context = self._format_context(documents)

            # Log nguồn tài liệu tìm được
            unique_files = {s.document for s in sources}
            logger.info(
                "RetrieverService: sources = %s | context_len=%d chars | expanded_queries=%d",
                list(unique_files),
                len(context),
                len(expanded_queries),
            )

            return RetrievalResult(
                documents=documents,
                sources=sources,
                context=context,
                retrieval_time=retrieval_time,
                is_empty=False,
                expanded_queries=expanded_queries,
            )

        except (EmptyRetrievalError, RetrieverException):
            raise
        except Exception as e:
            retrieval_time = time.monotonic() - start_time
            logger.error(
                "RetrieverService.retrieve error after %.3fs: %s",
                retrieval_time,
                str(e),
                exc_info=True,
            )
            raise RetrieverException(
                message=f"Lỗi truy vấn ChromaDB: {str(e)}",
                detail=str(e),
                code="RETRIEVER_ERROR",
            ) from e

    @property
    def search_type(self) -> str:
        """Chiến lược search đang dùng."""
        return self._search_type

    @property
    def top_k(self) -> int:
        """Số documents tối đa trả về."""
        return self._top_k
