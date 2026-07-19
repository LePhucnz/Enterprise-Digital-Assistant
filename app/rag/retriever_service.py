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
        documents:       List Document chunks từ ChromaDB.
        sources:         List SourceDocument (citations) đã format.
        context:         Chuỗi context đã format để inject vào LLM prompt.
        retrieval_time:  Thời gian retrieve (giây).
        is_empty:        True nếu không tìm được document nào.
    """

    documents: list[Document] = field(default_factory=list)
    sources: list[SourceDocument] = field(default_factory=list)
    context: str = ""
    retrieval_time: float = 0.0
    is_empty: bool = True


# ===========================================================
# Retriever Service
# ===========================================================


class RetrieverService:
    """
    Service cấp cao để retrieve documents từ ChromaDB.

    Wrap VectorStoreService với:
    - Hỗ trợ 3 search strategies.
    - Format context string cho LLM.
    - Extract source citations từ metadata.
    - Logging timing và document count.

    Attributes:
        _vector_store:     VectorStoreService instance (injected).
        _search_type:      Chiến lược tìm kiếm từ config.
        _top_k:            Số documents tối đa trả về.
        _score_threshold:  Ngưỡng score (chỉ dùng với threshold mode).
        _collection_name:  Tên ChromaDB collection.
    """

    def __init__(self, vector_store_service: VectorStoreService) -> None:
        """
        Khởi tạo RetrieverService.

        Args:
            vector_store_service: VectorStoreService instance (injected).
        """
        settings = get_settings()
        self._vector_store = vector_store_service
        self._search_type = settings.rag_search_type
        self._top_k = settings.retriever_top_k
        self._score_threshold = settings.rag_score_threshold
        self._collection_name = settings.collection_name

        logger.info(
            "RetrieverService initialized (search_type=%s, top_k=%d, threshold=%.2f).",
            self._search_type,
            self._top_k,
            self._score_threshold,
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

            header = f"[Tài liệu {i}: {file_name} | Trang {page} | Chunk #{chunk_id}]"
            parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, fallback_on_empty: bool = True) -> RetrievalResult:
        """
        Retrieve documents liên quan đến query từ ChromaDB.

        Args:
            query:             Câu hỏi từ người dùng.
            fallback_on_empty: Nếu True, trả về RetrievalResult rỗng thay vì raise error.
                               Nếu False, raise EmptyRetrievalError khi không có kết quả.

        Returns:
            RetrievalResult: Kết quả với documents, sources và context.

        Raises:
            EmptyRetrievalError: Nếu fallback_on_empty=False và không có kết quả.
            RetrieverException:  Nếu ChromaDB connection lỗi.
        """
        start_time = time.monotonic()

        logger.info(
            "RetrieverService.retrieve: query='%s...' (search_type=%s, top_k=%d).",
            query[:60],
            self._search_type,
            self._top_k,
        )

        try:
            retriever = self._build_retriever()
            documents = retriever.invoke(query)
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
                "RetrieverService: sources = %s | context_len=%d chars.",
                list(unique_files),
                len(context),
            )

            return RetrievalResult(
                documents=documents,
                sources=sources,
                context=context,
                retrieval_time=retrieval_time,
                is_empty=False,
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
