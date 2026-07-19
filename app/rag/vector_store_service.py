"""
Vector Store Service — ChromaDB operations cho Enterprise Knowledge Base.

Quản lý toàn bộ tương tác với ChromaDB persistent storage:
    - Tạo / load collection.
    - Thêm document chunks (với embeddings).
    - Xóa chunks theo doc_id.
    - Lấy retriever cho Phase 3B (RAG query).
    - Thống kê collection.

Sử dụng langchain-chroma (wrapper chính thức của LangChain cho ChromaDB).
ChromaDB 1.x API: chromadb.PersistentClient.

Config (từ settings.py):
    - vector_db_dir:    Đường dẫn lưu ChromaDB files.
    - collection_name:  Tên collection.
    - retriever_top_k:  Số documents trả về khi query.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.config.settings import get_settings
from app.core.exceptions import EmbeddingException, VectorStoreException
from app.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service quản lý ChromaDB vector store.

    Sử dụng langchain-chroma để tích hợp ChromaDB với LangChain.
    Persistent storage — dữ liệu tồn tại giữa các lần khởi động app.

    Attributes:
        _collection_name: Tên ChromaDB collection.
        _persist_dir:     Đường dẫn lưu ChromaDB files.
        _embedding_svc:   EmbeddingService instance.
        _vector_store:    Chroma instance (lazy-created).
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """
        Khởi tạo VectorStoreService.

        Args:
            embedding_service: EmbeddingService instance (injected).
        """
        settings = get_settings()
        self._collection_name = settings.collection_name
        self._persist_dir = str(Path(settings.vector_db_dir).resolve())
        self._embedding_svc = embedding_service
        self._vector_store: Optional[Chroma] = None

        # Tạo thư mục nếu chưa có
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "VectorStoreService initialized (collection='%s', path='%s').",
            self._collection_name,
            self._persist_dir,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_vector_store(self) -> Chroma:
        """
        Lấy hoặc tạo Chroma instance (lazy singleton).

        Chroma tự động load existing collection nếu đã có,
        hoặc tạo mới nếu chưa.

        Returns:
            Chroma: LangChain Chroma vector store instance.
        """
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_svc.get_embeddings(),
                persist_directory=self._persist_dir,
            )
            logger.debug(
                "VectorStoreService: loaded/created Chroma collection '%s'.",
                self._collection_name,
            )
        return self._vector_store

    # ------------------------------------------------------------------
    # Document Operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[Document],
        doc_id: str,
    ) -> int:
        """
        Thêm document chunks vào ChromaDB.

        Mỗi chunk đã có metadata (doc_id, chunk_id, file_name...) được
        inject bởi TextSplitterService. ChromaDB sẽ tự động embed
        nội dung text thông qua OllamaEmbeddings.

        Args:
            chunks: List Document chunks đã split và annotated.
            doc_id: UUID của document cha (để log).

        Returns:
            int: Số chunks đã thêm thành công.

        Raises:
            EmbeddingException: Nếu embedding model lỗi.
            VectorStoreException: Nếu ChromaDB lỗi.
        """
        if not chunks:
            logger.warning("VectorStoreService: no chunks to add for doc '%s'.", doc_id)
            return 0

        try:
            vs = self._get_vector_store()
            # Tạo IDs duy nhất cho mỗi chunk: {doc_id}_chunk_{i}
            ids = [f"{doc_id}_chunk_{chunk.metadata.get('chunk_id', i)}" for i, chunk in enumerate(chunks)]
            vs.add_documents(documents=chunks, ids=ids)

            logger.info(
                "VectorStoreService: added %d chunks for doc '%s' to collection '%s'.",
                len(chunks),
                doc_id,
                self._collection_name,
            )
            return len(chunks)

        except Exception as e:
            error_str = str(e).lower()
            logger.error(
                "VectorStoreService.add_documents error for doc '%s': %s",
                doc_id,
                str(e),
                exc_info=True,
            )
            if "embed" in error_str or "connect" in error_str or "ollama" in error_str:
                raise EmbeddingException(
                    model=self._embedding_svc.model_name,
                    detail=str(e),
                ) from e
            raise VectorStoreException(
                message=f"ChromaDB error khi thêm documents: {str(e)}",
                detail=str(e),
                code="VECTOR_STORE_ADD_ERROR",
            ) from e

    def delete_document(self, doc_id: str) -> bool:
        """
        Xóa tất cả chunks của một document khỏi ChromaDB.

        Dựa trên metadata filter `doc_id` — mỗi chunk đã có
        `metadata["doc_id"] = doc_id` được set bởi TextSplitterService.

        Args:
            doc_id: UUID của document cần xóa.

        Returns:
            bool: True nếu xóa thành công (ngay cả khi không có chunk nào).
        """
        try:
            vs = self._get_vector_store()
            # Lấy IDs của tất cả chunks thuộc doc này
            results = vs.get(where={"doc_id": doc_id})
            chunk_ids = results.get("ids", [])

            if chunk_ids:
                vs.delete(ids=chunk_ids)
                logger.info(
                    "VectorStoreService: deleted %d chunks for doc '%s'.",
                    len(chunk_ids),
                    doc_id,
                )
            else:
                logger.warning(
                    "VectorStoreService: no chunks found for doc '%s' in ChromaDB.",
                    doc_id,
                )
            return True

        except Exception as e:
            logger.error(
                "VectorStoreService.delete_document error for doc '%s': %s",
                doc_id,
                str(e),
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Stats & Retriever (Phase 3B ready)
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> dict:
        """
        Lấy thống kê về ChromaDB collection.

        Returns:
            dict: {
                "collection_name": str,
                "vector_db_path": str,
                "total_chunks": int,
                "embedding_model": str,
            }
        """
        try:
            vs = self._get_vector_store()
            total_chunks = vs._collection.count()
        except Exception:
            total_chunks = 0

        return {
            "collection_name": self._collection_name,
            "vector_db_path": self._persist_dir,
            "total_chunks": total_chunks,
            "embedding_model": self._embedding_svc.model_name,
        }

    def get_retriever(self, top_k: Optional[int] = None) -> VectorStoreRetriever:
        """
        Trả về LangChain Retriever để dùng trong RAG chain (Phase 3B).

        Args:
            top_k: Số documents trả về. Dùng settings.retriever_top_k nếu None.

        Returns:
            VectorStoreRetriever: Retriever tương thích với RunnableSequence.
        """
        settings = get_settings()
        k = top_k or settings.retriever_top_k
        vs = self._get_vector_store()
        return vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    @property
    def collection_name(self) -> str:
        """Tên ChromaDB collection đang dùng."""
        return self._collection_name

    @property
    def persist_directory(self) -> str:
        """Đường dẫn ChromaDB persistent storage."""
        return self._persist_dir
