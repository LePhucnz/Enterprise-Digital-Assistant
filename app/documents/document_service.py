"""
Document Service — Orchestrate toàn bộ document upload pipeline.

Luồng xử lý khi upload:
    1. Validate file type (extension check).
    2. Compute SHA256 hash → duplicate check.
    3. Save file bytes vào data/ directory.
    4. Tạo DocumentMetadata → save vào repository (status=PENDING).
    5. Load document qua LoaderFactory.
    6. Split thành chunks qua TextSplitterService.
    7. Embed + store vào ChromaDB qua VectorStoreService.
    8. Update status → INDEXED (hoặc FAILED nếu có lỗi).

DocumentService là entry point duy nhất cho tất cả document operations.
Không biết về FastAPI hay Streamlit — chỉ business logic.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config.settings import get_settings
from app.core.exceptions import (
    DocumentAlreadyExistsError,
    DocumentException,
    DocumentNotFoundError,
    EmbeddingException,
    LoaderException,
    UnsupportedFileTypeError,
    VectorStoreException,
)
from app.documents.loader_factory import DocumentLoaderFactory
from app.documents.models import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatsResponse,
    DocumentStatus,
)
from app.documents.repository import DocumentRepository
from app.rag.embedding_service import EmbeddingService
from app.rag.text_splitter import TextSplitterService
from app.rag.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Orchestrate toàn bộ document management pipeline.

    Chịu trách nhiệm:
        - Upload, validate, duplicate check.
        - Gọi Loader → Splitter → VectorStore theo đúng thứ tự.
        - Quản lý lifecycle của DocumentMetadata (PENDING → INDEXED / FAILED).
        - List, detail, delete, reindex documents.

    KHÔNG chịu trách nhiệm:
        - HTTP layer (→ app/api/v1/documents.py).
        - Embedding logic (→ EmbeddingService).
        - Persistence (→ DocumentRepository).

    Attributes:
        _repo:          DocumentRepository singleton.
        _loader_factory: DocumentLoaderFactory (static methods).
        _splitter:      TextSplitterService singleton.
        _vector_store:  VectorStoreService singleton.
        _data_dir:      Đường dẫn lưu raw files.
        _collection_name: Tên ChromaDB collection.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
    ) -> None:
        """
        Khởi tạo DocumentService với DI.

        Args:
            repository:          DocumentRepository (injected).
            embedding_service:   EmbeddingService (injected).
            vector_store_service: VectorStoreService (injected).
        """
        settings = get_settings()
        self._repo = repository
        self._embedding_svc = embedding_service
        self._vector_store = vector_store_service
        self._splitter = TextSplitterService()
        self._data_dir = Path(settings.data_dir)
        self._collection_name = settings.collection_name

        # Đảm bảo data dir tồn tại
        self._data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DocumentService initialized (data_dir='%s', collection='%s').",
            self._data_dir,
            self._collection_name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(file_bytes: bytes) -> str:
        """Tính SHA256 hash của file bytes."""
        return hashlib.sha256(file_bytes).hexdigest()

    def _save_file(self, file_bytes: bytes, filename: str) -> Path:
        """
        Lưu file bytes vào data directory.

        Tên file được đặt lại thành {doc_id}_{original_name} để tránh conflict.
        """
        safe_name = Path(filename).name  # Xóa path traversal
        file_path = self._data_dir / safe_name
        file_path.write_bytes(file_bytes)
        logger.debug("DocumentService: saved file to '%s'.", file_path)
        return file_path

    def _run_pipeline(self, doc: DocumentMetadata, file_path: Path) -> DocumentMetadata:
        """
        Chạy pipeline: Load → Split → Embed → Store.

        Cập nhật status của doc trong suốt quá trình.

        Args:
            doc:       DocumentMetadata đã tạo (status=PENDING).
            file_path: Đường dẫn đến file đã lưu trên disk.

        Returns:
            DocumentMetadata: Updated với status=INDEXED hoặc FAILED.
        """
        start_time = time.monotonic()

        # --- Step 1: Mark PROCESSING ---
        self._repo.update_status(doc.id, DocumentStatus.PROCESSING)
        logger.info("DocumentService: pipeline started for '%s' (doc_id=%s).", doc.file_name, doc.id)

        try:
            # --- Step 2: Load document ---
            load_start = time.monotonic()
            loader = DocumentLoaderFactory.create(file_path)
            try:
                raw_documents = loader.load()
            except Exception as e:
                raise LoaderException(file_name=doc.file_name, detail=str(e)) from e

            load_elapsed = time.monotonic() - load_start
            logger.info(
                "DocumentService: loaded '%s' → %d pages/rows in %.2fs.",
                doc.file_name,
                len(raw_documents),
                load_elapsed,
            )

            # --- Step 3: Split into chunks ---
            chunks = self._splitter.split(
                documents=raw_documents,
                doc_id=doc.id,
                file_name=doc.file_name,
                file_type=doc.file_type,
                collection_name=self._collection_name,
            )

            if not chunks:
                raise DocumentException(
                    message=f"Không thể tạo chunks từ file '{doc.file_name}'. File có thể rỗng.",
                    code="EMPTY_DOCUMENT",
                )

            logger.info(
                "DocumentService: '%s' split → %d chunks (chunk_size=%d).",
                doc.file_name,
                len(chunks),
                self._splitter.chunk_size,
            )

            # --- Step 4: Embed + Store ---
            embed_start = time.monotonic()
            chunk_count = self._vector_store.add_documents(chunks=chunks, doc_id=doc.id)
            embed_elapsed = time.monotonic() - embed_start

            logger.info(
                "DocumentService: embedded %d chunks for '%s' in %.2fs.",
                chunk_count,
                doc.file_name,
                embed_elapsed,
            )

            # --- Step 5: Mark INDEXED ---
            self._repo.update_status(doc.id, DocumentStatus.INDEXED, chunk_count=chunk_count)

            total_elapsed = time.monotonic() - start_time
            logger.info(
                "DocumentService: pipeline DONE for '%s' | chunks=%d | total_time=%.2fs.",
                doc.file_name,
                chunk_count,
                total_elapsed,
            )

            # Reload doc từ repo để lấy state mới nhất
            return self._repo.get(doc.id) or doc

        except (DocumentException, LoaderException, EmbeddingException, VectorStoreException) as e:
            elapsed = time.monotonic() - start_time
            logger.error(
                "DocumentService: pipeline FAILED for '%s' after %.2fs — %s",
                doc.file_name,
                elapsed,
                str(e),
                exc_info=True,
            )
            self._repo.update_status(doc.id, DocumentStatus.FAILED, error_message=str(e))
            raise

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(
                "DocumentService: pipeline UNEXPECTED FAILURE for '%s' after %.2fs — %s",
                doc.file_name,
                elapsed,
                str(e),
                exc_info=True,
            )
            self._repo.update_status(doc.id, DocumentStatus.FAILED, error_message=str(e))
            raise DocumentException(
                message=f"Lỗi không xác định khi xử lý '{doc.file_name}'.",
                detail=str(e),
                code="PIPELINE_ERROR",
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_document(self, file_bytes: bytes, filename: str) -> DocumentMetadata:
        """
        Upload và index một document vào Knowledge Base.

        Thực hiện đầy đủ pipeline: validate → save → load → split → embed → store.

        Args:
            file_bytes: Nội dung file dạng bytes.
            filename:   Tên file gốc (có extension).

        Returns:
            DocumentMetadata: Kết quả với status=INDEXED hoặc FAILED.

        Raises:
            UnsupportedFileTypeError:   Extension không hỗ trợ.
            DocumentAlreadyExistsError: File đã upload trước đó.
            LoaderException:            Không đọc được file.
            EmbeddingException:         Lỗi embedding.
            VectorStoreException:       Lỗi ChromaDB.
        """
        # --- Validate file type ---
        if not DocumentLoaderFactory.is_supported(filename):
            ext = Path(filename).suffix
            raise UnsupportedFileTypeError(file_name=filename, extension=ext)

        # --- Duplicate check (SHA256) ---
        file_hash = self._compute_hash(file_bytes)
        existing = self._repo.get_by_hash(file_hash)
        if existing:
            logger.warning(
                "DocumentService: duplicate file '%s' detected (existing doc_id=%s).",
                filename,
                existing.id,
            )
            raise DocumentAlreadyExistsError(
                file_name=filename,
                existing_doc_id=existing.id,
            )

        # --- Save file to disk ---
        doc_id = str(uuid.uuid4())
        # Rename file với prefix doc_id để tránh conflict
        safe_filename = f"{doc_id[:8]}_{Path(filename).name}"
        file_path = self._save_file(file_bytes, safe_filename)

        # --- Create DocumentMetadata ---
        doc = DocumentMetadata(
            id=doc_id,
            file_name=filename,
            file_type=Path(filename).suffix.lstrip(".").lower(),
            file_size=len(file_bytes),
            file_hash=file_hash,
            collection_name=self._collection_name,
            status=DocumentStatus.PENDING,
        )
        self._repo.save(doc)
        logger.info(
            "DocumentService.upload: doc_id=%s | file='%s' | size=%d bytes.",
            doc_id,
            filename,
            len(file_bytes),
        )

        # --- Run pipeline ---
        return self._run_pipeline(doc, file_path)

    def get_document(self, doc_id: str) -> DocumentMetadata:
        """
        Lấy thông tin chi tiết của một document.

        Args:
            doc_id: UUID của document.

        Returns:
            DocumentMetadata: Thông tin document.

        Raises:
            DocumentNotFoundError: Document không tồn tại.
        """
        doc = self._repo.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(doc_id)
        return doc

    def list_documents(self) -> DocumentListResponse:
        """
        Lấy danh sách tất cả documents.

        Returns:
            DocumentListResponse: List documents với thống kê.
        """
        docs = self._repo.list_all()
        indexed = sum(1 for d in docs if d.status == DocumentStatus.INDEXED)
        return DocumentListResponse(
            documents=docs,
            total=len(docs),
            indexed_count=indexed,
        )

    def delete_document(self, doc_id: str) -> DeleteDocumentResponse:
        """
        Xóa document khỏi Knowledge Base (ChromaDB + repository + file).

        Args:
            doc_id: UUID của document cần xóa.

        Returns:
            DeleteDocumentResponse: Kết quả xóa.

        Raises:
            DocumentNotFoundError: Document không tồn tại.
        """
        doc = self._repo.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(doc_id)

        # 1. Xóa chunks khỏi ChromaDB
        self._vector_store.delete_document(doc_id)

        # 2. Xóa file trên disk (best-effort, không fail nếu file đã mất)
        try:
            # Tìm file với prefix doc_id
            for file in self._data_dir.glob(f"{doc_id[:8]}_*"):
                file.unlink()
                logger.debug("DocumentService: deleted file '%s'.", file)
        except Exception as e:
            logger.warning("DocumentService: could not delete file for doc '%s': %s", doc_id, str(e))

        # 3. Xóa metadata khỏi repository
        self._repo.delete(doc_id)

        logger.info("DocumentService: deleted doc '%s' ('%s').", doc_id, doc.file_name)
        return DeleteDocumentResponse(
            doc_id=doc_id,
            success=True,
            message=f"Document '{doc.file_name}' đã được xóa thành công.",
        )

    def reindex_document(self, doc_id: str) -> DocumentMetadata:
        """
        Reindex một document — xóa chunks cũ và embed lại.

        Hữu ích khi:
        - Thay đổi chunk_size / chunk_overlap config.
        - Document trước đó bị FAILED, muốn thử lại.

        Args:
            doc_id: UUID của document cần reindex.

        Returns:
            DocumentMetadata: Updated sau khi reindex.

        Raises:
            DocumentNotFoundError: Document không tồn tại.
        """
        doc = self._repo.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(doc_id)

        logger.info("DocumentService: reindexing doc '%s' ('%s').", doc_id, doc.file_name)

        # Xóa chunks cũ khỏi ChromaDB
        self._vector_store.delete_document(doc_id)

        # Tìm file trên disk
        file_path: Optional[Path] = None
        for f in self._data_dir.glob(f"{doc_id[:8]}_*"):
            file_path = f
            break

        if not file_path or not file_path.exists():
            self._repo.update_status(
                doc_id,
                DocumentStatus.FAILED,
                error_message="File gốc không còn tồn tại trên disk. Hãy upload lại.",
            )
            raise DocumentException(
                message=f"File gốc của document '{doc.file_name}' không tìm thấy. Upload lại.",
                code="FILE_NOT_FOUND",
            )

        # Chạy lại pipeline
        return self._run_pipeline(doc, file_path)

    def get_collection_stats(self) -> DocumentStatsResponse:
        """
        Lấy thống kê về Knowledge Base collection.

        Returns:
            DocumentStatsResponse: Thống kê đầy đủ.
        """
        settings = get_settings()
        vs_stats = self._vector_store.get_collection_stats()
        docs = self._repo.list_all()
        indexed = sum(1 for d in docs if d.status == DocumentStatus.INDEXED)

        return DocumentStatsResponse(
            collection_name=vs_stats["collection_name"],
            vector_db_path=vs_stats["vector_db_path"],
            total_documents=len(docs),
            indexed_documents=indexed,
            total_chunks=vs_stats["total_chunks"],
            embedding_model=vs_stats["embedding_model"],
        )
