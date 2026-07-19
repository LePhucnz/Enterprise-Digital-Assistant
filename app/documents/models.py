"""
Document Models — Pydantic schemas cho Document Management.

Định nghĩa data structures cho:
    - DocumentStatus: Trạng thái của document trong pipeline.
    - DocumentMetadata: Thông tin đầy đủ về một document đã upload.
    - DocumentListResponse: Response cho GET /documents.
    - DocumentStatsResponse: Response cho GET /documents/stats.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ===========================================================
# Enums
# ===========================================================


class DocumentStatus(str, Enum):
    """
    Trạng thái của document trong indexing pipeline.

    Lifecycle:
        PENDING → PROCESSING → INDEXED
                            → FAILED
    """

    PENDING = "pending"         # Đã nhận file, chưa xử lý
    PROCESSING = "processing"   # Đang load / chunk / embed
    INDEXED = "indexed"         # Đã lưu vào ChromaDB thành công
    FAILED = "failed"           # Lỗi trong pipeline


class FileType(str, Enum):
    """Các định dạng file được hỗ trợ."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    MARKDOWN = "md"


# ===========================================================
# Core Document Model
# ===========================================================


class DocumentMetadata(BaseModel):
    """
    Thông tin đầy đủ về một document đã được upload và index.

    Đây là model trung tâm của Document Management system.
    Được lưu vào `data/documents.json` bởi DocumentRepository.

    Attributes:
        id:              UUID định danh document.
        file_name:       Tên file gốc (có extension).
        file_type:       Định dạng file (pdf, docx, txt, csv, md).
        file_size:       Kích thước file (bytes).
        file_hash:       SHA256 hash của nội dung file — dùng để duplicate check.
        chunk_count:     Số chunks đã được tạo và lưu vào ChromaDB.
        collection_name: Tên ChromaDB collection chứa document này.
        status:          Trạng thái hiện tại trong pipeline.
        created_at:      Thời điểm upload (ISO 8601 UTC).
        updated_at:      Thời điểm cập nhật cuối (ISO 8601 UTC).
        error_message:   Thông báo lỗi nếu status = FAILED.
    """

    id: str = Field(..., description="UUID định danh document.")
    file_name: str = Field(..., description="Tên file gốc (có extension).")
    file_type: str = Field(..., description="Định dạng file (pdf, docx, txt, csv, md).")
    file_size: int = Field(..., description="Kích thước file (bytes).")
    file_hash: str = Field(..., description="SHA256 hash — dùng để duplicate check.")
    chunk_count: int = Field(default=0, description="Số chunks đã index vào ChromaDB.")
    collection_name: str = Field(..., description="Tên ChromaDB collection.")
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        description="Trạng thái xử lý.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Thời điểm upload (ISO 8601 UTC).",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Thời điểm cập nhật cuối (ISO 8601 UTC).",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Thông báo lỗi nếu pipeline thất bại.",
    )

    def mark_processing(self) -> None:
        """Chuyển status sang PROCESSING và cập nhật updated_at."""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_indexed(self, chunk_count: int) -> None:
        """Chuyển status sang INDEXED sau khi embed xong."""
        self.status = DocumentStatus.INDEXED
        self.chunk_count = chunk_count
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Chuyển status sang FAILED với thông báo lỗi."""
        self.status = DocumentStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now(timezone.utc).isoformat()


# ===========================================================
# API Response Models
# ===========================================================


class DocumentListResponse(BaseModel):
    """Response cho GET /documents."""

    documents: list[DocumentMetadata] = Field(
        default_factory=list,
        description="Danh sách tất cả documents đã upload.",
    )
    total: int = Field(default=0, description="Tổng số documents.")
    indexed_count: int = Field(default=0, description="Số documents đã index thành công.")


class DocumentStatsResponse(BaseModel):
    """
    Response cho GET /documents/stats.

    Cung cấp thống kê về Knowledge Base collection.
    """

    collection_name: str = Field(..., description="Tên ChromaDB collection.")
    vector_db_path: str = Field(..., description="Đường dẫn đến ChromaDB storage.")
    total_documents: int = Field(default=0, description="Tổng số documents.")
    indexed_documents: int = Field(default=0, description="Số documents đã index.")
    total_chunks: int = Field(default=0, description="Tổng số chunks trong ChromaDB.")
    embedding_model: str = Field(..., description="Model embedding đang dùng.")


class DeleteDocumentResponse(BaseModel):
    """Response cho DELETE /documents/{doc_id}."""

    doc_id: str = Field(..., description="ID của document đã xóa.")
    success: bool = Field(..., description="True nếu xóa thành công.")
    message: str = Field(..., description="Thông báo kết quả.")
