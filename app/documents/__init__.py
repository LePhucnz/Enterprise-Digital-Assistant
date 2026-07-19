"""
Documents package — Document management & indexing pipeline.

Exports:
    DocumentService:     Main orchestrator cho upload/delete/list.
    DocumentMetadata:    Data model cho một document.
    DocumentStatus:      Enum trạng thái trong pipeline.
    DocumentRepository:  JSON persistence layer.
    DocumentLoaderFactory: Factory tạo LangChain loaders.
"""

from app.documents.document_service import DocumentService
from app.documents.loader_factory import DocumentLoaderFactory
from app.documents.models import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatsResponse,
    DocumentStatus,
)
from app.documents.repository import DocumentRepository

__all__ = [
    "DocumentService",
    "DocumentLoaderFactory",
    "DocumentMetadata",
    "DocumentStatus",
    "DocumentListResponse",
    "DocumentStatsResponse",
    "DeleteDocumentResponse",
    "DocumentRepository",
]
