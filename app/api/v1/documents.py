"""
Document API Endpoints — FastAPI router cho Document Management.

Endpoints:
    POST   /documents/upload          — Upload file và index vào ChromaDB
    GET    /documents                 — Danh sách tất cả documents
    GET    /documents/stats           — Thống kê Knowledge Base collection
    GET    /documents/{doc_id}        — Chi tiết một document
    DELETE /documents/{doc_id}        — Xóa document
    POST   /documents/{doc_id}/reindex — Reindex document

Không sửa bất kỳ chat endpoint nào.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.dependencies import get_document_service
from app.core.exceptions import (
    DocumentAlreadyExistsError,
    DocumentException,
    DocumentNotFoundError,
    EmbeddingException,
    LoaderException,
    UnsupportedFileTypeError,
    VectorStoreException,
)
from app.documents.models import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatsResponse,
)
from app.documents.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# ===========================================================
# Helper: Convert AppException → HTTPException
# ===========================================================


def _to_http_error(e: Exception) -> HTTPException:
    """
    Convert domain exception thành FastAPI HTTPException với status code phù hợp.

    Args:
        e: AppException hoặc subclass.

    Returns:
        HTTPException với code và detail phù hợp.
    """
    if isinstance(e, DocumentNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
    if isinstance(e, DocumentAlreadyExistsError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=e.message)
    if isinstance(e, UnsupportedFileTypeError):
        return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=e.message)
    if isinstance(e, (LoaderException, EmbeddingException, VectorStoreException, DocumentException)):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message,
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Lỗi không xác định: {str(e)}",
    )


# ===========================================================
# Endpoints
# ===========================================================


@router.post(
    "/upload",
    response_model=DocumentMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Upload và index document",
    description=(
        "Upload file và chạy toàn bộ pipeline: "
        "Load → Chunk → Embed (Ollama) → Store (ChromaDB). "
        "Hỗ trợ: PDF, DOCX, TXT, CSV, Markdown."
    ),
)
async def upload_document(
    file: UploadFile,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentMetadata:
    """
    Upload file và index vào Knowledge Base.

    Args:
        file:        UploadFile từ multipart/form-data.
        doc_service: Injected DocumentService.

    Returns:
        DocumentMetadata: Kết quả với status INDEXED hoặc FAILED.

    Raises:
        415: Định dạng file không hỗ trợ.
        409: File đã tồn tại (duplicate).
        422: Lỗi load hoặc embedding.
        500: Lỗi không xác định.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên file không hợp lệ.",
        )

    logger.info(
        "POST /documents/upload: filename='%s', content_type='%s'.",
        file.filename,
        file.content_type,
    )

    try:
        file_bytes = await file.read()
        result = doc_service.upload_document(
            file_bytes=file_bytes,
            filename=file.filename,
        )
        return result

    except (
        UnsupportedFileTypeError,
        DocumentAlreadyExistsError,
        LoaderException,
        EmbeddingException,
        VectorStoreException,
        DocumentException,
    ) as e:
        logger.warning("POST /documents/upload failed: %s", str(e))
        raise _to_http_error(e)
    except Exception as e:
        logger.error("POST /documents/upload unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi upload: {str(e)}",
        ) from e


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="Danh sách documents",
    description="Lấy danh sách tất cả documents đã upload, sắp xếp mới nhất trước.",
)
async def list_documents(
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """
    Lấy danh sách tất cả documents trong Knowledge Base.

    Returns:
        DocumentListResponse: List documents với tổng số và số đã index.
    """
    logger.info("GET /documents")
    try:
        return doc_service.list_documents()
    except Exception as e:
        logger.error("GET /documents error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/stats",
    response_model=DocumentStatsResponse,
    summary="Thống kê Knowledge Base",
    description="Lấy thống kê về ChromaDB collection: tổng chunks, documents, embedding model.",
)
async def get_collection_stats(
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentStatsResponse:
    """
    Lấy thống kê về Knowledge Base collection.

    Returns:
        DocumentStatsResponse: Thông tin collection và số chunks.
    """
    logger.info("GET /documents/stats")
    try:
        return doc_service.get_collection_stats()
    except Exception as e:
        logger.error("GET /documents/stats error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/{doc_id}",
    response_model=DocumentMetadata,
    summary="Chi tiết document",
    description="Lấy thông tin chi tiết của một document theo ID.",
)
async def get_document(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentMetadata:
    """
    Lấy thông tin chi tiết của một document.

    Args:
        doc_id:      UUID của document.
        doc_service: Injected DocumentService.

    Returns:
        DocumentMetadata: Chi tiết document.

    Raises:
        404: Document không tồn tại.
    """
    logger.info("GET /documents/%s", doc_id)
    try:
        return doc_service.get_document(doc_id)
    except DocumentNotFoundError as e:
        raise _to_http_error(e)
    except Exception as e:
        logger.error("GET /documents/%s error: %s", doc_id, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.delete(
    "/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Xóa document",
    description="Xóa document khỏi Knowledge Base (ChromaDB + metadata + file).",
)
async def delete_document(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponse:
    """
    Xóa document khỏi Knowledge Base.

    Args:
        doc_id:      UUID của document cần xóa.
        doc_service: Injected DocumentService.

    Returns:
        DeleteDocumentResponse: Kết quả xóa.

    Raises:
        404: Document không tồn tại.
    """
    logger.info("DELETE /documents/%s", doc_id)
    try:
        return doc_service.delete_document(doc_id)
    except DocumentNotFoundError as e:
        raise _to_http_error(e)
    except Exception as e:
        logger.error("DELETE /documents/%s error: %s", doc_id, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/{doc_id}/reindex",
    response_model=DocumentMetadata,
    summary="Reindex document",
    description=(
        "Xóa chunks cũ và embed lại toàn bộ document. "
        "Hữu ích khi thay đổi chunk_size hoặc khi document bị FAILED."
    ),
)
async def reindex_document(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentMetadata:
    """
    Reindex một document từ đầu.

    Args:
        doc_id:      UUID của document cần reindex.
        doc_service: Injected DocumentService.

    Returns:
        DocumentMetadata: Trạng thái sau khi reindex.

    Raises:
        404: Document không tồn tại.
        422: File gốc không còn trên disk.
    """
    logger.info("POST /documents/%s/reindex", doc_id)
    try:
        return doc_service.reindex_document(doc_id)
    except (DocumentNotFoundError, DocumentException) as e:
        raise _to_http_error(e)
    except Exception as e:
        logger.error(
            "POST /documents/%s/reindex error: %s",
            doc_id,
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
