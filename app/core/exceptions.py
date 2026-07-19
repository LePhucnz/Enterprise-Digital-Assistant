"""
Exceptions — Hệ thống exception phân cấp cho toàn bộ ứng dụng.

Mọi lỗi trong hệ thống đều phải kế thừa từ `AppException`.
Điều này cho phép:
    - Xử lý thống nhất tại API layer (catch `AppException`).
    - Phân biệt rõ loại lỗi để log và trả về HTTP status code phù hợp.
    - Dễ dàng thêm loại exception mới mà không ảnh hưởng handler hiện tại.

Hierarchy:
    AppException (base)
    ├── LLMException
    │   ├── LLMConnectionError   — Không kết nối được Ollama
    │   ├── LLMTimeoutError      — Request vượt quá timeout
    │   └── LLMGenerationError   — Lỗi trong quá trình generate
    ├── MemoryException
    │   └── SessionNotFoundError — Session ID không tồn tại
    └── ValidationException
        └── InvalidRequestError  — Request không hợp lệ
"""

from __future__ import annotations

from typing import Optional


# ===========================================================
# Base Exception
# ===========================================================


class AppException(Exception):
    """
    Base exception cho toàn bộ Enterprise Digital Assistant.

    Attributes:
        message: Thông điệp lỗi (tiếng Việt hoặc tiếng Anh).
        detail:  Chi tiết kỹ thuật bổ sung (dùng cho logging, không expose ra user).
        code:    Error code dạng string (dùng cho client xử lý lỗi).
    """

    def __init__(
        self,
        message: str,
        detail: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.code = code or self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


# ===========================================================
# LLM Exceptions
# ===========================================================


class LLMException(AppException):
    """Base class cho tất cả lỗi liên quan đến LLM."""


class LLMConnectionError(LLMException):
    """
    Không thể kết nối đến Ollama server.

    Thường xảy ra khi:
    - Ollama chưa được khởi động.
    - URL cấu hình sai trong OLLAMA_BASE_URL.
    - Firewall chặn kết nối đến port 11434.
    """

    def __init__(
        self,
        base_url: str,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Không thể kết nối đến Ollama tại {base_url!r}. Hãy kiểm tra Ollama đã chạy chưa.",
            detail=detail,
            code="LLM_CONNECTION_ERROR",
        )
        self.base_url = base_url


class LLMTimeoutError(LLMException):
    """
    LLM request vượt quá thời gian chờ cho phép.

    Thường xảy ra khi:
    - Model quá lớn, máy không đủ tài nguyên.
    - Prompt quá dài.
    - LLM_TIMEOUT config quá nhỏ.
    """

    def __init__(
        self,
        timeout_seconds: int,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"LLM request đã vượt quá {timeout_seconds}s. Thử rút ngắn câu hỏi hoặc tăng LLM_TIMEOUT.",
            detail=detail,
            code="LLM_TIMEOUT_ERROR",
        )
        self.timeout_seconds = timeout_seconds


class LLMGenerationError(LLMException):
    """
    Lỗi trong quá trình LLM generate response.

    Thường xảy ra khi:
    - Model trả về lỗi nội bộ.
    - Output không parse được.
    - Model bị OOM (out of memory).
    """

    def __init__(
        self,
        model: str,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Model {model!r} gặp lỗi trong quá trình generate. Vui lòng thử lại.",
            detail=detail,
            code="LLM_GENERATION_ERROR",
        )
        self.model = model


# ===========================================================
# Memory Exceptions
# ===========================================================


class MemoryException(AppException):
    """Base class cho tất cả lỗi liên quan đến Memory/Session."""


class SessionNotFoundError(MemoryException):
    """
    Session ID không tồn tại trong store.

    Xảy ra khi client gửi session_id cũ đã bị xóa
    hoặc session_id không hợp lệ.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session {session_id!r} không tồn tại hoặc đã hết hạn.",
            code="SESSION_NOT_FOUND",
        )
        self.session_id = session_id


# ===========================================================
# Validation Exceptions
# ===========================================================


class ValidationException(AppException):
    """Base class cho tất cả lỗi validation."""


class InvalidRequestError(ValidationException):
    """
    Request không hợp lệ — thiếu field bắt buộc hoặc giá trị không đúng.

    Thường bị bắt và convert sang HTTP 422 ở API layer.
    """

    def __init__(
        self,
        field: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=f"Trường '{field}' không hợp lệ: {reason}",
            code="INVALID_REQUEST",
        )
        self.field = field
        self.reason = reason


# ===========================================================
# Document Exceptions
# ===========================================================


class DocumentException(AppException):
    """Base class cho tất cả lỗi liên quan đến Document management."""


class DocumentNotFoundError(DocumentException):
    """
    Document ID không tồn tại trong repository.

    Xảy ra khi client gửi doc_id không hợp lệ hoặc đã bị xóa.
    """

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            message=f"Document '{doc_id}' không tồn tại hoặc đã bị xóa.",
            code="DOCUMENT_NOT_FOUND",
        )
        self.doc_id = doc_id


class DocumentAlreadyExistsError(DocumentException):
    """
    File đã được upload trước đó (duplicate check qua SHA256 hash).

    Xảy ra khi người dùng upload cùng một file hai lần.
    """

    def __init__(self, file_name: str, existing_doc_id: str) -> None:
        super().__init__(
            message=f"File '{file_name}' đã được upload trước đó (doc_id: {existing_doc_id}).",
            code="DOCUMENT_ALREADY_EXISTS",
        )
        self.file_name = file_name
        self.existing_doc_id = existing_doc_id


class UnsupportedFileTypeError(DocumentException):
    """
    Định dạng file không được hỗ trợ.

    Chỉ chấp nhận: PDF, DOCX, TXT, CSV, Markdown.
    """

    def __init__(self, file_name: str, extension: str) -> None:
        super().__init__(
            message=(
                f"Định dạng file '{extension}' không được hỗ trợ. "
                f"Chấp nhận: .pdf, .docx, .txt, .csv, .md, .markdown"
            ),
            code="UNSUPPORTED_FILE_TYPE",
        )
        self.file_name = file_name
        self.extension = extension


class LoaderException(DocumentException):
    """
    Lỗi khi load hoặc parse file.

    Thường xảy ra khi:
    - File bị corrupt.
    - File được password-protected (PDF).
    - File encoding không hợp lệ (TXT/CSV).
    """

    def __init__(self, file_name: str, detail: Optional[str] = None) -> None:
        super().__init__(
            message=f"Không thể đọc file '{file_name}'. Hãy kiểm tra file không bị lỗi.",
            detail=detail,
            code="LOADER_ERROR",
        )
        self.file_name = file_name


# ===========================================================
# Embedding Exceptions
# ===========================================================


class EmbeddingException(AppException):
    """
    Lỗi trong quá trình tạo embeddings.

    Thường xảy ra khi:
    - Ollama không có model embedding (nomic-embed-text chưa được pull).
    - Kết nối Ollama bị ngắt giữa chừng.
    - Text chunk quá dài vượt context window của embedding model.
    """

    def __init__(
        self,
        model: str,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Lỗi embedding với model '{model}'. Hãy kiểm tra model đã được pull về chưa.",
            detail=detail,
            code="EMBEDDING_ERROR",
        )
        self.model = model


# ===========================================================
# Vector Store Exceptions
# ===========================================================


class VectorStoreException(AppException):
    """Base class cho tất cả lỗi liên quan đến Vector Database."""


class CollectionNotFoundError(VectorStoreException):
    """
    ChromaDB collection chưa được tạo hoặc không tồn tại.

    Xảy ra khi truy vấn collection trước khi có document nào được index.
    """

    def __init__(self, collection_name: str) -> None:
        super().__init__(
            message=f"Collection '{collection_name}' chưa tồn tại. Hãy upload ít nhất một document.",
            code="COLLECTION_NOT_FOUND",
        )
        self.collection_name = collection_name


# ===========================================================
# Retriever Exceptions
# ===========================================================


class RetrieverException(AppException):
    """
    Base class cho tất cả lỗi liên quan đến Retrieval.

    Xảy ra trong quá trình tìm kiếm documents từ ChromaDB.
    """


class EmptyRetrievalError(RetrieverException):
    """
    Không tìm được document nào liên quan đến câu hỏi.

    Xảy ra khi:
    - ChromaDB collection rỗng (chưa có document nào được index).
    - Score của tất cả kết quả thấp hơn threshold.
    - Query không match với bất kỳ document nào.
    """

    def __init__(
        self,
        query: str,
        collection_name: str,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Không tìm thấy tài liệu liên quan đến câu hỏi trong collection '{collection_name}'. "
                f"Hãy upload tài liệu trước hoặc dùng chế độ Normal Chat."
            ),
            detail=detail,
            code="EMPTY_RETRIEVAL",
        )
        self.query = query
        self.collection_name = collection_name


# ===========================================================
# RAG Exceptions
# ===========================================================


class RAGException(AppException):
    """
    Lỗi tổng quát trong RAG pipeline.

    Thường xảy ra khi:
    - Retriever thành công nhưng LLM generation thất bại.
    - Context quá dài vượt context window.
    - Pipeline bị gián đoạn giữa chừng.
    """

    def __init__(
        self,
        stage: str,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"RAG pipeline thất bại tại giai đoạn '{stage}'. Vui lòng thử lại.",
            detail=detail,
            code="RAG_PIPELINE_ERROR",
        )
        self.stage = stage


# ===========================================================
# Citation Exceptions
# ===========================================================


class CitationException(AppException):
    """
    Lỗi khi format hoặc extract source citations từ retrieved documents.

    Không critical — thường chỉ log warning và trả về empty sources.
    """

    def __init__(
        self,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            message="Không thể trích xuất thông tin nguồn tài liệu.",
            detail=detail,
            code="CITATION_ERROR",
        )
