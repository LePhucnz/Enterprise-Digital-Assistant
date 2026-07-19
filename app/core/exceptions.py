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
