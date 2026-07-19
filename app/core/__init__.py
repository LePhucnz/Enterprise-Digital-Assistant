"""
Core package — Abstract base classes, exceptions và output parsers.

Exports:
    BaseChain:          Abstract base class cho tất cả LangChain chains.
    AppException:       Base exception class.
    LLMException:       Base LLM exception.
    LLMConnectionError: Không kết nối được Ollama.
    LLMTimeoutError:    LLM request timeout.
    LLMGenerationError: Lỗi generate response.
    MemoryException:    Base memory exception.
    SessionNotFoundError: Session không tồn tại.
    ValidationException: Base validation exception.
    InvalidRequestError: Request không hợp lệ.
    ChatOutputParser:   Parser chuẩn hóa LLM output.
    ParsedResponse:     Structured response model.
    ResponseMetadata:   Metadata model cho response.
"""

from app.core.base_chain import BaseChain
from app.core.exceptions import (
    AppException,
    InvalidRequestError,
    LLMConnectionError,
    LLMException,
    LLMGenerationError,
    LLMTimeoutError,
    MemoryException,
    SessionNotFoundError,
    ValidationException,
)
from app.core.output_parser import ChatOutputParser, ParsedResponse, ResponseMetadata

__all__ = [
    # Base
    "BaseChain",
    # Exceptions
    "AppException",
    "LLMException",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMGenerationError",
    "MemoryException",
    "SessionNotFoundError",
    "ValidationException",
    "InvalidRequestError",
    # Output Parser
    "ChatOutputParser",
    "ParsedResponse",
    "ResponseMetadata",
]
