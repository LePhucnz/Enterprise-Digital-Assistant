"""
Chat Models — Pydantic schemas cho Chat API và Service layer.

Định nghĩa data structures cho:
    - MessageRole: Vai trò trong hội thoại (user, assistant, system).
    - Message: Nội dung tin nhắn kèm vai trò.
    - ChatRequest: Request body cho /chat và /chat/sync.
    - ChatResponse: Response body cho non-streaming chat.
    - RAGChatResponse: Response chat kèm nguồn tài liệu tham chiếu.
    - ChatHistoryResponse: Response trả về danh sách lịch sử tin nhắn.
    - ClearHistoryResponse: Response cho thao tác xóa lịch sử session.
    - StreamChunk: Chunk format cho Server-Sent Events (SSE) streaming.
    - SourceDocument: Thông tin trích dẫn nguồn tài liệu.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Vai trò của người phát ngôn trong hội thoại."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceDocument(BaseModel):
    """Thông tin nguồn tài liệu trích dẫn."""

    document: str = Field(description="Tên file tài liệu gốc")
    page: Optional[int] = Field(default=None, description="Số trang tài liệu")
    chunk_id: Optional[int] = Field(default=None, description="ID chunk trong document")
    content: Optional[str] = Field(default=None, description="Nội dung trích đoạn")
    doc_id: Optional[str] = Field(default=None, description="UUID của document")


class Message(BaseModel):
    """Một tin nhắn trong phiên hội thoại."""

    role: MessageRole | str = Field(description="Vai trò người gửi (user / assistant / system)")
    content: str = Field(description="Nội dung tin nhắn")


class ChatRequest(BaseModel):
    """Request payload cho các API chat."""

    message: str = Field(..., description="Nội dung câu hỏi hoặc tin nhắn của người dùng")
    session_id: Optional[str] = Field(default=None, description="ID phiên làm việc (nếu None sẽ tự tạo mới)")
    model: Optional[str] = Field(default=None, description="Override LLM model name")
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Độ sáng tạo của mô hình (0.0 - 2.0)"
    )
    use_rag: bool = Field(default=False, description="Kích hoạt tìm kiếm RAG từ Knowledge Base")
    enable_query_expansion: Optional[bool] = Field(
        default=None, description="Bật/tắt kỹ thuật Query Expansion cho câu hỏi này (None = dùng cấu hình hệ thống)"
    )


class ChatResponse(BaseModel):
    """Response payload cho non-streaming chat."""

    session_id: str = Field(description="ID phiên làm việc")
    message: Message = Field(description="Tin nhắn phản hồi từ trợ lý AI")
    model: str = Field(description="Model LLM đã thực hiện phản hồi")
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Danh sách các trích dẫn tài liệu tham chiếu (nếu bật RAG)"
    )
    expanded_queries: Optional[List[str]] = Field(
        default=None, description="Danh sách các truy vấn mở rộng được sinh bởi Query Expansion"
    )


class RAGChatResponse(ChatResponse):
    """Response payload đặc thù cho RAG chat."""

    pass


class ChatHistoryResponse(BaseModel):
    """Response payload chứa lịch sử tin nhắn của một session."""

    session_id: str = Field(description="ID phiên làm việc")
    messages: List[Message] = Field(default_factory=list, description="Danh sách tin nhắn theo thứ tự thời gian")
    message_count: int = Field(default=0, description="Tổng số tin nhắn trong session")


class ClearHistoryResponse(BaseModel):
    """Response payload khi xóa lịch sử hội thoại."""

    session_id: str = Field(description="ID phiên làm việc")
    success: bool = Field(description="Kết quả xóa thành công hay không")
    message: str = Field(description="Thông báo chi tiết")


class StreamChunk(BaseModel):
    """Dữ liệu payload của từng event chunk trong SSE streaming."""

    session_id: Optional[str] = Field(default=None, description="ID phiên làm việc")
    content: str = Field(default="", description="Token nội dung đang sinh")
    is_done: bool = Field(default=False, description="Đánh dấu đã hoàn tất sinh câu trả lời")
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Danh sách nguồn tài liệu đính kèm ở chunk cuối"
    )
    expanded_queries: Optional[List[str]] = Field(
        default=None, description="Danh sách các truy vấn mở rộng được sinh bởi Query Expansion"
    )
    error: Optional[str] = Field(default=None, description="Thông báo lỗi nếu có gián đoạn")
