"""
Output Parser — Chuẩn hóa response từ LLM thành structured format.

Tách biệt logic parse output khỏi chain và service.
Thiết kế extensible để Phase 3 (RAG) có thể thêm `sources`, `documents`
mà không cần sửa ChatService hay ChatChain.

Cấu trúc output hiện tại:
    {
        "answer": "...",
        "metadata": {
            "model": "...",
            "elapsed_time": 1.23,
            "session_id": "..."
        }
    }

Cấu trúc output Phase 3 (RAG) — chỉ cần thêm field:
    {
        "answer": "...",
        "sources": ["doc1.pdf#p.3", "policy.docx#p.7"],
        "confidence": 0.87,
        "metadata": { ... }
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ===========================================================
# Response Data Models
# ===========================================================


class ResponseMetadata(BaseModel):
    """
    Metadata đính kèm theo mỗi AI response.

    Attributes:
        model:        Tên model đã generate response.
        elapsed_time: Thời gian xử lý (giây), làm tròn 3 chữ số.
        session_id:   Session ID của conversation.
        timestamp:    Thời điểm response được tạo (ISO 8601, UTC).
    """

    model: str = Field(..., description="Tên model LLM đã dùng.")
    elapsed_time: float = Field(
        default=0.0,
        description="Thời gian xử lý request (giây).",
    )
    session_id: str = Field(..., description="Session ID của conversation.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Thời điểm tạo response (ISO 8601 UTC).",
    )


class ParsedResponse(BaseModel):
    """
    Structured response từ AI sau khi qua OutputParser.

    Đây là format chuẩn cho tất cả non-streaming responses.
    Thiết kế để dễ mở rộng thêm field mà không breaking change.

    Attributes:
        answer:     Câu trả lời từ AI.
        metadata:   Thông tin kỹ thuật về request.
        sources:    (Phase 3) Danh sách tài liệu tham chiếu.
        confidence: (Phase 3) Độ tin cậy của câu trả lời (0.0 – 1.0).
        documents:  (Phase 3) Nội dung các chunks tài liệu đã dùng.
    """

    answer: str = Field(..., description="Câu trả lời từ AI assistant.")
    metadata: ResponseMetadata = Field(..., description="Metadata của response.")

    # --- Phase 3 RAG fields (None = chưa có) ---
    sources: Optional[list[str]] = Field(
        default=None,
        description="(RAG) Danh sách nguồn tài liệu tham chiếu.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="(RAG) Độ tin cậy của câu trả lời.",
    )
    documents: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="(RAG) Nội dung các document chunks đã dùng.",
    )


# ===========================================================
# Output Parser
# ===========================================================


class ChatOutputParser:
    """
    Parser chuẩn hóa raw LLM string output thành ParsedResponse.

    Không kế thừa LangChain BaseOutputParser vì chúng ta cần inject
    metadata (model, session_id, elapsed_time) từ context bên ngoài,
    điều mà LangChain BaseOutputParser không hỗ trợ trực tiếp.

    Cách sử dụng:
        parser = ChatOutputParser()
        result = parser.parse(
            raw_output="Xin chào!",
            model="qwen2.5:latest",
            session_id="abc-123",
            elapsed_time=1.23,
        )
        print(result.answer)    # "Xin chào!"
        print(result.metadata)  # ResponseMetadata(...)
    """

    def parse(
        self,
        raw_output: str,
        model: str,
        session_id: str,
        elapsed_time: float = 0.0,
        sources: Optional[list[str]] = None,
        confidence: Optional[float] = None,
        documents: Optional[list[dict[str, Any]]] = None,
    ) -> ParsedResponse:
        """
        Parse raw LLM string thành ParsedResponse.

        Args:
            raw_output:   Raw text từ LLM (sau StrOutputParser).
            model:        Tên model đã dùng.
            session_id:   Session ID của conversation.
            elapsed_time: Thời gian xử lý (giây).
            sources:      (RAG) Danh sách nguồn tài liệu.
            confidence:   (RAG) Độ tin cậy.
            documents:    (RAG) Document chunks.

        Returns:
            ParsedResponse: Structured response với đầy đủ metadata.
        """
        return ParsedResponse(
            answer=raw_output.strip(),
            metadata=ResponseMetadata(
                model=model,
                elapsed_time=round(elapsed_time, 3),
                session_id=session_id,
            ),
            sources=sources,
            confidence=confidence,
            documents=documents,
        )

    def parse_streaming_chunk(self, chunk: Any) -> str:
        """
        Xử lý một chunk trong streaming response.

        Hỗ trợ cả chuỗi văn bản (Ollama, OpenAI) và cấu trúc danh sách dict (Gemini).

        Args:
            chunk: Một token/chunk từ LLM stream.

        Returns:
            str: Chuỗi văn bản đã trích xuất từ chunk.
        """
        if chunk is None:
            return ""

        # Nếu là AIMessageChunk hoặc đối tượng có .content
        if hasattr(chunk, "content"):
            return self.parse_streaming_chunk(chunk.content)

        # Xử lý trường hợp Gemini trả về list các dict [{'type': 'text', 'text': ...}]
        if isinstance(chunk, list):
            parts = []
            for item in chunk:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        if isinstance(chunk, str):
            return chunk

        return str(chunk)
