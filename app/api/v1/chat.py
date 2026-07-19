"""
Chat API Endpoints — FastAPI router cho tất cả chat operations.

Endpoints:
  POST   /chat                     — Gửi tin nhắn (streaming SSE)
  POST   /chat/sync                — Gửi tin nhắn (non-streaming)
  GET    /chat/history/{session_id} — Lấy lịch sử chat
  DELETE /chat/history/{session_id} — Xóa lịch sử chat
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.models.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ClearHistoryResponse,
    StreamChunk,
)
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    summary="Gửi tin nhắn (Streaming)",
    description="Gửi tin nhắn đến AI và nhận response dưới dạng Server-Sent Events (SSE).",
)
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Streaming chat endpoint (Server-Sent Events).

    Client nhận response từng token một thay vì đợi toàn bộ.

    Args:
        request: ChatRequest với message và session info.
        chat_service: Injected ChatService instance.

    Returns:
        StreamingResponse: SSE stream với từng chunk.

    Raises:
        HTTPException 500: Nếu có lỗi trong quá trình xử lý.
    """
    logger.info(
        "POST /chat (streaming): session=%s, message=%s...",
        request.session_id,
        request.message[:50],
    )

    async def event_generator():
        """Generator tạo SSE events từ LLM stream."""
        session_id = None
        try:
            session_id, stream = await chat_service.astream_chat(
                message=request.message,
                session_id=request.session_id,
                model=request.model,
                temperature=request.temperature,
            )

            # Gửi session_id ngay lập tức (chunk đầu tiên)
            init_chunk = StreamChunk(session_id=session_id, content="", is_done=False)
            yield f"data: {init_chunk.model_dump_json()}\n\n"

            # Stream từng token
            async for token in stream:
                if token:
                    chunk = StreamChunk(
                        session_id=session_id,
                        content=token,
                        is_done=False,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            # Chunk cuối báo hiệu kết thúc
            done_chunk = StreamChunk(session_id=session_id, content="", is_done=True)
            yield f"data: {done_chunk.model_dump_json()}\n\n"

        except Exception as e:
            logger.error("Streaming error: %s", str(e), exc_info=True)
            error_data = json.dumps({"error": str(e), "session_id": session_id})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Tắt buffering cho nginx
        },
    )


@router.post(
    "/sync",
    response_model=ChatResponse,
    summary="Gửi tin nhắn (Non-streaming)",
    description="Gửi tin nhắn và chờ response hoàn chỉnh. Dùng cho testing và clients không hỗ trợ SSE.",
)
async def chat_sync(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Non-streaming chat endpoint.

    Chờ đến khi LLM generate xong toàn bộ response.

    Args:
        request: ChatRequest với message và session info.
        chat_service: Injected ChatService instance.

    Returns:
        ChatResponse: Response đầy đủ từ AI.

    Raises:
        HTTPException 500: Nếu có lỗi LLM.
    """
    logger.info(
        "POST /chat/sync: session=%s, message=%s...",
        request.session_id,
        request.message[:50],
    )

    try:
        return chat_service.chat(
            message=request.message,
            session_id=request.session_id,
            model=request.model,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error("Sync chat error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}",
        ) from e


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Lấy lịch sử chat",
)
async def get_chat_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatHistoryResponse:
    """
    Lấy toàn bộ conversation history của một session.

    Args:
        session_id: UUID của conversation session.
        chat_service: Injected ChatService.

    Returns:
        ChatHistoryResponse: Danh sách messages.
    """
    logger.info("GET /chat/history/%s", session_id)
    return chat_service.get_history(session_id)


@router.delete(
    "/history/{session_id}",
    response_model=ClearHistoryResponse,
    summary="Xóa lịch sử chat",
)
async def clear_chat_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> ClearHistoryResponse:
    """
    Xóa toàn bộ conversation history của một session.

    Args:
        session_id: UUID của conversation session.
        chat_service: Injected ChatService.

    Returns:
        ClearHistoryResponse: Kết quả thao tác.
    """
    logger.info("DELETE /chat/history/%s", session_id)
    return chat_service.clear_history(session_id)
