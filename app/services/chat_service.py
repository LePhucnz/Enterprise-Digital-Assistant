"""
Chat Service — Business logic cho chat operations.

Orchestrate luồng chat:
  1. Validate request.
  2. Tạo/lấy session.
  3. Gọi ChatChain.
  4. Trả về response.

Service layer không biết về FastAPI hay Streamlit — chỉ business logic.
"""

import logging
import uuid
from typing import AsyncGenerator, Optional

from app.chains.chat_chain import ChatChain
from app.memory.memory_manager import MemoryManager
from app.models.chat import (
    ChatHistoryResponse,
    ChatResponse,
    ClearHistoryResponse,
    Message,
    MessageRole,
)

logger = logging.getLogger(__name__)


class ChatService:
    """
    Business logic cho chat feature.

    Chịu trách nhiệm:
    - Quản lý session IDs.
    - Khởi tạo ChatChain đúng cấu hình.
    - Orchestrate invoke/streaming.
    - Trả về response đúng format.

    Attributes:
        _memory_manager: Singleton MemoryManager instance.
        _default_model: Tên model LLM mặc định.
        _default_temperature: Temperature mặc định.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        """
        Khởi tạo ChatService.

        Args:
            memory_manager: MemoryManager instance (injected).
        """
        from app.config.settings import get_settings

        self._memory_manager = memory_manager
        settings = get_settings()
        self._default_model = settings.ollama_model
        self._default_temperature = settings.llm_temperature
        logger.info("ChatService initialized.")

    def _ensure_session_id(self, session_id: Optional[str]) -> str:
        """
        Đảm bảo session_id hợp lệ. Tạo mới nếu None.

        Args:
            session_id: Session ID từ request (có thể None).

        Returns:
            str: Session ID hợp lệ.
        """
        if not session_id:
            new_id = str(uuid.uuid4())
            logger.info("Generated new session_id: %s", new_id)
            return new_id
        return session_id

    def _get_chain(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatChain:
        """
        Tạo ChatChain với cấu hình được yêu cầu.

        Args:
            model: Override model name.
            temperature: Override temperature.

        Returns:
            ChatChain: Chain đã được khởi tạo.
        """
        return ChatChain(
            memory_manager=self._memory_manager,
            model=model or self._default_model,
            temperature=temperature,
        )

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """
        Xử lý chat request (non-streaming).

        Args:
            message: Tin nhắn từ người dùng.
            session_id: Session ID (None → tạo mới).
            model: Override model.
            temperature: Override temperature.

        Returns:
            ChatResponse: Response đầy đủ từ AI.

        Raises:
            RuntimeError: Nếu có lỗi từ LLM.
        """
        session_id = self._ensure_session_id(session_id)
        used_model = model or self._default_model

        logger.info(
            "ChatService.chat: session=%s, model=%s, message=%s...",
            session_id,
            used_model,
            message[:50],
        )

        try:
            chain = self._get_chain(model=model, temperature=temperature)
            response_text = chain.invoke(input_text=message, session_id=session_id)

            return ChatResponse(
                session_id=session_id,
                message=Message(role=MessageRole.ASSISTANT, content=response_text),
                model=used_model,
            )
        except Exception as e:
            logger.error("ChatService.chat error: %s", str(e), exc_info=True)
            raise RuntimeError(f"Chat failed: {e}") from e

    async def astream_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, AsyncGenerator[str, None]]:
        """
        Xử lý chat request với streaming response.

        Args:
            message: Tin nhắn từ người dùng.
            session_id: Session ID (None → tạo mới).
            model: Override model.
            temperature: Override temperature.

        Returns:
            tuple[str, AsyncGenerator[str, None]]:
                - session_id: Session ID đã dùng.
                - generator: Async generator yield từng token.
        """
        session_id = self._ensure_session_id(session_id)

        logger.info(
            "ChatService.astream_chat: session=%s, message=%s...",
            session_id,
            message[:50],
        )

        chain = self._get_chain(model=model, temperature=temperature)
        generator = chain.astream(input_text=message, session_id=session_id)
        return session_id, generator

    def get_history(self, session_id: str) -> ChatHistoryResponse:
        """
        Lấy conversation history của một session.

        Args:
            session_id: Session ID.

        Returns:
            ChatHistoryResponse: Danh sách messages trong session.
        """
        raw_messages = self._memory_manager.get_messages(session_id)

        messages = [
            Message(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            )
            for m in raw_messages
        ]

        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            message_count=len(messages),
        )

    def clear_history(self, session_id: str) -> ClearHistoryResponse:
        """
        Xóa conversation history của một session.

        Args:
            session_id: Session ID.

        Returns:
            ClearHistoryResponse: Kết quả thao tác.
        """
        success = self._memory_manager.clear_session(session_id)
        return ClearHistoryResponse(
            session_id=session_id,
            success=success,
            message="History cleared successfully." if success else "Session not found.",
        )
