"""
Chat Service — Business logic cho chat operations.

Orchestrate luồng chat:
    1. Validate / ensure session_id.
    2. Gọi ChatChain (singleton, tạo một lần).
    3. Parse output qua ChatOutputParser.
    4. Trả về response đúng format.

Thay đổi so với Phase 1:
    - Chain được tạo một lần (singleton property) thay vì mỗi request.
    - Response time được tracking và log rõ ràng.
    - Output được parse qua ChatOutputParser → có metadata đầy đủ.
    - Exception được catch từ LLMException hierarchy, re-raise với context.
    - ChatService KHÔNG xử lý prompt, config hay memory trực tiếp.

Service layer không biết về FastAPI hay Streamlit — chỉ business logic.
"""

import logging
import time
import uuid
from typing import AsyncGenerator, Optional

from app.chains.chat_chain import ChatChain
from app.chains.rag_chain import RAGChain
from app.core.exceptions import AppException, LLMException, RetrieverException
from app.core.output_parser import ChatOutputParser, ParsedResponse
from app.memory.memory_manager import MemoryManager
from app.models.chat import (
    ChatHistoryResponse,
    ChatResponse,
    ClearHistoryResponse,
    Message,
    MessageRole,
    RAGChatResponse,
    SourceDocument,
)
from app.rag.retriever_service import RetrieverService

logger = logging.getLogger(__name__)


class ChatService:
    """
    Business logic cho chat feature.

    Chịu trách nhiệm:
        - Quản lý session IDs.
        - Gọi ChatChain hoặc RAGChain (singleton).
        - Parse output qua ChatOutputParser.
        - Trả về response đúng format cho API layer.

    KHÔNG chịu trách nhiệm:
        - Định nghĩa prompt (→ app/prompts/).
        - Cấu hình LLM (→ app/config/settings.py).
        - Lưu trữ memory (→ app/memory/).
        - Retrieval logic (→ RetrieverService).

    Attributes:
        _memory_manager:   Singleton MemoryManager instance.
        _retriever_service: RetrieverService instance (injected, optional).
        _output_parser:    ChatOutputParser instance.
        _default_model:    Tên model LLM mặc định từ config.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        retriever_service: Optional[RetrieverService] = None,
    ) -> None:
        """
        Khởi tạo ChatService.

        Args:
            memory_manager:    MemoryManager instance (injected qua DI).
            retriever_service: RetrieverService instance (injected, required cho RAG).
        """
        from app.config.settings import get_settings

        self._memory_manager = memory_manager
        self._retriever_service = retriever_service
        self._output_parser = ChatOutputParser()

        settings = get_settings()
        self._default_model = settings.ollama_model
        self._default_temperature = settings.llm_temperature

        # Chain singletons — tạo một lần, dùng cho tất cả request
        self._chain: Optional[ChatChain] = None
        self._rag_chain: Optional[RAGChain] = None

        logger.info(
            "ChatService initialized (model=%s, backend=%s, rag=%s).",
            self._default_model,
            memory_manager.backend_type,
            "enabled" if retriever_service else "disabled",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_session_id(self, session_id: Optional[str]) -> str:
        """
        Đảm bảo session_id hợp lệ. Tạo mới UUID nếu None.

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
        Trả về ChatChain singleton.

        Nếu request có override model/temperature, tạo chain mới cho request đó.
        Nếu dùng default, trả về singleton để tránh khởi tạo lại mỗi request.

        Args:
            model:       Override model name (None = dùng default).
            temperature: Override temperature (None = dùng default).

        Returns:
            ChatChain: Chain đã được khởi tạo.
        """
        # Nếu có override, tạo chain mới cho request này
        if model or temperature is not None:
            logger.debug(
                "ChatService: creating custom chain (model=%s, temp=%s).",
                model,
                temperature,
            )
            return ChatChain(
                memory_manager=self._memory_manager,
                model=model,
                temperature=temperature,
            )

        # Singleton chain cho default config
        if self._chain is None:
            logger.info("ChatService: creating default chain singleton.")
            self._chain = ChatChain(memory_manager=self._memory_manager)

        return self._chain

    def _get_rag_chain(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> RAGChain:
        """
        Trả về RAGChain singleton.

        Tương tự _get_chain() nhưng cho RAG mode.

        Args:
            model:       Override model name.
            temperature: Override temperature.

        Returns:
            RAGChain: Chain đã được khởi tạo.
        """
        if model or temperature is not None:
            return RAGChain(
                memory_manager=self._memory_manager,
                model=model,
                temperature=temperature,
            )

        if self._rag_chain is None:
            logger.info("ChatService: creating default RAGChain singleton.")
            self._rag_chain = RAGChain(memory_manager=self._memory_manager)

        return self._rag_chain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            message:     Tin nhắn từ người dùng.
            session_id:  Session ID (None → tạo mới).
            model:       Override model.
            temperature: Override temperature.

        Returns:
            ChatResponse: Response đầy đủ từ AI.

        Raises:
            LLMException:  Nếu có lỗi từ LLM (Connection, Timeout, Generation).
            AppException:  Nếu có lỗi ứng dụng khác.
        """
        session_id = self._ensure_session_id(session_id)
        used_model = model or self._default_model

        logger.info(
            "ChatService.chat: session=%s | model=%s | message='%s...'",
            session_id,
            used_model,
            message[:60],
        )

        start_time = time.monotonic()
        try:
            chain = self._get_chain(model=model, temperature=temperature)
            raw_response = chain.invoke(input_text=message, session_id=session_id)

            elapsed = time.monotonic() - start_time
            logger.info(
                "ChatService.chat: session=%s | elapsed=%.3fs | response_chars=%d",
                session_id,
                elapsed,
                len(raw_response),
            )

            # Parse thành ParsedResponse với đầy đủ metadata
            parsed: ParsedResponse = self._output_parser.parse(
                raw_output=raw_response,
                model=used_model,
                session_id=session_id,
                elapsed_time=elapsed,
            )

            return ChatResponse(
                session_id=session_id,
                message=Message(role=MessageRole.ASSISTANT, content=parsed.answer),
                model=used_model,
            )

        except LLMException:
            # Re-raise LLM exceptions as-is (đã có context đầy đủ)
            elapsed = time.monotonic() - start_time
            logger.error(
                "ChatService.chat ERROR: session=%s | elapsed=%.3fs",
                session_id,
                elapsed,
                exc_info=True,
            )
            raise
        except AppException:
            raise
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(
                "ChatService.chat UNEXPECTED ERROR: session=%s | elapsed=%.3fs | %s",
                session_id,
                elapsed,
                str(e),
                exc_info=True,
            )
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
            message:     Tin nhắn từ người dùng.
            session_id:  Session ID (None → tạo mới).
            model:       Override model.
            temperature: Override temperature.

        Returns:
            tuple[str, AsyncGenerator[str, None]]:
                - session_id: Session ID đã dùng.
                - generator:  Async generator yield từng token.
        """
        session_id = self._ensure_session_id(session_id)

        logger.info(
            "ChatService.astream_chat: session=%s | model=%s | message='%s...'",
            session_id,
            model or self._default_model,
            message[:60],
        )

        start_time = time.monotonic()

        async def _timed_stream() -> AsyncGenerator[str, None]:
            """Wrapper stream có log elapsed time khi kết thúc."""
            chain = self._get_chain(model=model, temperature=temperature)
            try:
                async for chunk in chain.astream(input_text=message, session_id=session_id):
                    yield self._output_parser.parse_streaming_chunk(chunk)
            finally:
                elapsed = time.monotonic() - start_time
                logger.info(
                    "ChatService.astream_chat: session=%s | stream_elapsed=%.3fs",
                    session_id,
                    elapsed,
                )

        return session_id, _timed_stream()

    def get_history(self, session_id: str) -> ChatHistoryResponse:
        """
        Lấy conversation history của một session.

        Args:
            session_id: Session ID.

        Returns:
            ChatHistoryResponse: Danh sách messages trong session.
        """
        logger.debug("ChatService.get_history: session=%s", session_id)
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
        logger.info("ChatService.clear_history: session=%s", session_id)
        success = self._memory_manager.clear_session(session_id)
        return ClearHistoryResponse(
            session_id=session_id,
            success=success,
            message="History cleared successfully." if success else "Session not found.",
        )
