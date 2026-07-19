"""
Chat Chain — Conversation chain với memory và streaming support.

Implement BaseChain cho basic chat flow theo chuẩn LCEL:
    Input → PromptTemplate → LLM → StrOutputParser → str response

Thay đổi so với Phase 1:
    - Prompt được import từ `app.prompts` (không còn hard-code ở đây).
    - Sử dụng `top_p` và `timeout` từ settings.
    - Exception được map thành `LLMException` hierarchy (app/core/exceptions.py).
    - Không thay đổi interface public: `invoke()` và `astream()` giữ nguyên.

Sử dụng RunnableWithMessageHistory để tích hợp chat history
theo cách khuyến nghị của LangChain v0.3+.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from app.config.settings import get_settings
from app.core.base_chain import BaseChain
from app.core.exceptions import (
    LLMConnectionError,
    LLMGenerationError,
    LLMTimeoutError,
)
from app.memory.memory_manager import MemoryManager
from app.prompts import get_chat_prompt_template

logger = logging.getLogger(__name__)


class ChatChain(BaseChain):
    """
    Conversation chain cơ bản với memory và streaming.

    Sử dụng RunnableWithMessageHistory của LangChain v0.3+
    để quản lý conversation history theo session_id.

    Luồng xử lý (LCEL):
        {input}
            → ChatPromptTemplate (System + History + Human)
            → ChatOllama
            → StrOutputParser
            → str response

    Prompt được lấy từ `app.prompts.get_chat_prompt_template()` —
    không hard-code trong class này.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Khởi tạo ChatChain.

        Args:
            memory_manager: MemoryManager instance để quản lý sessions.
            model:          Override model name. Dùng settings mặc định nếu None.
            temperature:    Override temperature. Dùng settings mặc định nếu None.
            system_prompt:  Không dùng nữa — kept for backward compatibility.
                            Prompt hiện được quản lý bởi `app.prompts`.
        """
        settings = get_settings()

        _model = model or settings.ollama_model
        _temperature = temperature if temperature is not None else settings.llm_temperature

        # Khởi tạo LLM với đầy đủ config từ settings
        llm = ChatOllama(
            model=_model,
            base_url=settings.ollama_base_url,
            temperature=_temperature,
            top_p=settings.llm_top_p,
            num_predict=settings.llm_max_tokens,
            num_ctx=settings.llm_context_window,
        )

        # Gọi parent constructor
        super().__init__(llm=llm)

        self._memory_manager = memory_manager
        self._model_name = _model
        self._timeout = settings.llm_timeout

        # Xây dựng chain
        self._chain = self._build_chain()

        logger.info(
            "ChatChain initialized (model=%s, temp=%.2f, top_p=%.2f, timeout=%ds).",
            _model,
            _temperature,
            settings.llm_top_p,
            settings.llm_timeout,
        )

    def _build_chain(self) -> RunnableWithMessageHistory:
        """
        Xây dựng LCEL chain: Prompt | LLM | Parser với history support.

        Prompt được lấy từ `app.prompts.get_chat_prompt_template()`.
        Không hard-code prompt trong method này.

        Returns:
            RunnableWithMessageHistory: Chain hoàn chỉnh có thể gọi invoke/astream.
        """
        # Lấy prompt template từ module quản lý prompt
        prompt = get_chat_prompt_template()

        # Core LCEL chain: prompt | llm | output parser
        core_chain = prompt | self.llm | StrOutputParser()

        # Wrap với RunnableWithMessageHistory để quản lý session history
        chain_with_history = RunnableWithMessageHistory(
            core_chain,
            get_session_history=self._memory_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_history

    def invoke(self, input_text: str, session_id: str = "default", **kwargs) -> str:
        """
        Gọi chain và trả về response đầy đủ (non-streaming).

        Args:
            input_text: Tin nhắn từ người dùng.
            session_id: UUID của conversation session.
            **kwargs:   Tham số bổ sung (ignored, kept for interface compat).

        Returns:
            str: Response đầy đủ từ LLM.

        Raises:
            LLMConnectionError:  Nếu không kết nối được Ollama.
            LLMTimeoutError:     Nếu request vượt quá timeout.
            LLMGenerationError:  Nếu LLM lỗi trong quá trình generate.
        """
        try:
            logger.debug(
                "ChatChain.invoke: session=%s, input='%s...'",
                session_id,
                input_text[:60],
            )
            config = {"configurable": {"session_id": session_id}}
            response = self._chain.invoke({"input": input_text}, config=config)
            logger.debug(
                "ChatChain.invoke: response length=%d chars.",
                len(response),
            )
            return response

        except ConnectionError as e:
            settings = get_settings()
            raise LLMConnectionError(
                base_url=settings.ollama_base_url,
                detail=str(e),
            ) from e
        except TimeoutError as e:
            raise LLMTimeoutError(
                timeout_seconds=self._timeout,
                detail=str(e),
            ) from e
        except Exception as e:
            # Bắt các lỗi khác từ httpx, ollama, langchain
            error_str = str(e).lower()
            if "connect" in error_str or "connection" in error_str:
                settings = get_settings()
                raise LLMConnectionError(
                    base_url=settings.ollama_base_url,
                    detail=str(e),
                ) from e
            if "timeout" in error_str or "timed out" in error_str:
                raise LLMTimeoutError(
                    timeout_seconds=self._timeout,
                    detail=str(e),
                ) from e
            logger.error("ChatChain.invoke error: %s", str(e), exc_info=True)
            raise LLMGenerationError(
                model=self._model_name,
                detail=str(e),
            ) from e

    async def astream(
        self,
        input_text: str,
        session_id: str = "default",
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Gọi chain với streaming response (async generator).

        Args:
            input_text: Tin nhắn từ người dùng.
            session_id: UUID của conversation session.
            **kwargs:   Tham số bổ sung.

        Yields:
            str: Từng token trong response.

        Raises:
            LLMConnectionError:  Nếu không kết nối được Ollama.
            LLMTimeoutError:     Nếu stream bị timeout.
            LLMGenerationError:  Nếu LLM lỗi trong quá trình generate.
        """
        try:
            logger.debug(
                "ChatChain.astream: session=%s, input='%s...'",
                session_id,
                input_text[:60],
            )
            config = {"configurable": {"session_id": session_id}}
            async for chunk in self._chain.astream({"input": input_text}, config=config):
                if chunk:
                    yield chunk

        except ConnectionError as e:
            settings = get_settings()
            raise LLMConnectionError(
                base_url=settings.ollama_base_url,
                detail=str(e),
            ) from e
        except TimeoutError as e:
            raise LLMTimeoutError(
                timeout_seconds=self._timeout,
                detail=str(e),
            ) from e
        except Exception as e:
            error_str = str(e).lower()
            if "connect" in error_str or "connection" in error_str:
                settings = get_settings()
                raise LLMConnectionError(
                    base_url=settings.ollama_base_url,
                    detail=str(e),
                ) from e
            if "timeout" in error_str or "timed out" in error_str:
                raise LLMTimeoutError(
                    timeout_seconds=self._timeout,
                    detail=str(e),
                ) from e
            logger.error("ChatChain.astream error: %s", str(e), exc_info=True)
            raise LLMGenerationError(
                model=self._model_name,
                detail=str(e),
            ) from e
