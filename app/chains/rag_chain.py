"""
RAG Chain — LangChain chain cho Retrieval-Augmented Generation.

Kế thừa BaseChain. KHÔNG sửa ChatChain.

Pipeline LCEL:
    {input, context, history}
        → get_rag_prompt_template() (System+Context+History+Human)
        → ChatOllama
        → StrOutputParser
        → str response

Khác biệt với ChatChain:
    - Nhận thêm tham số `context` (retrieved documents) trong invoke/astream.
    - Dùng get_rag_prompt_template() thay vì get_chat_prompt_template().
    - Context được inject vào prompt thông qua RunnablePassthrough.

Memory:
    - Vẫn dùng RunnableWithMessageHistory để giữ conversation history.
    - RAG chain nhớ câu hỏi trước — user không cần lặp context.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from app.config.settings import get_settings
from app.core.base_chain import BaseChain
from app.core.exceptions import (
    LLMConnectionError,
    LLMGenerationError,
    LLMTimeoutError,
    RAGException,
)
from app.memory.memory_manager import MemoryManager
from app.prompts import get_rag_prompt_template

logger = logging.getLogger(__name__)


class RAGChain(BaseChain):
    """
    RAG conversation chain với memory và streaming.

    Sử dụng get_rag_prompt_template() với {context} được inject
    từ RetrieverService (không phải từ retriever trực tiếp trong chain).

    Thiết kế: Context được inject ở ChatService level, không trong chain.
    Điều này giúp:
    1. Logging retrieval time tách biệt khỏi LLM time.
    2. Dễ fallback về normal chat khi không có context.
    3. RAGChain không phụ thuộc vào VectorStoreService.

    LCEL Flow:
        {input: str, context: str}
            → RAG PromptTemplate (System[context] + History + Human[input])
            → ChatOllama
            → StrOutputParser
            → str

    Attributes:
        _memory_manager: MemoryManager instance.
        _model_name:     Tên LLM model đang dùng.
        _timeout:        Timeout (giây) cho LLM request.
        _chain:          RunnableWithMessageHistory chain instance.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Khởi tạo RAGChain.

        Args:
            memory_manager: MemoryManager instance (injected).
            model:          Override model name. Dùng settings nếu None.
            temperature:    Override temperature. Dùng settings nếu None.
        """
        settings = get_settings()

        _model = model or settings.ollama_model
        _temperature = temperature if temperature is not None else settings.llm_temperature

        # Khởi tạo LLM — cùng config với ChatChain
        llm = ChatOllama(
            model=_model,
            base_url=settings.ollama_base_url,
            temperature=_temperature,
            top_p=settings.llm_top_p,
            num_predict=settings.llm_max_tokens,
            num_ctx=settings.llm_context_window,
        )

        super().__init__(llm=llm)

        self._memory_manager = memory_manager
        self._model_name = _model
        self._timeout = settings.llm_timeout

        # Build chain
        self._chain = self._build_chain()

        logger.info(
            "RAGChain initialized (model=%s, temp=%.2f, top_p=%.2f, timeout=%ds).",
            _model,
            _temperature,
            settings.llm_top_p,
            settings.llm_timeout,
        )

    def _build_chain(self) -> RunnableWithMessageHistory:
        """
        Xây dựng LCEL RAG chain với history support.

        Input keys: {"input": str, "context": str}
        → RAG Prompt (context injected vào system prompt)
        → LLM
        → StrOutputParser

        Wrap với RunnableWithMessageHistory để quản lý history theo session_id.

        Returns:
            RunnableWithMessageHistory: Hoàn chỉnh, có thể invoke/astream.
        """
        prompt = get_rag_prompt_template()

        # Core chain: prompt template nhận cả input lẫn context
        # context được pass qua input dict → template inject vào {context} trong system prompt
        core_chain = prompt | self.llm | StrOutputParser()

        # Wrap với history — history_messages_key="history" khớp với MessagesPlaceholder
        chain_with_history = RunnableWithMessageHistory(
            core_chain,
            get_session_history=self._memory_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_history

    def invoke(
        self,
        input_text: str,
        context: str = "",
        session_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Gọi RAG chain và trả về response đầy đủ (non-streaming).

        Args:
            input_text: Câu hỏi từ người dùng.
            context:    Context string từ RetrieverService (formatted chunks).
            session_id: UUID của conversation session.
            **kwargs:   Ignored (kept for interface compatibility với BaseChain).

        Returns:
            str: Response đầy đủ từ LLM.

        Raises:
            LLMConnectionError:  Nếu không kết nối được Ollama.
            LLMTimeoutError:     Nếu request vượt quá timeout.
            LLMGenerationError:  Nếu LLM lỗi trong quá trình generate.
            RAGException:        Nếu lỗi khác trong RAG pipeline.
        """
        try:
            logger.debug(
                "RAGChain.invoke: session=%s, input='%s...', context_len=%d.",
                session_id,
                input_text[:60],
                len(context),
            )
            config = {"configurable": {"session_id": session_id}}
            response = self._chain.invoke(
                {"input": input_text, "context": context},
                config=config,
            )
            logger.debug("RAGChain.invoke: response length=%d chars.", len(response))
            return response

        except ConnectionError as e:
            settings = get_settings()
            raise LLMConnectionError(base_url=settings.ollama_base_url, detail=str(e)) from e
        except TimeoutError as e:
            raise LLMTimeoutError(timeout_seconds=self._timeout, detail=str(e)) from e
        except Exception as e:
            error_str = str(e).lower()
            if "connect" in error_str or "connection" in error_str:
                settings = get_settings()
                raise LLMConnectionError(base_url=settings.ollama_base_url, detail=str(e)) from e
            if "timeout" in error_str or "timed out" in error_str:
                raise LLMTimeoutError(timeout_seconds=self._timeout, detail=str(e)) from e
            logger.error("RAGChain.invoke error: %s", str(e), exc_info=True)
            raise LLMGenerationError(model=self._model_name, detail=str(e)) from e

    async def astream(
        self,
        input_text: str,
        context: str = "",
        session_id: str = "default",
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Gọi RAG chain với streaming response.

        Args:
            input_text: Câu hỏi từ người dùng.
            context:    Context string từ RetrieverService.
            session_id: UUID của conversation session.
            **kwargs:   Ignored.

        Yields:
            str: Từng token trong response.

        Raises:
            LLMConnectionError:  Nếu không kết nối được Ollama.
            LLMTimeoutError:     Nếu stream bị timeout.
            LLMGenerationError:  Nếu LLM lỗi trong quá trình generate.
        """
        try:
            logger.debug(
                "RAGChain.astream: session=%s, input='%s...', context_len=%d.",
                session_id,
                input_text[:60],
                len(context),
            )
            config = {"configurable": {"session_id": session_id}}
            async for chunk in self._chain.astream(
                {"input": input_text, "context": context},
                config=config,
            ):
                if chunk:
                    yield chunk

        except ConnectionError as e:
            settings = get_settings()
            raise LLMConnectionError(base_url=settings.ollama_base_url, detail=str(e)) from e
        except TimeoutError as e:
            raise LLMTimeoutError(timeout_seconds=self._timeout, detail=str(e)) from e
        except Exception as e:
            error_str = str(e).lower()
            if "connect" in error_str or "connection" in error_str:
                settings = get_settings()
                raise LLMConnectionError(base_url=settings.ollama_base_url, detail=str(e)) from e
            if "timeout" in error_str or "timed out" in error_str:
                raise LLMTimeoutError(timeout_seconds=self._timeout, detail=str(e)) from e
            logger.error("RAGChain.astream error: %s", str(e), exc_info=True)
            raise LLMGenerationError(model=self._model_name, detail=str(e)) from e
