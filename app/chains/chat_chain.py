"""
Chat Chain — Conversation chain với memory và streaming support.

Implement BaseChain cho basic chat flow:
  User Message → PromptTemplate → LLM → Response

Sử dụng RunnableWithMessageHistory để tích hợp chat history
theo cách khuyến nghị của LangChain v0.3+.
"""

import logging
from typing import AsyncGenerator, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from app.config.settings import get_settings
from app.core.base_chain import BaseChain
from app.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# System prompt mặc định — sẽ được chuyển vào prompts/ ở Bước 2
_DEFAULT_SYSTEM_PROMPT = """Bạn là Enterprise Digital Assistant — trợ lý AI thông minh cho doanh nghiệp.

Vai trò của bạn:
- Trả lời câu hỏi chuyên nghiệp, chính xác và ngắn gọn.
- Hỗ trợ phân tích tài liệu, tóm tắt thông tin và giải thích khái niệm phức tạp.
- Sử dụng ngôn ngữ phù hợp với ngữ cảnh (tiếng Việt hoặc tiếng Anh theo câu hỏi).

Quy tắc:
- Luôn trả lời bằng ngôn ngữ người dùng đang dùng.
- Nếu không chắc, hãy thừa nhận giới hạn của mình.
- Không bịa đặt thông tin không có trong context."""


class ChatChain(BaseChain):
    """
    Conversation chain cơ bản với memory và streaming.

    Sử dụng RunnableWithMessageHistory của LangChain v0.3+
    để quản lý conversation history theo session_id.

    Luồng xử lý:
        UserMessage
            → ChatPromptTemplate (System + History + Human)
            → ChatOllama
            → StrOutputParser
            → str response
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
            model: Override model name. Dùng settings mặc định nếu None.
            temperature: Override temperature. Dùng settings mặc định nếu None.
            system_prompt: Custom system prompt. Dùng default nếu None.
        """
        settings = get_settings()

        # Khởi tạo LLM
        llm = ChatOllama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )

        # Gọi parent constructor
        super().__init__(llm=llm)

        self._memory_manager = memory_manager
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

        # Xây dựng chain
        self._chain = self._build_chain()
        logger.info(
            "ChatChain initialized (model=%s, temp=%.2f)",
            model or settings.ollama_model,
            temperature if temperature is not None else settings.llm_temperature,
        )

    def _build_chain(self) -> RunnableWithMessageHistory:
        """
        Xây dựng LangChain runnable chain với history support.

        Cấu trúc prompt:
            - SystemMessage: Định nghĩa vai trò và hành vi của AI.
            - MessagesPlaceholder("history"): Chat history từ memory.
            - HumanMessage: Tin nhắn hiện tại từ người dùng.

        Returns:
            RunnableWithMessageHistory: Chain có thể gọi invoke/astream.
        """
        # Prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # Core chain: prompt | llm | parser
        core_chain = prompt | self.llm | StrOutputParser()

        # Wrap với history management
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
            **kwargs: Tham số bổ sung.

        Returns:
            str: Response đầy đủ từ LLM.

        Raises:
            Exception: Nếu có lỗi từ LLM hoặc network.
        """
        try:
            logger.debug("ChatChain.invoke: session=%s, input=%s...", session_id, input_text[:50])
            config = {"configurable": {"session_id": session_id}}
            response = self._chain.invoke({"input": input_text}, config=config)
            logger.debug("ChatChain.invoke: response length=%d chars", len(response))
            return response
        except Exception as e:
            logger.error("ChatChain.invoke error: %s", str(e), exc_info=True)
            raise

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
            **kwargs: Tham số bổ sung.

        Yields:
            str: Từng token trong response.

        Raises:
            Exception: Nếu có lỗi từ LLM hoặc network.
        """
        try:
            logger.debug("ChatChain.astream: session=%s, input=%s...", session_id, input_text[:50])
            config = {"configurable": {"session_id": session_id}}
            async for chunk in self._chain.astream({"input": input_text}, config=config):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error("ChatChain.astream error: %s", str(e), exc_info=True)
            raise
