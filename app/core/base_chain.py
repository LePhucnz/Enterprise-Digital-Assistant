"""
Base Chain — Abstract base class cho tất cả LangChain chains.

Mọi chain trong hệ thống phải kế thừa từ class này để đảm bảo
interface nhất quán và dễ mở rộng.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class BaseChain(ABC):
    """
    Abstract base class cho tất cả LangChain chains.

    Định nghĩa interface chung mà mọi chain phải implement.
    Tuân theo Interface Segregation Principle của SOLID.

    Attributes:
        llm: LangChain chat model instance.
        memory: Optional memory instance cho conversation history.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[Any] = None,
    ) -> None:
        """
        Khởi tạo BaseChain.

        Args:
            llm: LangChain chat model (e.g., ChatOllama).
            memory: Optional memory instance.
        """
        self.llm = llm
        self.memory = memory
        logger.debug(
            "Initialized %s with model=%s, memory=%s",
            self.__class__.__name__,
            getattr(llm, "model", "unknown"),
            type(memory).__name__ if memory else "None",
        )

    @abstractmethod
    def invoke(self, input_text: str, **kwargs: Any) -> str:
        """
        Gọi chain với input và trả về response.

        Args:
            input_text: Câu hỏi hoặc tin nhắn từ người dùng.
            **kwargs: Các tham số bổ sung.

        Returns:
            str: Response từ LLM.
        """
        raise NotImplementedError

    @abstractmethod
    async def astream(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Gọi chain với streaming response (async generator).

        Args:
            input_text: Câu hỏi hoặc tin nhắn từ người dùng.
            **kwargs: Các tham số bổ sung.

        Yields:
            str: Từng token/chunk trong response.
        """
        raise NotImplementedError
        # Đảm bảo Python nhận ra đây là generator
        yield  # type: ignore[misc]

    def clear_memory(self) -> None:
        """
        Xóa toàn bộ conversation history trong memory.
        Safe to call ngay cả khi không có memory.
        """
        if self.memory is not None:
            self.memory.clear()
            logger.info("%s: memory cleared.", self.__class__.__name__)
