"""
FastAPI Dependencies — Dependency Injection cho toàn bộ API.

Cung cấp các singleton instances thông qua FastAPI's dependency system.
Đảm bảo MemoryManager và ChatService được khởi tạo một lần và share.
"""

import logging
from functools import lru_cache

from app.memory.memory_manager import MemoryManager
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """
    Trả về singleton MemoryManager instance.

    Dùng @lru_cache để đảm bảo chỉ tạo một instance duy nhất
    trong toàn bộ vòng đời ứng dụng.

    Returns:
        MemoryManager: Singleton instance.
    """
    logger.info("Creating MemoryManager singleton.")
    return MemoryManager()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """
    Trả về singleton ChatService instance.

    Returns:
        ChatService: Singleton instance.
    """
    logger.info("Creating ChatService singleton.")
    memory_manager = get_memory_manager()
    return ChatService(memory_manager=memory_manager)
