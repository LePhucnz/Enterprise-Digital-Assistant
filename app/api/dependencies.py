"""
FastAPI Dependencies — Dependency Injection cho toàn bộ API.

Cung cấp các singleton instances thông qua FastAPI's dependency system.
Đảm bảo MemoryManager và ChatService được khởi tạo một lần và share
trong toàn bộ vòng đời ứng dụng.

Thay đổi Phase 2:
    - `get_memory_backend()`: Tạo `InMemoryBackend` (có thể swap sang Redis).
    - `get_memory_manager()`: Inject backend vào MemoryManager.
    - `get_chat_service()`: Không đổi interface.

Để swap sang Redis backend trong tương lai:
    1. Tạo `RedisBackend(BaseMemoryBackend)` trong `app/memory/backends/redis.py`.
    2. Thay `InMemoryBackend()` bằng `RedisBackend(url=settings.redis_url)` ở đây.
    3. Không cần sửa MemoryManager, ChatService hay API endpoints.
"""

import logging
from functools import lru_cache

from app.config.settings import get_settings
from app.memory.memory_manager import InMemoryBackend, MemoryManager
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_memory_backend() -> InMemoryBackend:
    """
    Trả về singleton MemoryBackend instance.

    Hiện tại dùng InMemoryBackend.
    Để đổi sang Redis, chỉ cần sửa hàm này — không đổi gì khác.

    Returns:
        InMemoryBackend: Singleton memory storage backend.
    """
    settings = get_settings()
    logger.info("Creating InMemoryBackend singleton (max_history=%d).", settings.memory_max_history)
    return InMemoryBackend(max_history=settings.memory_max_history)


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """
    Trả về singleton MemoryManager instance.

    Inject backend vào MemoryManager theo Dependency Injection.

    Returns:
        MemoryManager: Singleton instance.
    """
    logger.info("Creating MemoryManager singleton.")
    backend = get_memory_backend()
    return MemoryManager(backend=backend)


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
