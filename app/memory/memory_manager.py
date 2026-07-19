"""
Memory Manager — Quản lý conversation memory theo session.

Kiến trúc Strategy Pattern:
    MemoryManager không biết backend lưu trữ là gì.
    Nó chỉ làm việc thông qua interface `BaseMemoryBackend`.

    Để thay đổi storage backend (Redis, SQLite, PostgreSQL...),
    chỉ cần:
        1. Tạo class mới implement `BaseMemoryBackend`.
        2. Inject vào `MemoryManager` qua DI.
        3. Không cần sửa `MemoryManager`, `ChatService` hay `ChatChain`.

Backends có sẵn:
    - InMemoryBackend: Lưu trong RAM (default, dùng cho dev/test).

Backends có thể thêm (Phase 3+):
    - RedisBackend:    Lưu trong Redis (production, multi-process).
    - SQLiteBackend:   Lưu trong file SQLite (persistent single-node).
    - PostgresBackend: Lưu trong PostgreSQL (enterprise, HA).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


# ===========================================================
# Abstract Backend Interface (Strategy Pattern)
# ===========================================================


class BaseMemoryBackend(ABC):
    """
    Abstract interface cho tất cả memory storage backends.

    Mọi backend phải implement 3 phương thức cốt lõi:
        - get_session_history: Lấy/tạo history cho session.
        - clear_session:       Xóa messages của session.
        - delete_session:      Xóa hoàn toàn session.

    Tuân theo Interface Segregation Principle — interface nhỏ gọn,
    chỉ những gì thực sự cần thiết.
    """

    @abstractmethod
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Lấy hoặc tạo mới ChatMessageHistory cho session.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            BaseChatMessageHistory: History object cho session.
        """
        raise NotImplementedError

    @abstractmethod
    def clear_session(self, session_id: str) -> bool:
        """
        Xóa tất cả messages của session (giữ lại session ID).

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            bool: True nếu session tồn tại và đã clear, False nếu không tìm thấy.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """
        Xóa hoàn toàn session khỏi storage.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            bool: True nếu đã xóa, False nếu không tìm thấy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, session_id: str) -> list[dict]:
        """
        Lấy danh sách messages dưới dạng list[dict].

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            list[dict]: Danh sách {'role': str, 'content': str}.
        """
        raise NotImplementedError

    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """
        Kiểm tra session có tồn tại trong storage không.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            bool: True nếu session đã tồn tại.
        """
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """
        Liệt kê tất cả session IDs đang active.

        Returns:
            list[str]: Danh sách session IDs.
        """
        raise NotImplementedError


# ===========================================================
# In-Memory Backend (Default Implementation)
# ===========================================================


class InMemoryBackend(BaseMemoryBackend):
    """
    In-memory storage backend sử dụng Python dict.

    Phù hợp cho: development, testing, single-process deployment.
    Không phù hợp cho: multi-process, data persistence across restarts.

    Attributes:
        _store: Dict mapping session_id → InMemoryChatMessageHistory.
        _max_history: Giới hạn số message mỗi session.
    """

    def __init__(self, max_history: int = 20) -> None:
        """
        Khởi tạo InMemoryBackend.

        Args:
            max_history: Số message tối đa mỗi session (từ config).
        """
        self._store: dict[str, InMemoryChatMessageHistory] = {}
        self._max_history = max_history
        logger.debug("InMemoryBackend initialized (max_history=%d).", max_history)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Lấy hoặc tạo mới InMemoryChatMessageHistory cho session."""
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
            logger.info("InMemoryBackend: created new session '%s'.", session_id)
        return self._store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """Xóa messages của session nhưng giữ lại key."""
        if session_id in self._store:
            self._store[session_id].clear()
            logger.info("InMemoryBackend: cleared session '%s'.", session_id)
            return True
        logger.warning("InMemoryBackend: session '%s' not found for clearing.", session_id)
        return False

    def delete_session(self, session_id: str) -> bool:
        """Xóa hoàn toàn session khỏi dict."""
        if session_id in self._store:
            del self._store[session_id]
            logger.info("InMemoryBackend: deleted session '%s'.", session_id)
            return True
        logger.warning("InMemoryBackend: session '%s' not found for deletion.", session_id)
        return False

    def get_messages(self, session_id: str) -> list[dict]:
        """Trả về messages dưới dạng list[dict] với keys 'role' và 'content'."""
        if session_id not in self._store:
            return []
        history = self._store[session_id]
        return [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
            }
            for msg in history.messages
        ]

    def session_exists(self, session_id: str) -> bool:
        """Kiểm tra session có trong dict không."""
        return session_id in self._store

    def list_sessions(self) -> list[str]:
        """Trả về danh sách tất cả session IDs."""
        return list(self._store.keys())

    @property
    def active_sessions_count(self) -> int:
        """Số sessions đang active."""
        return len(self._store)


# ===========================================================
# Memory Manager — Facade over Backend
# ===========================================================


class MemoryManager:
    """
    Facade quản lý conversation history qua pluggable backend.

    MemoryManager không implement storage logic — nó delegate
    tất cả sang `BaseMemoryBackend`. Điều này cho phép swap
    backend (Redis, SQLite...) mà không sửa gì ở tầng trên.

    Pattern: Facade + Strategy.

    Attributes:
        _backend: Storage backend (bất kỳ class nào implement BaseMemoryBackend).
    """

    def __init__(self, backend: BaseMemoryBackend) -> None:
        """
        Khởi tạo MemoryManager với backend cụ thể.

        Args:
            backend: Storage backend implement BaseMemoryBackend.
                     Inject từ `app/api/dependencies.py`.
        """
        self._backend = backend
        logger.info(
            "MemoryManager initialized with backend: %s.",
            type(backend).__name__,
        )

    # ------------------------------------------------------------------
    # Core methods (delegate to backend)
    # ------------------------------------------------------------------

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Lấy hoặc tạo mới ChatMessageHistory cho session.

        Đây là callback function được truyền vào RunnableWithMessageHistory.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            BaseChatMessageHistory: History object tương thích với LangChain.
        """
        return self._backend.get_session_history(session_id)

    def clear_session(self, session_id: str) -> bool:
        """
        Xóa messages của session (giữ lại session).

        Args:
            session_id: UUID định danh session.

        Returns:
            bool: True nếu thành công.
        """
        return self._backend.clear_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        Xóa hoàn toàn session khỏi storage.

        Args:
            session_id: UUID định danh session.

        Returns:
            bool: True nếu đã xóa.
        """
        return self._backend.delete_session(session_id)

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Lấy danh sách messages dạng dict.

        Args:
            session_id: UUID định danh session.

        Returns:
            list[dict]: Danh sách {'role': str, 'content': str}.
        """
        return self._backend.get_messages(session_id)

    def session_exists(self, session_id: str) -> bool:
        """Kiểm tra session có tồn tại không."""
        return self._backend.session_exists(session_id)

    def list_sessions(self) -> list[str]:
        """Liệt kê tất cả session IDs đang active."""
        return self._backend.list_sessions()

    @property
    def active_sessions_count(self) -> int:
        """Số sessions đang active."""
        if hasattr(self._backend, "active_sessions_count"):
            return self._backend.active_sessions_count  # type: ignore[union-attr]
        return len(self._backend.list_sessions())

    @property
    def backend_type(self) -> str:
        """Tên class của backend đang dùng (để log/debug)."""
        return type(self._backend).__name__
