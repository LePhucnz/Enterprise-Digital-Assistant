"""
Memory Manager — Quản lý conversation memory theo session.

Lưu trữ và cung cấp memory cho từng conversation session.
Sử dụng in-memory dict làm storage (có thể swap sang Redis sau này).
"""

import logging
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Quản lý conversation history theo session_id.

    Sử dụng in-memory dictionary để lưu trữ ChatMessageHistory
    cho từng session. Thiết kế để dễ dàng thay thế bằng
    Redis-backed storage trong tương lai.

    Pattern: Singleton (khuyến nghị dùng qua dependency injection).

    Attributes:
        _store: Dict mapping session_id → ChatMessageHistory.
        _max_history: Số lượng message tối đa mỗi session.
    """

    def __init__(self) -> None:
        """Khởi tạo MemoryManager với empty store."""
        self._store: dict[str, InMemoryChatMessageHistory] = {}
        settings = get_settings()
        self._max_history: int = settings.memory_max_history
        logger.info(
            "MemoryManager initialized (max_history=%d).",
            self._max_history,
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Lấy hoặc tạo mới ChatMessageHistory cho một session.

        Nếu session_id chưa tồn tại, tạo mới ChatMessageHistory rỗng.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            BaseChatMessageHistory: History object cho session.
        """
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
            logger.info("Created new session: %s", session_id)
        return self._store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """
        Xóa toàn bộ history của một session.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            bool: True nếu session tồn tại và đã được xóa, False nếu không tìm thấy.
        """
        if session_id in self._store:
            self._store[session_id].clear()
            logger.info("Cleared session history: %s", session_id)
            return True
        logger.warning("Session not found for clearing: %s", session_id)
        return False

    def delete_session(self, session_id: str) -> bool:
        """
        Xóa hoàn toàn một session khỏi store.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            bool: True nếu đã xóa, False nếu không tìm thấy.
        """
        if session_id in self._store:
            del self._store[session_id]
            logger.info("Deleted session: %s", session_id)
            return True
        logger.warning("Session not found for deletion: %s", session_id)
        return False

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Lấy danh sách messages của một session dưới dạng dict.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            list[dict]: Danh sách messages với keys 'role' và 'content'.
        """
        if session_id not in self._store:
            return []

        history = self._store[session_id]
        messages = []
        for msg in history.messages:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})
        return messages

    def session_exists(self, session_id: str) -> bool:
        """
        Kiểm tra session có tồn tại không.

        Args:
            session_id: UUID định danh conversation session.

        Returns:
            bool: True nếu session đã tồn tại.
        """
        return session_id in self._store

    def list_sessions(self) -> list[str]:
        """
        Liệt kê tất cả session IDs đang active.

        Returns:
            list[str]: Danh sách session IDs.
        """
        return list(self._store.keys())

    @property
    def active_sessions_count(self) -> int:
        """Số lượng sessions đang active."""
        return len(self._store)
