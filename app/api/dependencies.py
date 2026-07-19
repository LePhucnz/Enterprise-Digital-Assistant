"""
FastAPI Dependencies — Dependency Injection cho toàn bộ API.

Cung cấp các singleton instances thông qua FastAPI's dependency system.
Đảm bảo tất cả services được khởi tạo một lần và share trong toàn bộ
vòng đời ứng dụng.

Phase 2 (Chat):
    - get_memory_backend():  InMemoryBackend (swap sang Redis chỉ cần đổi đây).
    - get_memory_manager():  Inject backend vào MemoryManager.
    - get_chat_service():    ChatService singleton.

Phase 3A (Documents):
    - get_embedding_service():    OllamaEmbeddings singleton.
    - get_vector_store_service(): ChromaDB singleton.
    - get_document_repository():  JSON metadata repository.
    - get_document_service():     Document pipeline orchestrator.
"""

import logging
from functools import lru_cache

from app.config.settings import get_settings
from app.documents.document_service import DocumentService
from app.documents.repository import DocumentRepository
from app.memory.memory_manager import InMemoryBackend, MemoryManager
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store_service import VectorStoreService
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


# ===========================================================
# Chat Dependencies (Phase 1 & 2 — không đổi)
# ===========================================================


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


# ===========================================================
# Document Dependencies (Phase 3A — mới)
# ===========================================================


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Trả về singleton EmbeddingService (OllamaEmbeddings).

    Returns:
        EmbeddingService: Singleton instance.
    """
    logger.info("Creating EmbeddingService singleton.")
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    """
    Trả về singleton VectorStoreService (ChromaDB persistent).

    Returns:
        VectorStoreService: Singleton instance.
    """
    logger.info("Creating VectorStoreService singleton.")
    embedding_service = get_embedding_service()
    return VectorStoreService(embedding_service=embedding_service)


@lru_cache(maxsize=1)
def get_document_repository() -> DocumentRepository:
    """
    Trả về singleton DocumentRepository (JSON persistence).

    Returns:
        DocumentRepository: Singleton instance.
    """
    logger.info("Creating DocumentRepository singleton.")
    return DocumentRepository()


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    """
    Trả về singleton DocumentService (pipeline orchestrator).

    Returns:
        DocumentService: Singleton instance.
    """
    logger.info("Creating DocumentService singleton.")
    return DocumentService(
        repository=get_document_repository(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
    )
