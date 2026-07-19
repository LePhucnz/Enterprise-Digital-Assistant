"""
RAG package — Retrieval-Augmented Generation infrastructure.

Cung cấp các building blocks cho Phase 3:
    - TextSplitterService:  Chunk documents cho embedding.
    - EmbeddingService:     OllamaEmbeddings wrapper.
    - VectorStoreService:   ChromaDB operations.
    - RetrieverService:     Retriever logic với các chiến lược tìm kiếm.

Exports:
    TextSplitterService
    EmbeddingService
    VectorStoreService
    RetrieverService
"""

from app.rag.embedding_service import EmbeddingService
from app.rag.retriever_service import RetrieverService
from app.rag.text_splitter import TextSplitterService
from app.rag.vector_store_service import VectorStoreService

__all__ = [
    "TextSplitterService",
    "EmbeddingService",
    "VectorStoreService",
    "RetrieverService",
]

