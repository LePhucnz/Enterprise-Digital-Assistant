"""
RAG package — Retrieval-Augmented Generation infrastructure.

Cung cấp các building blocks cho Phase 3:
    - TextSplitterService:  Chunk documents cho embedding.
    - EmbeddingService:     OllamaEmbeddings wrapper.
    - VectorStoreService:   ChromaDB operations.

Phase 3B sẽ thêm:
    - RAGChain:    LangChain RAG pipeline dùng retriever từ VectorStoreService.
    - RAGService:  Orchestrator cho RAG query flow.

Exports:
    TextSplitterService
    EmbeddingService
    VectorStoreService
"""

from app.rag.embedding_service import EmbeddingService
from app.rag.text_splitter import TextSplitterService
from app.rag.vector_store_service import VectorStoreService

__all__ = [
    "TextSplitterService",
    "EmbeddingService",
    "VectorStoreService",
]
