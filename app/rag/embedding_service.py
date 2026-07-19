"""
Embedding Service — Wrapper cho OllamaEmbeddings.

Sử dụng model nomic-embed-text (hoặc model trong config)
để convert text thành vector representations.

Điều kiện tiên quyết:
    Model embedding phải được pull về trước:
        ollama pull nomic-embed-text

Config (từ settings.py):
    - ollama_embedding_model: Tên model (default: nomic-embed-text:latest)
    - ollama_base_url:        URL Ollama server
"""

from __future__ import annotations

import logging

from langchain_ollama import OllamaEmbeddings

from app.config.settings import get_settings
from app.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service quản lý OllamaEmbeddings instance.

    Cung cấp:
        - `get_embeddings()`: Trả về OllamaEmbeddings (dùng với ChromaDB).
        - `embed_query()`:    Embed một query string (dùng cho Phase 3B similarity search).

    Design: Lazy initialization — chỉ tạo OllamaEmbeddings khi cần,
    không kết nối Ollama khi import module.

    Attributes:
        _model:    Tên model embedding.
        _base_url: Ollama server URL.
        _embeddings: OllamaEmbeddings instance (lazy-created).
    """

    def __init__(self) -> None:
        """Khởi tạo EmbeddingService với config từ settings."""
        settings = get_settings()
        self._model = settings.ollama_embedding_model
        self._base_url = settings.ollama_base_url
        self._embeddings: OllamaEmbeddings | None = None
        logger.info(
            "EmbeddingService initialized (model=%s, url=%s).",
            self._model,
            self._base_url,
        )

    def get_embeddings(self) -> OllamaEmbeddings:
        """
        Trả về OllamaEmbeddings instance (lazy singleton).

        OllamaEmbeddings được truyền trực tiếp vào Chroma() constructor
        để ChromaDB tự động embed documents khi add.

        Returns:
            OllamaEmbeddings: LangChain embedding function.
        """
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self._model,
                base_url=self._base_url,
            )
            logger.debug("EmbeddingService: created OllamaEmbeddings instance.")
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed một query string thành vector.

        Dùng ở Phase 3B cho similarity search trước khi gửi vào retriever.

        Args:
            text: Query string cần embed.

        Returns:
            list[float]: Vector embedding.

        Raises:
            EmbeddingException: Nếu Ollama không kết nối được hoặc model lỗi.
        """
        try:
            embeddings = self.get_embeddings()
            vector = embeddings.embed_query(text)
            logger.debug(
                "EmbeddingService: embedded query (%d chars) → vector dim=%d.",
                len(text),
                len(vector),
            )
            return vector
        except Exception as e:
            error_str = str(e).lower()
            logger.error("EmbeddingService.embed_query error: %s", str(e), exc_info=True)
            raise EmbeddingException(model=self._model, detail=str(e)) from e

    @property
    def model_name(self) -> str:
        """Tên model embedding đang dùng."""
        return self._model
