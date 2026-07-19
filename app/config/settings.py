"""
Application Settings.

Sử dụng Pydantic BaseSettings để đọc cấu hình từ file .env.
Mọi thông số cấu hình đều tập trung tại đây — không hardcode ở bất kỳ file nào khác.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Cấu hình toàn cục của ứng dụng.

    Tất cả giá trị được đọc từ file .env.
    Dùng `get_settings()` để lấy instance (cached singleton).
    """

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL của Ollama server.",
    )
    ollama_model: str = Field(
        default="qwen2.5:latest",
        description="Model LLM chính.",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text:latest",
        description="Model dùng để tạo embeddings.",
    )

    # ------------------------------------------------------------------
    # LLM Parameters
    # ------------------------------------------------------------------
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature cho LLM (0.0 = deterministic, 2.0 = creative).",
    )
    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Số token tối đa trong response.",
    )
    llm_streaming: bool = Field(
        default=True,
        description="Bật/tắt streaming response.",
    )

    # ------------------------------------------------------------------
    # FastAPI
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="Host của FastAPI server.")
    api_port: int = Field(default=8000, description="Port của FastAPI server.")
    api_prefix: str = Field(default="/api/v1", description="Prefix cho tất cả API routes.")
    debug: bool = Field(default=True, description="Chế độ debug.")

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    memory_type: Literal["buffer", "summary"] = Field(
        default="buffer",
        description="Loại memory: 'buffer' hoặc 'summary'.",
    )
    memory_max_history: int = Field(
        default=20,
        gt=0,
        description="Số lượng message tối đa lưu trong buffer memory.",
    )
    memory_summary_max_tokens: int = Field(
        default=500,
        gt=0,
        description="Số token tối đa cho summary memory.",
    )

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Kích thước mỗi chunk khi split document.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Số ký tự overlap giữa các chunk.",
    )
    retriever_top_k: int = Field(
        default=4,
        gt=0,
        description="Số lượng document chunks trả về từ retriever.",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: str = Field(default="data", description="Thư mục lưu raw data (PDF, DOCX...).")
    vector_db_dir: str = Field(
        default="vector_db",
        description="Thư mục lưu ChromaDB persistent storage.",
    )

    # ------------------------------------------------------------------
    # App Info
    # ------------------------------------------------------------------
    app_name: str = Field(
        default="Enterprise Digital Assistant",
        description="Tên ứng dụng.",
    )
    app_version: str = Field(default="1.0.0", description="Phiên bản ứng dụng.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_must_be_less_than_chunk_size(cls, v: int, info) -> int:
        """Đảm bảo chunk_overlap < chunk_size."""
        # Lấy chunk_size từ data nếu đã được validate
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) phải nhỏ hơn chunk_size ({chunk_size})."
            )
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Trả về Settings instance (singleton, cached).

    Dùng @lru_cache để đảm bảo chỉ load .env một lần duy nhất.

    Returns:
        Settings: Instance cấu hình toàn cục.
    """
    return Settings()
