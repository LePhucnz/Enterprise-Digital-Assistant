"""
Application Settings.

Sử dụng Pydantic BaseSettings để đọc cấu hình từ file .env.
Mọi thông số cấu hình đều tập trung tại đây — không hardcode ở bất kỳ file nào khác.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Cấu hình toàn cục của ứng dụng.

    Tất cả giá trị được đọc từ file .env.
    Dùng `get_settings()` để lấy instance (cached singleton).
    """

    # ------------------------------------------------------------------
    # LLM Provider
    # ------------------------------------------------------------------
    llm_provider: Literal["ollama", "openai", "gemini"] = Field(
        default="ollama",
        description="Provider LLM mặc định: 'ollama', 'openai', hoặc 'gemini'.",
    )

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
    # OpenAI
    # ------------------------------------------------------------------
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API Key cho OpenAI.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model LLM OpenAI.",
    )
    openai_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1",
        description="Base URL của OpenAI.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model embedding của OpenAI.",
    )

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="API Key cho Google Gemini.",
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Alias API Key cho Google Gemini (LangChain / Google SDK).",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Model LLM Google Gemini.",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Model embedding của Google Gemini.",
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
    llm_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling — chỉ lấy các token có cumulative probability <= top_p.",
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
    llm_timeout: int = Field(
        default=120,
        gt=0,
        description="Timeout (giây) cho mỗi LLM request.",
    )
    llm_context_window: int = Field(
        default=32768,
        gt=0,
        description="Context window size (tokens) của model.",
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
    rag_search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        default="similarity",
        description="Chiến lược tìm kiếm: similarity, mmr (Max Marginal Relevance), hoặc similarity_score_threshold.",
    )
    rag_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ngưỡng score tối thiểu khi dùng similarity_score_threshold search.",
    )
    collection_name: str = Field(
        default="enterprise_kb",
        description="Tên ChromaDB collection lưu trữ document embeddings.",
    )

    # ------------------------------------------------------------------
    # Query Expansion (RAG)
    # ------------------------------------------------------------------
    rag_enable_query_expansion: bool = Field(
        default=True,
        description="Bật/tắt kỹ thuật Query Expansion để mở rộng phạm vi tìm kiếm.",
    )
    rag_expansion_num_queries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Số lượng query biến thể được sinh ra để mở rộng tìm kiếm.",
    )
    rag_expansion_top_k: int = Field(
        default=6,
        gt=0,
        description="Số lượng document chunks tối đa sau khi fuse kết quả Query Expansion.",
    )
    rag_expansion_rrf_k: int = Field(
        default=60,
        gt=0,
        description="Hằng số k trong thuật toán Reciprocal Rank Fusion (RRF).",
    )
    rag_relevance_distance_threshold: float = Field(
        default=0.50,
        ge=0.0,
        description=(
            "Ngưỡng khoảng cách L2 tối đa để một chunk được xem là liên quan. "
            "Chunks có distance > threshold sẽ bị loại bỏ (không gửi vào LLM). "
            "Giá trị thấp hơn = strict hơn. Thường: in-domain < 0.4, out-of-domain > 0.5."
        ),
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
    app_version: str = Field(default="2.0.0", description="Phiên bản ứng dụng.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level.",
    )
    prompt_locale: Literal["vi", "en"] = Field(
        default="vi",
        description="Ngôn ngữ mặc định cho system prompt (vi = tiếng Việt, en = English).",
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
