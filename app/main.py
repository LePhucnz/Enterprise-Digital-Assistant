"""
FastAPI Application Entry Point.

Khởi tạo FastAPI app, cấu hình middleware, đăng ký routers
và thiết lập logging.
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.config.settings import get_settings

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------


def setup_logging(log_level: str = "INFO") -> None:
    """
    Cấu hình logging cho toàn bộ ứng dụng.

    Args:
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Giảm noise từ thư viện bên thứ ba
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


# ------------------------------------------------------------------
# Application Factory
# ------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Tạo và cấu hình FastAPI application.

    Sử dụng Application Factory pattern để dễ test và tái sử dụng.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Tạo FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Enterprise Digital Assistant API

Trợ lý AI doanh nghiệp chạy hoàn toàn local với Ollama.

### Features
- 💬 **Chat** với streaming response (SSE)
- 🧠 **Conversation Memory** — lưu lịch sử chat theo session
- 📄 **RAG** — upload PDF và hỏi về nội dung tài liệu *(Bước 4)*
- 🔧 **Tools** — calculator, thời gian thực, tìm kiếm tài liệu *(Bước 5)*

### Tech Stack
- **LLM**: Ollama + qwen2.5:latest
- **Framework**: LangChain + FastAPI
- **Vector DB**: ChromaDB
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        debug=settings.debug,
    )

    # ------------------------------------------------------------------
    # CORS Middleware
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Giới hạn origins trong production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Register Routers
    # ------------------------------------------------------------------
    api_prefix = settings.api_prefix

    app.include_router(health_router, prefix=api_prefix)
    app.include_router(chat_router, prefix=api_prefix)

    logger.info("Routers registered with prefix: %s", api_prefix)

    # ------------------------------------------------------------------
    # Root endpoint
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect info cho root endpoint."""
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{api_prefix}/health",
        }

    @app.on_event("startup")
    async def on_startup():
        logger.info("Application startup complete. API ready.")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Application shutdown.")

    return app


# ------------------------------------------------------------------
# App Instance
# ------------------------------------------------------------------
app = create_app()
