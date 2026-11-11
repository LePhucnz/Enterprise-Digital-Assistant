"""
Health Check Endpoint.

Cung cấp endpoint kiểm tra trạng thái ứng dụng và Ollama connection.
"""

import logging

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    """Response cho health check endpoint."""

    status: str
    app_name: str
    version: str
    ollama_connected: bool
    ollama_model: str
    active_provider: str = "ollama"
    openai_configured: bool = False
    gemini_configured: bool = False
    details: dict


@router.get("", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """
    Kiểm tra trạng thái ứng dụng.

    - Kiểm tra kết nối đến Ollama server.
    - Kiểm tra cấu hình API keys (OpenAI, Gemini).
    - Trả về thông tin version và cấu hình.

    Returns:
        HealthResponse: Trạng thái hiện tại của ứng dụng.
    """
    settings = get_settings()
    ollama_connected = False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_connected = response.status_code == 200
    except Exception as e:
        logger.warning("Ollama health check failed: %s", str(e))

    openai_configured = bool(settings.openai_api_key)
    gemini_configured = bool(settings.gemini_api_key or settings.google_api_key)

    # Status: healthy nếu có ít nhất 1 backend hoạt động
    is_healthy = ollama_connected or openai_configured or gemini_configured
    status = "healthy" if is_healthy else "degraded"

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.app_version,
        ollama_connected=ollama_connected,
        ollama_model=settings.ollama_model,
        active_provider=settings.llm_provider,
        openai_configured=openai_configured,
        gemini_configured=gemini_configured,
        details={
            "ollama_url": settings.ollama_base_url,
            "llm_provider": settings.llm_provider,
            "openai_configured": openai_configured,
            "gemini_configured": gemini_configured,
            "debug_mode": settings.debug,
            "memory_type": settings.memory_type,
        },
    )
