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
    details: dict


@router.get("", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """
    Kiểm tra trạng thái ứng dụng.

    - Kiểm tra kết nối đến Ollama server.
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

    status = "healthy" if ollama_connected else "degraded"

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.app_version,
        ollama_connected=ollama_connected,
        ollama_model=settings.ollama_model,
        details={
            "ollama_url": settings.ollama_base_url,
            "debug_mode": settings.debug,
            "memory_type": settings.memory_type,
        },
    )
