"""
LLM Factory — Khởi tạo Chat Model linh hoạt theo Provider và Model.

Hỗ trợ đồng thời:
    - Ollama (Local LLM: qwen2.5, llama3.2...)
    - OpenAI / OpenRouter (API Key: gpt-4o-mini, gpt-4o, claude...)
    - Google Gemini (API Key: gemini-3.8-flash, gemini-3.5-flash-lite...)

Tự động nhận diện provider từ tên model hoặc cấu hình LLM_PROVIDER.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def infer_provider(model_name: Optional[str] = None) -> str:
    """
    Tự động suy luận provider từ tên model.

    Args:
        model_name: Tên model (e.g. "gemini-3.8-flash", "openai/gpt-4o-mini", "qwen2.5:latest").

    Returns:
        str: "gemini", "openai", hoặc "ollama".
    """
    settings = get_settings()

    if not model_name:
        return settings.llm_provider

    m = model_name.strip().lower()

    # Gemini detection
    if m.startswith(("gemini", "models/gemini", "gemma")):
        return "gemini"

    # OpenAI / OpenRouter detection
    if m.startswith((
        "gpt-",
        "openai/",
        "anthropic/",
        "claude-",
        "meta-llama/",
        "mistralai/",
        "deepseek/",
        "google/",
        "text-davinci",
    )):
        return "openai"

    # Ollama detection (local models usually have ':' or known local tags)
    if ":" in m or m.startswith(("qwen", "llama", "mistral", "phi", "vicuna")):
        return "ollama"

    return settings.llm_provider


def get_chat_model(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
    streaming: Optional[bool] = None,
) -> BaseChatModel:
    """
    Khởi tạo instance BaseChatModel phù hợp với provider và model.

    Args:
        model:       Tên model cần dùng. Nếu None, dùng default từ config của provider.
        temperature: Nhiệt độ sinh văn bản (0.0 - 2.0).
        provider:    Chỉ định rõ provider ("ollama", "openai", "gemini"). Nếu None, tự động suy luận.
        streaming:   Bật/tắt streaming.

    Returns:
        BaseChatModel: LangChain Chat Model (ChatOllama, ChatOpenAI, hoặc ChatGoogleGenerativeAI).
    """
    settings = get_settings()
    selected_provider = provider or infer_provider(model)
    _temp = temperature if temperature is not None else settings.llm_temperature
    _streaming = streaming if streaming is not None else settings.llm_streaming

    logger.info(
        "get_chat_model: provider=%s | model=%s | temperature=%.2f",
        selected_provider,
        model,
        _temp,
    )

    if selected_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.gemini_api_key or settings.google_api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY chưa được cấu hình. Vui lòng thêm vào file .env."
            )

        model_name = model or settings.gemini_model
        if model_name.startswith("models/"):
            model_name = model_name[len("models/"):]

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=_temp,
            max_output_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    elif selected_provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY chưa được cấu hình. Vui lòng thêm vào file .env."
            )

        model_name = model or settings.openai_model
        base_url = settings.openai_base_url or "https://api.openai.com/v1"

        # Tự động chuyển base_url sang OpenRouter nếu dùng key sk-or-v1-
        if api_key.startswith("sk-or-v1-") and "api.openai.com" in base_url:
            base_url = "https://openrouter.ai/api/v1"

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=_temp,
            max_tokens=settings.llm_max_tokens,
            streaming=_streaming,
            timeout=settings.llm_timeout,
        )

    else:
        # Default: Ollama (Local)
        from langchain_ollama import ChatOllama

        model_name = model or settings.ollama_model

        return ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=_temp,
            top_p=settings.llm_top_p,
            num_predict=settings.llm_max_tokens,
            num_ctx=settings.llm_context_window,
        )
