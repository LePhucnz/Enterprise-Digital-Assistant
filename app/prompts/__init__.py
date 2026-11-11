"""
Prompts Package — Quản lý tập trung tất cả LangChain PromptTemplates.

Mọi prompt trong hệ thống đều được định nghĩa tại đây.
Không viết prompt trực tiếp trong service hay chain.

Exports:
    get_chat_prompt_template: Trả về ChatPromptTemplate cho basic chat.
    get_rag_prompt_template:  Trả về ChatPromptTemplate cho RAG chat (Phase 3).
    get_query_expansion_prompt_template: Trả về ChatPromptTemplate cho Query Expansion.
    SYSTEM_PROMPT:            System prompt string (để test hoặc debug).
"""

from app.prompts.chat_prompts import (
    SYSTEM_PROMPT,
    get_chat_prompt_template,
    get_query_expansion_prompt_template,
    get_rag_prompt_template,
)

__all__ = [
    "SYSTEM_PROMPT",
    "get_chat_prompt_template",
    "get_rag_prompt_template",
    "get_query_expansion_prompt_template",
]
