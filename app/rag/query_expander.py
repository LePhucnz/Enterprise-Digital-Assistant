"""
Query Expander — Dịch vụ mở rộng truy vấn (Query Expansion) cho RAG.

Sử dụng LLM để phân tích câu hỏi ban đầu của người dùng và sinh ra các biến thể
truy vấn theo nhiều góc nhìn (synonyms, domain terminology, sub-aspects).
Giúp tăng recall và độ bao phủ khi tìm kiếm trong Vector Store, tránh tình trạng
RAG quá cứng nhắc khi câu hỏi của người dùng ngắn hoặc dùng từ ngữ không khớp chính xác.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config.settings import get_settings
from app.prompts import get_query_expansion_prompt_template

logger = logging.getLogger(__name__)

# ===========================================================
# Từ khóa out-of-domain — Nếu query chứa các từ này, bỏ qua
# mở rộng (tránh tốn thời gian LLM cho câu hỏi ngoài phạm vi)
# ===========================================================
_OUT_OF_DOMAIN_KEYWORDS: frozenset[str] = frozenset([
    # Lao động / nhân sự
    "nghỉ phép", "ngày phép", "phép năm", "nghỉ thai sản", "thai sản",
    "nghỉ bệnh", "nghỉ ốm", "bảo hiểm xã hội", "bhxh", "bhyt",
    "lương thưởng", "tiền lương", "tăng lương", "phụ cấp", "trợ cấp",
    "kỷ luật", "sa thải", "thôi việc", "nghỉ việc", "hợp đồng lao động",
    "nội quy lao động", "thưởng tết", "thưởng cuối năm",
    # Hành chính
    "văn phòng phẩm", "mua sắm", "đặt vé", "đặt phòng", "công tác phí",
    "hoàn ứng", "tạm ứng", "thanh toán chi phí",
    # IT
    "password", "mật khẩu", "wifi", "vpn", "phần mềm",
])


class QueryExpander:
    """
    Sinh các truy vấn mở rộng (Query Expansion) từ câu hỏi gốc.

    Pipeline:
        Câu hỏi gốc (e.g., "quy định nghỉ phép")
            → Query Expansion Prompt Template
            → ChatOllama (temperature thấp để tập trung)
            → StrOutputParser
            → Parsing & Deduplication
            → Danh sách [câu hỏi gốc, biến thể 1, biến thể 2, biến thể 3]

    Ưu điểm:
        - Mở rộng phạm vi tìm kiếm sang thuật ngữ văn bản nội bộ.
        - Giữ lại câu hỏi gốc ở vị trí đầu tiên.
        - Fallback an toàn về [câu hỏi gốc] nếu LLM gặp sự cố hoặc timeout.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> None:
        """
        Khởi tạo QueryExpander.

        Args:
            llm: BaseChatModel instance (injected). Nếu None sẽ tạo ChatOllama từ settings.
            model: Override model name nếu cần.
            temperature: Nhiệt độ khi sinh truy vấn (mặc định 0.3).
        """
        settings = get_settings()

        if llm is not None:
            self._llm = llm
        else:
            _model = model or settings.ollama_model
            self._llm = ChatOllama(
                model=_model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
                top_p=0.9,
                num_predict=512,
                num_ctx=settings.llm_context_window,
            )

        self._prompt = get_query_expansion_prompt_template()
        self._chain = self._prompt | self._llm | StrOutputParser()

        logger.info(
            "QueryExpander initialized (model=%s, temperature=%.2f).",
            getattr(self._llm, "model", "custom"),
            temperature,
        )

    @staticmethod
    def _clean_query_line(line: str) -> str:
        """
        Làm sạch một dòng truy vấn sinh ra từ LLM.

        Loại bỏ các định dạng list như số thứ tự (1., 1), 1 -), bullet point (-, *, •),
        dấu ngoặc kép bao quanh và khoảng trắng thừa.
        """
        line = line.strip()
        # Loại bỏ số thứ tự hoặc ký tự đầu dòng (1., 1), 1 -, *, -, •)
        cleaned = re.sub(r"^(?:\d+\s*[\.\)\-:]|[-*•–—])\s*", "", line).strip()
        # Loại bỏ ngoặc kép hoặc backticks nếu có
        cleaned = cleaned.strip('"\'`')
        return cleaned

    def _parse_output(self, raw_output: str, original: str, max_count: int) -> list[str]:
        """
        Parse raw text từ LLM thành danh sách truy vấn hợp lệ.

        Args:
            raw_output: Kết quả chuỗi từ LLM.
            original: Câu hỏi gốc.
            max_count: Số lượng biến thể tối đa.

        Returns:
            list[str]: Danh sách gồm [câu hỏi gốc, *biến thể].
        """
        lines = raw_output.strip().split("\n")
        results = [original]
        seen = {original.strip().lower()}

        for line in lines:
            cleaned = self._clean_query_line(line)
            if not cleaned:
                continue

            # Bỏ qua nếu dòng là text dẫn giải phổ biến
            if cleaned.lower().startswith(("dưới đây là", "truy vấn", "câu hỏi mở rộng")):
                continue

            cleaned_lower = cleaned.lower()
            if cleaned_lower not in seen:
                seen.add(cleaned_lower)
                results.append(cleaned)
                if len(results) >= max_count + 1:
                    break

        return results

    def expand(self, query: str, num_queries: int = 3) -> list[str]:
        """
        Mở rộng câu hỏi của người dùng (đồng bộ).

        Args:
            query: Câu hỏi ban đầu.
            num_queries: Số lượng truy vấn biến thể cần sinh.

        Returns:
            list[str]: Danh sách các truy vấn mở rộng (bắt đầu bằng câu hỏi gốc).
        """
        original = query.strip()
        if not original:
            return []

        if num_queries <= 0:
            return [original]

        # Kiểm tra out-of-domain — bỏ qua LLM nếu rõ ràng ngoài phạm vi
        query_lower = original.lower()
        if any(kw in query_lower for kw in _OUT_OF_DOMAIN_KEYWORDS):
            logger.info(
                "QueryExpander.expand: query '%s...' contains out-of-domain keyword. "
                "Skipping LLM expansion to save time.",
                original[:50],
            )
            return [original]

        logger.info(
            "QueryExpander.expand: expanding '%s...' into %d queries.",
            original[:50],
            num_queries,
        )

        try:
            raw_output = self._chain.invoke({"input": original, "num_queries": num_queries})
            expanded = self._parse_output(raw_output, original, num_queries)
            logger.info(
                "QueryExpander.expand: successfully generated %d queries: %s",
                len(expanded),
                expanded,
            )
            return expanded
        except Exception as e:
            logger.warning(
                "QueryExpander.expand failed (%s). Gracefully falling back to original query.",
                str(e),
            )
            return [original]

    async def aexpand(self, query: str, num_queries: int = 3) -> list[str]:
        """
        Mở rộng câu hỏi của người dùng (bất đồng bộ).

        Args:
            query: Câu hỏi ban đầu.
            num_queries: Số lượng truy vấn biến thể cần sinh.

        Returns:
            list[str]: Danh sách các truy vấn mở rộng (bắt đầu bằng câu hỏi gốc).
        """
        original = query.strip()
        if not original:
            return []

        if num_queries <= 0:
            return [original]

        # Kiểm tra out-of-domain — bỏ qua LLM nếu rõ ràng ngoài phạm vi
        query_lower = original.lower()
        if any(kw in query_lower for kw in _OUT_OF_DOMAIN_KEYWORDS):
            logger.info(
                "QueryExpander.aexpand: query '%s...' contains out-of-domain keyword. "
                "Skipping LLM expansion to save time.",
                original[:50],
            )
            return [original]

        logger.info(
            "QueryExpander.aexpand: expanding '%s...' into %d queries.",
            original[:50],
            num_queries,
        )

        try:
            raw_output = await self._chain.ainvoke({"input": original, "num_queries": num_queries})
            expanded = self._parse_output(raw_output, original, num_queries)
            logger.info(
                "QueryExpander.aexpand: successfully generated %d queries: %s",
                len(expanded),
                expanded,
            )
            return expanded
        except Exception as e:
            logger.warning(
                "QueryExpander.aexpand failed (%s). Gracefully falling back to original query.",
                str(e),
            )
            return [original]
