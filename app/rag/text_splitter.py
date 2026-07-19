"""
Text Splitter Service — Chia nhỏ documents thành chunks để embedding.

Sử dụng RecursiveCharacterTextSplitter của LangChain —
đây là splitter phổ biến nhất, hoạt động tốt với mọi loại văn bản.

Chiến lược split:
    Thử chia theo thứ tự ưu tiên:
        1. Đoạn văn (\\n\\n)
        2. Câu (\\n)
        3. Từ (space)
        4. Ký tự (fallback)

Config (từ settings.py):
    - chunk_size:    Số ký tự tối đa mỗi chunk.
    - chunk_overlap: Số ký tự overlap giữa các chunk liền kề.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class TextSplitterService:
    """
    Service chia nhỏ LangChain Documents thành chunks phù hợp để embed.

    Inject metadata (chunk_id, chunk_total) vào mỗi chunk để sau này
    VectorStoreService có thể filter và delete theo doc_id.

    Attributes:
        _splitter: RecursiveCharacterTextSplitter instance (singleton).
    """

    def __init__(self) -> None:
        """
        Khởi tạo TextSplitterService với config từ settings.

        chunk_size và chunk_overlap được đọc từ Settings —
        không hard-code ở đây.
        """
        settings = get_settings()
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            "TextSplitterService initialized (chunk_size=%d, overlap=%d).",
            self._chunk_size,
            self._chunk_overlap,
        )

    def split(
        self,
        documents: list[Document],
        doc_id: str,
        file_name: str,
        file_type: str,
        collection_name: str,
    ) -> list[Document]:
        """
        Chia documents thành chunks và inject metadata.

        Args:
            documents:       List[Document] từ loader (mỗi page/row là một Document).
            doc_id:          UUID của document cha — để filter/delete sau này.
            file_name:       Tên file gốc.
            file_type:       Extension của file (pdf, docx, ...).
            collection_name: Tên ChromaDB collection.

        Returns:
            list[Document]: Các chunks với metadata đầy đủ.

        Raises:
            ValueError: Nếu documents rỗng.
        """
        if not documents:
            logger.warning("TextSplitterService: received empty documents list for '%s'.", file_name)
            return []

        # Split tất cả documents thành chunks
        chunks = self._splitter.split_documents(documents)

        if not chunks:
            logger.warning(
                "TextSplitterService: no chunks produced for '%s'. File có thể rỗng.",
                file_name,
            )
            return []

        # Inject metadata vào mỗi chunk để truy vết sau này
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "chunk_id": i,
                "chunk_total": total,
                "collection_name": collection_name,
                # Giữ nguyên metadata gốc từ loader (page, source, row...)
            })

        logger.info(
            "TextSplitterService: '%s' → %d chunks (from %d pages/rows).",
            file_name,
            total,
            len(documents),
        )
        return chunks

    @property
    def chunk_size(self) -> int:
        """Chunk size đang dùng."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Chunk overlap đang dùng."""
        return self._chunk_overlap
