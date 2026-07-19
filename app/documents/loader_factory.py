"""
Document Loader Factory — Factory Pattern cho LangChain document loaders.

Mỗi định dạng file có một LangChain loader phù hợp.
Factory giúp tập trung logic chọn loader — không scatter ở nhiều nơi.

Hỗ trợ:
    .pdf      → PyPDFLoader       (langchain-community)
    .docx     → Docx2txtLoader    (langchain-community + docx2txt)
    .txt      → TextLoader        (langchain-core)
    .csv      → CSVLoader         (langchain-community)
    .md/.markdown → TextLoader    (đơn giản, không cần unstructured)

Cách mở rộng:
    Để thêm định dạng mới (ví dụ .pptx):
        1. Thêm extension vào `_LOADER_MAP`.
        2. Import loader phù hợp.
        3. Không cần sửa bất kỳ code nào khác.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader

from app.core.exceptions import UnsupportedFileTypeError

logger = logging.getLogger(__name__)

# Mapping từ extension (lowercase) → loader class
_LOADER_MAP: dict[str, type[BaseLoader]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".md": TextLoader,       # TextLoader đơn giản, không cần unstructured
    ".markdown": TextLoader,
}


class DocumentLoaderFactory:
    """
    Factory tạo LangChain loader phù hợp với từng định dạng file.

    Sử dụng static methods — không cần khởi tạo instance.

    Design Pattern: Factory Method.
    Nguyên tắc: Open/Closed — mở rộng bằng cách thêm vào `_LOADER_MAP`,
                không sửa logic hiện tại.
    """

    @staticmethod
    def create(file_path: str | Path) -> BaseLoader:
        """
        Tạo LangChain loader phù hợp với extension của file.

        Args:
            file_path: Đường dẫn đầy đủ đến file cần load.

        Returns:
            BaseLoader: LangChain document loader instance.

        Raises:
            UnsupportedFileTypeError: Nếu extension không được hỗ trợ.
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        loader_class = _LOADER_MAP.get(extension)
        if loader_class is None:
            raise UnsupportedFileTypeError(
                file_name=path.name,
                extension=extension or "(no extension)",
            )

        logger.debug(
            "DocumentLoaderFactory: '%s' → %s",
            path.name,
            loader_class.__name__,
        )

        # TextLoader cần encoding argument để đọc UTF-8 đúng cách
        if loader_class is TextLoader:
            return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

        return loader_class(str(path))

    @staticmethod
    def supported_extensions() -> list[str]:
        """
        Trả về danh sách extensions được hỗ trợ.

        Returns:
            list[str]: Ví dụ ['.pdf', '.docx', '.txt', '.csv', '.md', '.markdown'].
        """
        return list(_LOADER_MAP.keys())

    @staticmethod
    def is_supported(file_name: str) -> bool:
        """
        Kiểm tra file có extension được hỗ trợ không.

        Args:
            file_name: Tên file hoặc đường dẫn.

        Returns:
            bool: True nếu extension được hỗ trợ.
        """
        extension = Path(file_name).suffix.lower()
        return extension in _LOADER_MAP
