"""
Document Repository — JSON file-based persistence cho DocumentMetadata.

Lưu trữ metadata của tất cả documents trong `data/documents.json`.
Repository pattern: tách biệt storage logic khỏi business logic.

Thiết kế:
    - Thread-safe với file locking (dùng threading.Lock).
    - Tự tạo file nếu chưa tồn tại.
    - Dễ swap sang SQLite/PostgreSQL sau này nếu cần.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from app.config.settings import get_settings
from app.documents.models import DocumentMetadata, DocumentStatus

logger = logging.getLogger(__name__)

# Tên file lưu metadata trong data_dir
_METADATA_FILE = "documents.json"


class DocumentRepository:
    """
    JSON file-based repository cho DocumentMetadata.

    Lưu metadata vào `{data_dir}/documents.json` dưới dạng JSON array.
    Sử dụng threading.Lock để đảm bảo thread-safety khi đọc/ghi.

    Pattern: Repository (tách biệt persistence khỏi domain logic).

    Attributes:
        _file_path: Đường dẫn đến file JSON.
        _lock:      Threading lock để đảm bảo concurrent safety.
    """

    def __init__(self) -> None:
        """Khởi tạo repository, tạo file nếu chưa tồn tại."""
        settings = get_settings()
        data_dir = Path(settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        self._file_path = data_dir / _METADATA_FILE
        self._lock = threading.Lock()

        # Tạo file rỗng nếu chưa có
        if not self._file_path.exists():
            self._write_all([])
            logger.info("DocumentRepository: created metadata file at '%s'.", self._file_path)
        else:
            logger.info(
                "DocumentRepository: loaded from '%s' (%d documents).",
                self._file_path,
                len(self._read_all()),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_all(self) -> list[dict]:
        """Đọc toàn bộ records từ JSON file."""
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("DocumentRepository: JSON file invalid, resetting.")
            return []

    def _write_all(self, records: list[dict]) -> None:
        """Ghi toàn bộ records vào JSON file (overwrite)."""
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, doc: DocumentMetadata) -> None:
        """
        Lưu hoặc cập nhật một DocumentMetadata.

        Nếu doc.id đã tồn tại → update. Nếu chưa → insert.

        Args:
            doc: DocumentMetadata cần lưu.
        """
        with self._lock:
            records = self._read_all()
            # Update nếu đã tồn tại
            for i, record in enumerate(records):
                if record.get("id") == doc.id:
                    records[i] = doc.model_dump()
                    self._write_all(records)
                    logger.debug("DocumentRepository: updated doc '%s'.", doc.id)
                    return
            # Insert mới
            records.append(doc.model_dump())
            self._write_all(records)
            logger.info("DocumentRepository: saved new doc '%s' (%s).", doc.id, doc.file_name)

    def get(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Lấy DocumentMetadata theo ID.

        Args:
            doc_id: UUID của document.

        Returns:
            DocumentMetadata nếu tìm thấy, None nếu không.
        """
        with self._lock:
            for record in self._read_all():
                if record.get("id") == doc_id:
                    return DocumentMetadata(**record)
        return None

    def get_by_hash(self, file_hash: str) -> Optional[DocumentMetadata]:
        """
        Lấy DocumentMetadata theo SHA256 hash của file.

        Dùng để duplicate check trước khi upload.

        Args:
            file_hash: SHA256 hex string.

        Returns:
            DocumentMetadata nếu file đã tồn tại, None nếu chưa.
        """
        with self._lock:
            for record in self._read_all():
                if record.get("file_hash") == file_hash:
                    return DocumentMetadata(**record)
        return None

    def list_all(self) -> list[DocumentMetadata]:
        """
        Lấy danh sách tất cả documents.

        Returns:
            list[DocumentMetadata]: Sắp xếp theo created_at mới nhất trước.
        """
        with self._lock:
            records = self._read_all()
        docs = [DocumentMetadata(**r) for r in records]
        # Sắp xếp mới nhất trước
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

    def delete(self, doc_id: str) -> bool:
        """
        Xóa document khỏi repository.

        Args:
            doc_id: UUID của document.

        Returns:
            bool: True nếu đã xóa, False nếu không tìm thấy.
        """
        with self._lock:
            records = self._read_all()
            original_count = len(records)
            records = [r for r in records if r.get("id") != doc_id]
            if len(records) < original_count:
                self._write_all(records)
                logger.info("DocumentRepository: deleted doc '%s'.", doc_id)
                return True
        logger.warning("DocumentRepository: doc '%s' not found for deletion.", doc_id)
        return False

    def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Cập nhật trạng thái của document sau mỗi bước pipeline.

        Args:
            doc_id:        UUID của document.
            status:        Trạng thái mới.
            chunk_count:   Số chunks (chỉ set khi status=INDEXED).
            error_message: Lỗi (chỉ set khi status=FAILED).

        Returns:
            bool: True nếu cập nhật thành công.
        """
        doc = self.get(doc_id)
        if not doc:
            return False

        if status == DocumentStatus.PROCESSING:
            doc.mark_processing()
        elif status == DocumentStatus.INDEXED:
            doc.mark_indexed(chunk_count)
        elif status == DocumentStatus.FAILED:
            doc.mark_failed(error_message or "Unknown error")

        self.save(doc)
        logger.info(
            "DocumentRepository: doc '%s' status → %s (chunks=%d).",
            doc_id,
            status.value,
            chunk_count,
        )
        return True

    @property
    def count(self) -> int:
        """Tổng số documents trong repository."""
        with self._lock:
            return len(self._read_all())
