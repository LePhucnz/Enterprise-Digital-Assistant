"""
API Client cho Streamlit Frontend.

Encapsulate tất cả HTTP calls đến FastAPI backend.
Tách biệt hoàn toàn UI logic khỏi API communication.
"""

import json
import logging
from typing import Generator, Optional

import httpx
import requests

logger = logging.getLogger(__name__)


class APIClient:
    """
    HTTP client gọi FastAPI backend từ Streamlit.

    Attributes:
        base_url: Base URL của FastAPI server.
        api_prefix: API prefix (e.g., '/api/v1').
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        timeout: int = 120,
    ) -> None:
        """
        Khởi tạo APIClient.

        Args:
            base_url: Base URL của FastAPI server.
            api_prefix: API prefix.
            timeout: Request timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix
        self.timeout = timeout
        self._api_url = f"{self.base_url}{self.api_prefix}"

    def _url(self, path: str) -> str:
        """Build full URL từ path."""
        return f"{self._api_url}/{path.lstrip('/')}"

    def health_check(self) -> dict:
        """
        Kiểm tra trạng thái API server.

        Returns:
            dict: Health status hoặc error info.
        """
        try:
            response = requests.get(self._url("/health"), timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError:
            return {"status": "offline", "error": "Cannot connect to API server."}
        except Exception as e:
            logger.error("Health check failed: %s", str(e))
            return {"status": "error", "error": str(e)}

    def chat_sync(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        use_rag: bool = False,
    ) -> dict:
        """
        Gửi tin nhắn (non-streaming).

        Args:
            message: Tin nhắn từ người dùng.
            session_id: Session ID.
            model: Override model.
            temperature: Override temperature.
            use_rag: True để sử dụng RAG.

        Returns:
            dict: Response từ API.
        """
        payload = {
            "message": message,
            "session_id": session_id,
            "model": model,
            "temperature": temperature,
            "use_rag": use_rag,
        }
        try:
            response = requests.post(
                self._url("/chat/sync"),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("chat_sync failed: %s", str(e))
            raise

    def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        use_rag: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Gửi tin nhắn với streaming response (SSE).

        Args:
            message: Tin nhắn từ người dùng.
            session_id: Session ID.
            model: Override model.
            temperature: Override temperature.
            use_rag: True để sử dụng RAG.

        Yields:
            dict: Mỗi StreamChunk từ server.
        """
        payload = {
            "message": message,
            "session_id": session_id,
            "model": model,
            "temperature": temperature,
            "use_rag": use_rag,
        }

        with requests.post(
            self._url("/chat"),
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        try:
                            chunk = json.loads(data_str)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON chunk: %s", data_str)
                            continue


    def get_history(self, session_id: str) -> dict:
        """
        Lấy conversation history.

        Args:
            session_id: Session ID.

        Returns:
            dict: ChatHistoryResponse.
        """
        try:
            response = requests.get(
                self._url(f"/chat/history/{session_id}"),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("get_history failed: %s", str(e))
            raise

    def clear_history(self, session_id: str) -> dict:
        """
        Xóa conversation history.

        Args:
            session_id: Session ID.

        Returns:
            dict: ClearHistoryResponse.
        """
        try:
            response = requests.delete(
                self._url(f"/chat/history/{session_id}"),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("clear_history failed: %s", str(e))
            raise

    # ------------------------------------------------------------------
    # Document API (Phase 3A)
    # ------------------------------------------------------------------

    def upload_document(self, file_bytes: bytes, filename: str) -> dict:
        """
        Upload file vào Knowledge Base.

        Args:
            file_bytes: Nội dung file dạng bytes.
            filename:   Tên file gốc.

        Returns:
            dict: DocumentMetadata response.
        """
        try:
            response = requests.post(
                self._url("/documents/upload"),
                files={"file": (filename, file_bytes)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            # Trả về error detail từ server
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise ValueError(error_detail) from e
        except Exception as e:
            logger.error("upload_document failed: %s", str(e))
            raise

    def list_documents(self) -> dict:
        """
        Lấy danh sách tất cả documents.

        Returns:
            dict: DocumentListResponse với danh sách documents.
        """
        try:
            response = requests.get(
                self._url("/documents"),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("list_documents failed: %s", str(e))
            raise

    def delete_document(self, doc_id: str) -> dict:
        """
        Xóa một document khỏi Knowledge Base.

        Args:
            doc_id: UUID của document.

        Returns:
            dict: DeleteDocumentResponse.
        """
        try:
            response = requests.delete(
                self._url(f"/documents/{doc_id}"),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("delete_document failed: %s", str(e))
            raise

    def get_collection_stats(self) -> dict:
        """
        Lấy thống kê Knowledge Base collection.

        Returns:
            dict: DocumentStatsResponse.
        """
        try:
            response = requests.get(
                self._url("/documents/stats"),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("get_collection_stats failed: %s", str(e))
            return {
                "total_documents": 0,
                "indexed_documents": 0,
                "total_chunks": 0,
                "collection_name": "N/A",
                "embedding_model": "N/A",
            }

    def reindex_document(self, doc_id: str) -> dict:
        """
        Reindex một document từ đầu.

        Args:
            doc_id: UUID của document.

        Returns:
            dict: DocumentMetadata sau reindex.
        """
        try:
            response = requests.post(
                self._url(f"/documents/{doc_id}/reindex"),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise ValueError(error_detail) from e
        except Exception as e:
            logger.error("reindex_document failed: %s", str(e))
            raise

