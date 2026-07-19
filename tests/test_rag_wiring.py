"""
Tests for RAG Wiring — Kiểm thử logic định tuyến (routing) và fallback của RAG chat.

Sử dụng unittest.mock để mock các thành phần hạ tầng (RetrieverService, ChatChain, RAGChain),
giúp chạy test nhanh và không cần Ollama/ChromaDB thật.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.core.output_parser import ParsedResponse, ResponseMetadata
from app.models.chat import Message, MessageRole
from app.services.chat_service import ChatService
from app.rag.retriever_service import RetrievalResult, SourceDocument


class TestRAGWiring(unittest.TestCase):
    """Kiểm thử tích hợp các luồng switch giữa RAG và Normal Chat."""

    def setUp(self) -> None:
        # Mock MemoryManager
        self.mock_memory_manager = MagicMock()
        self.mock_memory_manager.backend_type = "mock"
        self.mock_memory_manager.get_messages.return_value = []

        # Mock RetrieverService
        self.mock_retriever = MagicMock()

        # Khởi tạo ChatService với mock memory và mock retriever
        self.chat_service = ChatService(
            memory_manager=self.mock_memory_manager,
            retriever_service=self.mock_retriever,
        )

        # Mock ChatChain và RAGChain
        self.mock_chat_chain = MagicMock()
        self.mock_rag_chain = MagicMock()

        # Patch các chain singletons
        self.chat_service._chain = self.mock_chat_chain
        self.chat_service._rag_chain = self.mock_rag_chain

        # Mock output parser để parse response
        self.chat_service._output_parser = MagicMock()
        self.chat_service._output_parser.parse.side_effect = lambda raw_output, model, session_id, elapsed_time, **kwargs: ParsedResponse(
            answer=raw_output,
            metadata=ResponseMetadata(
                model=model,
                elapsed_time=elapsed_time,
                session_id=session_id,
            ),
        )

    def test_normal_chat_when_use_rag_is_false(self) -> None:
        """
        Test case: use_rag=False.

        Kỳ vọng: Gọi ChatChain, không gọi RetrieverService, trả về sources=None.
        """
        self.mock_chat_chain.invoke.return_value = "Câu trả lời thường"

        response = self.chat_service.chat(
            message="Xin chào",
            session_id="session-test-1",
            use_rag=False,
        )

        # Kiểm tra chain được gọi
        self.mock_chat_chain.invoke.assert_called_once_with(
            input_text="Xin chào",
            session_id="session-test-1",
        )
        self.mock_rag_chain.invoke.assert_not_called()
        self.mock_retriever.retrieve.assert_not_called()

        # Kiểm tra response
        self.assertEqual(response.message.content, "Câu trả lời thường")
        self.assertIsNone(response.sources)

    def test_rag_chat_when_use_rag_is_true(self) -> None:
        """
        Test case: use_rag=True và có retriever.

        Kỳ vọng: Gọi RetrieverService để lấy context, gọi RAGChain, trả về response kèm sources.
        """
        # Giả lập kết quả từ retriever
        fake_docs = [MagicMock()]
        fake_sources = [
            SourceDocument(
                document="tailieu.pdf",
                page=1,
                chunk_id=0,
                content="preview",
            )
        ]
        self.mock_retriever.retrieve.return_value = RetrievalResult(
            documents=fake_docs,
            sources=fake_sources,
            context="Nội dung tài liệu giả lập",
            retrieval_time=0.1,
            is_empty=False,
        )

        self.mock_rag_chain.invoke.return_value = "Câu trả lời RAG"

        response = self.chat_service.chat(
            message="Tìm quy trình nghỉ phép",
            session_id="session-test-2",
            use_rag=True,
        )

        # Kiểm tra retriever được gọi
        self.mock_retriever.retrieve.assert_called_once_with(
            "Tìm quy trình nghỉ phép",
            fallback_on_empty=True,
        )

        # Kiểm tra RAGChain được gọi với context
        self.mock_rag_chain.invoke.assert_called_once_with(
            input_text="Tìm quy trình nghỉ phép",
            context="Nội dung tài liệu giả lập",
            session_id="session-test-2",
        )
        self.mock_chat_chain.invoke.assert_not_called()

        # Kiểm tra response chứa sources
        self.assertEqual(response.message.content, "Câu trả lời RAG")
        self.assertIsNotNone(response.sources)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0]["document"], "tailieu.pdf")
        self.assertEqual(response.sources[0]["page"], 1)

    def test_graceful_fallback_when_retriever_is_none(self) -> None:
        """
        Test case: use_rag=True nhưng retriever_service là None (không cấu hình).

        Kỳ vọng: Fallback về ChatChain (normal chat), không lỗi, trả về sources=None.
        """
        # Tạo ChatService với retriever = None
        service_no_rag = ChatService(
            memory_manager=self.mock_memory_manager,
            retriever_service=None,
        )
        service_no_rag._chain = self.mock_chat_chain
        service_no_rag._output_parser = self.chat_service._output_parser

        self.mock_chat_chain.invoke.return_value = "Câu trả lời thường (fallback)"

        response = service_no_rag.chat(
            message="Hỏi gì đó",
            session_id="session-test-3",
            use_rag=True,
        )

        # Kiểm tra chỉ ChatChain được gọi
        self.mock_chat_chain.invoke.assert_called_once()
        self.assertEqual(response.message.content, "Câu trả lời thường (fallback)")
        self.assertIsNone(response.sources)

    def test_rag_with_empty_retrieval(self) -> None:
        """
        Test case: use_rag=True nhưng không tìm thấy tài liệu nào (retrieval rỗng).

        Kỳ vọng: Vẫn gọi RAGChain với empty context thông báo, không raise lỗi, trả về sources trống.
        """
        # Giả lập kết quả rỗng
        self.mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[],
            sources=[],
            context="Không có tài liệu liên quan được tìm thấy.",
            retrieval_time=0.05,
            is_empty=True,
        )

        self.mock_rag_chain.invoke.return_value = (
            "Tôi không tìm thấy thông tin này trong tài liệu."
        )

        response = self.chat_service.chat(
            message="Chính sách lương 2026",
            session_id="session-test-4",
            use_rag=True,
        )

        # Kiểm tra retriever được gọi
        self.mock_retriever.retrieve.assert_called_once()

        # Kiểm tra RAGChain vẫn được gọi với context thông báo rỗng
        self.mock_rag_chain.invoke.assert_called_once_with(
            input_text="Chính sách lương 2026",
            context="Không có tài liệu liên quan được tìm thấy.",
            session_id="session-test-4",
        )

        self.assertEqual(
            response.message.content, "Tôi không tìm thấy thông tin này trong tài liệu."
        )
        self.assertEqual(response.sources, [])


if __name__ == "__main__":
    unittest.main()
