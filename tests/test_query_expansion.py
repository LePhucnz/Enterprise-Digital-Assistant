"""
Unit Tests cho kỹ thuật Query Expansion trong RAG pipeline.

Kiểm thử:
1. QueryExpander:
   - Làm sạch từng dòng truy vấn (_clean_query_line)
   - Parse kết quả từ LLM, deduplication và giữ nguyên câu hỏi gốc (_parse_output)
   - Gọi chain và xử lý ngoại lệ (graceful fallback về [câu hỏi gốc])
2. Reciprocal Rank Fusion (RRF):
   - Kết hợp kết quả từ nhiều truy vấn
   - Document xuất hiện ở nhiều query được boost điểm cao hơn
   - Deduplicate tài liệu trùng lặp
   - Giới hạn đúng top_k
3. RetrieverService integration:
   - retrieve() khi bật Query Expansion
   - retrieve() khi tắt Query Expansion
4. ChatService & API payload integration:
   - ChatService trả về expanded_queries trong ChatResponse
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.chat import Message, MessageRole
from app.rag.query_expander import QueryExpander
from app.rag.retriever_service import RetrievalResult, RetrieverService, SourceDocument
from app.services.chat_service import ChatService


class TestQueryExpanderUnit(unittest.TestCase):
    """Kiểm thử unit cho QueryExpander."""

    def setUp(self) -> None:
        self.mock_llm = MagicMock()
        self.expander = QueryExpander(llm=self.mock_llm, temperature=0.3)

    def test_clean_query_line(self) -> None:
        """Kiểm tra làm sạch các tiền tố số thứ tự, gạch đầu dòng, dấu ngoặc kép."""
        cases = [
            ("1. Quy định về nghỉ phép năm", "Quy định về nghỉ phép năm"),
            ("2) Thủ tục xin nghỉ phép", "Thủ tục xin nghỉ phép"),
            ("- Chính sách nghỉ không lương", "Chính sách nghỉ không lương"),
            ("* Quyền hạn của người lao động", "Quyền hạn của người lao động"),
            ("• Điều kiện nghỉ phép", "Điều kiện nghỉ phép"),
            ('"Quy chế quản trị nội bộ"', "Quy chế quản trị nội bộ"),
            ("'Thẩm quyền ký duyệt đơn'", "Thẩm quyền ký duyệt đơn"),
            ("   3 - Quy trình chấm công   ", "Quy trình chấm công"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(QueryExpander._clean_query_line(raw), expected)

    def test_parse_output_deduplication_and_original(self) -> None:
        """Kiểm tra hàm _parse_output luôn giữ query gốc ở đầu và loại bỏ trùng lặp."""
        raw_output = """
1. Quy định về nghỉ phép
2. Thủ tục và quy trình xin nghỉ phép
3. Quy định về nghỉ phép
4. Các trường hợp nghỉ hưởng nguyên lương
5. Chế độ nghỉ phép năm
"""
        original = "Quy định về nghỉ phép"
        parsed = self.expander._parse_output(raw_output, original=original, max_count=3)

        # Query gốc phải ở vị trí đầu
        self.assertEqual(parsed[0], original)
        # Không có phần tử trùng lặp
        self.assertEqual(len(parsed), len(set([p.lower() for p in parsed])))
        # Tối đa original + 3 biến thể = 4
        self.assertLessEqual(len(parsed), 4)

    def test_expand_successful(self) -> None:
        """Kiểm tra expand() khi LLM chạy thành công."""
        self.expander._chain = MagicMock()
        self.expander._chain.invoke.return_value = (
            "1. Tiêu chuẩn thành viên HĐQT\n"
            "2. Quyền hạn và nhiệm vụ của Ban giám đốc\n"
            "3. Cơ cấu quản trị nội bộ doanh nghiệp\n"
        )

        results = self.expander.expand(query="quy chế quản trị", num_queries=3)

        self.assertEqual(results[0], "quy chế quản trị")
        self.assertIn("Tiêu chuẩn thành viên HĐQT", results)
        self.assertIn("Quyền hạn và nhiệm vụ của Ban giám đốc", results)
        self.assertIn("Cơ cấu quản trị nội bộ doanh nghiệp", results)
        self.assertEqual(len(results), 4)

    def test_expand_graceful_fallback_on_error(self) -> None:
        """Kiểm tra expand() tự động fallback về [original] nếu LLM gặp sự cố."""
        self.expander._chain = MagicMock()
        self.expander._chain.invoke.side_effect = Exception("Ollama connection timeout")

        results = self.expander.expand(query="chính sách lương thưởng", num_queries=3)

        # Không raise exception, trả về [original]
        self.assertEqual(results, ["chính sách lương thưởng"])

    def test_expand_empty_query(self) -> None:
        """Kiểm tra query rỗng hoặc khoảng trắng."""
        self.assertEqual(self.expander.expand("   "), [])
        self.assertEqual(self.expander.expand(""), [])


class TestReciprocalRankFusion(unittest.TestCase):
    """Kiểm thử thuật toán Reciprocal Rank Fusion (RRF)."""

    def _create_mock_doc(self, file_name: str, page: int, chunk_id: int, content: str):
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {
            "file_name": file_name,
            "page": page,
            "chunk_id": chunk_id,
            "doc_id": "doc-1",
        }
        return doc

    def test_rrf_boosting_and_deduplication(self) -> None:
        """
        Kiểm tra: Chunk xuất hiện ở nhiều danh sách truy vấn sẽ được boost điểm cao hơn
        và tài liệu trùng lặp được lọc sạch.
        """
        doc_a = self._create_mock_doc("policy.pdf", 1, 1, "Chính sách nghỉ phép chung")
        doc_b = self._create_mock_doc("policy.pdf", 2, 2, "Quy trình nộp đơn nghỉ phép")
        doc_c = self._create_mock_doc("policy.pdf", 3, 3, "Chế độ thai sản và ốm đau")

        # Query 1 trả về [doc_a, doc_b]
        # Query 2 trả về [doc_b, doc_c]
        # Query 3 trả về [doc_b]
        # -> doc_b xuất hiện ở cả 3 query, phải đứng vị trí số 1 trong kết quả cuối!
        results_by_query = [
            [doc_a, doc_b],
            [doc_b, doc_c],
            [doc_b],
        ]

        fused = RetrieverService._reciprocal_rank_fusion(
            results_by_query=results_by_query,
            k=60,
            top_k=5,
        )

        self.assertEqual(len(fused), 3)  # Đã deduplicate (doc_a, doc_b, doc_c)
        self.assertEqual(fused[0].page_content, doc_b.page_content)

    def test_rrf_respects_top_k(self) -> None:
        """Kiểm tra giới hạn top_k của RRF."""
        docs = [self._create_mock_doc("f.pdf", i, i, f"Content {i}") for i in range(10)]
        results_by_query = [docs[:5], docs[5:]]

        fused = RetrieverService._reciprocal_rank_fusion(
            results_by_query=results_by_query,
            k=60,
            top_k=4,
        )

        self.assertEqual(len(fused), 4)


class TestRetrieverServiceQueryExpansion(unittest.TestCase):
    """Kiểm thử tích hợp Query Expansion vào RetrieverService."""

    def setUp(self) -> None:
        self.mock_vs = MagicMock()
        self.mock_expander = MagicMock()
        self.retriever_service = RetrieverService(
            vector_store_service=self.mock_vs,
            query_expander=self.mock_expander,
        )
        self.mock_retriever = MagicMock()
        self.retriever_service._build_retriever = MagicMock(return_value=self.mock_retriever)

    def test_retrieve_with_expansion_enabled(self) -> None:
        """Kiểm thử retrieve() khi Query Expansion được bật."""
        self.mock_expander.expand.return_value = [
            "quy chế làm việc",
            "giờ giấc làm việc và chấm công",
            "nội quy lao động doanh nghiệp",
        ]

        mock_doc = MagicMock()
        mock_doc.page_content = "Nội dung quy chế làm việc"
        mock_doc.metadata = {"file_name": "noiquy.pdf", "page": 1, "chunk_id": 1}

        # Mock _retrieve_with_score_filter thay vì retriever.invoke (đã thay thế)
        self.retriever_service._retrieve_with_score_filter = MagicMock(return_value=[mock_doc])

        result = self.retriever_service.retrieve(
            query="quy chế làm việc",
            enable_query_expansion=True,
        )

        # Phải gọi expand() trên query_expander
        self.mock_expander.expand.assert_called_once_with(
            query="quy chế làm việc",
            num_queries=self.retriever_service._expansion_num_queries,
        )

        # Phải gọi _retrieve_with_score_filter cho cả 3 query
        self.assertEqual(self.retriever_service._retrieve_with_score_filter.call_count, 3)

        # Kết quả phải chứa expanded_queries (không tính original query)
        self.assertEqual(len(result.expanded_queries), 2)
        self.assertIn("giờ giấc làm việc và chấm công", result.expanded_queries)
        self.assertIn("nội quy lao động doanh nghiệp", result.expanded_queries)
        self.assertFalse(result.is_empty)

    def test_retrieve_with_expansion_disabled(self) -> None:
        """Kiểm thử retrieve() khi tắt Query Expansion."""
        mock_doc = MagicMock()
        mock_doc.page_content = "Nội dung trực tiếp"
        mock_doc.metadata = {"file_name": "direct.pdf", "page": 1, "chunk_id": 1}

        # Mock _retrieve_with_score_filter thay vì retriever.invoke (đã thay thế)
        self.retriever_service._retrieve_with_score_filter = MagicMock(return_value=[mock_doc])

        result = self.retriever_service.retrieve(
            query="câu hỏi trực tiếp",
            enable_query_expansion=False,
        )

        # Không được gọi expander
        self.mock_expander.expand.assert_not_called()

        # Chỉ gọi _retrieve_with_score_filter 1 lần duy nhất với câu hỏi gốc
        self.retriever_service._retrieve_with_score_filter.assert_called_once_with("câu hỏi trực tiếp")
        self.assertEqual(result.expanded_queries, [])


class TestChatServiceWithQueryExpansion(unittest.TestCase):
    """Kiểm thử luồng chat trong ChatService khi truyền enable_query_expansion."""

    def setUp(self) -> None:
        self.mock_memory = MagicMock()
        self.mock_memory.backend_type = "mock"
        self.mock_retriever = MagicMock()
        self.chat_service = ChatService(
            memory_manager=self.mock_memory,
            retriever_service=self.mock_retriever,
        )
        self.mock_rag_chain = MagicMock()
        self.chat_service._rag_chain = self.mock_rag_chain

        self.chat_service._output_parser = MagicMock()
        self.chat_service._output_parser.parse.side_effect = lambda raw_output, model, session_id, elapsed_time, **kwargs: MagicMock(
            answer=raw_output
        )

    def test_chat_service_passes_expansion_flag_and_returns_expanded_queries(self) -> None:
        """ChatService gọi retriever với enable_query_expansion và trả về trong ChatResponse."""
        fake_sources = [
            SourceDocument(
                document="doc.pdf",
                page=1,
                chunk_id=0,
                content="test",
            )
        ]
        self.mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[MagicMock()],
            sources=fake_sources,
            context="Context test",
            retrieval_time=0.1,
            is_empty=False,
            expanded_queries=["Biến thể 1", "Biến thể 2"],
        )
        self.mock_rag_chain.invoke.return_value = "Câu trả lời RAG chi tiết"

        response = self.chat_service.chat(
            message="câu hỏi ngắn",
            session_id="session-exp-1",
            use_rag=True,
            enable_query_expansion=True,
        )

        self.mock_retriever.retrieve.assert_called_once_with(
            "câu hỏi ngắn",
            fallback_on_empty=True,
            enable_query_expansion=True,
        )

        self.assertEqual(response.expanded_queries, ["Biến thể 1", "Biến thể 2"])
        self.assertEqual(response.message.content, "Câu trả lời RAG chi tiết")


if __name__ == "__main__":
    unittest.main()

