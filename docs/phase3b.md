# Phase 3B — Enterprise RAG Pipeline

Tài liệu thiết kế chi tiết và kiến trúc hệ thống RAG (Retrieval-Augmented Generation) cho Enterprise Digital Assistant.

---

## 1. RAG Architecture (Kiến trúc RAG)

Hệ thống RAG hoạt động theo kiến trúc phân tầng (Clean Architecture), tách biệt giữa tầng lưu trữ dữ liệu (Vector Store), tầng nghiệp vụ (RetrieverService, ChatService), tầng liên kết LCEL (RAGChain) và tầng hiển thị (FastAPI, Streamlit).

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    Streamlit Frontend                        │
 │  ┌───────────────────────────────────────────────────────┐  │
 │  │ Chat UI (hiển thị Markdown, trích dẫn tài liệu)       │  │
 │  │ Toggle: Use RAG, Streaming Response                   │  │
 │  └──────────────────────────┬────────────────────────────┘  │
 └─────────────────────────────┼───────────────────────────────┘
                               │ HTTP JSON / SSE (Stream)
 ┌─────────────────────────────▼───────────────────────────────┐
 │                      FastAPI Backend                          │
 │  ┌───────────────────────────────────────────────────────┐  │
 │  │ /api/v1/chat & /chat/sync (Router nhận use_rag)      │  │
 │  └──────────────────────────┬────────────────────────────┘  │
 │                             │ Call
 │  ┌──────────────────────────▼────────────────────────────┐  │
 │  │ ChatService (Orchestrator định tuyến RAG/Normal)      │  │
 │  └──┬──────────────────────────────────┬─────────────────┘  │
 │     │ 1. Gọi                           │ 2. Gọi             │
 │  ┌──▼────────────────────┐      ┌──────▼─────────────────┐  │
 │  │ RetrieverService      │      │ RAGChain               │  │
 │  │ (Similarity, MMR,     │      │ (BaseChain LCEL với    │  │
 │  │  Score Threshold)      │      │  {context} & history)  │  │
 │  └──┬────────────────────┘      └──────┬─────────────────┘  │
 │     │ Queries                          │ Generates          │
 └─────┼──────────────────────────────────┼────────────────────┘
       │                                  │
 ┌─────▼──────────────────────────────────▼────────────────────┐
 │                       Infrastructure                        │
 │  ┌────────────────────┐          ┌───────────────────────┐  │
 │  │ ChromaDB           │          │ Ollama Server (Local) │  │
 │  │ (Vector Store)     │          │ - qwen2.5 (LLM)       │  │
 │  │ - collection:      │          │ - nomic-embed (Embed) │  │
 │  │   enterprise_kb    │          │                       │  │
 │  └────────────────────┘          └───────────────────────┘  │
 └─────────────────────────────────────────────────────────────┘
```

---

## 2. Retrieval Flow (Luồng Tìm Kiếm)

Khi user gửi tin nhắn với `use_rag=True`:

1. **Nhận Query**: `ChatService` kiểm tra trạng thái và định tuyến sang RAG workflow.
2. **Truy vấn Vector DB**: `RetrieverService` nhận query và cấu hình search từ `settings`.
3. **Chiến lược Tìm kiếm (Search Strategies)**:
   - **Similarity Search**: Cosine similarity chuẩn để lấy top_k chunks tương đồng nhất.
   - **MMR (Max Marginal Relevance)**: Tối ưu hóa tính tương đồng đồng thời tối đa hóa sự đa dạng của thông tin để tránh trùng lặp nội dung.
   - **Similarity Score Threshold**: Lọc bỏ các document chunks có độ tương đồng thấp hơn `rag_score_threshold`.
4. **Deduplicate & Extract Sources**:
   - `_extract_sources()` đọc qua metadata của các chunks tìm được.
   - Trích xuất: `document` (file_name), `page` (được chuyển từ 0-indexed sang 1-indexed), `chunk_id`.
   - Lọc trùng (deduplicate) theo cặp `(document, page)`.
5. **Đóng gói Kết quả**: Trả về `RetrievalResult` gồm danh sách documents, structured sources, context string (đã format có header) và thời gian thực hiện.

---

## 3. Prompt Flow & LCEL Chain

### LCEL RAG Chain Flow
```
{input_text, context} 
       │
       ▼
 RAG Prompt Template (get_rag_prompt_template)
       │ (System Prompt + History + Human Message)
       ▼
 ChatOllama (Local LLM qwen2.5)
       │
       ▼
 StrOutputParser (String response)
```

### System Prompt (Cơ chế chống Hallucination)
System prompt được cấu hình nghiêm ngặt để đảm bảo AI trợ lý không bao giờ tự ý bịa đặt thông tin nằm ngoài phạm vi tài liệu:

* **Chỉ sử dụng Context**: Ép LLM chỉ trả lời bằng thông tin nằm trong thẻ `<context>`.
* **Fallback rõ ràng**: Khi context trống hoặc không chứa thông tin, trả lời chuẩn: *"Tôi không tìm thấy thông tin về vấn đề này trong tài liệu hiện có."*
* **Yêu cầu trích dẫn**: Chỉ định AI chỉ rõ tên tài liệu và số trang ở cuối câu trả lời.

---

## 4. Sequence Diagram (Sơ đồ Tuần tự RAG Chat)

```
Client (Streamlit)    FastAPI Router    ChatService    RetrieverService    ChromaDB     RAGChain    ChatOllama
       │                    │                │                 │               │            │           │
       │ POST /chat/sync    │                │                 │               │            │           │
       ├───────────────────>│                │                 │               │            │           │
       │ (use_rag=True)     │ chat()         │                 │               │            │           │
       │                    ├───────────────>│                 │               │            │           │
       │                    │                │ retrieve()      │               │            │           │
       │                    │                ├────────────────>│               │            │           │
       │                    │                │                 │ Query vectors │            │           │
       │                    │                │                 ├──────────────>│            │           │
       │                    │                │                 │ Documents     │            │           │
       │                    │                │                 │<──────────────┤            │           │
       │                    │                │ RetrievalResult │               │            │           │
       │                    │                │<────────────────┤               │            │           │
       │                    │                │                                 │ invoke()   │           │
       │                    │                ├─────────────────────────────────────────────>│           │
       │                    │                │                                 │            │ Generate  │
       │                    │                │                                 │            ├──────────>│
       │                    │                │                                 │            │ Response  │
       │                    │                │                                 │            │<──────────┤
       │                    │                │                                 │ Raw text   │           │
       │                    │                │                                 │<───────────┤           │
       │                    │ ChatResponse   │                                 │            │           │
       │                    │ (kèm sources)  │                                 │            │           │
       │                    │<───────────────┤                                 │            │           │
       │ JSON Response      │                │                                 │            │           │
       │<───────────────────┤                │                                 │            │           │
```

---

## 5. Source Citation (Cơ chế Trích dẫn Nguồn)

* **Trích dẫn đồng bộ (Sync)**:
  Nguồn tài liệu được đính kèm trực tiếp trong JSON response:
  ```json
  {
    "session_id": "8a8f15b8-d2ea-46b5",
    "message": {
      "role": "assistant",
      "content": "Theo chính sách nghỉ phép của doanh nghiệp, nhân viên có 12 ngày phép năm..."
    },
    "model": "qwen2.5:latest",
    "sources": [
      {
        "document": "Chinh_sach_nhan_su_2026.pdf",
        "page": 5,
        "chunk_id": 12
      }
    ]
  }
  ```

* **Trích dẫn bất đồng bộ (Streaming SSE)**:
  Vì RAG retrieval chạy đồng bộ trước khi LLM bắt đầu sinh câu trả lời, danh sách `sources` đã được xác định trước.
  Tuy nhiên, để tối ưu UX, danh sách `sources` sẽ được gửi kèm trong event chunk cuối cùng (`is_done=True`), giúp frontend vẽ ra panel tham khảo một cách mượt mà sau khi AI đã hiển thị xong câu trả lời.

---

## 6. Setup & Hướng dẫn Demo

### Bước 1: Khởi động API Backend
```bash
python run_api.py
```
FastAPI sẽ chạy tại địa chỉ `http://localhost:8000`. Bạn có thể truy cập `/docs` để kiểm tra tài liệu Swagger API.

### Bước 2: Khởi động Frontend
```bash
streamlit run run_frontend.py
```

### Bước 3: Quy trình Test/Demo RAG
1. **Upload tài liệu**: Tại giao diện Streamlit, dùng sidebar upload một tài liệu PDF (ví dụ: Quy chế lương, quy trình làm việc). Nhấn nút **Index** và chờ trạng thái báo indexed thành công.
2. **Test Chat thường (Normal Chat)**:
   - Tắt checkbox/toggle **Sử dụng Knowledge Base (Use RAG)**.
   - Hỏi: *"Quy trình xin nghỉ phép như thế nào?"*
   - Kiểm tra: AI sẽ trả lời chung chung hoặc báo không biết chính xác chính sách công ty. Không có panel nguồn tài liệu bên dưới.
3. **Test RAG Chat**:
   - Bật checkbox/toggle **Sử dụng Knowledge Base (Use RAG)**.
   - Hỏi câu tương tự.
   - Kiểm tra: AI sẽ trả lời chi tiết và chính xác theo văn bản đã upload, ở cuối hiển thị expander **📚 Nguồn tài liệu tham chiếu** chỉ rõ tên file và trang tài liệu tham khảo.
