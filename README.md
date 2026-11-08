# 🏢 Enterprise Digital Assistant

<p align="center">
  <strong>Trợ lý AI Doanh Nghiệp Toàn Diện — Bảo Mật Nội Bộ, Tích Hợp RAG & Chạy 100% Local với Ollama</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/LangChain-v0.3-1C3C3C?logo=langchain" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-orange" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-black" alt="Ollama" />
</p>

---

## 📖 Giới thiệu

**Enterprise Digital Assistant** là giải pháp trợ lý ảo thông minh dành cho doanh nghiệp, được thiết kế theo kiến trúc phân tầng chuẩn mực (**Clean Architecture**). Hệ thống vận hành hoàn toàn độc lập trong mạng nội bộ (On-Premise / Local) thông qua **Ollama**, đảm bảo **100% quyền riêng tư và an toàn dữ liệu**, không rò rỉ dữ liệu tài liệu hay lịch sử hội thoại ra bên ngoài internet.

Hệ thống kết hợp sức mạnh của mô hình ngôn ngữ lớn (**Qwen 2.5**), cơ chế tạo sinh có tăng cường truy xuất tri thức (**RAG - Retrieval-Augmented Generation**), cơ sở dữ liệu véc-tơ (**ChromaDB**), backend API hiệu năng cao (**FastAPI**) và giao diện người dùng trực quan (**Streamlit**).

---

## ✨ Tính năng nổi bật

### 1. 💬 Trò chuyện Thông minh & Phản hồi Thời gian thực (Chat Engine)
- **Streaming Response (SSE)**: Phản hồi từng token mượt mà theo thời gian thực qua giao thức Server-Sent Events.
- **Quản lý ngữ cảnh (Session & Memory)**: Lưu vết lịch sử trò chuyện theo phiên làm việc (`session_id`), hỗ trợ bộ nhớ đệm Buffer Memory hoặc Summary Memory.
- **Tùy biến tham số LLM**: Điều chỉnh trực tiếp Model (`qwen2.5`, `llama3.2`, `mistral`), Temperature (độ sáng tạo), Top-p ngay trên giao diện.

### 2. 📚 Quản lý Kho Tri thức Doanh nghiệp (Enterprise Knowledge Base)
- **Hỗ trợ đa định dạng tài liệu**: Upload và lập chỉ mục tự động cho `.pdf`, `.docx`, `.txt`, `.csv`, `.md`.
- **Pipeline trích xuất & Chunking**: Tự động phân tích cấu trúc, chia nhỏ văn bản thông minh bằng `RecursiveCharacterTextSplitter` với metadata tracking (tên file, số trang, chunk ID).
- **Chống trùng lặp (SHA-256 Checksum)**: Tự động phát hiện và ngăn chặn upload lặp lại các tài liệu trùng nội dung.
- **Quản lý vòng đời tài liệu**: Theo dõi trạng thái tài liệu (`PENDING` ➔ `PROCESSING` ➔ `INDEXED` / `FAILED`), hỗ trợ xóa và reindex tài liệu dễ dàng.

### 3. 🎯 Pipeline RAG Nâng cao & Chống Ảo giác (Advanced RAG)
- **Local Embeddings**: Chuyển đổi văn bản thành véc-tơ thông qua model `nomic-embed-text` chạy trên Ollama.
- **ChromaDB Vector Store**: Lưu trữ véc-tơ bền vững (persistent), hỗ trợ tìm kiếm ngữ nghĩa cực nhanh.
- **Đa chiến lược tìm kiếm**:
  - `Similarity Search`: Cosine similarity chuẩn xác.
  - `MMR (Max Marginal Relevance)`: Tối ưu độ liên quan và đa dạng hóa nội dung, giảm trùng lặp thông tin.
  - `Similarity Score Threshold`: Lọc bỏ triệt để các đoạn văn bản có độ tương đồng thấp.
- **Chống ảo giác (Anti-Hallucination Prompting)**: Ép mô hình chỉ trả lời dựa trên nội dung tài liệu trích xuất; nếu tài liệu không đề cập, trợ lý sẽ thông báo rõ ràng thay vì suy đoán sai lệch.
- **Trích dẫn nguồn tài liệu minh bạch (Source Citation)**: Trả về chính xác tên tài liệu, trang tài liệu và đoạn tham chiếu đi kèm câu trả lời.

---

## 🏗️ Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (run_frontend.py)                 │
│  ┌─────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │   Chat UI (Markdown / SSE)  │   │     Knowledge Base Sidebar      │  │
│  │   Toggle: Use RAG / Stream  │   │  Upload / List / Delete / Stats │  │
│  └──────────────┬──────────────┘   └────────────────┬────────────────┘  │
└─────────────────┼───────────────────────────────────┼───────────────────┘
                  │ HTTP JSON / SSE (Stream)          │ Multipart / HTTP
┌─────────────────▼───────────────────────────────────▼───────────────────┐
│                     FastAPI Backend (run_api.py)                        │
│  ┌─────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │     /api/v1/chat Router     │   │    /api/v1/documents Router     │  │
│  └──────────────┬──────────────┘   └────────────────┬────────────────┘  │
│                 │                                   │                   │
│  ┌──────────────▼──────────────┐   ┌────────────────▼────────────────┐  │
│  │         ChatService         │   │         DocumentService         │  │
│  │ (Orchestrator RAG / Chat)   │   │     (Pipeline Ingestion)        │  │
│  └───┬─────────────────────┬───┘   └───┬────────────┬────────────┬───┘  │
│      │                     │           │            │            │      │
│  ┌───▼───────────┐  ┌──────▼──────┐ ┌──▼───────┐ ┌──▼───────┐ ┌──▼───┐  │
│  │RetrieverService│  │  RAGChain & │ │ Loader  │ │Splitter │ │Repo  │  │
│  │(Sim/MMR/Thresh│  │  ChatChain  │ │ Factory │ │Service  │ │(JSON)│  │
│  └───────┬───────┘  └──────┬──────┘ └──────────┘ └──────────┘ └──────┘  │
└──────────┼─────────────────┼────────────────────────────────────────────┘
           │ Queries         │ Generates / Embeds
┌──────────▼─────────────────▼────────────────────────────────────────────┐
│                       Hạ Tầng Lưu Trữ & AI (Local)                      │
│  ┌─────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │   ChromaDB (vector_db/)     │   │       Ollama Server (Local)     │  │
│  │   Collection: enterprise_kb │   │ - LLM: qwen2.5:latest           │  │
│  │   Metadata + Embeddings     │   │ - Embed: nomic-embed-text       │  │
│  └─────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc Thư mục

```text
Enterprise-Digital-Assistant/
├── app/
│   ├── api/                     # Tầng API Routers & Dependencies
│   │   ├── dependencies.py      # Dependency injection (Services, Repositories)
│   │   └── v1/
│   │       ├── chat.py          # Endpoints chat (sync, streaming SSE, history)
│   │       ├── documents.py     # Endpoints quản lý tài liệu (upload, list, stats...)
│   │       └── health.py        # Health check endpoint (API, Ollama status)
│   ├── chains/                  # LangChain LCEL Chains
│   │   ├── chat_chain.py        # Standard Conversation Chain
│   │   └── rag_chain.py         # RAG Chain kết hợp Context + History
│   ├── config/
│   │   └── settings.py          # Quản lý cấu hình tập trung qua Pydantic Settings
│   ├── core/                    # Core abstractions & Tiện ích chung
│   │   ├── base_chain.py        # Lớp cơ sở cho các Chains
│   │   ├── exceptions.py        # Hệ thống mã lỗi phân cấp (Domain Exceptions)
│   │   └── output_parser.py     # Bộ xử lý output & streaming tokens
│   ├── documents/               # Quản lý tài liệu & lưu trữ
│   │   ├── document_service.py  # Orchestrator upload, chunk, embed, delete
│   │   ├── loader_factory.py    # Factory nạp tài liệu (PDF, DOCX, CSV, TXT, MD)
│   │   ├── models.py            # Pydantic schemas cho tài liệu
│   │   └── repository.py        # Lưu trữ metadata tài liệu (JSON repository)
│   ├── memory/
│   │   └── memory_manager.py    # Quản lý bộ nhớ hội thoại theo session_id
│   ├── models/
│   │   └── chat.py              # Pydantic schemas cho Chat API (Request/Response)
│   ├── prompts/
│   │   └── chat_prompts.py      # System prompts & template cho Chat và RAG
│   ├── rag/                     # Hạ tầng RAG
│   │   ├── embedding_service.py # Service bọc Ollama Embeddings
│   │   ├── retriever_service.py # Tìm kiếm đa chiến lược (Similarity, MMR, Threshold)
│   │   ├── text_splitter.py     # Chia nhỏ tài liệu và gán metadata véc-tơ
│   │   └── vector_store_service.py # Wrapper giao tiếp ChromaDB
│   ├── services/
│   │   └── chat_service.py      # Business logic điều phối Chat và RAG
│   └── main.py                  # Entry point FastAPI, CORS & Lifespan
├── frontend/
│   ├── api_client.py            # HTTP Client giao tiếp backend từ Streamlit
│   └── app.py                   # Giao diện người dùng Streamlit
├── docs/                        # Tài liệu kỹ thuật chi tiết
│   ├── phase3a.md               # Kiến trúc Knowledge Base Ingestion
│   └── phase3b.md               # Kiến trúc RAG Pipeline & Retrieval
├── tests/
│   └── test_rag_wiring.py       # Unit tests kiểm thử định tuyến RAG & Fallback
├── .env.example                 # Mẫu biến môi trường cấu hình
├── .gitignore                   # Cấu hình bỏ qua file trong Git
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── run_api.py                   # Script khởi chạy FastAPI server
└── run_frontend.py              # Script khởi chạy Streamlit frontend
```

---

## ⚙️ Yêu cầu Hệ thống

1. **Hệ điều hành**: Linux, macOS, hoặc Windows (hỗ trợ WSL2).
2. **Python**: Phiên bản `3.10` trở lên (khuyến nghị Python 3.11).
3. **Ollama**: Đã cài đặt và đang chạy dịch vụ tại `http://localhost:11434`.
   - Cài đặt Ollama từ trang chủ: [https://ollama.com](https://ollama.com)

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Clone kho mã nguồn
```bash
git clone https://github.com/LePhucnz/Enterprise-Digital-Assistant.git
cd Enterprise-Digital-Assistant
```

### Bước 2: Tạo môi trường ảo Python & Cài đặt thư viện
```bash
# Khởi tạo virtual environment
python3 -m venv venv

# Kích hoạt môi trường ảo:
# Trên Linux/macOS:
source venv/bin/activate
# Trên Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

# Nâng cấp pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Tải các mô hình cần thiết trên Ollama
Mở terminal và pull các model phục vụ Chat và Embedding:
```bash
# Tải model LLM chính (Qwen 2.5)
ollama pull qwen2.5:latest

# Tải model Embedding tạo vector cho tài liệu
ollama pull nomic-embed-text:latest
```

### Bước 4: Cấu hình biến môi trường
Tạo file `.env` từ mẫu `.env.example`:
```bash
cp .env.example .env
```
Bạn có thể tùy chỉnh các thông số trong file `.env` nếu cần (ví dụ port, model, chunk size, top-k...).

### Bước 5: Khởi động Backend FastAPI
Mở một terminal mới (đã kích hoạt venv) và chạy:
```bash
python run_api.py
```
- API Server sẽ khởi chạy tại: `http://localhost:8000`
- Tài liệu trực quan Swagger UI: `http://localhost:8000/docs`
- Tài liệu ReDoc: `http://localhost:8000/redoc`

### Bước 6: Khởi động Giao diện Streamlit Frontend
Mở một terminal khác (đã kích hoạt venv) và chạy:
```bash
streamlit run run_frontend.py
```
Giao diện web sẽ tự động mở tại trình duyệt theo địa chỉ: `http://localhost:8501`.

---

## 📡 Danh sách API Endpoints Chính

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| **GET** | `/api/v1/health` | Kiểm tra trạng thái hệ thống và kết nối Ollama |
| **POST** | `/api/v1/chat` | Chat thời gian thực dạng Streaming (Server-Sent Events) |
| **POST** | `/api/v1/chat/sync` | Chat đồng bộ (Non-streaming), trả về JSON kèm sources |
| **GET** | `/api/v1/chat/history/{session_id}` | Xem toàn bộ lịch sử tin nhắn của phiên làm việc |
| **DELETE** | `/api/v1/chat/history/{session_id}` | Xóa toàn bộ lịch sử chat của phiên làm việc |
| **POST** | `/api/v1/documents/upload` | Upload file tài liệu, chia đoạn và index vào ChromaDB |
| **GET** | `/api/v1/documents` | Lấy danh sách tất cả tài liệu đã upload |
| **GET** | `/api/v1/documents/stats` | Xem thống kê số lượng tài liệu, tổng số chunks |
| **GET** | `/api/v1/documents/{doc_id}` | Lấy thông tin chi tiết một tài liệu |
| **DELETE** | `/api/v1/documents/{doc_id}` | Xóa tài liệu khỏi bộ nhớ và loại bỏ véc-tơ tương ứng |
| **POST** | `/api/v1/documents/{doc_id}/reindex` | Lập chỉ mục lại véc-tơ cho tài liệu |

---

## 🧪 Kiểm thử (Testing)

Dự án bao gồm bộ kiểm thử tự động sử dụng `pytest` và mock hạ tầng:
```bash
pytest tests/test_rag_wiring.py -v
```

Nội dung kiểm thử kiểm tra toàn diện:
- Luồng chat thông thường khi tắt RAG (`use_rag=False`).
- Luồng chat RAG kích hoạt tìm kiếm tài liệu và đính kèm nguồn (`use_rag=True`).
- Cơ chế fallback an toàn khi không tìm thấy tài liệu phù hợp trong kho tri thức.

---

## 🗺️ Lộ trình Phát triển (Roadmap)

- [x] **Phase 1**: Xây dựng kiến trúc nền tảng FastAPI + Streamlit, streaming tokens, ChatChain cơ bản.
- [x] **Phase 2**: Tách lớp Clean Architecture, quản lý Session Memory, Output Parser, Exception Handling chuẩn hóa.
- [x] **Phase 3A**: Xây dựng Knowledge Base Ingestion Pipeline (Upload, Load đa định dạng, Chunking, ChromaDB Vector Store).
- [x] **Phase 3B**: Hoàn thiện RAG Chain, trích dẫn nguồn (Source Citations), cơ chế chống ảo giác và tích hợp giao diện.
- [ ] **Phase 4**: Agentic AI & Tool Calling (Tích hợp công cụ tính toán, tra cứu dữ liệu cơ sở dữ liệu SQL, Web Search nội bộ).

---

## 🤝 Đóng góp & Phát triển

Mọi đóng góp, báo cáo lỗi hoặc đề xuất tính năng mới đều được hoan nghênh:
1. Fork dự án.
2. Tạo nhánh tính năng mới (`git checkout -b feature/AmazingFeature`).
3. Commit các thay đổi (`git commit -m 'feat: Add some AmazingFeature'`).
4. Đẩy lên branch (`git push origin feature/AmazingFeature`).
5. Tạo Pull Request.

---

## 📄 Bản quyền (License)

Dự án được phát hành theo giấy phép mã nguồn mở nội bộ doanh nghiệp. Mọi thông tin chi tiết vui lòng liên hệ nhóm phát triển.
