# Phase 3A — Enterprise Knowledge Base

**Phiên bản**: 3.0.0  
**Ngày hoàn thành**: 2026-07-19  
**Mô tả**: Upload tài liệu → Chunk → Embed (Ollama) → Store (ChromaDB)

---

## 1. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────────┐  ┌──────────────────────────────────────┐ │
│  │  Chat UI     │  │    Knowledge Base Sidebar             │ │
│  │  (Phase 1)   │  │  Upload / List / Delete / Stats      │ │
│  └──────┬───────┘  └──────────────┬───────────────────────┘ │
└─────────┼──────────────────────────┼─────────────────────────┘
          │ HTTP                     │ HTTP
┌─────────▼──────────────────────────▼─────────────────────────┐
│                      FastAPI Backend                          │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐│
│  │  /api/v1/chat   │  │       /api/v1/documents             ││
│  │  (Phase 1, 2)   │  │ upload / list / get / delete /      ││
│  └────────┬────────┘  │ reindex / stats                     ││
│           │           └──────────────┬──────────────────────┘│
│  ┌────────▼────────┐  ┌──────────────▼──────────────────────┐│
│  │  ChatService    │  │         DocumentService              ││
│  │  ChatChain      │  │   (Pipeline Orchestrator)            ││
│  │  MemoryManager  │  └──┬───────────┬────────────┬─────────┘│
│  └─────────────────┘     │           │            │           │
│                    ┌─────▼──┐  ┌─────▼──┐  ┌─────▼──────────┐│
│                    │Loader  │  │Splitter│  │VectorStore     ││
│                    │Factory │  │Service │  │Service         ││
│                    └────────┘  └────────┘  └────────────────┘│
└───────────────────────────────────────────────────────────────┘
                                                │
                           ┌────────────────────▼──────────────┐
                           │          Infrastructure            │
                           │  ┌──────────────┐ ┌─────────────┐ │
                           │  │   ChromaDB   │ │  documents  │ │
                           │  │  vector_db/  │ │   .json     │ │
                           │  └──────────────┘ └─────────────┘ │
                           │  ┌──────────────┐                  │
                           │  │ Ollama Server│                  │
                           │  │nomic-embed   │                  │
                           │  └──────────────┘                  │
                           └───────────────────────────────────┘
```

---

## 2. Upload Flow (Pipeline)

```
User uploads file
        │
        ▼
[1] Validate extension (.pdf/.docx/.txt/.csv/.md)
        │ ❌ → UnsupportedFileTypeError (HTTP 415)
        ▼
[2] Compute SHA256 hash
        │
        ▼
[3] Duplicate check (DocumentRepository.get_by_hash)
        │ ❌ → DocumentAlreadyExistsError (HTTP 409)
        ▼
[4] Save file bytes → data/{doc_id[:8]}_{filename}
        │
        ▼
[5] Create DocumentMetadata (status=PENDING)
    → Save to data/documents.json
        │
        ▼
[6] PENDING → PROCESSING
        │
        ▼
[7] DocumentLoaderFactory.create(file_path)
    → Load raw pages/rows as List[Document]
        │ ❌ → LoaderException (HTTP 422)
        ▼
[8] TextSplitterService.split(documents)
    → RecursiveCharacterTextSplitter
    → Inject metadata: doc_id, chunk_id, file_name, file_type
        │
        ▼
[9] VectorStoreService.add_documents(chunks)
    → ChromaDB.add_documents()
    → OllamaEmbeddings (nomic-embed-text) auto-embed
        │ ❌ → EmbeddingException (HTTP 422)
        ▼
[10] Update status → INDEXED (chunk_count=N)
        │
        ▼
Return DocumentMetadata (HTTP 201)
```

---

## 3. Loader Flow

```
File extension
    ├── .pdf      → PyPDFLoader       → List[Document] (1 doc/page)
    ├── .docx     → Docx2txtLoader    → List[Document] (1 doc tổng)
    ├── .txt      → TextLoader        → List[Document] (1 doc tổng)
    ├── .csv      → CSVLoader         → List[Document] (1 doc/row)
    ├── .md       → TextLoader        → List[Document] (1 doc tổng)
    └── .markdown → TextLoader        → List[Document] (1 doc tổng)
```

Mỗi `Document` có:
- `page_content`: Nội dung text.
- `metadata`: Thông tin từ loader (page number, source path, row index...).

---

## 4. Embedding Flow

```
List[Document] (chunks)
        │
        ▼
TextSplitterService.split()
→ RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
→ Inject metadata vào mỗi chunk:
    {
        "doc_id": "uuid",
        "file_name": "report.pdf",
        "file_type": "pdf",
        "chunk_id": 0,
        "chunk_total": 42,
        "collection_name": "enterprise_kb"
    }
        │
        ▼
VectorStoreService.add_documents(chunks)
→ Chroma(collection_name, embedding_function=OllamaEmbeddings)
→ OllamaEmbeddings.embed_documents(texts)
→ POST http://localhost:11434/api/embeddings
→ {model: "nomic-embed-text", vectors: [[0.1, 0.2, ...]]}
→ ChromaDB stores: (id, vector, text, metadata)
        │
        ▼
data/vector_db/ (ChromaDB persistent files)
```

---

## 5. ChromaDB Storage

### Collection Structure

| Field | Value | Mô tả |
|---|---|---|
| `collection_name` | `enterprise_kb` | Từ config COLLECTION_NAME |
| `persist_directory` | `./vector_db/` | Từ config VECTOR_DB_DIR |

### Chunk Metadata trong ChromaDB

```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "annual_report_2024.pdf",
  "file_type": "pdf",
  "chunk_id": 5,
  "chunk_total": 42,
  "collection_name": "enterprise_kb",
  "source": "data/550e8400_annual_report_2024.pdf",
  "page": 2
}
```

### Delete Strategy

Khi xóa document, `VectorStoreService.delete_document(doc_id)` dùng ChromaDB `where` filter:
```python
results = vs.get(where={"doc_id": doc_id})
vs.delete(ids=results["ids"])
```

---

## 6. Sequence Diagram — Upload Document

```
Client          FastAPI         DocumentService    LoaderFactory    TextSplitter   ChromaDB
  │               │                   │                 │               │             │
  │ POST /upload  │                   │                 │               │             │
  │──────────────▶│                   │                 │               │             │
  │               │ upload_document() │                 │               │             │
  │               │──────────────────▶│                 │               │             │
  │               │                   │ validate ext    │               │             │
  │               │                   │ compute hash    │               │             │
  │               │                   │ duplicate check │               │             │
  │               │                   │ save file       │               │             │
  │               │                   │ create metadata │               │             │
  │               │                   │                 │               │             │
  │               │                   │ create(path)    │               │             │
  │               │                   │────────────────▶│               │             │
  │               │                   │◀────────────────│ loader        │             │
  │               │                   │                 │               │             │
  │               │                   │ loader.load()   │               │             │
  │               │                   │── documents ───▶               │             │
  │               │                   │                 │               │             │
  │               │                   │ split(documents)│               │             │
  │               │                   │────────────────────────────────▶│             │
  │               │                   │◀──────────────────── chunks ────│             │
  │               │                   │                                  │             │
  │               │                   │ add_documents(chunks) ──────────────────────▶│
  │               │                   │◀────────────────────────────── chunk_count ──│
  │               │                   │                                               │
  │               │ DocumentMetadata  │                                               │
  │               │◀──────────────────│                                               │
  │ 201 Created   │                   │                                               │
  │◀──────────────│                   │                                               │
```

---

## 7. Class Diagram

```
AppException
├── DocumentException
│   ├── DocumentNotFoundError
│   ├── DocumentAlreadyExistsError
│   ├── UnsupportedFileTypeError
│   └── LoaderException
├── EmbeddingException
└── VectorStoreException
    └── CollectionNotFoundError

DocumentService ──uses──▶ DocumentRepository
DocumentService ──uses──▶ DocumentLoaderFactory
DocumentService ──uses──▶ TextSplitterService
DocumentService ──uses──▶ VectorStoreService
VectorStoreService ──uses──▶ EmbeddingService
EmbeddingService ──wraps──▶ OllamaEmbeddings
VectorStoreService ──wraps──▶ Chroma (ChromaDB)

DocumentMetadata
    id: str (UUID)
    file_name: str
    file_type: str
    file_size: int
    file_hash: str (SHA256)
    chunk_count: int
    collection_name: str
    status: DocumentStatus (PENDING/PROCESSING/INDEXED/FAILED)
    created_at: str (ISO 8601)
    updated_at: str (ISO 8601)
    error_message: Optional[str]
```

---

## 8. Cấu trúc thư mục sau Phase 3A

```
app/
├── config/
│   └── settings.py          ← Thêm collection_name
├── core/
│   ├── base_chain.py
│   ├── exceptions.py        ← Thêm Document/Embedding/VectorStore exceptions
│   └── output_parser.py
├── documents/               ← [MỚI] Document Management
│   ├── __init__.py
│   ├── models.py            ← DocumentMetadata, DocumentStatus
│   ├── repository.py        ← JSON persistence
│   ├── loader_factory.py    ← Factory Pattern
│   └── document_service.py  ← Pipeline orchestrator
├── rag/                     ← [MỚI] RAG Infrastructure
│   ├── __init__.py
│   ├── text_splitter.py     ← RecursiveCharacterTextSplitter
│   ├── embedding_service.py ← OllamaEmbeddings
│   └── vector_store_service.py ← ChromaDB
├── api/
│   ├── dependencies.py      ← Thêm document service DI
│   └── v1/
│       ├── chat.py          ← Không đổi
│       ├── documents.py     ← [MỚI] Document endpoints
│       └── health.py        ← Không đổi
├── chains/                  ← Không đổi
├── memory/                  ← Không đổi
├── prompts/                 ← Không đổi
└── services/                ← Không đổi

data/
├── documents.json           ← Auto-created: document metadata
└── *.pdf, *.docx, ...       ← Uploaded files

vector_db/
└── *.sqlite3, ...           ← ChromaDB persistent files

frontend/
├── api_client.py            ← Thêm document methods
└── app.py                   ← Thêm Knowledge Base sidebar

docs/
└── phase3a.md               ← File này
```

---

## 9. API Endpoints

| Method | Endpoint | Status Code | Mô tả |
|---|---|---|---|
| `POST` | `/api/v1/documents/upload` | 201 / 409 / 415 / 422 | Upload & index file |
| `GET` | `/api/v1/documents` | 200 | Danh sách documents |
| `GET` | `/api/v1/documents/stats` | 200 | Collection statistics |
| `GET` | `/api/v1/documents/{id}` | 200 / 404 | Chi tiết document |
| `DELETE` | `/api/v1/documents/{id}` | 200 / 404 | Xóa document |
| `POST` | `/api/v1/documents/{id}/reindex` | 200 / 404 / 422 | Reindex document |

---

## 10. Config mới

| Key | Default | Mô tả |
|---|---|---|
| `COLLECTION_NAME` | `enterprise_kb` | Tên ChromaDB collection |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Model embedding |
| `VECTOR_DB_DIR` | `vector_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Kích thước mỗi chunk (chars) |
| `CHUNK_OVERLAP` | `200` | Overlap giữa chunks (chars) |

---

## 11. Design Patterns áp dụng

| Pattern | Nơi áp dụng | Lý do |
|---|---|---|
| **Factory Method** | `DocumentLoaderFactory` | Chọn loader phù hợp theo extension |
| **Repository** | `DocumentRepository` | Tách persistence khỏi business logic |
| **Facade** | `DocumentService` | Giao diện đơn giản cho pipeline phức tạp |
| **Strategy** | `BaseMemoryBackend` (Phase 2) | Swap storage backend dễ dàng |
| **Singleton** | All `get_*()` functions | Shared instances qua app lifecycle |
| **Dependency Injection** | FastAPI `Depends()` | Loose coupling, testability |

---

## 12. Chuẩn bị cho Phase 3B (RAG Query)

Phase 3A đã sẵn sàng tất cả infrastructure. Phase 3B chỉ cần thêm:

1. **`app/prompts/chat_prompts.py`**: `get_rag_prompt_template()` đã có sẵn.

2. **`app/rag/rag_chain.py`** (mới):
```python
from app.rag.vector_store_service import VectorStoreService
from app.prompts import get_rag_prompt_template

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough(), "history": ...}
    | get_rag_prompt_template()
    | llm
    | StrOutputParser()
)
```

3. **`app/chains/rag_chat_chain.py`** (mới): Kế thừa `BaseChain`.

4. **`ChatRequest`**: Bật `use_rag=True` để route sang RAG chain.

5. **`ChatService`**: Check `use_rag` flag → chọn `ChatChain` hoặc `RagChatChain`.

---

## 13. Setup & Kiểm tra

### Điều kiện tiên quyết

```bash
# 1. Cài dependencies mới
pip install langchain-chroma docx2txt

# 2. Pull embedding model
ollama pull nomic-embed-text

# 3. Kiểm tra import
python -c "from app.documents.document_service import DocumentService; print('OK')"
python -c "from app.rag.vector_store_service import VectorStoreService; print('OK')"
```

### Chạy và test

```bash
# Khởi động API
python run_api.py

# Test upload (Swagger UI)
# Mở http://localhost:8000/docs
# POST /api/v1/documents/upload → chọn file
# GET  /api/v1/documents/stats  → xem chunk count
```

### Checklist

- [ ] `ollama pull nomic-embed-text` thành công
- [ ] `pip install langchain-chroma docx2txt` thành công
- [ ] Upload PDF → status=indexed
- [ ] Upload DOCX → status=indexed
- [ ] Upload TXT → status=indexed
- [ ] Upload CSV → status=indexed
- [ ] Upload MD → status=indexed
- [ ] `GET /documents/stats` → total_chunks > 0
- [ ] Upload file lần 2 → HTTP 409 Conflict
- [ ] `DELETE /documents/{id}` → xóa thành công
- [ ] Chat vẫn hoạt động bình thường
- [ ] Streamlit sidebar hiển thị KB section
