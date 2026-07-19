"""
Streamlit Frontend — Enterprise Digital Assistant.

Main application file. Kết hợp sidebar và chat interface
để tạo giao diện người dùng hoàn chỉnh.

Chạy đúng cách (từ project root):
    streamlit run run_frontend.py

Đừng chạy trực tiếp (gây ModuleNotFoundError):
    streamlit run frontend/app.py  # SAI — frontend/ không có trong sys.path
"""

import logging
import uuid

import streamlit as st

from frontend.api_client import APIClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------

def init_session_state() -> None:
    """Khởi tạo các giá trị mặc định trong session state."""
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],           # List of {"role": str, "content": str}
        "model": "qwen2.5:latest",
        "temperature": 0.7,
        "streaming": True,
        "api_online": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------------------------------------------------------
# API Client (cached)
# ------------------------------------------------------------------

@st.cache_resource
def get_api_client() -> APIClient:
    """Trả về singleton APIClient (cached bởi Streamlit)."""
    return APIClient(base_url="http://localhost:8000", api_prefix="/api/v1")


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

def render_sidebar(client: APIClient) -> None:
    """
    Render sidebar với các cài đặt và thông tin hệ thống.

    Args:
        client: APIClient instance.
    """
    with st.sidebar:
        # Header
        st.markdown("## ⚙️ Cài đặt")
        st.divider()

        # API Status
        health = client.health_check()
        is_online = health.get("status") == "healthy"
        st.session_state["api_online"] = is_online

        if is_online:
            st.markdown(
                '<span class="status-online"></span> **API Online**',
                unsafe_allow_html=True,
            )
            ollama_ok = health.get("ollama_connected", False)
            if ollama_ok:
                st.success(f"🟢 Ollama: {health.get('ollama_model', 'unknown')}", icon=None)
            else:
                st.error("🔴 Ollama: Không kết nối được")
        else:
            st.markdown(
                '<span class="status-offline"></span> **API Offline**',
                unsafe_allow_html=True,
            )
            st.error("Không thể kết nối đến FastAPI server.\nHãy chạy: `uvicorn app.main:app`")

        st.divider()

        # Model Selection
        st.markdown("### 🤖 Model")
        model = st.selectbox(
            "Chọn model",
            options=["qwen2.5:latest", "qwen2.5:7b", "llama3.2:latest", "mistral:latest"],
            index=0,
            key="model_select",
            help="Model LLM để sử dụng (phải có trong Ollama)",
        )
        st.session_state["model"] = model

        # Temperature
        st.markdown("### 🌡️ Temperature")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state["temperature"],
            step=0.1,
            key="temp_slider",
            help="0.0 = Chính xác, 1.0 = Cân bằng, 2.0 = Sáng tạo",
        )
        st.session_state["temperature"] = temperature

        # Streaming toggle
        st.markdown("### ⚡ Streaming")
        streaming = st.toggle(
            "Bật Streaming Response",
            value=st.session_state["streaming"],
            key="streaming_toggle",
            help="Hiển thị response từng token thay vì chờ toàn bộ",
        )
        st.session_state["streaming"] = streaming

        st.divider()

        # Session Info
        st.markdown("### 📋 Session")
        session_short = st.session_state["session_id"][:8] + "..."
        st.caption(f"Session ID: `{session_short}`")
        st.caption(f"Tin nhắn: {len(st.session_state['messages'])}")

        # Clear history button
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, type="secondary"):
            try:
                client.clear_history(st.session_state["session_id"])
                st.session_state["messages"] = []
                st.session_state["session_id"] = str(uuid.uuid4())
                st.success("Đã xóa lịch sử chat!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

        st.divider()

        # ------------------------------------------------------------------
        # Knowledge Base Section
        # ------------------------------------------------------------------
        st.markdown("### 📚 Knowledge Base")

        # Collection Stats
        if st.session_state.get("api_online"):
            try:
                stats = client.get_collection_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Documents", stats.get("indexed_documents", 0))
                with col2:
                    st.metric("🧩 Chunks", stats.get("total_chunks", 0))
                st.caption(f"🧠 Model: `{stats.get('embedding_model', 'N/A')}`")
            except Exception:
                st.caption("Không lấy được thống kê KB.")

        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload tài liệu",
            type=["pdf", "docx", "txt", "csv", "md"],
            key="kb_file_uploader",
            help="Hỗ trợ: PDF, DOCX, TXT, CSV, Markdown",
        )

        if uploaded_file is not None:
            if st.button(
                f"⬆️ Index '{uploaded_file.name}'",
                use_container_width=True,
                type="primary",
                key="kb_upload_btn",
            ):
                _handle_document_upload(client, uploaded_file)

        # Document List
        st.markdown("**📂 Tài liệu đã upload:**")
        _render_document_list(client)

        st.divider()

        # App info
        st.markdown("""
        <div style='font-size: 0.75rem; color: rgba(255,255,255,0.4); text-align: center;'>
            Enterprise Digital Assistant v2.0<br>
            Powered by Ollama + LangChain + ChromaDB
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Knowledge Base Helper Functions
# ------------------------------------------------------------------


def _handle_document_upload(client: APIClient, uploaded_file) -> None:
    """
    Xử lý upload file vào Knowledge Base.

    Args:
        client:        APIClient instance.
        uploaded_file: Streamlit UploadedFile object.
    """
    with st.spinner(f"⏳ Đang index '{uploaded_file.name}'... (có thể mất vài phút)"):
        try:
            file_bytes = uploaded_file.read()
            result = client.upload_document(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
            )
            status = result.get("status", "unknown")
            if status == "indexed":
                chunk_count = result.get("chunk_count", 0)
                st.success(
                    f"✅ '{uploaded_file.name}' đã được index!\n"
                    f"🧩 {chunk_count} chunks trong ChromaDB."
                )
            elif status == "failed":
                error_msg = result.get("error_message", "Unknown error")
                st.error(f"❌ Index thất bại: {error_msg}")
            else:
                st.info(f"ℹ️ Trạng thái: {status}")
            st.rerun()
        except ValueError as e:
            # Lỗi từ server (duplicate, unsupported type...)
            st.warning(f"⚠️ {str(e)}")
        except Exception as e:
            st.error(f"❌ Lỗi upload: {str(e)}")
            logger.error("Document upload error: %s", str(e))


def _render_document_list(client: APIClient) -> None:
    """
    Render danh sách documents với nút xóa.

    Args:
        client: APIClient instance.
    """
    if not st.session_state.get("api_online"):
        st.caption("⚠️ API offline")
        return

    try:
        response = client.list_documents()
        documents = response.get("documents", [])

        if not documents:
            st.caption("📥 Chưa có tài liệu nào.")
            return

        # Hiển thị từng document
        for doc in documents:
            doc_id = doc.get("id", "")
            file_name = doc.get("file_name", "unknown")
            status = doc.get("status", "unknown")
            chunk_count = doc.get("chunk_count", 0)
            file_size = doc.get("file_size", 0)

            # Status icon
            status_icon = {
                "indexed": "✅",
                "processing": "⏳",
                "pending": "⏳",
                "failed": "❌",
            }.get(status, "❔")

            # Truncate long filenames
            display_name = file_name if len(file_name) <= 20 else file_name[:17] + "..."

            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"{status_icon} **{display_name}**  \n"
                        f"│ 🧩 {chunk_count} chunks | 💾 {file_size // 1024}KB",
                    )
                with col2:
                    if st.button(
                        "🗑️",
                        key=f"del_{doc_id}",
                        help=f"Xóa '{file_name}'",
                    ):
                        _handle_document_delete(client, doc_id, file_name)

    except Exception as e:
        st.caption(f"Lỗi tải danh sách: {str(e)[:50]}")
        logger.error("list_documents error: %s", str(e))


def _handle_document_delete(client: APIClient, doc_id: str, file_name: str) -> None:
    """
    Xử lý xóa document khỏi Knowledge Base.

    Args:
        client:    APIClient instance.
        doc_id:    UUID của document.
        file_name: Tên file để hiển thị thông báo.
    """
    try:
        client.delete_document(doc_id)
        st.success(f"✅ Đã xóa '{file_name}'!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Lỗi xóa: {str(e)}")
        logger.error("delete_document error: %s", str(e))


# ------------------------------------------------------------------
# Chat Interface
# ------------------------------------------------------------------

def render_chat_messages() -> None:
    """Render tất cả messages trong conversation."""
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)


def process_user_input(client: APIClient, user_input: str) -> None:
    """
    Xử lý input từ người dùng và gọi API.

    Args:
        client: APIClient instance.
        user_input: Tin nhắn từ người dùng.
    """
    # Thêm user message vào history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Hiển thị user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Gọi API và hiển thị response
    with st.chat_message("assistant", avatar="🤖"):
        if st.session_state["streaming"]:
            _handle_streaming_response(client, user_input)
        else:
            _handle_sync_response(client, user_input)


def _handle_streaming_response(client: APIClient, user_input: str) -> None:
    """Xử lý streaming response từ API."""
    message_placeholder = st.empty()
    full_response = ""
    session_id = st.session_state["session_id"]

    try:
        for chunk in client.chat_stream(
            message=user_input,
            session_id=session_id,
            model=st.session_state["model"],
            temperature=st.session_state["temperature"],
        ):
            if chunk.get("is_done"):
                # Cập nhật session_id nếu server trả về mới
                if chunk.get("session_id"):
                    st.session_state["session_id"] = chunk["session_id"]
                break

            # Cập nhật session_id từ chunk đầu tiên
            if chunk.get("session_id") and not full_response:
                st.session_state["session_id"] = chunk["session_id"]

            content = chunk.get("content", "")
            if content:
                full_response += content
                message_placeholder.markdown(full_response + "▌")

        # Hiển thị response hoàn chỉnh (bỏ cursor)
        message_placeholder.markdown(full_response)

        # Lưu vào history
        if full_response:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_response,
            })

    except Exception as e:
        error_msg = f"❌ Lỗi kết nối API: {str(e)}\n\nHãy đảm bảo FastAPI server đang chạy."
        message_placeholder.error(error_msg)
        logger.error("Streaming error: %s", str(e))


def _handle_sync_response(client: APIClient, user_input: str) -> None:
    """Xử lý non-streaming response từ API."""
    with st.spinner("Đang xử lý..."):
        try:
            response = client.chat_sync(
                message=user_input,
                session_id=st.session_state["session_id"],
                model=st.session_state["model"],
                temperature=st.session_state["temperature"],
            )

            # Cập nhật session_id
            if response.get("session_id"):
                st.session_state["session_id"] = response["session_id"]

            content = response.get("message", {}).get("content", "Không có response.")
            st.markdown(content)

            # Lưu vào history
            st.session_state["messages"].append({
                "role": "assistant",
                "content": content,
            })

        except Exception as e:
            error_msg = f"❌ Lỗi: {str(e)}"
            st.error(error_msg)
            logger.error("Sync chat error: %s", str(e))


# ------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------

def main() -> None:
    """Main entry point cho Streamlit application."""
    # ------------------------------------------------------------------
    # Page Configuration — phải là lệnh Streamlit đầu tiên mỗi rerun
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="Enterprise Digital Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # Custom CSS
    # ------------------------------------------------------------------
    st.markdown("""
    <style>
        /* Ẩn Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Chat messages */
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
        }

        .assistant-message {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
            color: #e8f4f8;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 85%;
            word-wrap: break-word;
        }

        /* Header */
        .app-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 20px 24px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        /* Status badge */
        .status-online {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }

        .status-offline {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #ff4757;
            border-radius: 50%;
            margin-right: 6px;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }

        /* Input area */
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.4);
            background: rgba(255, 255, 255, 0.05);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    init_session_state()
    client = get_api_client()

    # Sidebar
    render_sidebar(client)

    # Header
    st.markdown("""
    <div class="app-header">
        <h1 style="margin:0; font-size: 1.8rem;">🤖 Enterprise Digital Assistant</h1>
        <p style="margin: 8px 0 0 0; opacity: 0.7; font-size: 0.9rem;">
            Trợ lý AI doanh nghiệp · Powered by Ollama + LangChain · Local & Private
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Warning nếu API offline
    if not st.session_state["api_online"]:
        st.warning(
            "⚠️ **API Server chưa khởi động.** "
            "Vui lòng chạy lệnh sau trong terminal:\n\n"
            "```bash\nuvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n```",
            icon="⚠️",
        )

    # Welcome message nếu chưa có messages
    if not st.session_state["messages"]:
        st.info(
            "👋 **Xin chào!** Tôi là Enterprise Digital Assistant.\n\n"
            "Hãy đặt câu hỏi hoặc yêu cầu tôi hỗ trợ bất kỳ tác vụ nào. "
            "Tôi chạy hoàn toàn **local** — dữ liệu của bạn không rời khỏi máy tính.",
            icon="ℹ️",
        )

    # Chat messages
    render_chat_messages()

    # Chat input
    if user_input := st.chat_input(
        placeholder="Nhập câu hỏi của bạn... (Enter để gửi)",
        disabled=not st.session_state["api_online"],
    ):
        process_user_input(client, user_input)


if __name__ == "__main__":
    main()
