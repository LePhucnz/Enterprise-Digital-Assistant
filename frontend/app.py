"""
Streamlit Frontend — Enterprise Digital Assistant.

Giao diện người dùng hiện đại, tinh gọn theo tiêu chuẩn UI/UX Enterprise:
    - RAG (Knowledge Base), Query Expansion, và Streaming luôn BẬT mặc định (ẩn nút gạt).
    - Hỗ trợ linh hoạt 3 nguồn mô hình: Local (Ollama), OpenAI/OpenRouter API, Google Gemini API.
    - Bố cục trực quan, tối ưu không gian, dễ sử dụng.

Chạy ứng dụng (từ thư mục gốc dự án):
    streamlit run run_frontend.py
"""

import logging
import uuid
from typing import Optional

import streamlit as st
import streamlit.components.v1 as st_components

from frontend.api_client import APIClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------

def init_session_state() -> None:
    """Khởi tạo các giá trị mặc định trong session state."""
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],           # List of {"role": str, "content": str, ...}
        "model": "qwen2.5:latest",
        "temperature": 0.2,       # Mặc định tối ưu cho RAG (chính xác, không bịa đặt)
        # Các tính năng cốt lõi luôn BẬT mặc định
        "streaming": True,        # Luôn bật streaming
        "use_rag": True,          # Luôn bật RAG tìm kiếm tài liệu
        "enable_query_expansion": True,  # Luôn bật Query Expansion đa chiều
        "api_online": False,
        "selected_provider": "🏠 Local (Ollama)",
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
# Sidebar — Cấu hình & Quản lý Knowledge Base
# ------------------------------------------------------------------

def render_sidebar(client: APIClient) -> None:
    """
    Render sidebar gọn gàng, tinh tế:
    - Trạng thái kết nối các provider
    - Lựa chọn nguồn mô hình (Local vs Cloud API)
    - Quản lý kho tài liệu (Knowledge Base)
    - Thao tác phiên làm việc
    """
    with st.sidebar:
        # 1. Header & Status
        st.markdown("### 🏢 Enterprise Assistant")
        st.caption("Trợ lý số doanh nghiệp · RAG thông minh")

        # Kiểm tra trạng thái hệ thống
        health = client.health_check()
        is_online = health.get("status") == "healthy"
        st.session_state["api_online"] = is_online

        if is_online:
            ollama_ok = health.get("ollama_connected", False)
            openai_ok = health.get("openai_configured", False)
            gemini_ok = health.get("gemini_configured", False)

            # Badge trạng thái 3 providers
            status_cols = st.columns(3)
            with status_cols[0]:
                st.markdown(
                    f"<div class='status-badge {'active' if ollama_ok else 'inactive'}'>"
                    f"● Ollama</div>",
                    unsafe_allow_html=True,
                )
            with status_cols[1]:
                st.markdown(
                    f"<div class='status-badge {'active' if openai_ok else 'inactive'}'>"
                    f"● OpenAI</div>",
                    unsafe_allow_html=True,
                )
            with status_cols[2]:
                st.markdown(
                    f"<div class='status-badge {'active' if gemini_ok else 'inactive'}'>"
                    f"● Gemini</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.error("🔴 **API Server Offline**\nVui lòng khởi động backend FastAPI.")

        st.markdown("<hr class='compact-divider'/>", unsafe_allow_html=True)

        # 2. Chọn Nguồn & Mô hình AI
        st.markdown("##### 🤖 Mô hình AI")

        provider_options = [
            "🏠 Local (Ollama)",
            "🔑 API Key (OpenAI / OpenRouter)",
            "✨ API Key (Google Gemini)",
        ]

        provider_mode = st.radio(
            "Nguồn LLM",
            options=provider_options,
            index=0,
            key="provider_radio",
            label_visibility="collapsed",
        )
        st.session_state["selected_provider"] = provider_mode

        # Danh sách model theo từng provider
        if "Local" in provider_mode:
            model_options = [
                "qwen2.5:latest",
                "qwen2.5:7b",
                "llama3.2:latest",
                "mistral:latest",
            ]
            help_text = "Chạy hoàn toàn cục bộ qua Ollama — bảo mật tuyệt đối, không cần internet."
        elif "Google" in provider_mode:
            model_options = [
                "gemini-3.8-flash",
                "gemini-3.5-flash-lite",
                "gemini-flash-latest",
                "gemini-2.5-pro",
            ]
            help_text = "Google Gemini API — tốc độ cao, khả năng hiểu tài liệu dài xuất sắc."
        else:
            model_options = [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "deepseek/deepseek-chat",
            ]
            help_text = "OpenAI / OpenRouter API — phản hồi thông minh, suy luận sắc bén."

        selected_model = st.selectbox(
            "Chọn model",
            options=model_options,
            index=0,
            key=f"select_model_{provider_mode}",
            help=help_text,
        )
        st.session_state["model"] = selected_model

        # Badge thông báo tính năng tự động (RAG + Expansion + Streaming)
        st.markdown(
            """
            <div class='feature-pill'>
                ⚡ <b>RAG + Query Expansion + Streaming</b>: Đang hoạt động
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='compact-divider'/>", unsafe_allow_html=True)

        # 3. Kho tri thức (Knowledge Base)
        st.markdown("##### 📚 Kho tài liệu (Knowledge Base)")

        if st.session_state.get("api_online"):
            try:
                stats = client.get_collection_stats()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📄 Tài liệu", stats.get("indexed_documents", 0))
                with col_b:
                    st.metric("🧩 Chunks", stats.get("total_chunks", 0))
            except Exception:
                st.caption("Chưa kết nối được Vector DB.")

        # Uploader và danh sách tài liệu
        with st.expander("📁 Tải lên & Quản lý tài liệu", expanded=False):
            uploaded_file = st.file_uploader(
                "Thêm tài liệu mới",
                type=["pdf", "docx", "txt", "csv", "md"],
                key="kb_uploader",
                help="Hỗ trợ PDF, DOCX, TXT, CSV, MD",
            )
            if uploaded_file is not None:
                if st.button(
                    f"⬆️ Nạp '{uploaded_file.name}'",
                    use_container_width=True,
                    type="primary",
                    key="btn_upload_doc",
                ):
                    _handle_document_upload(client, uploaded_file)

            st.markdown("**Tài liệu đã nạp:**")
            _render_document_list(client)

        st.markdown("<hr class='compact-divider'/>", unsafe_allow_html=True)

        # 4. Thao tác phiên hội thoại
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            st.caption(f"Phiên: `{st.session_state['session_id'][:8]}`")
        with col_s2:
            st.caption(f"Tin nhắn: {len(st.session_state['messages'])}")

        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, type="secondary"):
            try:
                client.clear_history(st.session_state["session_id"])
                st.session_state["messages"] = []
                st.session_state["session_id"] = str(uuid.uuid4())
                st.success("Đã làm mới phiên chat!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

        # Cài đặt nâng cao (ẩn thu gọn)
        with st.expander("⚙️ Nâng cao", expanded=False):
            temperature = st.slider(
                "Nhiệt độ (Temperature)",
                min_value=0.0,
                max_value=1.5,
                value=float(st.session_state["temperature"]),
                step=0.05,
                key="temp_slider_adv",
                help="Thấp hơn = câu trả lời bám sát tài liệu hơn. Cao hơn = đa dạng từ ngữ hơn.",
            )
            st.session_state["temperature"] = temperature


# ------------------------------------------------------------------
# Knowledge Base Helpers
# ------------------------------------------------------------------

def _handle_document_upload(client: APIClient, uploaded_file) -> None:
    """Xử lý upload file vào Knowledge Base."""
    with st.spinner(f"⏳ Đang index '{uploaded_file.name}' vào Vector Store..."):
        try:
            file_bytes = uploaded_file.read()
            result = client.upload_document(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
            )
            status = result.get("status", "unknown")
            if status == "indexed":
                chunk_count = result.get("chunk_count", 0)
                st.success(f"✅ Đã index {chunk_count} chunks!")
            elif status == "failed":
                st.error(f"❌ Thất bại: {result.get('error_message', 'Unknown error')}")
            else:
                st.info(f"ℹ️ Trạng thái: {status}")
            st.rerun()
        except ValueError as e:
            st.warning(f"⚠️ {str(e)}")
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")


def _render_document_list(client: APIClient) -> None:
    """Render danh sách documents gọn gàng kèm nút xóa."""
    if not st.session_state.get("api_online"):
        st.caption("Chưa kết nối API.")
        return

    try:
        response = client.list_documents()
        documents = response.get("documents", [])

        if not documents:
            st.caption("Chưa có tài liệu nào trong kho.")
            return

        for doc in documents:
            doc_id = doc.get("id", "")
            file_name = doc.get("file_name", "unknown")
            chunk_count = doc.get("chunk_count", 0)
            file_size_kb = doc.get("file_size", 0) // 1024

            display_name = file_name if len(file_name) <= 22 else file_name[:19] + "..."

            col_doc, col_del = st.columns([5, 1])
            with col_doc:
                st.markdown(f"📄 **{display_name}**  \n<small>{chunk_count} chunks · {file_size_kb}KB</small>", unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key=f"del_{doc_id}", help=f"Xóa {file_name}"):
                    try:
                        client.delete_document(doc_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
    except Exception as e:
        st.caption(f"Lỗi: {str(e)[:40]}")


# ------------------------------------------------------------------
# Chat Components
# ------------------------------------------------------------------

def _render_query_expansion(expanded_queries: list[str]) -> None:
    """Render danh sách các câu truy vấn mở rộng từ Query Expansion."""
    if not expanded_queries:
        return

    with st.expander("🔍 Góc nhìn tìm kiếm mở rộng (Query Expansion)", expanded=False):
        for idx, q in enumerate(expanded_queries, 1):
            st.markdown(f"• *{q}*")


def _render_message_sources(sources: list[dict]) -> None:
    """Render nguồn tài liệu tham chiếu (citations) dưới dạng thẻ trích dẫn."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tài liệu tham chiếu ({len(sources)} đoạn)", expanded=False):
        for idx, src in enumerate(sources, 1):
            doc_name = src.get("document", "Unknown")
            page = src.get("page", 1)
            chunk_id = src.get("chunk_id", 0)
            st.markdown(
                f"**{idx}.** 📄 `{doc_name}` — **Trang {page}** *(Chunk #{chunk_id})*"
            )


def render_chat_messages() -> None:
    """Render lịch sử tin nhắn trong phiên làm việc."""
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources")
        expanded_queries = msg.get("expanded_queries")

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                if expanded_queries:
                    _render_query_expansion(expanded_queries)
                if sources:
                    _render_message_sources(sources)


def process_user_input(client: APIClient, user_input: str) -> None:
    """
    Gửi câu hỏi của người dùng đến API với:
    - use_rag: Luôn True
    - enable_query_expansion: Luôn True
    - streaming: Luôn True
    """
    # 1. Thêm user message vào UI
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Xử lý phản hồi từ Assistant (Streaming luôn bật)
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        session_id = st.session_state["session_id"]
        sources = None
        expanded_queries = None

        try:
            for chunk in client.chat_stream(
                message=user_input,
                session_id=session_id,
                model=st.session_state["model"],
                temperature=st.session_state["temperature"],
                use_rag=True,                  # LUÔN BẬT
                enable_query_expansion=True,   # LUÔN BẬT
            ):
                if chunk.get("is_done"):
                    if chunk.get("session_id"):
                        st.session_state["session_id"] = chunk["session_id"]
                    sources = chunk.get("sources")
                    expanded_queries = chunk.get("expanded_queries")
                    break

                if chunk.get("session_id") and not full_response:
                    st.session_state["session_id"] = chunk["session_id"]

                content = chunk.get("content", "")
                if content:
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            # Hiển thị hoàn chỉnh bỏ con trỏ
            message_placeholder.markdown(full_response)

            if expanded_queries:
                _render_query_expansion(expanded_queries)

            if sources:
                _render_message_sources(sources)

            # Lưu vào bộ nhớ lịch sử
            if full_response:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "expanded_queries": expanded_queries,
                })

        except Exception as e:
            error_msg = f"❌ Không thể hoàn tất phản hồi: {str(e)}"
            message_placeholder.error(error_msg)
            logger.error("Chat error: %s", str(e))


# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------

def main() -> None:
    """Entry point chính của ứng dụng."""
    st.set_page_config(
        page_title="Enterprise Digital Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # Custom CSS — Thiết kế giao diện chuyên nghiệp, gọn gàng, hợp mắt
    # ------------------------------------------------------------------
    st.markdown("""
    <style>
        /* Ẩn bớt Streamlit menu và footer thừa */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stAppDeployButton"] {display: none !important;}

        /* Tạo khoảng đệm trên cùng để nội dung không che khuất thanh công cụ */
        .block-container {
            padding-top: 3.5rem !important;
        }

        /* Đảm bảo header và toolbar luôn nhận sự kiện chuột */
        header[data-testid="stHeader"],
        .stAppHeader {
            background: transparent !important;
            display: block !important;
            visibility: visible !important;
            height: 3.5rem !important;
            z-index: 9999 !important;
            pointer-events: auto !important;
        }

        [data-testid="stToolbar"],
        .stAppToolbar {
            visibility: visible !important;
            display: flex !important;
            pointer-events: auto !important;
        }

        /* Nút floating tùy chỉnh để mở sidebar — luôn hiển thị */
        #eda-sidebar-toggle-btn {
            position: fixed !important;
            top: 10px !important;
            left: 10px !important;
            z-index: 999999 !important;
            background: #2563eb !important;
            color: #ffffff !important;
            border: 1px solid #1d4ed8 !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
            cursor: pointer !important;
            font-size: 16px !important;
            line-height: 1 !important;
            display: none; /* JS sẽ show khi sidebar đóng */
        }
        #eda-sidebar-toggle-btn:hover {
            background: #1d4ed8 !important;
            transform: scale(1.05) !important;
        }

        /* Header ứng dụng */
        .main-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(99, 102, 241, 0.04) 100%);
            border: 1px solid rgba(37, 99, 235, 0.15);
            padding: 16px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .main-header-title {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .main-header-desc {
            margin: 4px 0 0 0;
            opacity: 0.75;
            font-size: 0.85rem;
        }

        /* Pill thông tin tính năng */
        .feature-pill {
            background: rgba(37, 99, 235, 0.1);
            color: #2563eb;
            border: 1px solid rgba(37, 99, 235, 0.2);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.78rem;
            text-align: center;
            margin: 10px 0;
        }

        /* Status Badge nhỏ gọn */
        .status-badge {
            font-size: 0.75rem;
            padding: 4px 6px;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
        }
        .status-badge.active {
            background: rgba(34, 197, 94, 0.12);
            color: #16a34a;
            border: 1px solid rgba(34, 197, 94, 0.25);
        }
        .status-badge.inactive {
            background: rgba(148, 163, 184, 0.1);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        /* Thẻ gợi ý câu hỏi ban đầu */
        .welcome-card {
            background: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 12px;
            padding: 24px;
            margin: 10px 0 20px 0;
        }

        /* Divider tinh gọn */
        .compact-divider {
            margin: 14px 0;
            border: none;
            border-top: 1px solid rgba(128, 128, 128, 0.15);
        }

        /* Streamlit chat message spacing */
        [data-testid="stChatMessage"] {
            padding: 12px 16px;
            margin-bottom: 8px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # JavaScript: Nút floating toggle sidebar — LUÔN hiển thị
    # st.components.v1.html đảm bảo script thực sự chạy (không bị strip)
    # ------------------------------------------------------------------
    st_components.html("""
    <script>
    (function() {
        // Thử truy cập window.parent — cùng origin nên được phép
        var doc;
        try { doc = window.parent.document; } catch(e) { return; }

        // Inject CSS cho nút floating vào parent <head> (chỉ 1 lần)
        if (!doc.getElementById('eda-toggle-btn-css')) {
            var style = doc.createElement('style');
            style.id = 'eda-toggle-btn-css';
            style.textContent =
                '#eda-toggle-btn {'
                + '  position: fixed !important;'
                + '  top: 12px !important;'
                + '  left: 12px !important;'
                + '  z-index: 2147483647 !important;'
                + '  background: #2563eb !important;'
                + '  color: #fff !important;'
                + '  border: none !important;'
                + '  border-radius: 8px !important;'
                + '  padding: 4px 10px 5px !important;'
                + '  font-size: 22px !important;'
                + '  line-height: 1.2 !important;'
                + '  cursor: pointer !important;'
                + '  box-shadow: 0 3px 10px rgba(37,99,235,0.45) !important;'
                + '  display: block !important;'
                + '  opacity: 1 !important;'
                + '}'
                + '#eda-toggle-btn:hover {'
                + '  background: #1d4ed8 !important;'
                + '  transform: scale(1.1) !important;'
                + '}';
            doc.head.appendChild(style);
        }

        // Tạo nút nếu chưa có (chỉ 1 lần, tồn tại xuyên suốt)
        if (!doc.getElementById('eda-toggle-btn')) {
            var btn = doc.createElement('button');
            btn.id = 'eda-toggle-btn';
            btn.title = 'Mo / Dong thanh ben';
            btn.textContent = '\u00BB';
            btn.addEventListener('click', function() {
                // Click nút gốc của Streamlit để toggle sidebar
                var nativeBtn =
                    doc.querySelector('[data-testid="stExpandSidebarButton"]') ||
                    doc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                    doc.querySelector('button[aria-label="Expand sidebar"]') ||
                    doc.querySelector('button[aria-label="Collapse sidebar"]');
                if (nativeBtn) nativeBtn.click();
            });
            doc.body.appendChild(btn);
        }
    })();
    </script>
    """, height=1, scrolling=False)

    # ------------------------------------------------------------------
    init_session_state()
    client = get_api_client()

    # Render sidebar
    render_sidebar(client)

    # Header chính
    current_model_label = st.session_state["model"]
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1 class="main-header-title">🤖 Enterprise Digital Assistant</h1>
            <p class="main-header-desc">
                Trợ lý tra cứu quy chế, văn bản & điều hành doanh nghiệp · Tự động đối chiếu tài liệu nội bộ
            </p>
        </div>
        <div>
            <span style="font-size: 0.8rem; background: rgba(37,99,235,0.12); color: #2563eb; padding: 4px 10px; border-radius: 20px; font-weight: 600;">
                🧠 {current_model_label}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Warning nếu API chưa kết nối
    if not st.session_state["api_online"]:
        st.warning(
            "⚠️ **Backend FastAPI đang offline.** Vui lòng khởi động backend server bằng lệnh:\n\n"
            "```bash\n.venv/bin/python run_api.py\n```",
            icon="⚠️",
        )

    # Trạng thái rỗng: Lời chào & Gợi ý câu hỏi nhanh
    if not st.session_state["messages"]:
        st.markdown("""
        <div class="welcome-card">
            <h3 style="margin: 0 0 8px 0; font-size: 1.15rem;">👋 Xin chào! Tôi có thể giúp gì cho bạn hôm nay?</h3>
            <p style="margin: 0; opacity: 0.85; font-size: 0.92rem;">
                Hệ thống đã được kích hoạt tính năng <b>RAG Tìm kiếm tài liệu</b> và <b>Query Expansion đa chiều</b>. 
                Mọi câu hỏi sẽ tự động được tra cứu và đối chiếu chính xác từ kho tài liệu doanh nghiệp.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 💡 Câu hỏi gợi ý:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👥 Cơ cấu & Số lượng thành viên HĐQT?", use_container_width=True):
                process_user_input(client, "Hội đồng quản trị có bao nhiêu thành viên?")
                st.rerun()
        with col2:
            if st.button("📜 Thẩm quyền của Đại hội đồng cổ đông?", use_container_width=True):
                process_user_input(client, "Quyền hạn và thẩm quyền của Đại hội đồng cổ đông là gì?")
                st.rerun()
        with col3:
            if st.button("🔍 Nhiệm vụ của Ban Kiểm soát?", use_container_width=True):
                process_user_input(client, "Nhiệm vụ và quyền hạn của Ban Kiểm soát?")
                st.rerun()

    # Render danh sách tin nhắn hiện tại
    render_chat_messages()

    # Ô nhập tin nhắn
    if user_input := st.chat_input(
        placeholder="Đặt câu hỏi về quy chế, quy định hoặc thông tin doanh nghiệp... (Enter để gửi)",
        disabled=not st.session_state["api_online"],
    ):
        process_user_input(client, user_input)


if __name__ == "__main__":
    main()
