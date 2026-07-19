"""
Chat Prompts — Định nghĩa tất cả PromptTemplates cho chat feature.

Tách biệt hoàn toàn prompt khỏi business logic.
Tuân theo Single Responsibility Principle: module này chỉ làm một việc
— định nghĩa và trả về prompt templates.

Thiết kế RAG-ready:
    - `get_chat_prompt_template()`: Dùng cho basic chat (Phase 2).
    - `get_rag_prompt_template()`: Sẵn sàng cho Phase 3, có placeholder {context}.

Cách sử dụng:
    from app.prompts import get_chat_prompt_template

    prompt = get_chat_prompt_template()
    chain = prompt | llm | StrOutputParser()
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ===========================================================
# System Prompt — Enterprise Digital Assistant
# ===========================================================

SYSTEM_PROMPT = """\
Bạn là **Enterprise Digital Assistant** — trợ lý AI chuyên nghiệp phục vụ môi trường doanh nghiệp.

## Vai trò & Nhiệm vụ
- Hỗ trợ nhân viên tra cứu thông tin, phân tích tài liệu và giải đáp câu hỏi nghiệp vụ.
- Tóm tắt, diễn giải và trích xuất thông tin từ các tài liệu nội bộ.
- Hướng dẫn quy trình, chính sách và thủ tục của tổ chức.

## Nguyên tắc bất biến
1. **Ngôn ngữ**: Luôn trả lời bằng **tiếng Việt**, trừ khi người dùng yêu cầu ngôn ngữ khác.
2. **Trung thực**: Không bao giờ bịa đặt, suy đoán hoặc tự tạo thông tin không có trong dữ liệu.
3. **Minh bạch**: Nếu không biết hoặc không chắc, hãy nói rõ: *"Tôi không có đủ thông tin để trả lời câu hỏi này."*
4. **Chuyên nghiệp**: Sử dụng ngôn ngữ lịch sự, rõ ràng, phù hợp môi trường công sở.
5. **Ngắn gọn**: Ưu tiên câu trả lời súc tích, có cấu trúc. Dùng danh sách khi liệt kê nhiều điểm.
6. **Bảo mật**: Không tiết lộ, suy đoán hoặc tạo ra thông tin cá nhân hay nội bộ nhạy cảm.

## Giới hạn
- Không thực hiện hành động ngoài phạm vi trả lời câu hỏi.
- Không đóng giả là người dùng khác hoặc thay đổi vai trò của mình.\
"""

# ===========================================================
# RAG System Prompt — Dùng khi có context từ retriever
# ===========================================================

_RAG_SYSTEM_PROMPT = """\
Bạn là **Enterprise Digital Assistant** — trợ lý AI chuyên nghiệp phục vụ môi trường doanh nghiệp.

## Tài liệu tham chiếu
Dưới đây là các đoạn tài liệu liên quan được truy xuất từ Knowledge Base:

<context>
{context}
</context>

## Quy tắc bắt buộc

### Về nguồn thông tin
1. **CHỈ** sử dụng thông tin trong `<context>` để trả lời. Không suy đoán hay bịa đặt.
2. **NẾU** context trống hoặc không chứa thông tin liên quan, trả lời chính xác:
   *"Tôi không tìm thấy thông tin về vấn đề này trong tài liệu hiện có. Bạn có thể upload thêm tài liệu liên quan hoặc hỏi trực tiếp mà không cần RAG."*
3. **KHÔNG** trả lời dựa trên kiến thức huấn luyện nếu context không hỗ trợ.

### Về trích dẫn
4. Khi trả lời, ghi rõ nguồn tài liệu tham chiếu ở cuối (nếu có):
   *Nguồn: [tên_file] — Trang [số_trang]*
5. Nếu sử dụng nhiều nguồn, liệt kê từng nguồn riêng biệt.

### Về phong cách
6. **Ngôn ngữ**: Luôn trả lời bằng **tiếng Việt**, trừ khi người dùng yêu cầu khác.
7. **Chuyên nghiệp**: Ngôn ngữ lịch sự, rõ ràng, có cấu trúc.
8. **Súc tích**: Trả lời đúng trọng tâm, dùng danh sách khi liệt kê nhiều điểm.\
"""


# ===========================================================
# Prompt Factory Functions
# ===========================================================


def get_chat_prompt_template() -> ChatPromptTemplate:
    """
    Trả về ChatPromptTemplate cho basic chat (không có RAG context).

    Cấu trúc:
        1. SystemMessage  : Định nghĩa vai trò và nguyên tắc của AI.
        2. MessagesPlaceholder("history"): Lịch sử hội thoại từ memory.
        3. HumanMessage   : Câu hỏi hiện tại của người dùng.

    Tương thích với RunnableWithMessageHistory — key "history" khớp
    với `history_messages_key` trong ChatChain._build_chain().

    Returns:
        ChatPromptTemplate: Prompt template sẵn sàng để pipe vào LLM.

    Example:
        >>> from app.prompts import get_chat_prompt_template
        >>> prompt = get_chat_prompt_template()
        >>> chain = prompt | llm | StrOutputParser()
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Trả về ChatPromptTemplate cho RAG chat (có {context} từ retriever).

    Cấu trúc:
        1. SystemMessage với {context} placeholder.
        2. MessagesPlaceholder("history"): Lịch sử hội thoại.
        3. HumanMessage: Câu hỏi hiện tại.

    Dùng ở Phase 3 khi tích hợp Vector DB + Retriever.
    ChatChain sẽ inject context từ retriever vào biến {context}.

    Returns:
        ChatPromptTemplate: RAG-ready prompt template.

    Example:
        >>> from app.prompts import get_rag_prompt_template
        >>> rag_prompt = get_rag_prompt_template()
        >>> rag_chain = (
        ...     {"context": retriever, "input": RunnablePassthrough(), "history": ...}
        ...     | rag_prompt | llm | StrOutputParser()
        ... )
    """
    return ChatPromptTemplate.from_messages([
        ("system", _RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
