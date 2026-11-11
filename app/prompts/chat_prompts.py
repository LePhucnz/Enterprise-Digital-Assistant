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
Bạn là trợ lý AI tra cứu quy chế nội bộ doanh nghiệp.

<context>
{context}
</context>

QUY TẮC BẮT BUỘC:
1. **TRUNG THÀNH VỚI CONTEXT:** Chỉ sử dụng thông tin có sẵn trong <context> để trả lời đúng trọng tâm câu hỏi. Tuyệt đối không tự suy diễn, không dùng kiến thức bên ngoài, không bịa đặt thêm nội dung không có trong tài liệu.
2. **TỪ CHỐI KHI THIẾU DỮ LIỆU:** Nếu trong <context> không có thông tin chính xác để trả lời câu hỏi, BẮT BUỘC trả lời nguyên văn câu sau và dừng lại ngay:
   "Tôi không tìm thấy thông tin này trong tài liệu quy chế quản trị được cung cấp."
3. **CHỐNG LẶP & ĐỊNH DẠNG:** Trình bày ngắn gọn, súc tích bằng các gạch đầu dòng rõ ràng. Nghiêm cấm lặp lại ý hoặc sinh danh sách dài lặp đi lặp lại. Sau khi hoàn thành câu trả lời, hãy dừng lại ngay.
4. **TRÍCH DẪN:** Nêu rõ điều khoản hoặc nội dung quy định nếu có trong tài liệu. Khi kết thúc, ghi rõ: *Nguồn: [tên_file] — Trang [số_trang]*.\
"""

# ===========================================================
# Query Expansion Prompt — Dùng để mở rộng góc nhìn tìm kiếm
# ===========================================================

_QUERY_EXPANSION_SYSTEM_PROMPT = """\
Nhiệm vụ: Tạo ra {num_queries} câu hỏi tìm kiếm thay thế cho câu hỏi dưới đây.
Chỉ dùng các từ ngữ có thể xuất hiện trong tài liệu doanh nghiệp nội bộ (quy chế, quy định, điều lệ, nghị quyết, thẩm quyền, nghĩa vụ, hội đồng, đại hội, ban kiểm soát...).
Mỗi câu hỏi trên một dòng riêng. Không đánh số. Không dùng gạch đầu dòng. Không giải thích.\
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


def get_query_expansion_prompt_template() -> ChatPromptTemplate:
    """
    Trả về ChatPromptTemplate cho Query Expansion.

    Nhận 2 biến:
        - input: Câu hỏi ban đầu của người dùng.
        - num_queries: Số lượng biến thể câu hỏi cần sinh.

    Returns:
        ChatPromptTemplate: Prompt template để pipe vào LLM.
    """
    return ChatPromptTemplate.from_messages([
        ("system", _QUERY_EXPANSION_SYSTEM_PROMPT),
        ("human", "Câu hỏi: {input}\nTạo {num_queries} câu hỏi tìm kiếm thay thế:"),
    ])
