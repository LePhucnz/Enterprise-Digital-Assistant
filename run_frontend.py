"""
Run script — Khởi chạy Streamlit frontend.

Chạy từ thư mục gốc của project:
    streamlit run run_frontend.py

Lý do cần file này:
    Khi chạy `streamlit run frontend/app.py`, Streamlit thêm thư mục
    `frontend/` vào sys.path thay vì project root. Điều này khiến câu
    lệnh `from frontend.api_client import APIClient` trong frontend/app.py
    thất bại vì Python không tìm thấy package `frontend` trong sys.path.

    Bằng cách chạy `streamlit run run_frontend.py` từ project root,
    Python tự động thêm project root vào sys.path — đúng chuẩn Python
    packaging, không cần bất kỳ sys.path hack nào.

Cơ chế hoạt động:
    Streamlit re-execute toàn bộ script này mỗi khi có user interaction.
    `from frontend import app` khiến Python nạp module frontend/app.py,
    bao gồm cả phần module-level code (set_page_config, CSS) và sau đó
    gọi main() để render UI. Pattern này đảm bảo Streamlit hoạt động
    đúng với execution model của nó.
"""

# Entry point thuần túy — toàn bộ logic nằm trong frontend/app.py.
# Import và gọi main() để Streamlit re-execute đúng cách mỗi lần rerun.
from frontend import app as _frontend_app  # noqa: F401 — triggers module-level setup

_frontend_app.main()

