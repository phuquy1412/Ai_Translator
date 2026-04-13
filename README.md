# 💬 AI Manga Translator

AI Manga Translator là một công cụ tự động hóa hoàn toàn việc dịch thuật truyện tranh (Manga, Manhua, Manhwa, Comic) từ nhiều ngôn ngữ khác nhau sang tiếng Việt. Công cụ kết hợp sức mạnh của các mô hình AI hiện đại để xử lý hình ảnh, nhận diện văn bản và dịch thuật mượt mà.

## ✨ Tính năng nổi bật

- **Nhận diện bong bóng thoại tự động**: Sử dụng mô hình `YOLO` để khoanh vùng chính xác các bong bóng thoại trong trang truyện.
- **Xóa chữ thông minh**: Tự động xóa chữ gốc trong bong bóng thoại mà không làm hỏng nền (sử dụng OpenCV).
- **Trích xuất văn bản (OCR) mạnh mẽ**: Sử dụng `Chrome Lens OCR` để đọc văn bản độ chính xác cao đối với nhiều loại ngôn ngữ, kể cả chữ dọc của Nhật Bản.
- **Dịch thuật AI Đa nền tảng**: 
  - Tích hợp **Google Gemini API** (`gemini-2.5-flash`) cho tốc độ dịch siêu nhanh, hiểu ngữ cảnh truyện tranh và hỗ trợ dịch trực tiếp từ vùng crop của bong bóng.
  - Tích hợp **Local LLM (Ollama)** (như `translategemma:4b`) cho dịch thuật offline an toàn, không lo kiểm duyệt với prompt cực mạnh.
- **Trình chỉnh sửa văn bản tích hợp (Editor)**:
  - Tùy chỉnh màu sắc, phông chữ, kích thước, in nghiêng, in đậm và căn lề.
  - Tự động xuống dòng cho văn bản vừa vặn hoàn hảo vào bong bóng.
- **Xử lý hàng loạt (Batch Processing)**: Hỗ trợ kéo thả tải lên nhiều ảnh/trang truyện cùng lúc và thanh tiến trình (Progress Bar) theo thời gian thực.
- **Xuất file dễ dàng**: Tải nguyên tệp `.zip` chứa tất cả các trang đã hoàn tất dịch.

## 💻 Công nghệ sử dụng

- **Backend**: Python, Flask, Flask-SocketIO.
- **Computer Vision & AI**: OpenCV, Ultralytics YOLOv8, Pillow.
- **OCR**: Chrome Lens API (`chrome_lens_py`).
- **LLMs**: Google Gemini SDK, Requests (gọi API Ollama).

## 📂 Cấu trúc dự án chính

```text
Ai_Translator/
├── app.py                      # Flask Web Application (Xử lý định tuyến, Upload, Download, Rerender)
├── process_bubble.py           # Logic cốt lõi: Kết hợp OCR, Dịch, và Vẽ chữ lại
├── detect_bubbles.py           # Gọi mô hình YOLO nhận diện vị trí các bong bóng chữ
├── editor.py                   # Quản lý trình chỉnh sửa bong bóng thoại (fonts, sizes, text wrap...)
├── add_text.py                 # (Inferred) Hàm xóa chữ và vẽ chữ vào vùng ảnh
├── ocr/
│   └── chrome_lens_ocr.py      # Module trích xuất chữ viết bằng Lens API
├── translator/
│   ├── gemini_translator.py    # Class tích hợp Google Gemini API
│   ├── local_llm_translator.py # Tích hợp Local Model qua Ollama API
│   └── context_memory.py       # (Inferred) Bộ nhớ ngữ cảnh cho LLM cục bộ
├── model/
│   └── comic.pt                # Mô hình YOLO đã train để nhận diện truyện tranh (CẦN TẢI/HUẤN LUYỆN)
├── fonts/                      # Thư mục chứa các font chữ (.ttf, .otf) cho Editor
└── examples/                   # Thư mục ảnh dùng để test code
```

## 🚀 Hướng dẫn Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.8 trở lên.
- Tùy chọn: Đã cài đặt Ollama và tải model nếu dùng chức năng Local LLM (VD: `ollama run translategemma:4b`).

### 2. Cài đặt thư viện

Clone dự án và chạy các lệnh dưới đây để cài đặt những thư viện cần thiết:

```bash
pip install Flask Flask-SocketIO opencv-python numpy Pillow ultralytics google-generativeai requests python-dotenv chrome_lens_py tqdm asyncio
```

### 3. Cấu hình

**1. Mô hình YOLO:**
Đảm bảo bạn đã lưu một mô hình nhận diện bong bóng thoại (ví dụ `comic.pt` hoặc `yolov8n.pt`) vào thư mục `model/`. Mặc định mã nguồn tìm `model/comic.pt`.

**2. Font chữ:**
Tạo thư mục `fonts/` ở thư mục gốc của dự án và thêm một vài font chữ hỗ trợ tiếng Việt (VD: `ariali.ttf`) vào đó.

**3. Cấu hình API:**
Tạo file `.env` ở thư mục gốc của dự án để thiết lập các biến môi trường nếu dùng Google Gemini:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(Hoặc bạn có thể nhập API Key trực tiếp trên giao diện trình duyệt khi sử dụng).*

## 🛠️ Hướng dẫn Sử dụng

1. **Khởi chạy ứng dụng:**

```bash
python app.py
```

2. **Truy cập Giao diện Web:**
   Mở trình duyệt và truy cập vào địa chỉ `http://localhost:5000` hoặc `http://127.0.0.1:5000`.

3. **Thao tác:**
   - **Tải ảnh lên**: Chọn nhiều file ảnh truyện tranh.
   - **Chọn Ngôn ngữ Gốc**: Nhật, Trung, Hàn hoặc Anh.
   - **Chọn Nguồn AI**: Dùng Gemini hoặc Local LLM.
   - Nhấn **Translate** và chờ thanh tiến trình xử lý. Khi hoàn tất, bạn có thể chỉnh sửa lại text (Re-render) và Tải xuống bản `.zip`.