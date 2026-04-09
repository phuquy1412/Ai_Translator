from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
import io
import zipfile
import json

from process_bubble import process_many, process_single

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

LANGUAGES = {
    "ja": "Japanese (Manga)",
    "zh": "Chinese (Manhua)", 
    "ko": "Korean (Manhwa)",
    "en": "English (Comic)"
}

MODES = {
    "translate": "Auto Translate 🤖",
    "erase": "Erase Only 🗑️",
}

@app.route("/")
def home():
    return render_template("index.html", languages=LANGUAGES, modes=MODES)


@app.route("/translate", methods=["POST"])
def translate():
    files = request.files.getlist("files")
    src_lang = request.form.get("src_lang", "ja")
    mode = request.form.get("mode", "translate")

    if not files or files[0].filename == "":
        return jsonify({"error": "No files"}), 400

    # Đọc tất cả ảnh vào memory
    images = []
    for file in files:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            images.append({
                'name': os.path.splitext(file.filename)[0],
                'image': image
            })

    if not images:
        return jsonify({"error": "No valid images"}), 400

    # Progress callback
    def on_progress(current, total, message):
        socketio.emit("progress", {
            "current": current,
            "total": total,
            "message": message,
            "percent": int(current / total * 100)
        })

    # Xử lý tất cả ảnh
    results = process_many(
        images,
        src_lang=src_lang,
        mode=mode,
        progress_callback=on_progress
    )

    # Encode sang base64
    processed_images = []
    for result in results:
        _, buffer = cv2.imencode(".jpg", result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        processed_images.append({
            "name": result['name'],
            "data": encoded,
            "bubbles": result['bubbles']
        })

    socketio.emit("progress", {"message": "Hoàn tất!", "percent": 100})
    return render_template("translate.html", images=processed_images, mode=mode)


@app.route("/rerender", methods=["POST"])
def rerender():
    """
    Mode 2: User sửa text → render lại bubble đó.
    Nhận: ảnh gốc + tọa độ bubble + text mới
    Trả về: ảnh đã cập nhật
    """
    data = request.json
    image_data = data.get("image")
    x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
    new_text = data["text"]

    # Decode ảnh từ base64
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Xóa chữ cũ + vẽ chữ mới
    from add_text import xoa_chu_trong_bubble, ve_chu_vao_bubble
    image = xoa_chu_trong_bubble(image, x1, y1, x2, y2)
    image = ve_chu_vao_bubble(image, x1, y1, x2, y2, new_text)

    # Encode lại
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return jsonify({"image": encoded})


@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Tải tất cả ảnh về dạng ZIP"""
    images = json.loads(request.form.get("images_data", "[]"))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            image_bytes = base64.b64decode(img['data'])
            zf.writestr(f"{img['name']}_translated.jpg", image_bytes)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='manga_translated.zip'
    )


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)