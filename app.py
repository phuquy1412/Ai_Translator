from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
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
    "erase":     "Erase Only 🗑️",
}


@app.route("/")
def home():
    return render_template("index.html", languages=LANGUAGES, modes=MODES)


@app.route("/translate", methods=["POST"])
def translate():
    files      = request.files.getlist("files")
    src_lang   = request.form.get("src_lang", "ja")
    mode       = request.form.get("mode", "translate")
    ai_source  = request.form.get("ai_source", "local")       # "local" | "gemini"
    api_key    = request.form.get("gemini_api_key", "").strip() or None
    model_name = request.form.get("gemini_model", "").strip() or "gemini-2.5-flash"

    if not files or files[0].filename == "":
        return jsonify({"error": "No files"}), 400

    # Validate: nếu chọn gemini mà không có key thì báo lỗi
    if ai_source == "gemini" and not api_key:
        return jsonify({"error": "Gemini API key is required"}), 400

    # Đọc ảnh vào memory
    images = []
    for file in files:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            images.append({
                'name':  os.path.splitext(file.filename)[0],
                'image': image
            })

    if not images:
        return jsonify({"error": "No valid images"}), 400

    def on_progress(current, total, message):
        socketio.emit("progress", {
            "current": current,
            "total":   total,
            "message": message,
            "percent": int(current / total * 100)
        })

    results = process_many(
        images,
        src_lang=src_lang,
        mode=mode,
        progress_callback=on_progress,
        ai_source=ai_source,
        api_key=api_key,
        model_name=model_name
    )

    processed_images = []
    for result in results:
        _, buffer = cv2.imencode(".jpg", result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        processed_images.append({
            "name":    result['name'],
            "data":    encoded,
            "bubbles": result['bubbles']
        })

    socketio.emit("progress", {"message": "Hoàn tất!", "percent": 100})
    return render_template("translate.html", images=processed_images, mode=mode)


@app.route("/get-fonts", methods=["GET"])
def get_fonts():
    font_dir = os.path.join(app.root_path, "fonts")
    if not os.path.exists(font_dir):
        return jsonify([])
    fonts = [f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]
    return jsonify(fonts)


@app.route("/rerender", methods=["POST"])
def rerender():
    data       = request.json
    image_data = data.get("image")
    coords     = (int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))
    new_text   = data.get("text")
    font_name  = data.get("font_name", "ariali.ttf")
    font_size  = int(data.get("font_size", 20))
    font_color = data.get("font_color", "#000000")
    bold       = bool(data.get("bold", False))
    italic     = bool(data.get("italic", False))
    align      = data.get("align", "center")

    image_bytes = base64.b64decode(image_data.split(",")[-1])
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    from editor import BubbleEditor
    editor = BubbleEditor()
    result_img = editor.process_render(
        img, coords, new_text,
        font_name=font_name, font_size=font_size,
        font_color=font_color, bold=bold, italic=italic, align=align
    )

    _, buffer = cv2.imencode(".jpg", result_img)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"image": encoded})


@app.route("/download-zip", methods=["POST"])
def download_zip():
    data   = request.get_json()
    images = data.get("images", [])

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