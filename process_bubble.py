import cv2
from PIL import Image
from tqdm import tqdm
from detect_bubbles import detect_bubbles
from add_text import xoa_chu_trong_bubble, ve_chu_vao_bubble

# Cache OCR để không khởi tạo lại mỗi lần
_ocr_cache = {}

def get_ocr(lang):
    from ocr.chrome_lens_ocr import ChromeLensOCR
    if lang not in _ocr_cache:
        _ocr_cache[lang] = ChromeLensOCR(ocr_language=lang)
    return _ocr_cache[lang]


def get_translator(ai_source="local", api_key=None, model_name=None):
    """
    Trả về translator phù hợp theo ai_source.
    - "gemini": dùng GeminiTranslator với API key
    - "local": dùng local_llm_translator cũ
    """
    if ai_source == "gemini":
        from translator.gemini_translator import GeminiTranslator
        return GeminiTranslator(
            api_key=api_key,
            model_name=model_name or "gemini-2.5-flash"
        )
    else:
        from translator.local_llm_translator import translate_batch as local_batch
        # Wrap thành object có method translate_batch cho đồng nhất
        class LocalTranslator:
            def translate_batch(self, texts, source="ja", target="vi"):
                return local_batch(texts)
        return LocalTranslator()


def process_single(image, src_lang="ja", mode="translate", ai_source="local", api_key=None, model_name=None):
    """
    Xử lý 1 ảnh.

    Args:
        image: numpy array BGR
        src_lang: ngôn ngữ gốc (ja, zh, ko, en)
        mode: translate | erase | ocr_only
        ai_source: "local" | "gemini"
        api_key: Gemini API key (nếu dùng gemini)
        model_name: tên model Gemini (optional)

    Returns:
        dict: {
            'image': ảnh đã xử lý,
            'bubbles': [{ 'box': [x1,y1,x2,y2], 'original': text, 'translated': text }]
        }
    """
    # Detect bubbles
    boxes = detect_bubbles(image)
    if not boxes:
        return {'image': image, 'bubbles': []}

    bubble_info = []

    # Erase only
    if mode == "erase":
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            image = xoa_chu_trong_bubble(image, x1, y1, x2, y2)
            bubble_info.append({'box': [x1, y1, x2, y2], 'original': '', 'translated': ''})
        return {'image': image, 'bubbles': bubble_info}

    # OCR tất cả bubble
    ocr = get_ocr(src_lang)
    texts = []
    for box in tqdm(boxes, desc="OCR", leave=False):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        texts.append(ocr(crop_pil))

    # OCR only
    if mode == "ocr_only":
        for box, text in zip(boxes, texts):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bubble_info.append({'box': [x1, y1, x2, y2], 'original': text, 'translated': ''})
        return {'image': image, 'bubbles': bubble_info}

    # Translate
    print(f"Đang dịch với [{ai_source}]...")
    translator = get_translator(ai_source=ai_source, api_key=api_key, model_name=model_name)
    translated_texts = translator.translate_batch(texts, source=src_lang, target="vi")

    for box, original, translated in zip(boxes, texts, translated_texts):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        image = xoa_chu_trong_bubble(image, x1, y1, x2, y2)
        image = ve_chu_vao_bubble(image, x1, y1, x2, y2, translated)
        bubble_info.append({
            'box': [x1, y1, x2, y2],
            'original': original,
            'translated': translated
        })

    return {'image': image, 'bubbles': bubble_info}


def process_many(images, src_lang="ja", mode="translate", progress_callback=None,
                 ai_source="local", api_key=None, model_name=None):
    """
    Xử lý nhiều ảnh cùng lúc.

    Args:
        images: list of {'image': numpy array, 'name': str}
        src_lang: ngôn ngữ gốc
        mode: translate | erase | ocr_only
        progress_callback: function(current, total, message)
        ai_source: "local" | "gemini"
        api_key: Gemini API key
        model_name: tên model Gemini
    """
    results = []
    total = len(images)

    for i, img_data in enumerate(images):
        name = img_data['name']
        image = img_data['image']

        if progress_callback:
            progress_callback(i + 1, total, f"Đang xử lý {name}...")

        print(f"[{i+1}/{total}] {name}")
        result = process_single(
            image,
            src_lang=src_lang,
            mode=mode,
            ai_source=ai_source,
            api_key=api_key,
            model_name=model_name
        )

        results.append({
            'name': name,
            'image': result['image'],
            'bubbles': result['bubbles']
        })

    return results


if __name__ == "__main__":
    image = cv2.imread("examples/0.png")
    result = process_single(image, src_lang="ja", mode="translate", ai_source="local")
    cv2.imwrite("result_translated.png", result['image'])
    print("✅ Da luu result_translated.png")