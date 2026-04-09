import cv2
from PIL import Image
from tqdm import tqdm
from detect_bubbles import detect_bubbles
from ocr.chrome_lens_ocr import ChromeLensOCR
from add_text import xoa_chu_trong_bubble, ve_chu_vao_bubble
from translator.local_llm_translator import translate_batch

# Cache OCR để không khởi tạo lại mỗi lần
_ocr_cache = {}

def get_ocr(lang):
    if lang not in _ocr_cache:
        _ocr_cache[lang] = ChromeLensOCR(ocr_language=lang)
    return _ocr_cache[lang]


def process_single(image, src_lang="ja", mode="translate"):
    """
    Xử lý 1 ảnh.
    
    Args:
        image: numpy array BGR
        src_lang: ngôn ngữ gốc (ja, zh, ko, en)
        mode: translate | erase | ocr_only
        
    Returns:
        dict: {
            'image': ảnh đã xử lý,
            'bubbles': [{ 'box': [x1,y1,x2,y2], 'original': text, 'translated': text }]
        }
    """
    ocr = get_ocr(src_lang)

    # Detect bubbles
    boxes = detect_bubbles(image)
    if not boxes:
        return {'image': image, 'bubbles': []}

    bubble_info = []

    # Erase only - chỉ xóa chữ
    if mode == "erase":
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            image = xoa_chu_trong_bubble(image, x1, y1, x2, y2)
            bubble_info.append({
                'box': [x1, y1, x2, y2],
                'original': '',
                'translated': ''
            })
        return {'image': image, 'bubbles': bubble_info}

    # OCR tất cả bubble
    texts = []
    for box in tqdm(boxes, desc="OCR", leave=False):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(__import__('cv2').cvtColor(crop, __import__('cv2').COLOR_BGR2RGB))
        texts.append(ocr(crop_pil))

    # Ocr only - chỉ OCR không dịch
    if mode == "ocr_only":
        for box, text in zip(boxes, texts):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bubble_info.append({
                'box': [x1, y1, x2, y2],
                'original': text,
                'translated': ''
            })
        return {'image': image, 'bubbles': bubble_info}

    # Translate mode - dịch 1 lần toàn bộ
    print("Đang dịch...")
    translated_texts = translate_batch(texts)

    # Xóa chữ + vẽ chữ dịch
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


def process_many(images, src_lang="ja", mode="translate", progress_callback=None):
    """
    Xử lý nhiều ảnh cùng lúc.
    
    Args:
        images: list of {'image': numpy array, 'name': str}
        src_lang: ngôn ngữ gốc
        mode: translate | erase | ocr_only
        progress_callback: function(current, total, message) để update progress
        
    Returns:
        list of {'name': str, 'image': numpy array, 'bubbles': [...]}
    """
    results = []
    total = len(images)

    for i, img_data in enumerate(images):
        name = img_data['name']
        image = img_data['image']

       
        if progress_callback:
            progress_callback(i + 1, total, f"Đang xử lý {name}...")

        print(f"[{i+1}/{total}] {name}")
        result = process_single(image, src_lang=src_lang, mode=mode)

        results.append({
            'name': name,
            'image': result['image'],
            'bubbles': result['bubbles']
        })

    return results


if __name__ == "__main__":
    image = cv2.imread("examples/0.png")
    result = process_single(image, src_lang="ja", mode="translate")
    cv2.imwrite("result_translated.png", result['image'])
    print("✅ Da luu result_translated.png")