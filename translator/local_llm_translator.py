import requests
import json
from translator.context_memory import ContextMemory

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "translategemma:4b"

SYSTEM_PROMPT = """Bạn là dịch giả manga chuyên nghiệp người Việt.
Quy tắc:
- Dịch tự nhiên như người Việt nói chuyện hàng ngày
- Giữ cảm xúc và giọng điệu nhân vật
- Đại từ: tao/mày (thân mật), tôi/anh/chị (trang trọng)
- Thán từ: くそ→Chết tiệt, やばい→Toang rồi, すごい→Đỉnh thật
- Giữ nguyên tên nhân vật
- Câu ngắn giữ ngắn, mạnh mẽ
- KHÔNG giải thích, CHỈ trả về bản dịch"""

# Global context memory
context = ContextMemory()


def translate(text: str, model: str = DEFAULT_MODEL) -> str:
    if not text.strip():
        return text

    response = requests.post(OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context.get_prompt()}Dịch sang tiếng Việt:\n{text}"}
            ],
            "temperature": 0.3
        },
        timeout=30
    )
    result = response.json()["choices"][0]["message"]["content"].strip()
    context.add(text, result)
    return result


def translate_batch(texts: list, model: str = DEFAULT_MODEL, src_lang: str = "ja") -> list:
    if not texts:
        return []

    valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not valid:
        return texts

    valid_texts = [t for _, t in valid]

    lang_names = {"ja": "tiếng Nhật", "zh": "tiếng Trung", "ko": "tiếng Hàn", "en": "tiếng Anh"}
    src_name = lang_names.get(src_lang, "tiếng Nhật")

    numbered = {str(i+1): t for i, t in enumerate(valid_texts)}

    prompt = f"""{context.get_prompt()}Dịch toàn bộ hội thoại manga từ {src_name} sang tiếng Việt.
Các bubble đánh số theo thứ tự đọc — dịch có tính đến ngữ cảnh toàn trang.

Input:
{json.dumps(numbered, ensure_ascii=False, indent=2)}

Trả về JSON object: {{"1": "dịch 1", "2": "dịch 2"}}"""

    response = requests.post(OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        },
        timeout=60
    )

    result_text = response.json()["choices"][0]["message"]["content"].strip()

    if "```" in result_text:
        result_text = result_text.split("```")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]
    result_text = result_text.strip()

    try:
        translated_dict = json.loads(result_text)
        translations = [translated_dict.get(str(i+1), valid_texts[i]) for i in range(len(valid_texts))]
        for orig, trans in zip(valid_texts, translations):
            context.add(orig, trans)
    except json.JSONDecodeError:
        print("JSON lỗi, fallback từng câu...")
        translations = [translate(t, model) for t in valid_texts]

    result = list(texts)
    for (orig_idx, _), trans in zip(valid, translations):
        result[orig_idx] = trans

    return result


def clear_context():
    context.clear()