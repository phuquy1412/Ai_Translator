"""
Gemini Translator - Bản kết hợp tốt nhất
- Gửi ảnh crop trực tiếp (như người 2) để AI đọc chính xác hơn
- Batch nhiều ảnh/trang cùng lúc (như người 1) để tiết kiệm API call
- Safety settings BLOCK_NONE để không bị từ chối với nội dung action/violence
- Retry + exponential backoff khi gặp lỗi mạng
- Fallback thông minh: batch thất bại → single, quota → trả gốc ngay
- response_mime_type="application/json" để parse ổn định
"""

import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME      = "gemini-2.5-flash"
MAX_RETRIES     = 3
RETRY_BASE      = 0.5   # 0.5s → 1s → 2s

LANG_NAMES = {
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "en": "English",  "vi": "Vietnamese",
}

STYLE_PRESETS = {
    "default":  "",
    "casual":   "Dùng ngôn ngữ thân mật, trẻ trung, nhiều từ lóng.",
    "formal":   "Dùng ngôn ngữ trang trọng, lịch sự.",
    "action":   "Câu ngắn, mạnh, dứt khoát. Ưu tiên impact.",
    "romance":  "Nhẹ nhàng, tình cảm, giàu cảm xúc.",
}

SAFETY_OFF = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_NONE"},
]

TRANSLATION_RULES = """
QUY TẮC DỊCH:
1. ĐÂY LÀ HỘI THOẠI NÓI - phải nghe tự nhiên như người thật nói chuyện
2. DỊCH ĐẦY ĐỦ 100% SANG TIẾNG VIỆT - kể cả từ tiếng Anh ngắn (Money, Attack...)
3. TUYỆT ĐỐI KHÔNG dịch word-by-word, diễn đạt lại theo cách người Việt nói
4. Giữ nguyên cảm xúc, tính cách nhân vật qua cách dùng từ
5. Câu ngắn giữ ngắn, không thêm thắt dài dòng

HƯỚNG DẪN TIẾNG VIỆT:
- TÊN NHÂN VẬT: GIỮ NGUYÊN tên gốc, có thể Việt hóa nhẹ
  + Nhật: -san→anh/chị, -kun→bạn/cậu, senpai→tiền bối, sensei→thầy
  + Hàn: sunbae→tiền bối, oppa→anh, hyung→anh, noona→chị
  + Trung: sư huynh, sư đệ, đại nhân giữ nguyên
- Đại từ nhân xưng phù hợp quan hệ:
  + Bạn thân: tao/mày, tớ/cậu
  + Người yêu: anh/em, mình
  + Trang trọng: tôi/anh/chị
  + Gia đình: con/bố/mẹ/ông/bà
- Thán từ tự nhiên:
  + くそ/チクショウ → Đ*t/Chết tiệt/Khốn kiếp
  + やばい → Toang rồi/Xong đời
  + すごい → Đỉnh thật/Bá đạo
  + なに/何 → Cái gì/Hả
  + 大丈夫 → Ổn mà/Không sao
  + 아이고 → Ối trời/Chết tôi chưa
- Dùng khẩu ngữ tự nhiên: oke, ngon, tởm, đỉnh, toang, chill...
- TRÁNH: dịch sách giáo khoa, quá nhiều từ Hán Việt, giữ cấu trúc câu gốc
"""


# ── Main class ───────────────────────────────────────────────────────────────
class GeminiTranslator:
    """
    Translator dùng Google Gemini.
    Hỗ trợ: dịch text, dịch từ ảnh crop, batch nhiều trang.
    """

    def __init__(
        self,
        api_key: str = None,
        style: str = "default",
        custom_prompt: str = None,
        model_name: str = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Cần GEMINI_API_KEY. Set trong .env hoặc truyền vào api_key=")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name or MODEL_NAME)

        self.style_text = custom_prompt or STYLE_PRESETS.get(style, "")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_style_line(self, override: str = None) -> str:
        s = override or self.style_text
        return f"\nPhong cách: {s}" if s else ""

    def _call_with_retry(self, contents, use_json: bool = True) -> str:
        """Gọi Gemini với retry + exponential backoff. Trả về response.text."""
        gen_cfg = genai.types.GenerationConfig(
            temperature=0.1,
            **({"response_mime_type": "application/json"} if use_json else {}),
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(
                    contents,
                    generation_config=gen_cfg,
                    safety_settings=SAFETY_OFF,
                )
                return response.text.strip()

            except Exception as e:
                err = str(e)
                print(f"[Gemini] attempt {attempt+1}/{MAX_RETRIES} failed: {e}")

                # Quota → không retry, trả ngay
                if "429" in err or "quota" in err.lower():
                    print("⚠️  Quota exceeded! Dừng retry để tránh tốn thêm quota.")
                    raise

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE * (2 ** attempt)
                    print(f"   Retry sau {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _clean_json(text: str) -> str:
        """Bỏ markdown code fences nếu model trả về."""
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def translate_single(
        self,
        text: str,
        source: str = "ja",
        target: str = "vi",
        custom_prompt: str = None,
    ) -> str:
        """Dịch 1 đoạn text."""
        if not text or not text.strip():
            return text

        src = LANG_NAMES.get(source, "Japanese")
        tgt = LANG_NAMES.get(target, "Vietnamese")

        prompt = f"""Bạn là chuyên gia dịch manga/comic từ {src} sang {tgt}.
{TRANSLATION_RULES}{self._build_style_line(custom_prompt)}

Gợi ý bối cảnh: Dịch sát nghĩa, tự nhiên, phù hợp truyện tranh.

IMPORTANT: Trả về CHỈ bản dịch, không giải thích, không markdown.

Text gốc: {text}"""

        try:
            return self._call_with_retry([prompt], use_json=False)
        except Exception as e:
            print(f"[translate_single] error: {e}")
            return text

    def translate_batch(
        self,
        texts: List[str],
        source: str = "ja",
        target: str = "vi",
        custom_prompt: str = None,
    ) -> List[str]:
        """
        Dịch nhiều đoạn text trong 1 API call.
        Tự động fallback sang từng cái nếu batch thất bại.
        """
        if not texts:
            return []

        # Giữ vị trí các string rỗng
        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed:
            return texts

        to_translate = [t for _, t in indexed]
        translations = self._batch_internal(to_translate, source, target, custom_prompt)

        result = list(texts)
        for (orig_i, _), trans in zip(indexed, translations):
            result[orig_i] = trans
        return result

    def translate_from_crops(
        self,
        crops: List,                  # List[PIL.Image hoặc numpy array hoặc bytes]
        source: str = "ja",
        target: str = "vi",
        hint: str = None,
        custom_prompt: str = None,
    ) -> List[str]:
        """
        Dịch trực tiếp từ ảnh crop bubble — AI tự đọc chữ trong ảnh.
        Tốt hơn OCR riêng vì Gemini đọc cả font lạ, chữ nghiêng, chữ nhỏ.

        Args:
            crops: Danh sách ảnh (PIL.Image, numpy BGR array, hoặc bytes PNG/JPEG)
            hint: Gợi ý bối cảnh cho model
        """
        if not crops:
            return []

        src = LANG_NAMES.get(source, "Japanese")
        tgt = LANG_NAMES.get(target, "Vietnamese")
        hint_text = hint or "Dịch sát nghĩa, ngầu, phù hợp với hành động."

        prompt = f"""Bạn là Editor truyện tranh chuyên nghiệp dịch từ {src} sang {tgt}.
Nhiệm vụ: Dịch nội dung từ {len(crops)} bức ảnh bong bóng thoại dưới đây.
{TRANSLATION_RULES}{self._build_style_line(custom_prompt)}

Gợi ý bối cảnh: {hint_text}

IMPORTANT: Trả về ĐÚNG 1 JSON array với {len(crops)} chuỗi, theo THỨ TỰ GIỐNG HỆT.
Format: ["bản dịch 1", "bản dịch 2", ...]
Không đánh số, không giải thích."""

        # Chuẩn hóa crops về dạng Gemini chấp nhận
        contents = [prompt] + [self._normalize_crop(c) for c in crops]

        try:
            raw = self._call_with_retry(contents, use_json=True)
            result = json.loads(self._clean_json(raw))

            # Đảm bảo độ dài khớp
            result = result[:len(crops)]
            while len(result) < len(crops):
                result.append("...")
            return result

        except Exception as e:
            print(f"[translate_from_crops] error: {e}")
            return ["..."] * len(crops)

    def translate_pages_batch(
        self,
        pages_texts: Dict[str, List[str]],
        source: str = "ja",
        target: str = "vi",
        custom_prompt: str = None,
    ) -> Dict[str, List[str]]:
        """
        Dịch nhiều trang liên tiếp trong 1 API call.
        Giữ mạch truyện và giọng nhân vật nhất quán xuyên trang.

        Args:
            pages_texts: {"page_1": ["text1", "text2"], "page_2": [...], ...}
        """
        if not pages_texts:
            return {}

        src = LANG_NAMES.get(source, "Japanese")
        tgt = LANG_NAMES.get(target, "Vietnamese")

        prompt = f"""Bạn là chuyên gia dịch manga/comic từ {src} sang {tgt}.
Đây là các trang LIÊN TIẾP trong cùng 1 story. Giữ mạch truyện và giọng nhân vật nhất quán.
{TRANSLATION_RULES}{self._build_style_line(custom_prompt)}

Input (JSON - các trang liên tiếp):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

IMPORTANT: Trả về ĐÚNG JSON object với cấu trúc GIỐNG HỆT nhưng đã dịch.
Giữ nguyên tên page và thứ tự bubble. Không giải thích, không markdown."""

        try:
            raw = self._call_with_retry([prompt], use_json=True)
            return json.loads(self._clean_json(raw))

        except Exception as e:
            print(f"[translate_pages_batch] error: {e}, fallback sang từng trang")
            return {
                page: self.translate_batch(texts, source, target, custom_prompt)
                for page, texts in pages_texts.items()
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _batch_internal(
        self,
        texts: List[str],
        source: str,
        target: str,
        custom_prompt: str = None,
    ) -> List[str]:
        src = LANG_NAMES.get(source, "Japanese")
        tgt = LANG_NAMES.get(target, "Vietnamese")

        prompt = f"""Bạn là chuyên gia dịch manga/comic từ {src} sang {tgt}.
{TRANSLATION_RULES}{self._build_style_line(custom_prompt)}

Input (JSON array - mỗi item là 1 bubble):
{json.dumps(texts, ensure_ascii=False)}

IMPORTANT: Trả về ĐÚNG JSON array với {len(texts)} bản dịch theo THỨ TỰ GIỐNG HỆT.
Format: ["bản dịch 1", "bản dịch 2", ...]"""

        try:
            raw = self._call_with_retry([prompt], use_json=True)
            result = json.loads(self._clean_json(raw))

            if len(result) != len(texts):
                raise ValueError(f"Mismatch: expected {len(texts)}, got {len(result)}")
            return result

        except Exception as e:
            err = str(e)
            # Quota → trả gốc ngay, không fallback
            if "429" in err or "quota" in err.lower():
                print("⚠️  Quota! Trả về text gốc.")
                return texts

            print(f"[_batch_internal] fallback sang single: {e}")
            return [self.translate_single(t, source, target, custom_prompt) for t in texts]

    @staticmethod
    def _normalize_crop(crop) -> any:
        """Chuyển numpy/bytes/PIL về dạng Gemini hiểu."""
        import numpy as np

        # Bytes thô → giữ nguyên (Gemini nhận được)
        if isinstance(crop, bytes):
            import PIL.Image, io
            return PIL.Image.open(io.BytesIO(crop))

        # Numpy BGR (OpenCV) → PIL RGB
        if isinstance(crop, np.ndarray):
            import cv2
            import PIL.Image
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return PIL.Image.fromarray(rgb)

        # PIL Image → giữ nguyên
        return crop


# ── Convenience function (tương thích code cũ) ────────────────────────────────
def translate_crops(
    crops: List,
    hint: str = None,
    source: str = "ja",
    target: str = "vi",
    api_key: str = None,
) -> List[str]:
    """
    Drop-in replacement cho hàm translate_crops() của người 2.
    Gửi ảnh crop trực tiếp cho Gemini đọc + dịch.
    """
    translator = GeminiTranslator(api_key=api_key)
    return translator.translate_from_crops(crops, source=source, target=target, hint=hint)