from PIL import Image
import numpy as np
from chrome_lens_py import LensAPI
import asyncio

class ChromeLensOCR:
    def __init__(self, ocr_language="ja"):
        self.api = LensAPI()
        self.ocr_language = ocr_language

    def __call__(self, image):
        # Chuyển numpy array sang PIL nếu cần
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Chạy async
        return asyncio.run(self._process(image))

    async def _process(self, image):
        try:
            result = await self.api.process_image(
                image_path=image,
                ocr_language=self.ocr_language
            )
            return result.get("ocr_text", "")
        except Exception as e:
            print(f"Loi OCR: {e}")
            return ""
