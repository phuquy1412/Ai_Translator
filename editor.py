import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from add_text import wrap_text

class BubbleEditor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.font_dir = os.path.join(self.base_dir, "fonts")

    def get_available_fonts(self):
        if not os.path.exists(self.font_dir):
            return []
        return [f for f in os.listdir(self.font_dir) if f.endswith(('.ttf', '.otf'))]

    def clear_bubble(self, image, coords):
        x1, y1, x2, y2 = coords
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(crop)
            cv2.drawContours(mask, [biggest], -1, (255, 255, 255), -1)
            crop[mask[:,:,0] == 255] = 255
            image[y1:y2, x1:x2] = crop
        return image

    # ✅ Thụt lề đúng vào trong class, và thêm các tham số mới từ app.py
    def process_render(self, image, coords, text, font_name="ariali.ttf", font_size=20,
                       font_color="#000000", bold=False, italic=False, align="center"):
        x1, y1, x2, y2 = coords

        # Parse màu hex → RGB tuple
        font_color = font_color.lstrip("#")
        color = tuple(int(font_color[i:i+2], 16) for i in (0, 2, 4))

        # 1. Xóa nội dung cũ
        image = self.clear_bubble(image, coords)

        # 2. Chuyển sang PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 3. Cấu hình Font
        font_path = os.path.join(self.font_dir, font_name)
        if not os.path.exists(font_path):
            font_path = os.path.join(self.font_dir, "ariali.ttf")

        try:
            font = ImageFont.truetype(font_path, int(font_size))
        except:
            font = ImageFont.load_default()

        # 4. Wrap text và vẽ
        max_w = (x2 - x1) - 10
        lines = wrap_text(text, font, max_w)

        line_height = int(font_size) + 2
        total_h = len(lines) * line_height
        curr_y = (y1 + y2) // 2 - total_h // 2
        center_x = (x1 + x2) // 2

        # Xử lý align
        anchor_map = {"left": "lt", "center": "mt", "right": "rt"}
        anchor = anchor_map.get(align, "mt")
        x_map = {"left": x1 + 5, "center": center_x, "right": x2 - 5}
        draw_x = x_map.get(align, center_x)

        for line in lines:
            draw.text((draw_x, curr_y), line, font=font, fill=color,
                      anchor=anchor, stroke_width=1, stroke_fill=(255, 255, 255))
            curr_y += line_height

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)