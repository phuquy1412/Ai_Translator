import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def xoa_chu_trong_bubble(image, x1, y1, x2, y2):
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(crop)
        cv2.drawContours(mask, [biggest], -1, (255, 255, 255), -1)
        crop[mask == 255] = 255
        image[y1:y2, x1:x2] = crop
    return image

def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if font.getlength(test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def ve_chu_vao_bubble(image, x1, y1, x2, y2, text):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    bubble_width = x2 - x1
    font_size = max(10, bubble_width // 10)

    try:
        font = ImageFont.truetype("fonts/ariali.ttf", font_size)
    except:
        font = ImageFont.load_default()

    max_width = bubble_width - 20
    lines = wrap_text(text, font, max_width)

    line_height = font_size + 4
    total_height = len(lines) * line_height
    center_x = (x1 + x2) // 2
    start_y = (y1 + y2) // 2 - total_height // 2

    for line in lines:
        draw.text((center_x, start_y), line, font=font, fill=(0, 0, 0), anchor="mt")
        start_y += line_height

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)