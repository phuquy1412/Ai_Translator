import cv2
import numpy as np
from ultralytics import YOLO

def xoa_chu_trong_bubble(image, x1, y1, x2, y2):
    # Crop vùng bubble
    crop = image[y1:y2, x1:x2]
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Threshold tìm vùng trắng
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Tìm contour lớn nhất (bubble)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(crop)
        cv2.drawContours(mask, [biggest], -1, (255, 255, 255), -1)
        crop[mask == 255] = 255
        image[y1:y2, x1:x2] = crop
    
    return image

# Load model và ảnh
model = YOLO("model/model.pt")
image = cv2.imread("examples/1.png")

# Detect bubbles
results = model(image, verbose=False)[0]
boxes = results.boxes.data.tolist()

# Xóa chữ trong từng bubble
for box in boxes:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    image = xoa_chu_trong_bubble(image, x1, y1, x2, y2)

# Lưu kết quả
cv2.imwrite("result_xoa_chu.png", image)
print("Da luu result_xoa_chu.png")