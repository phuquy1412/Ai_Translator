import cv2
import numpy as np
import requests
from PIL import Image
from ocr.chrome_lens_ocr import ChromeLensOCR
from ultralytics import YOLO

def translate(text, model="translategemma:4b"):
    prompt = f"Dịch đoạn text sau sang tiếng Việt, chỉ trả về bản dịch, không giải thích:\n{text}"
    response = requests.post("http://localhost:11434/api/generate",
                             json={
                                 "model":model, "prompt":prompt,"stream":False
                                 }
                        )
    
    return response.json()["response"]

def detect_and_ocr(image_path, model_path):
    #B1: Load model
    print("Loading model...")
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    
    #B2: Detect
    print("Detecting bubbles")
    results = model(image, verbose=False)[0]
    boxes = results.boxes.data.tolist()
    
    #B3: OCR
    ocr = ChromeLensOCR(ocr_language="ja")

    #B4: Crop bubble va OCR 
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, confidence, class_id = box
        # Crop bubble
        bubble_image = image[int(y1):int(y2), int(x1):int(x2)]
        #Chuyen qua pil de OCR
        bubble_pil = Image.fromarray(cv2.cvtColor(bubble_image, cv2.COLOR_BGR2RGB))
        #OCR
        text = ocr(bubble_pil)
       
        #Dich
        translated_text = translate(text)

        print(f"Bubble {i+1}:")
        print(f"Text OCR: {text}")
        print(f"Text dich: {translated_text}")


detect_and_ocr("examples/0.png", "model/model.pt")