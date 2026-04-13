import cv2 #opencv xu ly anh
from ultralytics import YOLO #Train model detect object

def detect_bubbles_simple(image_path, model_path):
    #B1: Load model
    print("Loading model...")
    model = YOLO(model_path) #Tao object model 
    print("Load xong model")
    #B2: Load image
    print("Loading image...")
    image = cv2.imread(image_path) # Doc anh  
    print("Load xong image")
    #B3: Detect
    print("Detecting bubbles")
    results = model(image, verbose=False)[0]
    #model(image) → chạy inference
    #verbose=False → không in log chi tiết
    #[0] → lấy kết quả của ảnh đầu tiên
    #results chứa
    #results.boxes → bounding boxes
    #results.masks → segmentation (nếu model seg)
    #results.names → tên class
    #results.probs → xác suất

    #B4: Lay ket qua
    boxes = results.boxes.data.tolist()
    
    #B5: In ra ket qua
    for i, box in enumerate(boxes):
        x1,y1,x2,y2, confidence, class_id = box
        print(f"Bubble{i+1} : Toa do ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}) | do tin cay : {confidence:.2f}")
   
    
    # B6: Ve bubble len anh
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite("result.png", image)
    print(f"\nTong cong: {len(boxes)} bubbles")
    print("Da luu result.png")
    return boxes

detect_bubbles_simple("examples/1.webp", "model/model.pt")