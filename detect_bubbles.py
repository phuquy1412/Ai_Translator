import cv2
from ultralytics import YOLO

_model_cache = {}

def detect_bubbles(image, model_path="model/comic.pt"):
    """
    Detect speech bubbles trong ảnh manga.
    
    Args:
        image: numpy array (BGR) hoặc đường dẫn ảnh
        model_path: đường dẫn file model YOLO
        
    Returns:
        list: danh sách [x1, y1, x2, y2, confidence, class_id]
    """
    # Cache model tránh load lại mỗi lần
    if model_path not in _model_cache:
        print(f"Loading model {model_path}...")
        _model_cache[model_path] = YOLO(model_path)

    model = _model_cache[model_path]

    # Load ảnh nếu là đường dẫn
    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        return []

    # Detect
    results = model(image, verbose=False)[0]
    boxes = results.boxes.data.tolist()

    print(f"Tim thay {len(boxes)} bubbles")
    return boxes