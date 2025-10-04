from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def detect_objects(image_path: str):
    results = model(image_path)
    return results

if __name__ == "__main__":
    img_path = "bus.jpg"
    detection_results = detect_objects(img_path)
    result = detection_results[0]
    boxes = result.boxes
    names = result.names

    person_count = 0
    car_count = 0
    tank_count = 0

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        confidence = float(box.conf[0])
        print(f"Class: {class_name}, Confidence: {confidence:.2f}")