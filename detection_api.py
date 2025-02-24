from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
MODEL_PATH = "best_model.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Ensure images folder exists
IMAGE_SAVE_FOLDER = "images"
os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)

# Get the current image count based on existing files
def get_next_image_count():
    existing_files = [f for f in os.listdir(IMAGE_SAVE_FOLDER) if f.startswith("plastic") and f.endswith(".jpg")]
    numbers = [int(f.replace("plastic", "").replace(".jpg", "")) for f in existing_files if f.replace("plastic", "").replace(".jpg", "").isdigit()]
    return max(numbers, default=0) + 1

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Save image with unique name
    image_counter = get_next_image_count()
    image_filename = f"plastic{image_counter}.jpg"
    image_path = os.path.join(IMAGE_SAVE_FOLDER, image_filename)
    cv2.imwrite(image_path, img)
    
    # Perform prediction
    results = model(img, conf=0.5, iou=0.5)
    detections = []
    
    for result in results:
        if hasattr(result, "boxes") and result.boxes:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                if confidence >= 0.5:
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 2),
                        "bbox": bbox
                    })
    
    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True)
