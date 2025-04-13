from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load YOLO model
MODEL_PATH = "plastic_detection_best_model.pt"
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

def enhance_image_clahe(image, clip_limit=2.0, tile_grid_size=(4, 4)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image array."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image data")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced_img

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # Enhance image contrast
    img = enhance_image_clahe(img, clip_limit=2.0, tile_grid_size=(4, 4))

    # Save image with unique name
    image_counter = get_next_image_count()
    image_filename = f"plastic{image_counter}.jpg"
    image_path = os.path.join(IMAGE_SAVE_FOLDER, image_filename)
    cv2.imwrite(image_path, img)
    
    # Perform prediction
    results = model.predict(img, conf=0.5, iou=0.5)
    detections = []

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0]) if box.cls is not None else 0
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                bbox = box.xyxy[0].tolist() if box.xyxy is not None else []

                if confidence >= 0.5:
                    detections.append({
                        "class": model.names[class_id],  # ✅ Corrected class name retrieval
                        "confidence": round(confidence, 2),
                        "bbox": bbox
                    })
    
    return jsonify({"detections": detections, "saved_image": image_filename})

if __name__ == '__main__':
    app.run(debug=True)