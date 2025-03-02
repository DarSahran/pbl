import time
import torch
from ultralytics import YOLO

# Load models
models = {
    "YOLOv8n": YOLO("yolov8n.pt"),
    "YOLOv5n": YOLO("yolov5nu.pt"),
    "YOLOv11n": YOLO("yolo11n.pt")  # Placeholder for YOLOv11n
}

# Dummy input image (adjust the path as needed)
image_path = "test.jpg"

# Measure inference time
for model_name, model in models.items():
    start_time = time.time()
    results = model(image_path)  # Perform inference
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"{model_name} Inference Time: {inference_time:.4f} seconds")
