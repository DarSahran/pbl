import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 hand detection model (pre-trained or custom)
MODEL_PATH = "yolov8n-hand.pt"  # Replace with your trained hand detection model
model = YOLO(MODEL_PATH)

# Set up camera
CAMERA_INDEX = 0  # Adjust based on your Raspberry Pi setup
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for better processing
    frame_resized = cv2.resize(frame, (640, 640))

    # Perform hand detection
    results = model(frame_resized)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()  # Confidence score
        label = "Hand"

        # Draw bounding box
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, f"{label} ({conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Detection", frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
