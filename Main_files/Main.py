import cv2
import torch
from ultralytics import YOLO
import gtts
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize camera
cap = cv2.VideoCapture(1)

def detect_hand_objects():
    ret, frame = cap.read()
    if not ret:
        return "Error capturing image"
    
    results = model(frame)
    
    detected_objects = []
    for obj in results[0].names.values():
        detected_objects.append(obj)

    if "hand" in detected_objects:
        detected_objects.remove("hand")  # Remove hand itself
        if detected_objects:
            description = f"You are holding a {detected_objects[0]}"
        else:
            description = "Your hand is empty."
    else:
        description = "No hand detected."

    tts = gtts.gTTS(description)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Plays the audio

    return description

detect_hand_objects()
