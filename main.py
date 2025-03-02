import cv2
import time
import multiprocessing
import speech_recognition as sr
import pyttsx3
import requests
from ultralytics import YOLO

# Constants
YOLO_MODEL = 'yolov5nu.pt'  # Lightweight model for embedded systems
CAMERA_INDEX = 1  # Adjust based on device
DETECTION_INTERVAL = 3  # Time in seconds to analyze surroundings
FRAME_SIZE = 416  # Resize frame for faster processing
WAKE_WORD = "hey assistant"

# Relevant object classes
RELEVANT_CLASSES = ['person', 'chair', 'table', 'bottle', 'cell phone', 'laptop', 'pen', 'book', 'keyboard', 'mouse', 'cup', 'backpack', 'tv', 'door', 'window', 'stairs']

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1)
engine.setProperty('voice', engine.getProperty('voices')[1].id)  # Set to female voice

def speak(text):
    """ Uses pyttsx3 to speak the given text aloud. """
    engine.say(text)
    engine.runAndWait()

def detect_objects(frame):
    """ Runs YOLO object detection and returns detected objects and results. """
    results = yolo_model(frame)[0]
    detected_objects = {}
    
    for box in results.boxes:
        label = results.names[int(box.cls)]
        if label in RELEVANT_CLASSES:
            detected_objects[label] = detected_objects.get(label, 0) + 1
    
    return detected_objects, results

def describe_object_online(object_name):
    """ Fetches a short description of the object from an online API. """
    try:
        response = requests.get(f"https://api.duckduckgo.com/?q={object_name}&format=json")
        data = response.json()
        description = data.get("AbstractText", "I couldn't find information about this object.")
        return description
    except Exception as e:
        return "I couldn't retrieve information at the moment."

def process_hand_object():
    """ Captures an image, detects the object in hand, and fetches a description. """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        detected_objects, _ = detect_objects(frame)
        
        if detected_objects:
            object_name = max(detected_objects, key=detected_objects.get)
            description = describe_object_online(object_name)
            response = f"You are holding a {object_name}. {description}"
        else:
            response = "I couldn't detect any object in your hand."
    else:
        response = "Failed to capture an image."
    
    print(response)
    speak(response)

def process_surroundings():
    """ Captures frames for 3 seconds and analyzes the surroundings. """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    start_time = time.time()
    detected_items = {}
    
    while time.time() - start_time < DETECTION_INTERVAL:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frame_objects, _ = detect_objects(frame)
            for obj, count in frame_objects.items():
                detected_items[obj] = max(detected_items.get(obj, 0), count)
    
    cap.release()
    
    if detected_items:
        response = "I see " + ", ".join([f"{count} {obj}" for obj, count in detected_items.items()]) + " around you."
    else:
        response = "I don't see any familiar objects around."
    
    print(response)
    speak(response)

def voice_command_listener():
    """ Listens for wake word once, then processes commands without requiring it again. """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("I'm in standby mode. Say 'Hey Assistant' to activate.")
        print("Listening for wake word...")
        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                
                if WAKE_WORD in text:
                    speak("How can I assist you?")
                    while True:
                        print("Listening for command...")
                        audio = recognizer.listen(source)
                        command = recognizer.recognize_google(audio).lower()
                        print(f"Command: {command}")
                        
                        if "in hand" in command:
                            process_hand_object()
                        elif "surrounding" in command:
                            process_surroundings()
                        elif "exit" in command or "stop" in command:
                            speak("Goodbye!")
                            return
                        else:
                            speak("I didn't understand that. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
                speak("I couldn't understand. Please repeat.")

def show_camera():
    """ Opens the camera feed and overlays detected objects in real-time. """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        detected_objects, results = detect_objects(frame)
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls)]
            conf = box.conf[0].item()
            if label in RELEVANT_CLASSES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    voice_thread = multiprocessing.Process(target=voice_command_listener)
    voice_thread.start()
    voice_thread.join()