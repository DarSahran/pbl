import multiprocessing
import pyttsx3
import cv2
import queue
import speech_recognition as sr
from ultralytics import YOLO

# Constants
YOLO_MODEL = 'yolov5nu.pt'  # Lightweight YOLO model
CAMERA_INDEX = 1  # Set to 1 since your camera is at index 1
FRAME_SIZE = 416  # Resize frame for detection
RELEVANT_CLASSES = ['person', 'chair', 'table', 'bottle', 'laptop', 'mouse', 'pen', 'cell phone', 'car', 'bus', 'bike', 'door', 'bag']

# Global variables
frame_queue = multiprocessing.Queue(maxsize=1)

yolo_model = YOLO(YOLO_MODEL)  # Load YOLO model

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

def speak(text):
    """ Use pyttsx3 to speak the given text aloud in a conversational way. """
    engine.say(text)
    engine.runAndWait()

def camera_capture():
    """ Continuously captures frames and updates the queue. """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put(resized_frame)
    cap.release()

def process_surroundings():
    """ Processes the surroundings and provides an accurate spoken response. """
    if frame_queue.empty():
        speak("Koi frame nahi mila, thodi der baad try karo.")
        return
    
    frame = frame_queue.get()
    results = yolo_model(frame)[0]
    detections = []
    
    for box in results.boxes:
        class_id = int(box.cls)
        label = results.names[class_id]
        if label in RELEVANT_CLASSES:
            detections.append(label)
    
    if detections:
        response = f"Maine dekha hai: {', '.join(detections)}"
    else:
        response = "Kuch khas nahi dikh raha."
    
    print(response)
    speak(response)

def voice_command_listener():
    """ Listens for voice commands and responds interactively. """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            print("Listening for a command...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")

                # Hinglish Command Processing
                if "yah kya hai" in text or "mere aaspaas kya hai" in text:
                    process_surroundings()
                elif "camera dikhao" in text or "camera on karo" in text:
                    speak("Camera feed on kiya ja raha hai.")
                elif "band ho jao" in text or "system stop karo" in text:
                    speak("Goodbye!")
                    break
                else:
                    speak("Mujhe yeh samajh nahi aaya.")
            except Exception as e:
                print(f"Error: {e}")
                speak("Maaf karna, samajh nahi aaya.")

if __name__ == '__main__':
    camera_thread = multiprocessing.Process(target=camera_capture)
    voice_thread = multiprocessing.Process(target=voice_command_listener)
    
    camera_thread.start()
    voice_thread.start()
    
    camera_thread.join()
    voice_thread.join()
