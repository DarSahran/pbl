import cv2
import speech_recognition as sr
import torch
import pyttsx3
import time

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Load YOLOv5 Model (small model for faster inference)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Load YOLOv5 small model

# Initialize Speech Recognition
recognizer = sr.Recognizer()

def voice_to_text():
    """Capture voice input and convert to text"""
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text.lower()
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Could not request results")
            return None

def text_to_speech(text):
    """Convert text to speech and output it to the user"""
    engine.say(text)
    engine.runAndWait()

def analyze_frame():
    """Capture a frame and analyze it using YOLO"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        results.show()  # Display the detected object
        objects_detected = results.names  # Get the object names detected
        cap.release()
        cv2.destroyAllWindows()
        return objects_detected
    return []

def analyze_surroundings():
    """Analyze surroundings by capturing 5 frames and ensuring object consistency"""
    cap = cv2.VideoCapture(0)
    detected_objects = []

    for i in range(5):  # Capture 5 frames in 1 second
        ret, frame = cap.read()
        if ret:
            results = model(frame)
            detected_objects.append(results.names)  # Add detected object names

    cap.release()
    cv2.destroyAllWindows()

    # Check for consistency in detected objects (simplified logic)
    detected_objects = [item for sublist in detected_objects for item in sublist]  # Flatten list
    unique_objects = list(set(detected_objects))  # Get unique objects

    if len(unique_objects) > 1:
        return f"The surroundings have the following objects: {', '.join(unique_objects)}"
    else:
        return "The surroundings appear to be clear with no significant objects detected."

def main():
    while True:
        # Get voice input from the user
        command = voice_to_text()

        if command:
            if "what's this" in command:
                # Scenario 1: What's this? - Detect object in front
                print("Detecting object in front...")
                detected_objects = analyze_frame()
                if detected_objects:
                    text_to_speech(f"The detected object is: {', '.join(detected_objects)}")
                else:
                    text_to_speech("Sorry, I could not detect any object.")
            
            elif "describe my surroundings" in command or "what's in my surroundings" in command:
                # Scenario 2: Describe surroundings - Analyze 5 frames
                print("Analyzing surroundings...")
                surroundings_description = analyze_surroundings()
                text_to_speech(surroundings_description)
            
            else:
                text_to_speech("Sorry, I did not understand that command.")
        
        time.sleep(1)  # Add a small delay before next command detection

if __name__ == "__main__":
    main()
