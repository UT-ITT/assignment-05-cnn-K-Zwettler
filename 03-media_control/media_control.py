import cv2
from tensorflow.keras.models import load_model
import time
import numpy as np
from pynput.keyboard import Key, Controller
import sys

# image size
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)

# load the trained model
model = load_model("gesture_recognition.keras")
# for pynput
keyboard = Controller()

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])
# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

last_action = None
last_prediction_time = 0

# set the labels
labels = {
    0: "volume_up",
    1: "pause",
    2: "skip",
    3: "no_gesture"
}

# get the cropped image of the hand
def get_hand_crop(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow("mask", mask)

    # get the contour of the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: 
        return None
    
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        return None

    x, y, w, h = cv2.boundingRect(c)

    hand_crop = frame[y:y+h, x:x+w]

    return hand_crop

# predict the action by giving the image of the cropped hand to the model
def predict(cropped):
    resized = cv2.resize(cropped, SIZE)
    reshaped = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    reshaped = reshaped.astype(np.float32) / 255.0
    prediction = model.predict(reshaped)
    class_id = np.argmax(prediction)
    return labels[class_id]

# function to execute the predicted action
def execute(action):
    if action == "volume_up":
        keyboard.press(Key.media_volume_up)
        keyboard.release(Key.media_volume_up)
    elif action == "pause":
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    elif action == "skip":
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)

# main loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("no frame received")
        break
    # show the camera frame
    cv2.imshow("camera", frame)

    current_time = time.time()

    # predict after every second
    if current_time - last_prediction_time > 1:
        cropped = get_hand_crop(frame)  

        if cropped is not None:
            action = predict(cropped)
            print(action)

            if action != "no_gesture" and action != last_action:
                execute(action)
                last_action = action
        
        last_prediction_time = current_time


    # Wait for a key press and check if it's the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()