import cv2
import numpy as np
import pyautogui
import time
# from tensorflow.keras.models import load_model
from keras.models import load_model
import keras
import mss
from PIL import Image, ImageDraw
import os

# Load the trained model
# keras.config.enable_unsafe_deserialization()
# model = load_model("my_model_latest.h5")
model = load_model("my_model_latest.h5", compile=False)
model.compile(optimizer='adam', loss='mse')

# Define the region of the screen to capture (you may need to adjust this for your game)
monitor = {'top': 32, 'left': 0, 'width': 800, 'height': 600}

# Preprocessing function (same as training)
def preprocess(img):
    img = cv2.resize(img, (200, 66))   # resize to model input size
    img = img / 255.0 - 0.5             # normalize
    return img

# Control the car based on steering angle
# def control_car(prediction):
#     if prediction > 0.1:
#         pyautogui.keyDown('d')
#         time.sleep(0.1)
#         pyautogui.keyUp('d')
        
#     elif prediction < -0.1:
#         pyautogui.keyDown('a')
#         time.sleep(0.1)
#         pyautogui.keyUp('a')
        
#     else:
#         pyautogui.keyDown('w')
#         time.sleep(0.1)
#         pyautogui.keyUp('w')
forward_counter = 0

def control_car(prediction):
    global forward_counter
    prediction = float(prediction)  # Make sure it's a float

    if prediction > 0.3:
        # Extreme Right Turn
        pyautogui.keyDown('d')
        time.sleep(0.2)
        pyautogui.keyUp('d')
        print("Extreme Right Turn")
    
    elif prediction > 0.15:
        # Moderate Right Turn
        pyautogui.keyDown('d')
        time.sleep(0.1)
        pyautogui.keyUp('d')
        print("Moderate Right Turn")
    
    elif prediction > 0.03:
        # Slight Right
        pyautogui.keyDown('d')
        time.sleep(0.05)
        pyautogui.keyUp('d')
        print("Slight Right Turn")
    
    elif prediction < -0.3:
        # Extreme Left Turn
        pyautogui.keyDown('a')
        time.sleep(0.2)
        pyautogui.keyUp('a')
        print("Extreme Left Turn")
    
    elif prediction < -0.15:
        # Moderate Left Turn
        pyautogui.keyDown('a')
        time.sleep(0.1)
        pyautogui.keyUp('a')
        print("Moderate Left Turn")
    
    elif prediction < -0.03:
        # Slight Left
        pyautogui.keyDown('a')
        time.sleep(0.05)
        pyautogui.keyUp('a')
        print("Slight Left Turn")
    
    # else:
    # # Drive Straight
    # pyautogui.keyDown('w')
    # time.sleep(0.06)
    # pyautogui.keyUp('w')
    # print("Going Straight")

    # Go straight but slower
    forward_counter += 1
    if forward_counter % 2 == 0:  # Only move forward every 2 frame
        pyautogui.keyDown('w')
        time.sleep(0.05)  # Shorter press time for slower acceleration
        pyautogui.keyUp('w')
        print("Moving Forward Slowly")

    # OPTIONAL: Reverse if stuck (prediction is weirdly large)
    if abs(prediction) > 1.5:
        print("Reversing!")
        pyautogui.keyDown('s')
        time.sleep(0.3)  # Reverse for 0.3 seconds
        pyautogui.keyUp('s')

# Main loop
with mss.mss() as sct:
    print("[INFO] Starting self-driving...")
    time.sleep(2)  # 2 sec delay to switch to game window

    while True:
        # cv2.imshow('Captured Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


        # Capture screen
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        
        # Preprocess
        frame_processed = preprocess(frame)
        frame_processed = np.expand_dims(frame_processed, axis=0)

        # Predict steering angle
        prediction = model.predict(frame_processed)[0][0]
        print(f"Predicted Steering Angle: {prediction:.2f}")

        # Control the car
        control_car(prediction)
        print(prediction)

        # Getting the image out in the folder
        # Convert the screenshot (numpy array) to a PIL Image
        output_image = Image.fromarray(screenshot)                              # Comment the whole section out to stop capturing output images
                                                                                #
        # Draw on the image                                                     #
        draw = ImageDraw.Draw(output_image)                                     #
        draw.text((28, 36), f"{prediction:.2f}", fill=(225, 0, 0))              #
                                                                                #
        # Make sure output folder exists                                        #
        output_folder = "captured_frames"                                       #
        os.makedirs(output_folder, exist_ok=True)                               #
                                                                                #
        timestamp = int(time.time() * 1000)  # milliseconds                     #
        image_path = os.path.join(output_folder, f"frame_{timestamp}.png")      #
        output_image.save(image_path)                                           #

        # Optional: small sleep to slow down loop (depends on game speed)
        time.sleep(0.05)

