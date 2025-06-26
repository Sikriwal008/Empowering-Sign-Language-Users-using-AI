import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import time
import pickle
import requests
from collections import Counter

def send_to_gemini():
    url = "http://localhost:5000/correct_sentence"
    try:
        response = requests.post(url, json={"sentence": sentence})
        if response.ok:
            result = response.json()
            corrected = result['corrected_sentence']
            print("Corrected:", corrected)
            sentence_var.set(f"Corrected : {corrected}")
        else:
            print("Error:", response.json().get("error"))
    except Exception as e:
        print("Request failed:", e)

# Load trained model and label encoder
model = load_model("sign_language_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Mediapipe setup for hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# GUI setup
window = tk.Tk()
window.title("Sign Language To Text Conversion")
window.geometry("1200x700")
window.configure(bg='lightgrey')

video_label = tk.Label(window)
video_label.grid(row=1, column=0, padx=20, pady=20)

landmark_label = tk.Label(window)
landmark_label.grid(row=1, column=1, padx=20, pady=20)

char_label = tk.Label(window, text="Character :", font=("Courier", 24), bg='lightgrey')
char_label.grid(row=2, column=0, sticky='w', padx=20)

sentence_var = tk.StringVar()
sentence_var.set("Sentence :")
sentence_label = tk.Label(window, textvariable=sentence_var, font=("Courier", 24), bg='lightgrey')
sentence_label.grid(row=3, column=0, sticky='w', padx=20)

sentence = ""
last_prediction_time = 0
last_detection_time = time.time()
last_character = ""
recent_predictions = []  # List to store recent predictions

# Buttons
def clear_text():
    global sentence
    sentence = ""
    sentence_var.set("Sentence :")

def speak_text():
    import pyttsx3
    engine = pyttsx3.init()
    text = sentence_var.get().replace("Sentence :", "").strip()
    if text:
        engine.say(text)
        engine.runAndWait()

tk.Button(window, text="Clear", command=clear_text, font=("Courier", 16)).grid(row=4, column=1, sticky='e', padx=10, pady=10)
tk.Button(window, text="Speak", command=speak_text, font=("Courier", 16)).grid(row=4, column=1, sticky='w', padx=10, pady=10)
tk.Button(window, text="Correct Sentence", command=send_to_gemini, font=("Courier", 16)).grid(row=6, column=1, padx=10, pady=10)

cap = cv2.VideoCapture(0)

# Helper functions
def extract_landmarks(results):
    """
    Returns a (126,) flattened landmarks array for two hands sorted by wrist x-coordinate,
    or None if no landmarks.
    """
    if not results.multi_hand_landmarks:
        return None

    hands_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = [ [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark ]
        hands_data.append(np.array(coords).flatten())  # 63
    # Pad if only one hand detected
    if len(hands_data) == 1:
        hands_data.append(np.zeros(63, dtype=float))
    # Sort by wrist x (first element)
    hands_data.sort(key=lambda arr: arr[0])
    return np.concatenate(hands_data)  # 126

def is_flat_palm(landmarks_63):
    """
    landmarks_63: first 63 values for one hand
    Detect flat palm by checking finger tip vs mcp Y positions
    """
    # indices for tip and mcp joints
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    # reshape
    pts = np.array(landmarks_63).reshape(21, 3)
    tips = pts[tip_ids]
    mcps = pts[mcp_ids]
    wrist = pts[0]
    extended = all(tips[i][1] < mcps[i][1] for i in range(4))
    upright = all(abs(tips[i][2] - wrist[2]) < 0.1 for i in range(4))
    # near face heuristic: wrist Y < 0.3
    near_face = wrist[1] < 0.3
    return extended and upright and near_face

def update():
    global sentence, last_prediction_time, last_detection_time, last_character, recent_predictions

    ret, frame = cap.read()
    if not ret:
        window.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Display video
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img.resize((400, 300)))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Prepare white canvas for landmarks
    white_img = np.ones((300, 300, 3), dtype=np.uint8) * 255

    # Extract and handle landmarks
    data = extract_landmarks(results)
    delete_gesture = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                white_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
        # detect flat palm on first hand
        first_hand = [ [lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark ]
        if is_flat_palm(np.array(first_hand).flatten()):
            delete_gesture = True

    # Prediction logic
    now = time.time()
    # Remove predictions older than 2 seconds
    recent_predictions = [p for p in recent_predictions if p[1] >= now - 2]

    if delete_gesture and sentence:
        sentence = sentence[:-1]
        sentence_var.set(f"Sentence : {sentence}")
        last_prediction_time = now
    elif data is not None:
        # Predict
        X = data.reshape(1, -1)
        preds = model.predict(X, verbose=0)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]
        char = label_encoder.inverse_transform([idx])[0]

        # Add to recent predictions if confidence is sufficient
        if confidence >= 0.7:
            recent_predictions.append((char, now))

        # Display current prediction with color coding
        if confidence >= 0.995:
            color = "green"
        elif confidence >= 0.7:
            color = "orange"
        else:
            color = "red"
        char_label.config(text=f"Character : {char} ({confidence*100:.1f}%)", fg=color)
        last_detection_time = now

        # Add to sentence based on most frequent prediction
        if now - last_prediction_time > 1 and recent_predictions:
            counter = Counter([p[0] for p in recent_predictions])
            if counter:  # Ensure counter is not empty
                most_common_char, count = counter.most_common(1)[0]
                if count >= 5:  # Threshold: at least 5 predictions
                    sentence += most_common_char
                    sentence_var.set(f"Sentence : {sentence}")
                    last_prediction_time = now
                    recent_predictions = []  # Clear to prevent repetition
    else:
        # No hands: add space after delay
        if now - last_detection_time > 4.0 and sentence and not sentence.endswith(' '):
            sentence += ' '
            sentence_var.set(f"Sentence : {sentence}")
            last_detection_time = now

    # Display landmarks canvas
    lm_img = Image.fromarray(cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB))
    lm_imgtk = ImageTk.PhotoImage(image=lm_img)
    landmark_label.imgtk = lm_imgtk
    landmark_label.configure(image=lm_imgtk)

    window.after(10, update)

update()
window.mainloop()
cap.release()