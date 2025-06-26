import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import pickle

# Load model and label encoder
model = load_model('sign_language_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables
sequence = ""
last_pred = ""
last_time = time.time()
no_hand_start = None

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    hands_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        hands_data.append(np.array(coords).flatten())
    
    if len(hands_data) == 1:
        hands_data.append(np.zeros(63))  # pad for second hand

    # sort by x position (left first)
    hands_data.sort(key=lambda x: x[0])
    return np.concatenate(hands_data)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks = extract_landmarks(results)

    now = time.time()
    pred_text = "No hands"
    color = (0, 0, 255)  # Default red

    if landmarks is not None:
        no_hand_start = None  # reset no-hand timer
        input_data = landmarks.reshape(1, -1)
        probs = model.predict(input_data)[0]
        confidence = np.max(probs)
        pred_class = le.inverse_transform([np.argmax(probs)])[0]

        # Determine prediction color based on confidence
        if confidence >= 0.9:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.7:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red

        pred_text = f"{pred_class}: {confidence:.2f}"

        # Add to sequence if strong prediction
        if confidence >= 0.99:
            if pred_class != last_pred or (now - last_time > 1.0):
                sequence += pred_class
                last_pred = pred_class
                last_time = now
    else:
        if no_hand_start is None:
            no_hand_start = now
        elif now - no_hand_start > 2:
            if not sequence.endswith(" "):
                sequence += " "
            last_pred = ""

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display prediction and constructed word
    cv2.putText(frame, pred_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # cv2.putText(frame, f"Sequence: {sequence.strip()}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Sign Language Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
