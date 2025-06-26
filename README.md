# Empowering Sign-Language Users Using AI 🤟💬

A real-time AI-powered system that interprets hand gestures into text using MediaPipe, OpenCV, and a trained deep learning model. It includes a user-friendly GUI for live prediction, gesture-based sentence construction, and integrates Google Gemini (via a Flask API) for grammar correction and speech output.

---
Dataset:-https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred

---

## 🔍 Features

- 🎥 **Real-Time Hand Detection:** Uses MediaPipe to extract 21 landmarks per hand (up to 2 hands), creating 126-dimensional feature vectors.
- 🔠 **Gesture Classification:** Trained a lightweight Keras-based neural network to classify 36 alphanumeric gestures (A-Z, 0-9).
- 🧠 **Noise-Resilient Prediction:** Uses a confidence threshold and sliding window voting to stabilize predictions.
- ✋ **Smart Controls:**  
  - Auto-inserts space when hands are absent for 4+ seconds.  
  - "Flat palm" gesture deletes the last character.  
- 🖥️ **Desktop GUI:** Built with Tkinter, includes live video feed, hand landmark display, and sentence builder.
- 🧠 **Grammar Correction:** Sends raw sentences to a local Flask API that queries Google Gemini for polished output.
- 🔊 **Speech Output:** Reads the corrected sentence aloud using pyttsx3 (offline TTS).

---

## 🧰 Tech Stack

- **Programming Language:** Python  
- **ML/DL:** TensorFlow, Keras  
- **Computer Vision:** OpenCV, MediaPipe  
- **GUI:** Tkinter + PIL  
- **Web API:** Flask  
- **AI Integration:** Google Gemini (via `google.generativeai`)  
- **Others:** NumPy, Requests, Pyttsx3, Pickle


