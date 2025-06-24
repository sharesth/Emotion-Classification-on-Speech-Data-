import streamlit as st
import numpy as np
import librosa
import joblib
import os

# Title
st.title("🎙️ Speech Emotion Recognition App")

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("emotion_model.pkl")
    return model

model = load_model()

# Feature extraction
def extract_features(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# Upload audio
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        features = extract_features(uploaded_file)
        prediction = model.predict([features])[0]
        st.success(f"Predicted Emotion: **{prediction.upper()}** 🎯")
    except Exception as e:
        st.error(f"Error processing file: {e}")
