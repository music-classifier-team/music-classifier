
import streamlit as st
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(file):
    try:
        audio, sr = librosa.load(file, duration=30)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return np.hstack((mfccs_mean, chroma_mean, zcr_mean, tempo)).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

st.title("🎵 Music Genre Classifier")

uploaded_file = st.file_uploader("Upload a .wav music file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Extracting features and predicting..."):
        features = extract_features(uploaded_file)
        if features is not None:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            predicted_genre = label_encoder.inverse_transform(prediction)[0]
            st.success(f"Predicted Genre: **{predicted_genre}**")
