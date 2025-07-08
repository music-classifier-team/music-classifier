
# Music Genre Classification Using Audio Features

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from tracking import log_metrics, plot_f1_scores, save_conf_matrix

file_path = "data/gtzan/genres"
# 1. Feature extractor

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Shape: (n_mfcc,)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)  # Shape: (n_chroma,)

        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)  # Scalar

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)  # Scalar
        tempo = tempo.item() if np.ndim(tempo) > 0 else tempo

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid, axis=1)  # Shape: (1,)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)  # Shape: (1,)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth, axis=1)  # Shape: (1,)

        rmse = librosa.feature.rms(y=audio)
        rmse_mean = np.mean(rmse, axis=1)  # Shape: (1,)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)  # Shape: (n_bands,)
        
    
        return np.hstack((
            mfccs_mean,                
            chroma_mean,               
            np.array([zcr_mean]),      
            np.array([tempo]),         
            spectral_centroid_mean,    
            spectral_rolloff_mean,     
            spectral_bandwidth_mean,   
            rmse_mean,                 
            contrast_mean              
        ))
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# ======= Dataset Creation =======
def create_dataset(directory):
    genres = os.listdir(directory)
    print(f"\n✅ Found genres: {genres}\n")
    data, labels = [], []

    for genre in genres:
        genre_path = os.path.join(directory, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"🎵 Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if file.endswith(('.wav', '.au')):
                file_path = os.path.join(genre_path, file)
                features = extract_features(file_path)
                if features is not None:
                    data.append(features)
                    labels.append(genre)
    return np.array(data), np.array(labels)
# === MAIN EXECUTION ===


data_path = "data/gtzan/genres"

# 3. Load and verify dataset
X, y = create_dataset(data_path)

print(f"\n🧪 X shape: {np.shape(X)}")
print(f"🧪 y shape: {np.shape(y)}\n")

# Exit if dataset is empty
if len(X) == 0:
    print("🚫 ERROR: No audio features extracted! Exiting script.")
    exit()

# 4. Encode & scale
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.8, random_state=10)

def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log + Visualize
    precisions, recalls, f1s, supports = log_metrics(y_test, y_pred, label_encoder, model_name=model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confusion_path = f"results/confusion_matrix_{model_name}_{timestamp}.png"
    f1_path = f"results/f1_scores_{model_name}_{timestamp}.png"

    plot_f1_scores(label_encoder, f1s, save_path=f1_path)
    save_conf_matrix(y_test, y_pred, label_encoder, save_path=confusion_path)

    print(f"\n📝 {model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    return model

# 6. Train model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = SVC(kernel='linear', probability=True)
# model.fit(X_train, y_train)

models = {
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='mlogloss'),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1)
}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    trained_model = train_and_evaluate(model, name)


# 8. Save model and scalers
joblib.dump(trained_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\n✅ Model, scaler, and label encoder saved as .pkl files.")
