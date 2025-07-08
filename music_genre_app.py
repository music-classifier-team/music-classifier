import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from pydub import AudioSegment
from PIL import Image
import os
import tempfile
import io

# Constants
IMG_SIZE = (128, 128)
class_names = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# Load Model (Cached)
@st.cache_resource
def load_trained_model():
    return load_model("clean_cnn_model.keras")

model = load_trained_model()

# Audio to Spectrogram
def audio_to_spectrogram_image(audio_path):
    y, sr = librosa.load(audio_path, duration=29.5)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Save spectrogram as image
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='gray')
    plt.tight_layout(pad=0)

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_img.name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Load image and prepare for prediction
    img = Image.open(temp_img.name).convert("L")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array, temp_img.name

# Plot All Genre Confidences
def plot_confidences(prediction):
    confidences = prediction.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(class_names, confidences, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Genre Confidence Scores")
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + 0.3, f"{bar.get_width():.2f}")
    st.pyplot(fig)



def convert_audio_to_wav(uploaded_file):
    """Convert uploaded audio file (e.g. .au, .mp3) to WAV for playback."""
    input_suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
    temp_input.write(uploaded_file.read())
    temp_input.flush()

    audio = AudioSegment.from_file(temp_input.name)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_wav.name, format="wav")

    return temp_wav.name

# Streamlit App
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier (GTZAN CNN)")
st.markdown("Upload a 30-second audio clip to classify its genre using a CNN model trained from scratch on the GTZAN dataset.")

uploaded_file = st.file_uploader("🎧 Upload Audio File", type=["wav", "au", "mp3", "ogg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    st.audio(file_bytes, format="audio/wav" if uploaded_file.name.endswith(".wav") else "audio/mp3")
    # playable_audio_path = convert_audio_to_wav(uploaded_file)
    # with open(playable_audio_path, 'rb') as f:
    #     st.audio(f.read(), format="audio/wav")


    # Save to temporary file for Librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    with st.spinner("🔍 Analyzing audio..."):
        try:
            img_array, spectrogram_path = audio_to_spectrogram_image(tmp_path)
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[0][predicted_index]

            # Display prediction
            st.subheader("📌 Prediction Result")
            st.success(f"🎼 **Predicted Genre:** {predicted_class.upper()}")
            st.info(f"🧠 **Model Confidence:** {confidence:.2%}")

            # Top 3 genres
            top_3 = np.argsort(prediction[0])[::-1][:3]
            st.markdown("### 🏆 Top 3 Predictions")
            for idx in top_3:
                st.write(f"- {class_names[idx]}: {prediction[0][idx]:.2%}")

            # Optional: Show spectrogram
            if st.checkbox("🖼️ Show Spectrogram"):
                st.image(spectrogram_path, caption="Mel Spectrogram", use_column_width=True)

            # Plot confidence bar chart
            st.markdown("### 📊 Prediction Confidence")
            plot_confidences(prediction)

            # Download spectrogram
            with open(spectrogram_path, "rb") as f:
                btn = st.download_button(
                    label="📥 Download Spectrogram",
                    data=f,
                    file_name="spectrogram.png",
                    mime="image/png"
                )

            # Clean up
            os.remove(tmp_path)
            os.remove(spectrogram_path)

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")


