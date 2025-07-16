# 🎵 Music Genre Classifier (GTZAN)

This project is a machine learning pipeline for classifying music genres using the [GTZAN dataset](http://marsyas.info/downloads/datasets.html). It supports both **feature-based classical ML models** and a **deep learning CNN approach** using spectrogram images and transfer learning.

---

## Features

- **Classical ML**: Extracts audio features (MFCCs, chroma, spectral, etc.) and trains models like RandomForest, SVM, XGBoost, LightGBM, and MLP.
- **Deep Learning**: Converts audio to mel spectrogram images and uses a VGG16-based CNN for classification.
- **Data Augmentation**: Augments audio files to improve model robustness.
- **Tracking & Visualization**: Logs metrics, plots F1-scores, and confusion matrices.
- **Streamlit App**: Upload an audio file and get instant genre predictions with confidence scores.

---

## Project Structure

```
music-classifier/
│
├── augment_audio.py           # Audio augmentation script
├── cnn.py                     # CNN training on spectrogram images
├── music_genre_classifier.py  # Feature-based ML pipeline
├── music_genre_app.py         # Streamlit web app for genre prediction
├── tracking.py                # Metrics logging and visualization
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files/folders to ignore in git
├── model_scores.csv           # (gitignored) Model evaluation logs
├── data/                      # (gitignored) Raw GTZAN dataset
├── augmented_audio/           # (gitignored) Augmented audio files
├── gtzan_spectrograms/        # (gitignored) Spectrogram images
├── cnn_results/               # (gitignored) CNN result images
└── results/                   # (gitignored) Other result images
```

---

## Setup

1. **Clone the repository**  
   ```sh
   git clone https://github.com/<your-org>/music-classifier.git
   cd music-classifier
   ```

2. **Install dependencies**  
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Download the GTZAN dataset**  
   - Place the dataset in `data/gtzan/genres/` (see `.gitignore` for ignored data).

---

## Usage

### 1. **Classical ML Pipeline**

Extract features and train models:
```sh
python music_genre_classifier.py
```
- Logs metrics to `model_scores.csv`
- Saves confusion matrices and F1-score plots

### 2. **Audio Augmentation**

Augment the dataset for better generalization:
```sh
python augment_audio.py
```
- Outputs to `augmented_audio/`

### 3. **CNN Training**

Train a VGG16-based CNN on spectrogram images:
```sh
python cnn.py
```
- Outputs model to `transfer_vgg16_model.keras`
- Saves results in `cnn_results/`

### 4. **Web App**

Run the Streamlit app for interactive predictions:
```sh
streamlit run music_genre_app.py
```
- Upload a `.wav`, `.mp3`, `.au`, or `.ogg` file and get genre predictions.

---

## Results

- **Model metrics** are logged in `model_scores.csv`
- **Confusion matrices** and **F1-score plots** are saved in `cnn_results/` and `results/`

---

## Notes

- All data, models, and results folders are gitignored by default.
- You can customize the augmentation and model parameters in the respective scripts.
- The project is modular: you can use either the classical ML or CNN approach independently.

---

## License

This project is for educational and research purposes. Please respect the GTZAN dataset license.

---

## Credits

- [GTZAN dataset](http://marsyas.info/downloads/datasets.html)
- [Librosa](https://librosa.org/), [scikit-learn](https://scikit-learn.org/), [Keras](https://keras.io/), [Streamlit](https://streamlit.io/), and others.
