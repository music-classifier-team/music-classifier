import os
import librosa
import soundfile as sf
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# === Paths ===
INPUT_DIR = 'data/gtzan/genres'  # Your original .au files
OUTPUT_DIR = 'augmented_audio'   # Where to save augmented .wav files
AUGS_PER_FILE = 2                # 2 augmentations per original

# === Define augmentations ===
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
])

# === Create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)

for genre in os.listdir(INPUT_DIR):
    genre_path = os.path.join(INPUT_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    out_genre_path = os.path.join(OUTPUT_DIR, genre)
    os.makedirs(out_genre_path, exist_ok=True)

    audio_files = [f for f in os.listdir(genre_path) if f.endswith('.au')]

    for file in tqdm(audio_files, desc=f"Augmenting {genre}"):
        in_path = os.path.join(genre_path, file)
        try:
            y, sr = librosa.load(in_path, sr=None)
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue

        for i in range(1, AUGS_PER_FILE + 1):
            y_aug = augment(samples=y, sample_rate=sr)
            out_filename = file.replace('.au', f'_aug{i}.wav')
            out_path = os.path.join(out_genre_path, out_filename)
            sf.write(out_path, y_aug, sr)


