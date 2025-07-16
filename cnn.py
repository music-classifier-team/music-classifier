import os
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from keras import layers, models, Sequential
from keras.utils import image_dataset_from_directory 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import RandomContrast, RandomZoom, RandomRotation
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tracking import log_metrics, plot_f1_scores, save_conf_matrix
from keras.callbacks import ReduceLROnPlateau
from keras.applications.vgg16 import VGG16, preprocess_input




AUDIO_DIR = 'augmented_audio'
OUTPUT_DIR = 'gtzan_spectrograms'
IMG_SIZE = (128, 128)
FILES_PER_GENRE = 200 


# create the image spectograms from the audio files



# def create_spectrogram(audio_path, output_path):
#     try:
#         y, sr = librosa.load(audio_path, duration=29.5)
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#         S_DB = librosa.power_to_db(S, ref=np.max)

#         fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
#         plt.axis('off')
#         librosa.display.specshow(S_DB, sr=sr, cmap='gray')
#         plt.tight_layout(pad=0)
#         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#         print(f"✅ Saved spectrogram: {output_path}")
#     except Exception as e:
#         print(e)


# for genre in os.listdir(AUDIO_DIR):
#     genre_path = os.path.join(AUDIO_DIR, genre)
#     output_genre_path = os.path.join(OUTPUT_DIR, genre)
#     os.makedirs(output_genre_path, exist_ok=True)

#      # Get all .au and .wav files in this genre folder, sort and pick top N
#     wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
#     selected_files = sorted(wav_files)[:FILES_PER_GENRE]

#     for filename in tqdm(selected_files, desc=f"Processing {genre}"):
#         audio_path = os.path.join(genre_path, filename)
#         output_image_path = os.path.join(output_genre_path, filename.replace('.wav', '.png'))
#         print(output_image_path)

#         if not os.path.exists(output_image_path):
#             create_spectrogram( audio_path, output_image_path)
    


# === TRANSFER LEARNING WITH VGG16 ON SPECTROGRAM IMAGES ===

# === CONFIGURATION ===
DATASET_DIR = 'gtzan_spectrograms'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60
SEED = 42

# === Convert grayscale to RGB ===
def to_rgb(image):
    return tf.image.grayscale_to_rgb(image)

# === LOAD DATASETS ===
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

# === CLASS INFO ===
class_names = train_ds.class_names
num_classes = len(class_names)

# === PREPROCESSING WITH VGG16 NORM ===
train_ds = train_ds.map(lambda x, y: (preprocess_input(to_rgb(x)), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(to_rgb(x)), y))

# === PERFORMANCE ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === BASE MODEL ===
base_model = VGG16(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Initially freeze all layers

# === UNFREEZE LAST BLOCK FOR FINE-TUNING ===
for layer in base_model.layers:
    if 'block5' in layer.name:
        layer.trainable = True

# === COMPILE FULL MODEL ===
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === CALLBACKS ===
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("transfer_vgg16_model.keras", save_best_only=True)
]

# === TRAINING ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


#  Evaluate 
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
#  Label encoder for reports 
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

#  Log and visualize 
precisions, recalls, f1s, supports = log_metrics(
    y_true, y_pred,
    label_encoder=label_encoder,
    model_name='CleanCNN',
    log_file='model_scores.csv'
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("cnn_results", exist_ok=True)
confusion_path = f"cnn_results/confusion_matrix_{timestamp}.png"
f1_path = f"cnn_results/f1_scores{timestamp}.png"
plot_f1_scores(label_encoder, f1s, save_path=f1_path)
save_conf_matrix(y_true, y_pred, label_encoder, save_path=confusion_path)


# === SAVE FINAL MODEL ===
model.save("transfer_vgg16_model.keras")
print("\u2705 Model saved as transfer_vgg16_model.keras")
