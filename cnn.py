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




AUDIO_DIR = 'data/gtzan/genres'
OUTPUT_DIR = 'gtzan_spectrograms'
IMG_SIZE = (128, 128)
FILES_PER_GENRE = 100 

# === Data Augmentation ===
# data_augmentation = Sequential([
#     RandomZoom(0.1),
#     RandomContrast(0.1),
#     RandomRotation(0.05),
# ])

# create the image spectograms from the audio files



# def create_spectrogram(audio_path, output_path):
    # try:
    #     y, sr = librosa.load(audio_path, duration=29.5)
    #     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #     S_DB = librosa.power_to_db(S, ref=np.max)

    #     fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    #     plt.axis('off')
    #     librosa.display.specshow(S_DB, sr=sr, cmap='gray')
    #     plt.tight_layout(pad=0)
    #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)
    #     print(f"✅ Saved spectrogram: {output_path}")
    # except Exception as e:
    #     print(e)


# for genre in os.listdir(AUDIO_DIR):
    # genre_path = os.path.join(AUDIO_DIR, genre)
    # output_genre_path = os.path.join(OUTPUT_DIR, genre)
    # os.makedirs(output_genre_path, exist_ok=True)

    #  # Get all .wav files in this genre folder, sort and pick top N
    # wav_files = [f for f in os.listdir(genre_path) if f.endswith('.au')]
    # selected_files = sorted(wav_files)[:FILES_PER_GENRE]

    # for filename in tqdm(selected_files, desc=f"Processing {genre}"):
    #     audio_path = os.path.join(genre_path, filename)
    #     output_image_path = os.path.join(output_genre_path, filename.replace('.au', '.png'))
    #     print(output_image_path)

    #     if not os.path.exists(output_image_path):
    #         create_spectrogram( audio_path, output_image_path)
    


# === Configuration ===
DATASET_DIR = OUTPUT_DIR
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 80
SEED = 42

# === Load dataset (grayscale) ===
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

# === Get class names ===
class_names = train_ds.class_names
num_classes = len(class_names)

# === Normalize pixels ===
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# === Optimize performance ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === Compute Class Weights ===
y_all = []
for _, labels in train_ds.unbatch():
    y_all.append(labels.numpy())

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_all),
    y=y_all
)
class_weights = dict(enumerate(class_weights))

# Build CNN 

model = models.Sequential([
    layers.InputLayer(shape=(128, 128, 1)),

    # data_augmentation,  # Apply augmentation

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),


    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("best_cnn_model.h5", save_best_only=True)
]


# === Train the model ===
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS
# )

# Flatten all training labels


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
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

#  Save model 
# model.save("clean_cnn_model.h5")
model.save("clean_cnn_model.keras")
print("✅ Clean CNN model saved as clean_cnn_model.keras")