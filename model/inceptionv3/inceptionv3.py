import os
import tensorflow as tf
from keras import layers, models, applications, callbacks
import numpy as np

# ==========================================
# 0 & 1. SETUP LINGKUNGAN DAN INFORMASI DATASET
# ==========================================
tf.keras.utils.set_random_seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset/data_stratified'))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

IMG_SIZE = (224, 224) # InceptionV3 fleksibel dan mendukung input 224x224
BATCH_SIZE = 32 
EPOCHS_STAGE_1 = 30
EPOCHS_STAGE_2 = 50

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# PENANGANAN IMBALANCE: CLASS WEIGHTS DINAMIS
# ==========================================
print("Mengekstrak label dari train_dataset untuk menghitung class weights dinamis...")

train_labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)
train_labels = train_labels.flatten()

count_0 = np.sum(train_labels == 0)
count_1 = np.sum(train_labels == 1)
total_train_data = count_0 + count_1

weight_for_0 = total_train_data / (2.0 * count_0)
weight_for_1 = total_train_data / (2.0 * count_1)
class_weight = {0: weight_for_0, 1: weight_for_1}

# ==========================================
# 2. PRAPEMROSESAN & AUGMENTASI
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
    layers.RandomRotation(factor=0.05),
    layers.RandomBrightness(factor=0.1),
    layers.RandomContrast(factor=0.1),
    layers.GaussianNoise(stddev=0.1)
], name="data_augmentation")

# ==========================================
# 3. ARSITEKTUR MODEL (TRANSFER LEARNING)
# ==========================================
def build_model():
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    
    x = data_augmentation(inputs)
    
    # Prapemrosesan khusus InceptionV3 (mengubah skala piksel ke rentang -1 hingga 1)
    x = applications.inception_v3.preprocess_input(x)
    
    # Base Model: InceptionV3
    base_model = applications.InceptionV3(
        weights='imagenet', 
        include_top=False, 
        input_tensor=x
    )
    
    base_model.trainable = False
    
    # Custom Classifier Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

model, base_model = build_model()

# ==========================================
# 5. CALLBACKS
# ==========================================
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='inceptionv3_best.keras', # Disimpan dengan nama file model terkait
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# 4. STRATEGI PELATIHAN (TWO-STAGE FINE-TUNING)
# ==========================================

print("\n--- MULAI TAHAP 1: Feature Extraction ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

history_stage_1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE_1,
    class_weight=class_weight,
    callbacks=callbacks_list
)

print("\n--- MULAI TAHAP 2: Fine-Tuning ---")
base_model.trainable = True

# InceptionV3 menggunakan banyak layer BatchNormalization, pastikan tetap beku!
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

history_stage_2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE_2,
    class_weight=class_weight,
    callbacks=callbacks_list
)

print("\nPelatihan selesai! Bobot terbaik telah disimpan di 'inceptionv3_best.keras'.")