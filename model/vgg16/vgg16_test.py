import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP PATH DAN PARAMETER
# ==========================================
# Arahkan ke folder test
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset/data_aug'))
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# 2. MUAT DATASET TEST
# ==========================================
print(f"Mencari data test di: {TEST_DIR}")
# PENTING: shuffle=False wajib digunakan saat testing agar prediksi 
# dan label asli tidak saling tertukar posisinya!
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False 
)

# ==========================================
# 3. MUAT MODEL TERBAIK
# ==========================================
print("\nMemuat model vgg16_best.keras...")
# Pastikan file vgg16_best.keras berada di direktori yang sama dengan skrip ini
model = tf.keras.models.load_model('vgg16_best.keras')

# ==========================================
# 4. EVALUASI SEDERHANA
# ==========================================
print("\nMengevaluasi model pada data test...")
# Mengeksekusi evaluate() bawaan Keras
results = model.evaluate(test_dataset, verbose=1)

# Mencetak hasil (Indeks 0 biasanya loss, sisanya adalah metrics)
print("\nHasil Evaluasi Keras:")
for name, value in zip(model.metrics_names, results):
    print(f"- {name}: {value:.4f}")

# ==========================================
# 5. PREDIKSI MENDALAM & MATRIKS EVALUASI
# ==========================================
print("\nMemulai prediksi satu per satu untuk evaluasi detail...")

# 5.1 Mengambil label asli dari test_dataset
y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0).flatten()

# 5.2 Melakukan prediksi (menghasilkan angka probabilitas dari sigmoid)
y_pred_prob = model.predict(test_dataset)

# 5.3 Mengubah probabilitas menjadi kelas biner (Threshold 0.5)
# Jika > 0.5 maka kelas 1 (Pneumonia), jika <= 0.5 maka kelas 0 (Normal)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

classes = ["NORMAL (0)", "PNEUMONIA (1)"]

# ==========================================
# 6. REPORT & VISUALISASI
# ==========================================
print("\n==========================================")
print("           CLASSIFICATION REPORT          ")
print("==========================================")
print(classification_report(y_true, y_pred, target_names=classes))

print("\nMembuat visualisasi Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 14}) # Memperbesar font angka
plt.title('Confusion Matrix - Uji Coba VGG16 - B', fontsize=16)
plt.ylabel('Label Asli (True Label)', fontsize=12)
plt.xlabel('Prediksi Model (Predicted Label)', fontsize=12)
plt.tight_layout()

# Menyimpan gambar dengan nama spesifik untuk VGG16
save_path = 'confusion_matrix_vgg16_B.png'
plt.savefig(save_path, dpi=300)
print(f"Selesai! Gambar Confusion Matrix disimpan di: {os.path.abspath(save_path)}")