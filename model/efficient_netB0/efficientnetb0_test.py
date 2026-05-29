import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP PATH DAN PARAMETER
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset/data_aug'))
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# 2. MUAT DATASET TEST
# ==========================================
print(f"Mencari data test di: {TEST_DIR}")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False # Wajib False agar urutan label tidak berantakan
)

# ==========================================
# 3. MUAT MODEL TERBAIK
# ==========================================
print("\nMemuat model efficientnetb0_best.keras...")
model = tf.keras.models.load_model('efficientnetb0_best.keras')

# ==========================================
# 4. EVALUASI SEDERHANA
# ==========================================
print("\nMengevaluasi model pada data test...")
results = model.evaluate(test_dataset, verbose=1)

print("\nHasil Evaluasi Keras (EfficientNetB0):")
for name, value in zip(model.metrics_names, results):
    print(f"- {name}: {value:.4f}")

# ==========================================
# 5. PREDIKSI MENDALAM & MATRIKS EVALUASI
# ==========================================
print("\nMemulai prediksi satu per satu untuk evaluasi detail...")

y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0).flatten()
y_pred_prob = model.predict(test_dataset)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

classes = ["NORMAL (0)", "PNEUMONIA (1)"]

# ==========================================
# 6. REPORT & VISUALISASI
# ==========================================
print("\n==========================================")
print("     CLASSIFICATION REPORT EFFICIENTNET   ")
print("==========================================")
print(classification_report(y_true, y_pred, target_names=classes))

print("\nMembuat visualisasi Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Menggunakan warna hijau untuk membedakan dengan model lain
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 14})
plt.title('Confusion Matrix - Uji Coba EfficientNetB0 - B', fontsize=16)
plt.ylabel('Label Asli (True Label)', fontsize=12)
plt.xlabel('Prediksi Model (Predicted Label)', fontsize=12)
plt.tight_layout()

save_path = 'confusion_matrix_efficientnetb0_B.png'
plt.savefig(save_path, dpi=300)
print(f"Selesai! Gambar Confusion Matrix disimpan di: {os.path.abspath(save_path)}")