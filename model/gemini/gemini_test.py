import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(csv_file, output_filename, title):
    # 1. Membaca data CSV
    df = pd.read_csv(csv_file)
    
    # 2. Mengambil label asli dan label prediksi
    y_true = df['true_label']
    y_pred = df['pred_label']
    
    # Mendefinisikan kelas agar urutannya konsisten
    classes = ['NORMAL', 'PNEUMONIA']
    
    # 3. Menghitung confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # 4. Membuat visualisasi (Heatmap)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14}) # Memperbesar font angka
    
    plt.title(title, fontsize=16)
    plt.ylabel('Label Asli (True Label)', fontsize=12)
    plt.xlabel('Prediksi Model (Predicted Label)', fontsize=12)
    plt.tight_layout()
    
    # 5. Menyimpan gambar
    save_path = f'{output_filename}.png'
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix disimpan di: {save_path}")
    plt.close() # Menutup plot agar tidak tumpang tindih

# ==========================================
# EKSEKUSI FUNGSI
# ==========================================

# Untuk file hasil_ori.csv
generate_confusion_matrix(
    csv_file='hasil_ori.csv', 
    output_filename='confusion_matrix_gemini_A', 
    title='Confusion Matrix - Gemini - A'
)

# Untuk file hasil_aug.csv
generate_confusion_matrix(
    csv_file='hasil_aug.csv', 
    output_filename='confusion_matrix_gemini_B', 
    title='Confusion Matrix - Gemini - B'
)