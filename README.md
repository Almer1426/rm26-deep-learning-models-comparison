# 🫁 CNN Pneumonia Detection

Sistem deteksi pneumonia dari citra **chest X-ray** menggunakan **Convolutional Neural Network (CNN)** dengan pendekatan **Transfer Learning** dan **Two-Stage Fine-Tuning**. Proyek ini membandingkan performa **5 arsitektur model** yang berbeda pada dua variasi dataset, serta menyediakan **web application** lengkap untuk klasifikasi secara real-time.

> **ComVis26** — Computer Vision, Semester 4

---

## 📑 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Arsitektur Model](#-arsitektur-model)
- [Strategi Pelatihan](#-strategi-pelatihan)
- [Struktur Proyek](#-struktur-proyek)
- [Dataset](#-dataset)
- [Cara Penggunaan](#-cara-penggunaan)
  - [Training Model](#1-training-model)
  - [Testing Model](#2-testing-model)

---

## ✨ Fitur Utama

- 🧠 **5 Arsitektur Model** — VGG16, DenseNet121, ResNet50, InceptionV3, dan EfficientNetB0
- 📊 **2 Variasi Dataset** — Dataset Original (Stratified) & Dataset Augmented
- 🔬 **Evaluasi Lengkap** — Classification Report & Confusion Matrix untuk setiap model
- ⚖️ **Penanganan Class Imbalance** — Dynamic Class Weights berbasis distribusi data aktual
- 🎯 **Two-Stage Fine-Tuning** — Feature Extraction → Full Fine-Tuning dengan learning rate scheduling
- 🌐 **Web App Full-Stack** — Frontend React + Backend FastAPI untuk prediksi real-time
- 🚀 **Production Deployment** — Backend di Railway, model weights di Hugging Face Hub
- 🎨 **UI Premium** — Glassmorphism, animasi halus, dan desain responsif

---

## 🧠 Arsitektur Model

Semua model CNN menggunakan pendekatan **Transfer Learning** dengan bobot pre-trained dari ImageNet. Lapisan top (classifier) diganti dengan arsitektur kustom:

```
Input (224×224×3) → Data Augmentation → Preprocess Input → Base Model (frozen/unfrozen)
    → GlobalAveragePooling2D → Dropout(0.5) → Dense(1, sigmoid)
```

| Model | Parameter | Input Size | Preprocessing |
|-------|-----------|------------|---------------|
| **VGG16** | ~138M | 224×224 | BGR conversion + mean subtraction |
| **DenseNet121** | ~8M | 224×224 | Pixel scaling |
| **ResNet50** | ~25M | 224×224 | BGR conversion + mean subtraction |
| **InceptionV3** | ~24M | 224×224 | Scale ke [-1, 1] |
| **EfficientNetB0** | ~5M | 224×224 | EfficientNet-specific |
| **Gemini 3.1 Flash Lite** | LLM | Variabel | — |

---

## 🎯 Strategi Pelatihan

### Two-Stage Fine-Tuning

| | Tahap 1: Feature Extraction | Tahap 2: Fine-Tuning |
|---|---|---|
| **Base Model** | Frozen (seluruh layer) | Unfrozen (BatchNorm tetap frozen) |
| **Yang dilatih** | Custom Classifier Head saja | Seluruh model |
| **Learning Rate** | `0.001` | `0.00001` |
| **Max Epochs** | 30 | 50 |

### Callbacks

| Callback | Konfigurasi |
|----------|-------------|
| **EarlyStopping** | Monitor `val_loss`, patience 10, restore best weights |
| **ModelCheckpoint** | Simpan model terbaik berdasarkan `val_loss` terendah |
| **ReduceLROnPlateau** | Kurangi LR ×0.5 jika stagnan 5 epoch, min LR `1e-6` |

### Penanganan Class Imbalance

Class weights dihitung secara **dinamis** dari distribusi data training aktual:
```python
weight = total_samples / (2.0 × count_per_class)
```

### On-the-fly Data Augmentation (saat Training)

| Teknik | Parameter | Catatan |
|--------|-----------|---------|
| Random Zoom | ±5% | Scale tipis |
| Random Rotation | ±5% (~18°) | Rotate tipis |
| Random Brightness | ±10% | — |
| Random Contrast | ±10% | — |
| Gaussian Noise | σ = 0.1 | Simulasi noise ringan |

> ⚠️ **Horizontal Flip sengaja TIDAK digunakan** untuk menjaga validitas anatomi posisi jantung pada citra X-ray.

---

## 📁 Struktur Proyek

```
CNN_Pneumonia_Detection/
│
├── 📄 README.md
├── 📄 .gitignore
│
├── model/                              # Semua kode model & eksperimen
│   │
│   ├── dataset/                        # Dataset gambar (tidak di-track Git)
│   │   ├── data_ori/                   #   Dataset original
│   │   │   ├── train/
│   │   │   │   ├── NORMAL/
│   │   │   │   └── PNEUMONIA/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── data_stratified/            #   Dataset original (stratified re-split)
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── data_aug/                   #   Dataset augmented
│   │       └── test/
│   │           ├── normal/
│   │           └── pneumonia/
│   │
│   ├── vgg16/                          # Model VGG16
│   │   ├── vgg16.py                    #   Training script (two-stage)
│   │   ├── vgg16_test.py               #   Testing & evaluasi
│   │   ├── confusion_matrix_vgg16_A.png
│   │   └── confusion_matrix_vgg16_B.png
│   │
│   ├── densenet121/                    # Model DenseNet121
│   │   ├── densenet121.py              #   Training script (two-stage)
│   │   ├── densenet121_test.py         #   Testing & evaluasi
│   │   ├── plan.txt                    #   Rencana & dokumentasi pelatihan
│   │   ├── confusion_matrix_densenet121_A.png
│   │   └── confusion_matrix_densenet121_B.png
│   │
│   ├── resnet50/                       # Model ResNet50
│   │   ├── resnet50.py                 #   Training script (two-stage)
│   │   ├── resnet50_test.py            #   Testing & evaluasi
│   │   ├── confusion_matrix_resnet50_A.png
│   │   └── confusion_matrix_resnet50_B.png
│   │
│   ├── inceptionv3/                    # Model InceptionV3
│   │   ├── inceptionv3.py              #   Training script (two-stage)
│   │   ├── inceptionv3_test.py         #   Testing & evaluasi
│   │   ├── confusion_matrix_inceptionv3_A.png
│   │   └── confusion_matrix_inceptionv3_B.png
│   │
│   ├── efficient_netB0/                # Model EfficientNetB0
│   │   ├── efficientnetb0.py           #   Training script (two-stage)
│   │   ├── efficientnetb0_test.py      #   Testing & evaluasi
│   │   ├── confusion_matrix_efficientnetb0_A.png
│   │   └── confusion_matrix_efficientnetb0_B.png
│   │
│   ├── gemini/                         # Model Gemini (LLM-based)
│   │   ├── gemini_script.py            #   Script evaluasi batch via API
│   │   ├── gemini_test.py              #   Confusion matrix generator
│   │   ├── hasil_ori.csv               #   Hasil prediksi dataset A
│   │   ├── hasil_aug.csv               #   Hasil prediksi dataset B
│   │   ├── confusion_matrix_gemini_A.png
│   │   └── confusion_matrix_gemini_B.png
│   │
│   └── manipulation/                   # Utilitas data
│       ├── data_augmentation.py        #   Script augmentasi gambar (Albumentations)
│       └── kocok.py                    #   Script stratified split (70/15/15)
│
├── app/                                # Web Application
│   ├── front/
│   │   └── index.html                  # Frontend React SPA (single-file)
│   │
│   └── back/
│       └── comvis-backend/             # Backend FastAPI
│           ├── main.py                 #   App entry point & endpoint definitions
│           ├── model.py                #   Model loading dari HF Hub & inference
│           ├── preprocess.py           #   Image preprocessing (resize, RGB)
│           ├── requirements.txt        #   Python dependencies
│           ├── railway.toml            #   Railway deployment config
│           ├── .gitignore
│           └── README.md               #   Dokumentasi backend
│
└── .vscode/
    └── settings.json                   # Konfigurasi VS Code
```

---

## 📦 Dataset

Proyek ini menggunakan dataset citra **chest X-ray** dari https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia yang dibagi menjadi dua variasi:

| Kode | Nama | Deskripsi |
|------|------|-----------|
| **Dataset A** | `data_stratified` | Dataset original yang di-split ulang secara stratified (70/15/15) |
| **Dataset B** | `data_aug` | Dataset yang telah di-augmentasi |

### Stratified Split (`kocok.py`)

Script `model/manipulation/kocok.py` menggabungkan **seluruh gambar** dari split original (train + val + test), mengocak secara acak (seed 42), lalu membagi ulang dengan rasio:

| Split | Rasio |
|-------|-------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

File di-rename secara otomatis menjadi format konsisten: `{split}_{kelas}_{nomor}.jpeg` (contoh: `train_n_1.jpeg`, `val_p_5.jpeg`).

### Offline Data Augmentation (`data_augmentation.py`)

Teknik augmentasi menggunakan library [Albumentations](https://albumentations.ai/) untuk menghasilkan **Dataset B**:

| Teknik | Probabilitas | Parameter |
|--------|-------------|-----------|
| Horizontal Flip | 50% | — |
| Rotation | 50% | ±15° |
| Random Brightness/Contrast | 50% | ±20% |
| Gaussian Blur | 30% | Kernel 3–5 |
| Gaussian Noise | 30% | Variance 10–50 |

Setiap gambar menghasilkan **4 variasi augmentasi**.

> ⚠️ **Dataset tidak di-track oleh Git** karena ukurannya yang besar. Hanya file `.gitkeep` yang di-commit untuk mempertahankan struktur folder.

---

## 🚀 Cara Penggunaan

### Prasyarat

```bash
# Untuk training & testing model CNN
pip install tensorflow numpy scikit-learn matplotlib seaborn pillow

# Untuk augmentasi data (opsional)
pip install albumentations opencv-python

# Untuk model Gemini (opsional)
pip install google-genai pillow
```

### 1. Training Model

Semua model menggunakan pola training yang sama (two-stage fine-tuning). Jalankan dari direktori masing-masing model:

```bash
cd model

# Training VGG16
python vgg16/vgg16.py

# Training DenseNet121
python densenet121/densenet121.py

# Training ResNet50
python resnet50/resnet50.py

# Training InceptionV3
python inceptionv3/inceptionv3.py

# Training EfficientNetB0
python efficient_netB0/efficientnetb0.py
```

**Mengganti dataset:**

Script training secara default menggunakan `data_stratified` (Dataset A). Untuk mengganti ke Dataset B:
- Ubah path `BASE_DIR` di script training, arahkan ke `dataset/data_aug`

Model terbaik otomatis disimpan berdasarkan `val_loss` terendah (contoh: `densenet121_best.keras`).

### 2. Testing Model

Jalankan script testing dari direktori masing-masing model:

```bash
cd model

# Testing VGG16
python vgg16/vgg16_test.py

# Testing DenseNet121
python densenet121/densenet121_test.py

# Testing ResNet50
python resnet50/resnet50_test.py

# Testing InceptionV3
python inceptionv3/inceptionv3_test.py

# Testing EfficientNetB0
python efficient_netB0/efficientnetb0_test.py

```

**Output testing berupa:**
- 📋 **Classification Report** — Precision, Recall, F1-Score per kelas
- 📊 **Confusion Matrix** — Disimpan sebagai file `.png`

Setiap model menghasilkan dua confusion matrix:
- `confusion_matrix_*_A.png` → Hasil evaluasi pada **Dataset A** (Original Stratified)
- `confusion_matrix_*_B.png` → Hasil evaluasi pada **Dataset B** (Augmented)
