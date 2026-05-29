import os
import shutil
import random

# ==========================================
# KONFIGURASI
# ==========================================
# Path relatif karena skrip dijalankan dari dalam folder 'densenet121'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOURCE_DIRS = [
    os.path.join(BASE_DIR, 'data', 'train'), 
    os.path.join(BASE_DIR, 'data', 'val')
]
OUTPUT_DIR = os.path.join(BASE_DIR, 'data_stratified') # Folder baru agar data asli aman

CLASSES = ["NORMAL", "PNEUMONIA"]
VAL_RATIO = 0.15 # 15% untuk validasi, 85% untuk training
SEED = 42

# Mengunci seed agar hasil kocokan selalu sama jika skrip diulang
random.seed(SEED)

def process_split():
    print(f"Target Folder Output: {OUTPUT_DIR}\n")
    
    # 1. Buat struktur folder output baru
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
            
    # 2. Proses setiap kelas secara terpisah untuk menjamin rasio seimbang (Stratified)
    for cls in CLASSES:
        all_files = []
        
        # Kumpulkan semua path file gambar dari folder train dan val asli
        for src_dir in SOURCE_DIRS:
            cls_dir = os.path.join(src_dir, cls)
            if os.path.exists(cls_dir):
                # Hanya mengambil file dengan ekstensi gambar
                files = [
                    os.path.join(cls_dir, f) for f in os.listdir(cls_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                all_files.extend(files)
        
        if len(all_files) == 0:
            print(f"Peringatan: Tidak ada gambar ditemukan untuk kelas {cls}!")
            continue
            
        # Kocok ulang (Shuffle) semua data di kelas ini
        random.shuffle(all_files)
        
        # Hitung titik potong (index) untuk memisahkan data
        split_idx = int(len(all_files) * (1 - VAL_RATIO))
        
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        print(f"Kelas {cls:<10} -> Total: {len(all_files)} | Train (85%): {len(train_files)} | Val (15%): {len(val_files)}")
        
        # 3. Fungsi bantuan untuk menyalin file ke folder tujuan
        def copy_files(file_list, split_name):
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(OUTPUT_DIR, split_name, cls, file_name)
                shutil.copy2(file_path, dest_path) # copy2 mempertahankan metadata file
        
        # Eksekusi penyalinan
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')

if __name__ == "__main__":
    print("Memulai proses penggabungan, pengocokan, dan pemisahan ulang data...")
    process_split()
    print("\nSelesai! Data yang sudah dirapikan tersedia di folder 'data_stratified'.")