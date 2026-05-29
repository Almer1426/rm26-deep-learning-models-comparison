import os
import csv
import time
from google import genai
import sys
from PIL import Image
from datetime import datetime

client = genai.Client(api_key="API_KEY")
MODEL_ID = 'gemini-3.1-flash-lite-preview'
PROMPT = "Act as an expert radiologist. Carefully examine this chest X-ray strictly based on the visual pixel data. Look for clinical signs of pneumonia, such as focal or " \
"diffuse opacities, infiltrates, or lung consolidation. Ignore any text or markers on the image. Respond with EXACTLY ONE WORD: either 'NORMAL' or 'PNEUMONIA'. Do not include any other explanation."

def log_print(message):
    """Fungsi bantuan agar setiap output print memiliki timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def get_processed_filenames(csv_path):
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])
    return processed

def evaluate(dataset_dirs, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    processed_files = get_processed_filenames(csv_path)
    print(f"[*] Ditemukan {len(processed_files)} gambar yang sudah diproses di {csv_path}.")

    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, mode='a', newline='')
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(['nama file', 'true_label', 'pred_label'])

    count = 0
    
    try:
        for dir_path in dataset_dirs:
            if not os.path.exists(dir_path):
                continue
                
            true_label = os.path.basename(dir_path).upper()
            
            # Urutkan file agar prosesnya berurutan rapi (penting untuk konsistensi)
            for filename in sorted(os.listdir(dir_path)):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                # 3. FITUR RESUME: Skip jika gambar sudah ada di CSV
                if filename in processed_files:
                    continue
                    
                # 4. FITUR LIMIT HARIAN: Stop jika sudah mencapai batas
                if count >= 400:
                    print(f"\n[INFO] Batas harian tercapai ({count} request). Proses dihentikan.")
                    return # Keluar dari fungsi evaluate_dataset
                    
                img_path = os.path.join(dir_path, filename)
                
                while True:
                    try:
                        img = Image.open(img_path)
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[PROMPT, img]
                        )
                        
                        pred_label = response.text.strip().upper()
                            
                        writer.writerow([filename, true_label, pred_label])
                        csv_file.flush() 
                        
                        count += 1
                        log_print(f"[{count}/400] Berhasil: {filename} | Asli: {true_label} | Prediksi: {pred_label}")
                        
                        time.sleep(40)
                        
                        # Jika proses berhasil, break dari loop while untuk lanjut ke gambar selanjutnya
                        break 
                        
                    except Exception as e:
                        log_print(f"[ERROR] Gagal saat memproses {filename}!")
                        log_print(f"Detail Error: {e}")
                        log_print("Menunggu 2 menit (120 detik) sebelum mencoba re-run lagi...")
                        
                        # Tunggu lalu otomatis mengulang siklus while dari atas
                        time.sleep(120)
                    
    finally:
        # Selalu tutup file dengan aman meskipun program dihentikan paksa (Ctrl+C)
        csv_file.close()
        print(f"--- Sesi selesai. Total request dijalankan: {count} ---")

if __name__ == "__main__":
    test_set_a = [
        'data/test/NORMAL', 
        'data/test/PNEUMONIA'
    ]

    test_set_b = [
        'data_aug/test/normal', 
        'data_aug/test/pneumonia'
    ]

    # evaluate(test_set_a, 'gemini/hasil_ori.csv')
    evaluate(test_set_b, 'gemini/hasil_aug.csv')
