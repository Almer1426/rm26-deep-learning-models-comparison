import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os

def augmentation():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.OneOf([
            A.Rotate(
                limit=5, 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=0, 
                p=1.0
            ),
            A.Affine(
                translate_percent=0.0,
                scale=(0.95, 1.05),
                rotate=0,
                p=1.0
            )
        ], p = 0.5),

        A.GaussNoise(
            std_range=(0.01, 0.05),
            p=1.0
        )
    ])

def process (input_path, output_path, transform):
    os.makedirs(output_path, exist_ok=True)
    files = [f for f in os.listdir(input_path)]

    for filename in files:
        path = os.path.join(input_path, filename)
        img = cv2.imread(path)
        
        if img is not None:
            augmented = transform(image=img)['image']
            save_path = os.path.join(output_path, f"aug_{filename}")
            cv2.imwrite(save_path, augmented)

if __name__ == "__main__":
    dataset_map = {
        "data/test/NORMAL": "data_aug/test/normal",
        "data/test/PNEUMONIA": "data_aug/test/pneumonia"
    }
    
    test_transform = augmentation()
    
    for source, destination in dataset_map.items():
        if os.path.exists(source):
            process(source, destination, test_transform)
        else:
            print(f"Peringatan: Folder {source} tidak ditemukan!")

    print("Seluruh proses pembuatan Test Set B Selesai!")
