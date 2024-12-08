import cv2
import numpy as np
import os

masks_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\masks\train"

output_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\masks2\train"

os.makedirs(output_dir, exist_ok=True)

for mask_name in os.listdir(masks_dir):
    mask_path = os.path.join(masks_dir, mask_name)
    mask = cv2.imread(mask_path)
    mask = np.where(mask == 255, 1, mask)  # Заменяем 255 на 1
    output_path = os.path.join(output_dir, mask_name)
    cv2.imwrite(output_path, mask)


masks_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\masks\val"

output_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\masks2\val"

os.makedirs(output_dir, exist_ok=True)

for mask_name in os.listdir(masks_dir):
    mask_path = os.path.join(masks_dir, mask_name)
    mask = cv2.imread(mask_path)
    mask = np.where(mask == 255, 1, mask)  # Заменяем 255 на 1
    output_path = os.path.join(output_dir, mask_name)
    cv2.imwrite(output_path, mask)
