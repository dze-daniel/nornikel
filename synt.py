import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from raindrop.config import cfg
from raindrop.dropgenerator import generate_label, generateDrops

def see_plot(pict, color='gray', size=(5,5), title=''):
    plt.figure(figsize=size)
    plt.imshow(pict, cmap=color)
    plt.title(title)
    plt.xticks()
    plt.show()

def synt_gereration(real_img_fullpath, real_msk_fullpath, cfg, verbose=False):
    img = cv2.imread(real_img_fullpath)[:,:,::-1]
    msk = cv2.imread(real_msk_fullpath)[:,:,::-1]
    (h, w, c) = img.shape

    if verbose:
        mrg = img.copy()
        mrg[:,:,0] = mrg[:,:,0]//5*4+msk[:,:,0]//5
        see_plot(np.concatenate([img,msk,mrg],axis=1), size=(20,20), title='REAL image-mask-merged')
    
    # Main synthetic generation
    List_of_Drops, label_map = generate_label(h, w, cfg)
    synt_img, _, synt_msk = generateDrops(real_img_fullpath, cfg, List_of_Drops)
    
    synt_img = np.array(synt_img)
    synt_msk = np.array(synt_msk)
    
    if len(synt_msk.shape) == 3: 
        synt_msk = synt_msk[:,:,0]
    synt_msk = np.stack((synt_msk,)*3, axis=-1)
    
    synt_msk = (((msk > 0).astype(int) + (synt_msk > 0).astype(int)) > 0).astype(np.uint8) * 255
    
    if verbose:
        merged = synt_img.copy()
        merged[:,:,0] = merged[:,:,0]//5*4 + synt_msk[:,:,0]//5
        for_plot = np.concatenate([synt_img, synt_msk, merged], axis=1)
        see_plot(for_plot, size=(20,20), title=f'SYNT image-mask-merged')
        
    return synt_img, synt_msk

# Paths to real images and masks
real_images_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_open_dataset\open_img"  # Folder with real images
real_masks_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_open_dataset\open_msk"  # Folder with real masks
output_dir = "open_synt"  # Folder to save generated data

os.makedirs(output_dir, exist_ok=True)

images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)
masks_dir = os.path.join(output_dir, "masks")
os.makedirs(masks_dir, exist_ok=True)

images_dir_train = os.path.join(images_dir, "train")
os.makedirs(images_dir_train, exist_ok=True)
images_dir_val = os.path.join(images_dir, "val")
os.makedirs(images_dir_val, exist_ok=True)

labels_dir_train = os.path.join(labels_dir, "train")
os.makedirs(labels_dir_train, exist_ok=True)
labels_dir_val = os.path.join(labels_dir, "val")
os.makedirs(labels_dir_val, exist_ok=True)

masks_dir_train = os.path.join(masks_dir, "train")
os.makedirs(masks_dir_train, exist_ok=True)
masks_dir_val = os.path.join(masks_dir, "val")
os.makedirs(masks_dir_val, exist_ok=True)

# Get list of real images and masks
real_images = sorted([os.path.join(real_images_dir, f) for f in os.listdir(real_images_dir) if f.endswith('.jpg')])
real_masks = sorted([os.path.join(real_masks_dir, f) for f in os.listdir(real_masks_dir) if f.endswith('.png')])

# Generate 1000 synthetic samples
full_q = 1200
for i in range(full_q):
    img_idx = i % len(real_images)  # Loop through the available images if less than 1000
    real_img_fullpath = real_images[img_idx]
    real_msk_fullpath = real_masks[img_idx]
    
    synt_img, synt_msk = synt_gereration(real_img_fullpath, real_msk_fullpath, cfg, verbose=False)
    
    # Save generated images and masks
    if i<=full_q*0.8:
        cv2.imwrite(os.path.join(images_dir_train, f"synt_img_{i:04d}.jpg"), synt_img[:,:,::-1])
        cv2.imwrite(os.path.join(masks_dir_train, f"synt_img_{i:04d}.png"), synt_msk[:,:,::-1])
    else:
        cv2.imwrite(os.path.join(images_dir_val, f"synt_img_{i:04d}.jpg"), synt_img[:,:,::-1])
        cv2.imwrite(os.path.join(masks_dir_val, f"synt_img_{i:04d}.png"), synt_msk[:,:,::-1])
        
print("1200 synthetic images and masks have been generated.")
