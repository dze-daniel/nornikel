import os
from ultralytics import YOLO
from PIL import Image

our_model = YOLO(r"C:\Users\dzebo\Downloads\2\2\nornikel_dockerfile\big_model.pt")
their_model = YOLO(r"C:\Users\dzebo\Downloads\baseline.pt")

masks_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_open_dataset\masks"
images_dir = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_open_dataset\open_img"
masks = os.listdir(masks_dir)

for mask in masks:
    image = Image.open(os.path.join(images_dir, mask).replace("png", "jpg"))
    our_res = our_model(image)
    their_res = their_model(image)
    image.save(os.path.join(masks_dir, mask).replace("png", "jpg"))
    our_res[0].save(os.path.join(masks_dir, mask+"our.png"))
    their_res[0].save(os.path.join(masks_dir, mask+"their.png"))
