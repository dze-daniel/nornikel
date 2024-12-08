from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# Укажите путь к каталогу с бинарными масками и каталог для сохранения преобразованных масок
masks_dir_train = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\masks2\train"
labels_dir_train = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\labels\train"
num_classes = 1

masks_dir_val = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\masks2\val"
labels_dir_val = r"C:\Users\dzebo\Downloads\2\2\train_dataset\cv_synt_dataset\open_synt\labels\val"

print("train")
convert_segment_masks_to_yolo_seg(masks_dir=masks_dir_train, output_dir=labels_dir_train, classes = 1)

print("val")
convert_segment_masks_to_yolo_seg(masks_dir=masks_dir_val, output_dir=labels_dir_val, classes=1)

