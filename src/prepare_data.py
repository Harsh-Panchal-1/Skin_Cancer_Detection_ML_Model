import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

RAW_DIR = "data/raw_ham"
OUT_DIR = "data/dataset"

CLASSES = ['Solar or actinic keratosis','Basal cell carcinoma','Pigmented benign keratosis','Dermatofibroma','Melanoma, NOS','Nevus','Squamous cell carcinoma, NOS']

def prepare_split(target_size=(28,28)):
    """Create stratified train/val/test split (80%/10%/10%), resizing all to 28x28."""
    meta = pd.read_csv(os.path.join(RAW_DIR, "ham10000_metadata_2025-08-30.csv"))
    meta = meta[meta['diagnosis_3'].isin(CLASSES)].copy()
    meta['filename'] = meta['isic_id'] + ".jpg"

    # stratified split
    train, temp = train_test_split(meta, test_size=0.2, stratify=meta['diagnosis_3'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['diagnosis_3'], random_state=42)

    # function to copy + resize
    def copy_and_resize(df, split_name):
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split_name, cls), exist_ok=True)
        for _, row in df.iterrows():
            src = os.path.join(RAW_DIR, row['filename'])
            dst = os.path.join(OUT_DIR, split_name, row['diagnosis_3'], row['filename'])
            if os.path.exists(src):
                try:
                    img = load_img(src, target_size=target_size)  # resize
                    img.save(dst)
                except Exception as e:
                    print("Error resizing:", src, e)

    copy_and_resize(train, "train")
    copy_and_resize(val, "val")
    copy_and_resize(test, "test")

    return train, val, test

def augment_balance(train_df, target_size=(28,28)):
    """Augment minority classes until balanced (with brightness/contrast)."""
    train_dir = os.path.join(OUT_DIR, "train")
    max_count = train_df['diagnosis_3'].value_counts().max()

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],   # as per paper
        fill_mode="nearest"
    )

    balanced_meta = []

    for cls in CLASSES:
        cls_dir = os.path.join(train_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        cls_df = train_df[train_df['diagnosis_3'] == cls]
        balanced_meta.append(cls_df)

        n_to_add = max_count - len(cls_df)
        if n_to_add <= 0:
            continue

        images = list(cls_df['filename'])
        i = 0
        pbar = tqdm(total=n_to_add, desc=f"Augmenting {cls}")
        while n_to_add > 0:
            img_path = os.path.join(RAW_DIR, images[i % len(images)])
            if not os.path.exists(img_path):
                i += 1
                continue

            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for batch in datagen.flow(x, batch_size=1):
                new_name = f"aug_{cls}_{i}_{np.random.randint(1e6)}.jpg"
                save_path = os.path.join(cls_dir, new_name)
                array_to_img(batch[0]).save(save_path)

                new_row = {"isic_id": new_name.replace(".jpg",""), "diagnosis_3": cls, "filename": new_name}
                balanced_meta.append(pd.DataFrame([new_row]))
                n_to_add -= 1
                pbar.update(1)
                break
            i += 1
        pbar.close()

    balanced_df = pd.concat(balanced_meta, ignore_index=True)
    balanced_df.to_csv(os.path.join(OUT_DIR, "train_balanced.csv"), index=False)
    return balanced_df
