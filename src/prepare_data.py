import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# --- Configuration ---
RAW_DIR = "data/raw_ham"
OUT_DIR = "data/dataset_balanced_offline" # New directory for this dataset
METADATA_FILE = os.path.join(RAW_DIR, "ham10000_metadata_2025-08-30.csv")
TARGET_SIZE = (128, 128)
TARGET_COUNT_PER_CLASS = 6500 # The target number of images per class after balancing

CLASS_MAP = {
    'Solar or actinic keratosis': 'akiec', 'Basal cell carcinoma': 'bcc',
    'Pigmented benign keratosis': 'bkl', 'Dermatofibroma': 'df',
    'Melanoma, NOS': 'mel', 'Nevus': 'nv', 'Vascular lesions': 'vasc',
    'Squamous cell carcinoma, NOS': 'vasc'
}
# --- End Configuration ---

def prepare_balanced_offline_dataset():
    """Creates a balanced offline dataset by over/under-sampling the training set."""
    if os.path.exists(OUT_DIR):
        print(f"'{OUT_DIR}' already exists. Please delete it to regenerate.")
        return

    meta = pd.read_csv(METADATA_FILE)
    meta['dx_short'] = meta['diagnosis_3'].map(CLASS_MAP)
    meta.dropna(subset=['dx_short'], inplace=True)
    meta['filename'] = meta['isic_id'] + ".jpg"

    # Step 1: Create initial train/val/test split from the original data
    train_df, temp_df = train_test_split(meta, test_size=0.2, stratify=meta['dx_short'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx_short'], random_state=42)

    # Step 2: Balance the training DataFrame
    print("Balancing training data...")
    samples = []
    for cls in train_df['dx_short'].unique():
        class_df = train_df[train_df['dx_short'] == cls]
        if len(class_df) > TARGET_COUNT_PER_CLASS: # Undersample
            samples.append(class_df.sample(TARGET_COUNT_PER_CLASS, random_state=42))
        else: # Keep all original samples for oversampling later
            samples.append(class_df)
    balanced_train_df = pd.concat(samples)

    # Step 3: Set up ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                 shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

    # Step 4: Copy/Resize/Augment images to the new directory
    print("Processing and saving images...")
    for split_name, df in [('train', balanced_train_df), ('val', val_df), ('test', test_df)]:
        is_training_set = (split_name == 'train')
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            cls = row['dx_short']
            split_dir = os.path.join(OUT_DIR, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            
            src_path = os.path.join(RAW_DIR, row['filename'])
            dst_path = os.path.join(split_dir, row['filename'])

            if os.path.exists(src_path):
                img = load_img(src_path, target_size=TARGET_SIZE)
                img.save(dst_path) # Save the original (or undersampled) image

        # Augment only the training set
        if is_training_set:
            for cls in df['dx_short'].unique():
                class_dir = os.path.join(OUT_DIR, split_name, cls)
                current_count = len(os.listdir(class_dir))
                num_to_augment = TARGET_COUNT_PER_CLASS - current_count

                if num_to_augment > 0:
                    print(f"Augmenting class '{cls}' with {num_to_augment} new images...")
                    original_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
                    i = 0
                    for _ in tqdm(range(num_to_augment)):
                        img = load_img(original_images[i % len(original_images)])
                        x = img_to_array(img)
                        x = x.reshape((1,) + x.shape)
                        
                        # Generate one augmented image
                        aug_iter = datagen.flow(x, batch_size=1)
                        aug_image = next(aug_iter)[0].astype('uint8')
                        
                        save_path = os.path.join(class_dir, f'aug_{i}.jpg')
                        array_to_img(aug_image).save(save_path)
                        i += 1
    
    print("Balanced offline dataset created successfully.")

if __name__ == '__main__':
    prepare_balanced_offline_dataset()