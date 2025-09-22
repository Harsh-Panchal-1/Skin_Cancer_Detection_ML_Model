import tensorflow as tf
from tensorflow.keras import layers
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def get_datasets():
    """Loads the pre-augmented offline dataset."""
    data_dir = "data/dataset_balanced_offline"
    
    normalization_layer = layers.Rescaling(1./255)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "train"), labels="inferred", label_mode="int",
        class_names=CLASSES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "val"), labels="inferred", label_mode="int",
        class_names=CLASSES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "test"), labels="inferred", label_mode="int",
        class_names=CLASSES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Only apply normalization
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), test_ds.prefetch(tf.data.AUTOTUNE)