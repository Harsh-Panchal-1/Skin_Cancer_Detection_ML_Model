import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = (28, 28)
BATCH_SIZE = 128
CLASSES = ['Solar or actinic keratosis','Basal cell carcinoma','Pigmented benign keratosis','Dermatofibroma','Melanoma, NOS','Nevus','Squamous cell carcinoma, NOS']

def get_datasets():
    """Load train/val/test datasets with normalization and augmentation (train only)."""

    # Augmentation for training
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ])

    normalization_layer = layers.Rescaling(1./255)

    # Train
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/dataset/train",
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Val
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/dataset/val",
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Test
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/dataset/test",
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Apply normalization & augmentation
    train_ds = train_ds.map(lambda x,y: (data_augmentation(normalization_layer(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds  = test_ds.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), test_ds.prefetch(tf.data.AUTOTUNE)
