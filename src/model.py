import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(28,28,3), num_classes=7):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(32, activation='relu'),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model
