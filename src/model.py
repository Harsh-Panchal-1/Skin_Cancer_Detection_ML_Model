import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

INPUT_SHAPE = (128, 128, 3)

def se_block(input_tensor, ratio=8):
    """Creates a Squeeze-and-Excitation block for channel-wise attention."""
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return layers.multiply([input_tensor, se])

def build_cnn(input_shape=INPUT_SHAPE, num_classes=7):
    """Builds a denser custom CNN with an integrated attention mechanism."""
    
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    # Block 3 with Attention
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x)  # <-- ATTENTION LAYER
    x = layers.MaxPooling2D()(x)

    # Block 4 with Attention
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x)  # <-- ATTENTION LAYER
    x = layers.MaxPooling2D()(x)

    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.summary()
    return model