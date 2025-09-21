import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from src.data_pipeline import get_datasets, CLASSES
from src.model import build_cnn # Use your advanced custom model

def train_model():
    """Compiles and trains the model on the balanced offline dataset."""
    train_ds, val_ds, test_ds = get_datasets()
    model = build_cnn(num_classes=len(CLASSES))
    
    callbacks_list = [
        callbacks.ModelCheckpoint("outputs/models/best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    ]

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy", # Standard loss for a balanced dataset
        metrics=["accuracy"]
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks_list
    )

    model.save("outputs/models/final_model.keras")
    return model, history, test_ds