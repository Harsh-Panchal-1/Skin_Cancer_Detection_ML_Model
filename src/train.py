import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from src.data_pipeline import get_datasets
from src.model import build_cnn

def train_model():
    train_ds, val_ds, test_ds = get_datasets()

    model = build_cnn()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_cb = callbacks.ModelCheckpoint("outputs/models/best_model.keras", save_best_only=True, monitor="val_loss", mode="min")
    # earlystop_cb  = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduce_cb     = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[checkpoint_cb, reduce_cb]
    )

    model.save("outputs/models/final_model.keras")
    return model, history, test_ds
