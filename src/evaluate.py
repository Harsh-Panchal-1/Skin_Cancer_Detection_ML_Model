import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.data_pipeline import get_datasets, CLASSES

def evaluate_model(model_path="outputs/models/best_model.keras"):
    """
    Loads the best model, evaluates it on the test set, 
    and returns labels, predictions, and prediction probabilities.
    """
    _, _, test_ds = get_datasets()

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Generating predictions on the test set...")
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    print("\n--- Confusion Matrix (Raw)---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # --- THE FIX IS HERE ---
    # Return the probabilities needed for the ROC curve
    return y_true, y_pred, y_probs