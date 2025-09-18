import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from src.data_pipeline import get_datasets

CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']

def evaluate_model(model_path="outputs/models/best_model.keras"):
    _, _, test_ds = get_datasets()

    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([y.numpy() for x,y in test_ds], axis=0)
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=CLASSES))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Multiclass AUC
    y_true_bin = label_binarize(y_true, classes=range(len(CLASSES)))
    auc_macro = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
    print("Macro AUC:", auc_macro)

    # Regression-style metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print("MSE:", mse, "MAE:", mae, "RMSE:", rmse)
