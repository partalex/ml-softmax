import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.shared import load_multiclass_csv, INPUT_FILE


# ============================================================
# Reference softmax (sklearn)
# ============================================================

def train_softmax_reference(
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_val: np.ndarray,
        labels_val: np.ndarray
) -> None:
    """
    Train and evaluate a reference softmax classifier using sklearn.
    Args:
        features_train (np.ndarray): Training feature matrix.
        labels_train (np.ndarray): Training label vector.
        features_val (np.ndarray): Validation feature matrix.
        labels_val (np.ndarray): Validation label vector.
    """
    model = LogisticRegression(
        # multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )

    model.fit(features_train, labels_train)

    y_train_pred: np.ndarray = model.predict(features_train)
    y_val_pred: np.ndarray = model.predict(features_val)

    train_acc: float = accuracy_score(labels_train, y_train_pred)
    val_acc: float = accuracy_score(labels_val, y_val_pred)

    print("=== Reference Softmax (sklearn) ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")


if __name__ == "__main__":
    features, labels = load_multiclass_csv(INPUT_FILE)

    features_train, features_val, labels_train, labels_val = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_softmax_reference(features_train, labels_train, features_val, labels_val)
