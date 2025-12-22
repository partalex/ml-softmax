import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.shared import load_multiclass_csv, INPUT_FILE


# ============================================================
# Reference softmax (sklearn)
# ============================================================

def train_softmax_reference(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
) -> None:
    """
    Train and evaluate a reference softmax classifier using sklearn.
    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training label vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation label vector.
    """
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )

    model.fit(X_train, y_train)

    y_train_pred: np.ndarray = model.predict(X_train)
    y_val_pred: np.ndarray = model.predict(X_val)

    train_acc: float = accuracy_score(y_train, y_train_pred)
    val_acc: float = accuracy_score(y_val, y_val_pred)

    print("=== Reference Softmax (sklearn) ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    X, y = load_multiclass_csv(INPUT_FILE)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_softmax_reference(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
