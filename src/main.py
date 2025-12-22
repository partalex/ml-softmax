import time
import numpy as np

from src.shared import load_multiclass_csv, INPUT_FILE


# ============================================================
# Data utilities
# ============================================================


def train_val_split(
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.2,
        seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and validation sets.
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,).
        val_ratio (float): Proportion of the dataset to include in the validation split.
        seed (int): Random seed for reproducibility.
    Returns:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training label vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation label vector.
    """
    rng = np.random.default_rng(seed)
    indices: np.ndarray = rng.permutation(len(X))

    split: int = int(len(X) * (1.0 - val_ratio))
    train_idx: np.ndarray = indices[:split]
    val_idx: np.ndarray = indices[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ============================================================
# Softmax classifier
# ============================================================

class SoftmaxClassifier:
    def __init__(self, n_features: int, n_classes: int) -> None:
        self.W: np.ndarray = 0.01 * np.random.randn(n_features, n_classes)
        self.b: np.ndarray = np.zeros(n_classes)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Compute the softmax probabilities.
        Args:
            logits (np.ndarray): Logits of shape (n_samples, n_classes).
        Returns:
            np.ndarray: Softmax probabilities of shape (n_samples, n_classes).
        """
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits: np.ndarray = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute class probabilities.
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        logits: np.ndarray = X @ self.W + self.b
        return self._softmax(logits)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).
        Returns:
            float: Cross-entropy loss.
        """
        probs: np.ndarray = self.forward(X)
        n: int = X.shape[0]
        log_likelihood: np.ndarray = -np.log(probs[np.arange(n), y] + 1e-12)
        return float(np.mean(log_likelihood))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the accuracy of the model on the given data.
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).
        Returns:
            float: Accuracy as a float between 0 and 1.
        """
        probs: np.ndarray = self.forward(X)
        preds: np.ndarray = np.argmax(probs, axis=1)
        return float(np.mean(preds == y))

    def gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients of the loss with respect to weights and biases.
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).
        Returns:
            tuple[np.ndarray, np.ndarray]: Gradients of weights and biases.
        """
        n: int = X.shape[0]
        probs: np.ndarray = self.forward(X)
        probs[np.arange(n), y] -= 1.0
        probs /= n

        dW: np.ndarray = X.T @ probs
        db: np.ndarray = np.sum(probs, axis=0)

        return dW, db

    def update(self, dW: np.ndarray, db: np.ndarray, lr: float) -> None:
        """
        Update the model parameters using gradient descent.
        Args:
            dW (np.ndarray): Gradient of weights.
            db (np.ndarray): Gradient of biases.
            lr (float): Learning rate.
        """
        self.W -= lr * dW
        self.b -= lr * db


# ============================================================
# Training loop (mini-batch SGD)
# ============================================================

def train(
        model: SoftmaxClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        seed: int = 0
) -> dict[str, list[float]]:
    """
    Train the SoftmaxClassifier using mini-batch SGD.
    Args:
        model (SoftmaxClassifier): The model to train.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training label vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation label vector.
        learning_rate (float): Learning rate.
        batch_size (int): Size of each mini-batch.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
    Returns:
        dict[str, list[float]]: Training history containing loss and accuracy.
    """
    rng = np.random.default_rng(seed)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    n: int = len(X_train)

    for epoch in range(epochs):
        start_time: float = time.time()

        indices: np.ndarray = rng.permutation(n)

        for start in range(0, n, batch_size):
            end: int = start + batch_size
            batch_idx: np.ndarray = indices[start:end]

            X_batch: np.ndarray = X_train[batch_idx]
            y_batch: np.ndarray = y_train[batch_idx]

            dW, db = model.gradients(X_batch, y_batch)
            model.update(dW, db, learning_rate)

        epoch_time: float = time.time() - start_time

        train_loss: float = model.loss(X_train, y_train)
        val_acc: float = model.accuracy(X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(f"Epoch {epoch + 1:3d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | time={epoch_time:.3f}s")

    return history


# ============================================================
# Main
# ============================================================

BATCH_SIZE: int = 16
EPOCHS: int = 100
LEARNING_RATE: float = 0.1

if __name__ == "__main__":
    X, y = load_multiclass_csv(INPUT_FILE)

    X_train, y_train, X_val, y_val = train_val_split(X, y)

    n_features: int = X.shape[1]
    n_classes: int = int(np.max(y)) + 1

    model = SoftmaxClassifier(n_features, n_classes)

    history = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    print("Final train accuracy:", model.accuracy(X_train, y_train))
    print("Final val accuracy:", model.accuracy(X_val, y_val))
