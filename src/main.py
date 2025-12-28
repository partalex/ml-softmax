import time
import numpy as np

from src.shared import load_multiclass_csv, INPUT_FILE


# ============================================================
# Data utilities
# ============================================================


def train_val_split(
        features: np.ndarray,
        labels: np.ndarray,
        val_ratio: float = 0.2,
        seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and validation sets.
    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Label vector of shape (n_samples,).
        val_ratio (float): Proportion of the dataset to include in the validation split.
        seed (int): Random seed for reproducibility.
    Returns:
        x_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training label vector.
        x_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation label vector.
    """
    rng = np.random.default_rng(seed)
    indices: np.ndarray = rng.permutation(len(features))

    split: int = int(len(features) * (1.0 - val_ratio))
    train_idx: np.ndarray = indices[:split]
    val_idx: np.ndarray = indices[split:]

    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


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

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute class probabilities.
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        logits: np.ndarray = features @ self.W + self.b
        return self._softmax(logits)

    def loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): True labels of shape (n_samples,).
        Returns:
            float: Cross-entropy loss.
        """
        probs: np.ndarray = self.forward(features)
        n: int = features.shape[0]
        log_likelihood: np.ndarray = -np.log(probs[np.arange(n), labels] + 1e-12)
        return float(np.mean(log_likelihood))

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the accuracy of the model on the given data.
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): True labels of shape (n_samples,).
        Returns:
            float: Accuracy as a float between 0 and 1.
        """
        probs: np.ndarray = self.forward(features)
        preds: np.ndarray = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))

    def gradients(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients of the loss with respect to weights and biases.
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): True labels of shape (n_samples,).
        Returns:
            tuple[np.ndarray, np.ndarray]: Gradients of weights and biases.
        """
        n: int = features.shape[0]
        probs: np.ndarray = self.forward(features)
        probs[np.arange(n), labels] -= 1.0
        probs /= n

        dW: np.ndarray = features.T @ probs
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
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_val: np.ndarray,
        labels_val: np.ndarray,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        seed: int = 0
) -> dict[str, list[float]]:
    """
    Train the SoftmaxClassifier using mini-batch SGD.
    Args:
        model (SoftmaxClassifier): The model to train.
        features_train (np.ndarray): Training feature matrix.
        labels_train (np.ndarray): Training label vector.
        features_val (np.ndarray): Validation feature matrix.
        labels_val (np.ndarray): Validation label vector.
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

    n: int = len(features_train)

    for epoch in range(epochs):
        start_time: float = time.time()

        indices: np.ndarray = rng.permutation(n)

        for start in range(0, n, batch_size):
            end: int = start + batch_size
            batch_idx: np.ndarray = indices[start:end]

            X_batch: np.ndarray = features_train[batch_idx]
            y_batch: np.ndarray = labels_train[batch_idx]

            dW, db = model.gradients(X_batch, y_batch)
            model.update(dW, db, learning_rate)

        epoch_time: float = time.time() - start_time

        train_loss: float = model.loss(features_train, labels_train)
        val_acc: float = model.accuracy(features_val, labels_val)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(f"Epoch {epoch + 1:3d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | time={epoch_time:.3f}s")

    return history


BATCH_SIZE: int = 32  # mb
# BATCH_SIZE: list[int] = [8, 16, 32, 64, 128]
EPOCHS: int = 100
LEARNING_RATE: float = 0.1  # alpha
# LEARNING_RATE: list[float] = [0.001, 0.01, 0.1, 0.5, 1]

if __name__ == "__main__":
    features, labels = load_multiclass_csv(INPUT_FILE)

    # todo: make also test sets
    x_train, y_train, x_val, y_val = train_val_split(features, labels)

    n_features = features.shape[1]
    n_classes = int(np.max(labels)) + 1

    model = SoftmaxClassifier(n_features, n_classes)

    history = train(
        model=model,
        features_train=x_train,
        labels_train=y_train,
        features_val=x_val,
        labels_val=y_val,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    print("Final train accuracy:", model.accuracy(x_train, y_train))
    print("Final val accuracy:", model.accuracy(x_val, y_val))
