import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.shared import load_multiclass_csv, INPUT_FILE


# ============================================================
# Data utilities
# ============================================================

def train_val_test_split(
        features: np.ndarray,
        labels: np.ndarray,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Label vector of shape (n_samples,).
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for test.
        seed (int): Random seed.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and their sum < 1.0")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))

    n_total = len(features)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return (
        features[train_idx], labels[train_idx],
        features[val_idx], labels[val_idx],
        features[test_idx], labels[test_idx],
    )


# ============================================================
# Softmax classifier
# ============================================================

class SoftmaxClassifier:
    def __init__(self, n_features: int, n_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W: np.ndarray = 0.01 * rng.standard_normal((n_features, n_classes))
        self.b: np.ndarray = np.zeros(n_classes, dtype=np.float64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Compute the softmax probabilities.

        Args:
            logits (np.ndarray): Logits of shape (n_samples, n_classes).


            np.ndarray: Softmax probabilities of shape (n_samples, n_classes).
        """
        logits = logits - np.max(logits, axis=1, keepdims=True)  # numeric stability
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
        Compute the cross-entropy loss for integer class labels.

        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): True labels (class indices) of shape (n_samples,).

        Returns:
            float: Mean cross-entropy loss.
        """
        probs: np.ndarray = self.forward(features)
        n: int = features.shape[0]
        # add epsilon for numeric stability
        log_likelihood: np.ndarray = -np.log(probs[np.arange(n), labels] + 1e-12)
        return float(np.mean(log_likelihood))

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the accuracy of the model.

        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): True labels (class indices) of shape (n_samples,).

        Returns:
            float: Accuracy in [0, 1].
        """
        probs: np.ndarray = self.forward(features)
        preds: np.ndarray = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))

    def gradients(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of the loss w.r.t. W and b.

        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): True labels (class indices) of shape (n_samples,).

        Returns:
            (dW, db): Gradients for weights and biases.
        """
        n: int = features.shape[0]
        probs: np.ndarray = self.forward(features)

        # dL/dlogits = probs - one_hot(labels)
        probs[np.arange(n), labels] -= 1.0
        probs /= n

        dW: np.ndarray = features.T @ probs
        db: np.ndarray = np.sum(probs, axis=0)

        return dW, db

    def update(self, dW: np.ndarray, db: np.ndarray, lr: float) -> None:
        """
        Update parameters using gradient descent.

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

    Returns:
        history dict with keys:
            train_loss, train_acc, val_acc, epoch_time
    """
    rng = np.random.default_rng(seed)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
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
        train_acc: float = model.accuracy(features_train, labels_train)
        val_acc: float = model.accuracy(features_val, labels_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch + 1:3d} | loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | time={epoch_time:.3f}s"
        )

    return history


# ============================================================
# Plotting
# ============================================================

def plot_history(
        history: dict[str, list[float]],
        out_dir: str,
        prefix: str,
        save: bool = False,
) -> None:
    """
    Plot training history and optionally save plots to disk.

    Args:
        history (dict[str, list[float]]): Training history.
        save (bool): Whether to save plots.
        out_dir (str): Output directory for plots.
        prefix (str): Filename prefix.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    if save:
        os.makedirs(out_dir, exist_ok=True)

    # ---------- LOSS ----------
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    if save:
        plt.savefig(
            os.path.join(out_dir, f"{prefix}_train_loss.png"),
            dpi=150,
            bbox_inches="tight"
        )

    # ---------- ACCURACY ----------
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(
            os.path.join(out_dir, f"{prefix}_accuracy.png"),
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# Hyperparameters
# ============================================================

# BATCH_SIZE: int = 32
# EPOCHS: int = 100
# LEARNING_RATE: float = 0.1
BATCH_SIZES: list[int] = [8, 16, 32, 64, 128]
LEARNING_RATES: list[float] = [0.001, 0.01, 0.1, 0.5, 1]
EPOCHS: int = 100

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    features, labels = load_multiclass_csv(INPUT_FILE)

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(
        features, labels, val_ratio=0.2, test_ratio=0.1, seed=42
    )

    n_features = features.shape[1]
    n_classes = int(np.max(labels)) + 1

    model = SoftmaxClassifier(n_features, n_classes, seed=42)

    best_val_acc: float = 0.0
    best_params: dict[str, float | int] = {}
    results: list[dict[str, float | int]] = []

    for batch_size in BATCH_SIZES:
        for lr in LEARNING_RATES:
            print("=" * 60)
            print(f"Training with batch_size={batch_size}, learning_rate={lr}")

            model = SoftmaxClassifier(n_features, n_classes, seed=42)

            history = train(
                model=model,
                features_train=x_train,
                labels_train=y_train,
                features_val=x_val,
                labels_val=y_val,
                learning_rate=lr,
                batch_size=batch_size,
                epochs=EPOCHS,
                seed=0,
            )
            plot_history(
                history,
                out_dir="../out/graph",
                prefix=f"res_lr{lr}_bs{batch_size}",
                save=True,
            )

            max_val_acc = max(history["val_acc"])

            results.append({"batch_size": batch_size, "learning_rate": lr, "max_val_acc": max_val_acc})

            if max_val_acc > best_val_acc:
                best_val_acc = max_val_acc
                best_params = {"batch_size": batch_size, "learning_rate": lr}

            print(f"Finished: batch_size={batch_size}, lr={lr} | best_val_acc={max_val_acc:.4f}")

    print("\n" + "#" * 60)
    print("BEST HYPERPARAMETERS FOUND")
    print(f"Batch size     : {best_params['batch_size']}")
    print(f"Learning rate  : {best_params['learning_rate']}")
    print(f"Best val acc   : {best_val_acc:.4f}")
    print("#" * 60)
