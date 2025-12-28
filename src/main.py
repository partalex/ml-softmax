import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.shared import load_multiclass_csv, INPUT_FILE, RES_DIR, PLOT_DIR


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
# Convergence estimation
# ============================================================

def estimate_convergence_epoch(
        val_acc: list[float],
        tail: int = 5,
        tol: float = 0.002
) -> int:
    """
    Estimate epoch of convergence as the first epoch where val_acc reaches
    a value close to the final value.

    final_value = mean of last `tail` epochs
    converged when val_acc >= final_value - tol

    Returns:
        int: epoch index (1-based)
    """
    if len(val_acc) == 0:
        return 0

    tail = min(tail, len(val_acc))
    final_value = float(np.mean(val_acc[-tail:]))
    threshold = final_value - tol

    for i, a in enumerate(val_acc):
        if a >= threshold:
            return i + 1  # 1-based

    return len(val_acc)


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
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits: np.ndarray = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, features: np.ndarray) -> np.ndarray:
        logits: np.ndarray = features @ self.W + self.b
        return self._softmax(logits)

    def loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        probs: np.ndarray = self.forward(features)
        n: int = features.shape[0]
        log_likelihood: np.ndarray = -np.log(probs[np.arange(n), labels] + 1e-12)
        return float(np.mean(log_likelihood))

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        probs: np.ndarray = self.forward(features)
        preds: np.ndarray = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))

    def gradients(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n: int = features.shape[0]
        probs: np.ndarray = self.forward(features)

        probs[np.arange(n), labels] -= 1.0
        probs /= n

        dW: np.ndarray = features.T @ probs
        db: np.ndarray = np.sum(probs, axis=0)

        return dW, db

    def update(self, dW: np.ndarray, db: np.ndarray, lr: float) -> None:
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
# Plotting + saving
# ============================================================

def plot_history(
        history: dict[str, list[float]],
        save: bool,
        out_dir: str,
        prefix: str,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    if save:
        os.makedirs(out_dir, exist_ok=True)

    # LOSS
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    if save:
        plt.savefig(os.path.join(out_dir, f"{prefix}_train_loss.png"), dpi=150, bbox_inches="tight")

    # ACCURACY
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(os.path.join(out_dir, f"{prefix}_accuracy.png"), dpi=150, bbox_inches="tight")

    plt.show()


# ============================================================
# Tables (console + CSV)
# ============================================================

def print_float_table(title: str, table: np.ndarray, learning_rates: list[float], batch_sizes: list[int],
                      fmt: str) -> None:
    print("\n" + title)
    header = "lr\\bs | " + " | ".join(f"{bs:>10d}" for bs in batch_sizes)
    print(header)
    print("-" * len(header))
    for i, lr in enumerate(learning_rates):
        row = f"{lr:<5} | " + " | ".join(f"{table[i, j]:{fmt}}" for j in range(len(batch_sizes)))
        print(row)


def print_int_table(title: str, table: np.ndarray, learning_rates: list[float], batch_sizes: list[int]) -> None:
    print("\n" + title)
    header = "lr\\bs | " + " | ".join(f"{bs:>10d}" for bs in batch_sizes)
    print(header)
    print("-" * len(header))
    for i, lr in enumerate(learning_rates):
        row = f"{lr:<5} | " + " | ".join(f"{int(table[i, j]):>10d}" for j in range(len(batch_sizes)))
        print(row)


def save_table_csv(path: str, table: np.ndarray, learning_rates: list[float], batch_sizes: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr\\bs"] + batch_sizes)
        for i, lr in enumerate(learning_rates):
            writer.writerow([lr] + list(table[i, :]))


# ============================================================
# Hyperparameters grid
# ============================================================

BATCH_SIZES: list[int] = [8, 16, 32, 64, 128]
LEARNING_RATES: list[float] = [0.001, 0.01, 0.1, 0.5, 1.0]
EPOCHS: int = 100

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    features, labels = load_multiclass_csv(INPUT_FILE)

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(
        features, labels, val_ratio=0.2, test_ratio=0.1, seed=42
    )

    n_features: int = features.shape[1]
    n_classes: int = int(np.max(labels)) + 1

    # Tables: rows = learning rates, cols = batch sizes
    avg_epoch_time_table = np.zeros((len(LEARNING_RATES), len(BATCH_SIZES)), dtype=np.float64)
    conv_epoch_table = np.zeros((len(LEARNING_RATES), len(BATCH_SIZES)), dtype=np.int64)
    total_time_to_conv_table = np.zeros((len(LEARNING_RATES), len(BATCH_SIZES)), dtype=np.float64)
    max_val_acc_table = np.zeros((len(LEARNING_RATES), len(BATCH_SIZES)), dtype=np.float64)

    best_val_acc: float = -1.0
    best_params: dict[str, float | int] = {"learning_rate": 0.0, "batch_size": 0}

    for i_lr, lr in enumerate(LEARNING_RATES):
        for j_bs, bs in enumerate(BATCH_SIZES):
            print("\n" + "=" * 70)
            print(f"Training: lr={lr}, batch_size={bs}")

            model = SoftmaxClassifier(n_features, n_classes, seed=42)

            history = train(
                model=model,
                features_train=x_train,
                labels_train=y_train,
                features_val=x_val,
                labels_val=y_val,
                learning_rate=lr,
                batch_size=bs,
                epochs=EPOCHS,
                seed=0,
            )

            # average epoch time
            avg_epoch_time = float(np.mean(history["epoch_time"]))
            avg_epoch_time_table[i_lr, j_bs] = avg_epoch_time

            # convergence epoch estimate (based on val_acc reaching near-final value)
            conv_epoch = estimate_convergence_epoch(history["val_acc"], tail=5, tol=0.002)
            conv_epoch_table[i_lr, j_bs] = conv_epoch

            # total time to convergence
            total_time_to_conv = avg_epoch_time * conv_epoch
            total_time_to_conv_table[i_lr, j_bs] = total_time_to_conv

            # max validation accuracy (for selecting best params)
            max_val_acc = float(np.max(history["val_acc"]))
            max_val_acc_table[i_lr, j_bs] = max_val_acc

            if max_val_acc > best_val_acc:
                best_val_acc = max_val_acc
                best_params = {"learning_rate": lr, "batch_size": bs}

            print(
                f"avg_epoch_time={avg_epoch_time:.4f}s | conv_epoch={conv_epoch} | total_to_conv={total_time_to_conv:.2f}s")
            print(f"max_val_acc={max_val_acc:.4f}")

    # ----- Print tables -----
    print_float_table(
        title="Average epoch time (s)  [rows=learning_rate, cols=batch_size]",
        table=avg_epoch_time_table,
        learning_rates=LEARNING_RATES,
        batch_sizes=BATCH_SIZES,
        fmt=">10.4f"
    )

    print_int_table(
        title="Epochs to convergence  [rows=learning_rate, cols=batch_size]",
        table=conv_epoch_table,
        learning_rates=LEARNING_RATES,
        batch_sizes=BATCH_SIZES
    )

    print_float_table(
        title="Total time to convergence (s)  [rows=learning_rate, cols=batch_size]",
        table=total_time_to_conv_table,
        learning_rates=LEARNING_RATES,
        batch_sizes=BATCH_SIZES,
        fmt=">10.2f"
    )

    print_float_table(
        title="Max validation accuracy  [rows=learning_rate, cols=batch_size]",
        table=max_val_acc_table,
        learning_rates=LEARNING_RATES,
        batch_sizes=BATCH_SIZES,
        fmt=">10.4f"
    )

    # ----- Save tables -----
    save_table_csv(f"{RES_DIR}/avg_epoch_time.csv", avg_epoch_time_table, LEARNING_RATES, BATCH_SIZES)
    save_table_csv(f"{RES_DIR}/epochs_to_convergence.csv", conv_epoch_table, LEARNING_RATES, BATCH_SIZES)
    save_table_csv(f"{RES_DIR}/total_time_to_convergence.csv", total_time_to_conv_table, LEARNING_RATES, BATCH_SIZES)
    save_table_csv(f"{RES_DIR}/max_val_acc.csv", max_val_acc_table, LEARNING_RATES, BATCH_SIZES)

    print("\n" + "#" * 70)
    print("BEST CONFIGURATION (by max val accuracy)")
    print(f"learning_rate = {best_params['learning_rate']}")
    print(f"batch_size    = {best_params['batch_size']}")
    print(f"max_val_acc   = {best_val_acc:.4f}")
    print("#" * 70)

    # ----- Final training with best params -----
    best_lr = float(best_params["learning_rate"])
    best_bs = int(best_params["batch_size"])

    final_model = SoftmaxClassifier(n_features, n_classes, seed=42)
    final_history = train(
        model=final_model,
        features_train=x_train,
        labels_train=y_train,
        features_val=x_val,
        labels_val=y_val,
        learning_rate=best_lr,
        batch_size=best_bs,
        epochs=EPOCHS,
        seed=0,
    )

    plot_history(
        final_history,
        save=True,
        out_dir=f"{PLOT_DIR}",
        prefix=f"best_lr{best_lr}_bs{best_bs}"
    )

    print("Final train accuracy:", final_model.accuracy(x_train, y_train))
    print("Final val accuracy:", final_model.accuracy(x_val, y_val))
    print("Final test accuracy:", final_model.accuracy(x_test, y_test))
