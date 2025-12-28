import csv
import numpy as np

INPUT_FILE: str = "../res/multiclass_data.csv"


def load_multiclass_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a multiclass classification dataset from a CSV file.
    The last column is assumed to be the class label.
    Args:
        path (str): Path to the CSV file.
    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,).
    """
    data: list[list[float]] = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])

    data_np: np.ndarray = np.array(data, dtype=np.float64)

    features: np.ndarray = data_np[:, :-1]
    labels: np.ndarray = data_np[:, -1].astype(np.int64)

    return features, labels


def cross_entropy_loss(
        y_true: np.float64,
        y_pred: np.float64
) -> float:
    """
   Compute the categorical cross-entropy loss.
   Args:
       y_true (np.ndarray):
           Ground truth labels encoded as one-hot vectors.
           Shape: (n_samples, n_classes).
       y_pred (np.ndarray):
           Predicted class probabilities (after softmax).
           Shape: (n_samples, n_classes).
   Returns:
       loss (float): Mean cross-entropy loss over all samples.
   """
    epsilon: float = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))
