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

    X: np.ndarray = data_np[:, :-1]
    y: np.ndarray = data_np[:, -1].astype(np.int64)

    return X, y
