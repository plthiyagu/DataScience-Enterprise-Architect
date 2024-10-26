import numpy as np

# FIXME: Optimize and improve these distance metrics


def euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean Distance between two points.

    Args:
        x (np.ndarray): the features of the first point.
        y (np.ndarray): the features of the second point.

    Returns:
        distances (np.ndarray): the euclidean distance value.
    """
    distances = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            distances[i, j] = np.sqrt(np.sum(np.square(x[i] - y[j])))

    return distances


def manhattan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Manhattan Distance (MD) between two points.

    Args:
        x (np.ndarray): the features of the first point.
        y (np.ndarray): the features of the second point.

    Returns:
        distances (np.ndarray): the MD value.
    """
    distances = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            distances[i, j] = np.sum(np.abs(x[i] - y[j]))

    return distances


def chebyshev(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Chebyshev Distance (CD) between two points.

    Args:
        x (np.ndarray): the features of the first point.
        y (np.ndarray): the features of the second point.

    Returns:
        distances (np.ndarray): the CD value.
    """
    distances = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            distances[i, j] = np.max(np.abs(x[i] - y[j]))

    return distances


def minkowski(x: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """
    Calculates the Minkowski Distance (MiD) between two points.

    Args:
        x (np.ndarray): the features of the first point.
        y (np.ndarray): the features of the second point.
        p (float): the power parameter of the Minkowski metric.

    Returns:
        distances (np.ndarray): the MiD value.
    """
    distances = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            distances[i, j] = np.sum(np.abs(x[i] - y[j]) ** p) ** (1 / p)

    return distances
