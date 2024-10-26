import numpy as np


def binary_cross_entropy(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9,
) -> np.ndarray:
    """
    Applies the Binary Cross Entropy (BCE) loss function.

    Args:
        y (np.ndarray): the true targets.
        y_hat (np.ndarray): the predicted targets.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the loss function with respect
            to the real targets and the predicted targets.
    """
    if derivative:
        return y_hat - y

    y1 = y * np.log(y_hat + epsilon)
    y2 = (1 - y) * np.log(1 - y_hat + epsilon)
    return (-1 * (1 / y.shape[0])) * np.sum(y1, y2)


def cross_entropy(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9,
) -> np.ndarray:
    """
    Applies the Cross Entropy (CE) loss function.

    Args:
        y (np.ndarray): the true targets.
        y_hat (np.ndarray): the predicted targets.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the loss function with respect
            to the real targets and the predicted targets.
    """
    if derivative:
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -(y / y_hat) + (1 - y) / (1 - y_hat)

    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    y1 = -y * np.log(y_hat + epsilon)
    y2 = (1 - y) * np.log(1 - y_hat + epsilon)
    return y1 - y2
