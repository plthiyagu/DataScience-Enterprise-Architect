import numpy as np


def linear(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Applies the Linear activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        np.ndarray: the output of the linear function
            for the given numpy array.
    """
    if derivative:
        return np.ones_like(x)

    return x


def relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Applies the ReLU activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        np.ndarray: the output of the ReLu function
            for the given numpy array.
    """
    if derivative:
        return np.where(x <= 0, 0, 1)

    return np.maximum(0, x)


def elu(
    x: np.ndarray, derivative: bool = False, epsilon: np.float32 = 1e-9
) -> np.ndarray:
    """
    Applies the ELU activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the ELU function
            for the given numpy array.
    """
    if derivative:
        return np.where(x <= 0, elu(x) + 1.0, 1)

    return np.where(x <= 0, (np.exp(x + epsilon) - 1), x)


def leaky_relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Applies the Leaky ReLU activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        np.ndarray: the output of the Leaky ReLu function
            for the given numpy array.
    """
    if derivative:
        return np.where(x <= 0, 0.001, 1)

    return np.where(x <= 0, 0.001 * x, x)


def tanh(
    x: np.ndarray, derivative: bool = False, epsilon: np.float32 = 1e-9
) -> np.ndarray:
    """
    Applies the TanH (Hyperbolic tangent) activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the TanH function
            for the given numpy array.
    """
    if derivative:
        return 1.0 - np.square(tanh(x))

    e_x = np.exp(x + epsilon)
    negative_e_x = np.exp((-1.0 * x) + epsilon)
    return (e_x - negative_e_x) / (e_x + negative_e_x)


def sigmoid(
    x: np.ndarray, derivative: bool = False, epsilon: np.float32 = 1e-9
) -> np.ndarray:
    """
    Applies the Sigmoid activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the sigmoid function
            for the given numpy array.
    """
    if derivative:
        return sigmoid(x) * (1.0 - sigmoid(x))

    return 1.0 / (1.0 + np.exp(-1 * x + epsilon))


def softmax(
    x: np.ndarray, derivative: bool = False, epsilon: np.float32 = 1e-9
) -> np.ndarray:
    """
    Applies the Softmax activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the softmax function
            for the given numpy array.
    """
    if derivative:
        p = softmax(x)
        return p * (1 - p)

    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True) + epsilon)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def softplus(
    x: np.ndarray, derivative: bool = False, epsilon: np.float32 = 1e-9
) -> np.ndarray:
    """
    Applies the SoftPlus activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the softplus function
            for the given numpy array.
    """
    if derivative:
        return 1 / (1 + np.exp(-x + epsilon))

    return np.log(1 + np.exp(x + epsilon))


def selu(
    x: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9,
    alpha: np.float32 = 1.6732632423543772848170429916717,
    scale: np.float32 = 1.0507009873554804934193349852946,
) -> np.ndarray:
    """
    Applies the SELU activation function.

    Args:
        x (np.ndarray): the features array.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the selu function
            for the given numpy array.
    """
    if derivative:
        return scale * np.where(x > 0.0, 1, alpha * np.exp(x + epsilon))

    return scale * np.where(x >= 0.0, x, alpha * (np.exp(x + epsilon) - 1))
