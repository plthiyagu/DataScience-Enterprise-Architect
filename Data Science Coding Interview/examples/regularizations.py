import numpy as np


def l1(
    weights: np.array, reg_lambda: float = 1e-02, derivative: bool = False
) -> np.ndarray:
    """
    Applies the L1 Regularization technique (also called Lasso Regression)
    to avoid overfit towards better generalization.

    Args:
        weights (np.array): the model's weights.
        reg_lambda (float, optional): the lambda regularization
            factor. Defaults to 1e-02.
        derivative (bool, optional): whether to use its derivative
            function or not. Defaults to False.

    Returns:
        np.ndarray: the regularized weights.
    """
    if derivative:
        return reg_lambda * np.sign(weights)

    return reg_lambda * np.sum(np.abs(weights))


def l2(
    weights: np.array, reg_lambda: float = 1e-02, derivative: bool = False
) -> np.ndarray:
    """
    Applies the L2 Regularization technique (also called Ridge Regression)
    to avoid overfit towards better generalization.

    Args:
        weights (np.array): the model's weights.
        reg_lambda (float, optional): the lambda regularization
            factor. Defaults to 1e-02.
        derivative (bool, optional): whether to use its derivative
            function or not. Defaults to False.

    Returns:
        np.ndarray: the regularized weights.
    """
    if derivative:
        return 2 * reg_lambda * weights

    return reg_lambda * np.sum(weights**2)
