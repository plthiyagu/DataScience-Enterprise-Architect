from .utils import convert_array_numpy
from .encoders import OneHotEncoder
from typing import Union
import numpy as np


def mean_squared_error(
    y: np.ndarray, y_hat: np.ndarray, derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MSE, respectively.
    """
    if derivative:
        return y_hat - y

    return np.sum((y_hat - y) ** 2) / y.shape[0]


def root_mean_squared_error(
    y: np.ndarray, y_hat: np.ndarray, derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Root Mean Squared Error (RMSE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the RMSE, respectively.
    """
    if derivative:
        raise NotImplementedError

    return np.sqrt(np.sum((y_hat - y) ** 2) / y.shape[0])


def mean_absolute_error(
    y: np.ndarray, y_hat: np.ndarray, derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MAE, respectively.
    """
    if derivative:
        return np.where(y_hat > y, 1, -1) / y.shape[0]

    return np.sum(np.abs(y - y_hat)) / (y.shape[0])


def median_absolute_error(
    y: np.ndarray, y_hat: np.ndarray, derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Median Absolute Error (MedAE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MedAE, respectively.
    """
    if derivative:
        raise NotImplementedError

    return np.median(np.abs(y - y_hat))


def mean_absolute_percentage_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9,
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MAPE, respectively.
    """
    if derivative:
        raise NotImplementedError

    score = np.sum(
        [
            np.abs(y[i] - y_hat[i]) / np.maximum(epsilon, np.abs(y[i]))
            for i in range(y.shape[0])
        ]
    )
    return score / (y.shape[0])


def mean_squared_logarithmic_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9,
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Squared Logarithmic Error (MSLE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MSLE, respectively.
    """
    if derivative:
        raise NotImplementedError

    score = np.sum((np.log(1 + y + epsilon) - np.log(1 + y_hat + epsilon)) ** 2)
    return score / y.shape[0]


def max_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Max Error (ME).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the ME, respectively.
    """
    if derivative:
        raise NotImplementedError

    return np.max(np.abs(y - y_hat))


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> np.float32:
    """
    Calculates the R Squared (R2).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the R2 error.
    """
    # sum of the squared residuals
    u = ((y - y_hat) ** 2).sum()

    # total sum of squares
    v = ((y - y_hat.mean()) ** 2).sum()

    return 1 - (u / v)


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> np.float32:
    """
    Calculates the Accuracy score.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the Accuracy score.
    """
    score = (y == np.squeeze(y_hat)).astype(int)
    return np.sum(score) / y.shape[0]


def precision(y: np.ndarray, y_hat: np.ndarray, average: str = "binary") -> np.float32:
    """
    Calculates the Precision score.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        average (str): how the metric will be calculated.
            Defaults to binary.

    Returns:
        np.float32: the value of the Precision score.
    """
    _valid_averages = ["binary", "micro", "macro", "weighted"]

    # validating the average value
    try:
        assert average in _valid_averages
    except AssertionError as error:
        raise ValueError(
            f"Average should be {_valid_averages}, got {average}.\n"
        ) from error

    if average == "binary":
        tp = np.sum((y == 1) & (y_hat == 1))
        fp = np.sum((y == 0) & (y_hat == 1))
        return tp / (tp + fp)

    if average == "micro":
        # calculate globally by counting the total true positives and false positives
        tp = np.sum((y == 1) & (y_hat == 1))
        fp = np.sum((y == 0) & (y_hat == 1))
        return tp / (tp + fp)

    # if macro, calculate for each label, and find their unweighted mean
    # otherwise, if weighted calculate for each label, and find their weighted mean
    # based on the true positive classes
    unique_classes = np.unique(y)
    precisions = []
    weights = []

    for c in unique_classes:
        _y = np.where(y == c, 1, 0)
        _y_hat = np.where(y_hat == c, 1, 0)
        _tp = np.sum((_y == 1) & (_y_hat == 1))
        _fp = np.sum((_y == 0) & (_y_hat == 1))
        _precision = _tp / (_tp + _fp)
        precisions.append(_precision)
        weights.append(_y.shape[0] / y.shape[0])

    if average == "macro":
        return sum(precisions) / len(precisions)

    if average == "weighted":
        return np.dot(precisions, weights) / len(precisions)


def recall(y: np.ndarray, y_hat: np.ndarray, average: str = "binary") -> np.float32:
    """
    Calculates the Recall score.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        average (str): how the metric will be calculated.
            Defaults to binary.

    Returns:
        np.float32: the value of the Recall score.
    """
    _valid_averages = ["binary", "micro", "macro", "weighted"]

    # validating the average value
    try:
        assert average in _valid_averages
    except AssertionError as error:
        raise ValueError(
            f"Average should be {_valid_averages}, got {average}.\n"
        ) from error

    if average == "binary":
        tp = np.sum((y == 1) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))
        return tp / (tp + fn)

    if average == "micro":
        # calculate globally by counting the total true positives and false positives
        tp = np.sum((y == 1) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))
        return tp / (tp + fn)

    # if macro, calculate for each label, and find their unweighted mean
    # otherwise, if weighted calculate for each label, and find their weighted mean
    # based on the true positive classes
    unique_classes = np.unique(y)
    recalls = []
    weights = []

    for c in unique_classes:
        _y = np.where(y == c, 1, 0)
        _y_hat = np.where(y_hat == c, 1, 0)
        _tp = np.sum((_y == 1) & (_y_hat == 1))
        _fn = np.sum((_y == 1) & (_y_hat == 0))
        _recall = _tp / (_tp + _fn)
        recalls.append(_recall)
        weights.append(_y.shape[0] / y.shape[0])

    if average == "macro":
        return sum(recalls) / len(recalls)

    if average == "weighted":
        return np.dot(recalls, weights) / len(recalls)


def f1_score(y: np.ndarray, y_hat: np.ndarray, average: str = "binary") -> np.float32:
    """
    Calculates the F1-Score (F1).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        average (str): how the metric will be calculated.
            Defaults to binary.

    Returns:
        np.float32: the value of the F1 Score.
    """
    _valid_averages = ["binary", "micro", "macro", "weighted"]

    # validating the average value
    try:
        assert average in _valid_averages
    except AssertionError as error:
        raise ValueError(
            f"Average should be {_valid_averages}, got {average}.\n"
        ) from error

    if average == "binary":
        op = precision(y, y_hat) * recall(y, y_hat)
        div = precision(y, y_hat) + recall(y, y_hat)
        return 2 * (op / div)

    if average == "micro":
        op = precision(y, y_hat, "micro") * recall(y, y_hat, "micro")
        div = precision(y, y_hat, "micro") + recall(y, y_hat, "micro")
        return 2 * (op / div)

    if average == "macro":
        op = precision(y, y_hat, "macro") * recall(y, y_hat, "macro")
        div = precision(y, y_hat, "macro") + recall(y, y_hat, "macro")
        return 2 * (op / div)

    if average == "weighted":
        op = precision(y, y_hat, "weighted") * recall(y, y_hat, "weighted")
        div = precision(y, y_hat, "weighted") + recall(y, y_hat, "weighted")
        return 2 * (op / div)


def confusion_matrix(
    y: np.ndarray, y_hat: np.ndarray, labels: list = None, normalize: bool = None
) -> Union[np.int16, np.float32]:
    """
    Calculates the Confusion Matrix.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        labels (str): list of labels to index the matrix.
            This may be used to reorder or select a subset of labels.
        normalize (bool): normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population. If None,
            confusion matrix will not be normalized. Defaults to None.

    Returns:
        cm (np.int16, np.float32): the confusion matrix.
    """
    _valid_normalizations = ["true", "pred", "all"]

    if labels is not None:
        _labels = convert_array_numpy(labels).reshape(-1)
    else:
        _labels = np.sort(np.unique(y)).reshape(-1)

    # validating the normalize value
    if normalize is not None:
        try:
            assert normalize in _valid_normalizations
        except AssertionError as error:
            raise ValueError(
                f"Normalize value should be {_valid_normalizations}, got {normalize} instead.\n"
            ) from error

    # creating the confusion matrix
    cm = np.zeros((_labels.shape[0], _labels.shape[0]))

    # creating a combination of the labels so we can calculate the metrics
    # e.g.: labels = [0, 1, 2] => combination = [[0, 0], [0, 1], [0, 2], [1, 0], ..., [2, 2]]
    # and then we calculate the metrics for the true class 0 and prediction 1, true class 0
    # and prediction 1, and so on...
    labels_combinations = np.array(np.meshgrid(_labels, _labels)).T.reshape(-1, 2)

    for i, j in labels_combinations:
        _y = np.where(y == i, 1, 0)
        _y_hat = np.where(y_hat == j, 1, 0)

        # filling the index of the true class i and the predicted
        # class j with its occurrencies
        cm[i, j] = _y[(_y_hat == 1) == 1].sum()

    if normalize is None:
        cm = cm.astype(int)
    else:
        if normalize == "pred":
            # divide the counts by the sum of each column
            for j in range(cm.shape[1]):
                cm[:, j] /= cm[:, j].sum()
        elif normalize == "true":
            # divide the counts by the sum of each row
            for i in range(cm.shape[0]):
                cm[i, :] /= cm[i, :].sum()
        elif normalize == "all":
            # divide the counts by the sum of the entire matrix
            cm /= cm.sum()

    return cm


def true_positive_rate(y: np.ndarray, y_hat: np.ndarray) -> np.float32:
    """
    Calculates the True Positive Rate (TPR).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the TPR.
    """
    tp = np.sum((y == 1) & (y_hat == 1))
    fn = np.sum((y == 1) & (y_hat == 0))
    return tp / (tp + fn)


def false_positive_rate(y: np.ndarray, y_hat: np.ndarray) -> np.float32:
    """
    Calculates the False Positive Rate (FPR).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the FPR.
    """
    fp = np.sum((y == 0) & (y_hat == 1))
    tn = np.sum((y == 0) & (y_hat == 0))
    return fp / (fp + tn)


def roc_auc_score(
    y: np.ndarray, y_hat: np.ndarray, average: str = "micro"
) -> np.float32:
    """
    Calculates the False Positive Rate (FPR).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        average (str): how the metric will be calculated.
            Defaults to binary.

    Returns:
        np.float32: the value of the FPR.
    """
    _valid_averages = ["micro", "macro", "weighted"]

    # validating the average value
    try:
        assert average in _valid_averages
    except AssertionError as error:
        raise ValueError(
            f"Average should be {_valid_averages}, got {average}.\n"
        ) from error

    tprs = []
    fprs = []
    n_classes = len(np.unique(y))

    # checking whether its a binary classification or not
    if n_classes == 2:
        # ordering the prediction (y_hat) and the true y
        ordered_indexes = np.argsort(y_hat)[::-1]
        y_hat = y_hat[ordered_indexes]
        y = y[ordered_indexes]

        tpr, fpr = [], []

        unique_y_hat = np.unique(y_hat)[::-1]
        thresholds = [np.inf]
        thresholds.extend(unique_y_hat)

        for threshold in thresholds:
            prediction = (y_hat >= threshold).astype(int)
            _tpr = true_positive_rate(y=y, y_hat=prediction)
            _fpr = false_positive_rate(y=y, y_hat=prediction)

            tpr.append(_tpr)
            fpr.append(_fpr)

        tprs.append(tpr)
        fprs.append(fpr)
    else:
        # transforming the classes array into one hot encoder
        ohe = OneHotEncoder(sparse_output=False)
        one_hot_y = ohe.fit_transform(y.reshape(-1, 1))

        # iterating over the classes (applying the One vs Rest approach)
        for i in range(n_classes):
            _tpr, _fpr = [], []

            # ordering the prediction (y_hat) and the true y
            ordered_indexes = np.argsort(y_hat[:, i])[::-1]
            _y_hat = y_hat[:, i][ordered_indexes]
            _y = one_hot_y[:, i][ordered_indexes]

            unique_y_hat = np.unique(y_hat[:, i])[::-1]
            thresholds = [np.inf]
            thresholds.extend(unique_y_hat)

            for threshold in thresholds:
                _prediction = (_y_hat >= threshold).astype(int)
                _tpr.append(true_positive_rate(y=_y, y_hat=_prediction))
                _fpr.append(false_positive_rate(y=_y, y_hat=_prediction))

            tprs.append(_tpr)
            fprs.append(_fpr)

    aucs = []

    # iterating over the tprs and fprs for each class
    for tpr, fpr in zip(tprs, fprs):
        auc = 0

        # calculating the area under the curve using the trapezodial rule
        for j in range(1, len(tpr)):
            auc += ((fpr[j] - fpr[j - 1]) * (tpr[j] + tpr[j - 1])) / 2

        aucs.append(auc)

    if n_classes == 2:
        return np.mean(aucs)

    # multiplying the auc of each class for its weights (the number of
    # occurrences in data) and then dividing by the size of the data
    if average == "weighted":
        for i in range(n_classes):
            class_occurrences = np.where(y == i, 1, 0).sum()
            aucs[i] = (class_occurrences / y.shape[0]) * aucs[i]

        return np.sum(aucs)

    return np.mean(aucs)
