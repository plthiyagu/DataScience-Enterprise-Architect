from typing import Tuple, Union, List
from abc import ABC
from .utils import convert_array_numpy
import numpy as np


class BaseScaler(ABC):
    """
    Scalers base class.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """
        Abstract method to the fit the scaler.
        """

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract method to use the fitted scaler to transform the data.
        """

    def fit_transform(self, X: np.ndarray, y: np.array) -> np.ndarray:
        """
        Abstract method to fit the scaler and then used it to transform the data.
        """

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract method to use the fitted scaler to inverse transform the data
        (get the original value).
        """


class MinMaxScaler(BaseScaler):
    """
    Creates a class (inherited from BaseScaler) for the MinMaxScaler.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        copy: bool = True,
        clip: bool = False,
    ) -> None:
        """
        Creates a MinMaxScaler's instance.

        Args:
            feature_range (Tuple[float, float], optional): the range of
                the new values after applying the scaler. Defaults to (0, 1).
            copy (bool, optional): whether to create a copy of the
                transformed values or not. Defaults to True.
            clip (bool, optional): whether to clip the scaler's
                output to be within the feature range or not. Defaults to False.
        """
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.feature_names_in_ = None

    def fit(self, X: Union[np.ndarray, List], y: np.array = None) -> None:
        """
        Fits the MinMaxScaler.

        Args:
            X (Union[np.ndarray, List]): the features array.
            y (np.array, optional): the targets array (will be
                ignore). Defaults to None.
        """
        X = convert_array_numpy(X)
        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.data_max_ = X.max(axis=0)
        self.data_min_ = X.min(axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max_ - self.data_min_
        )
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.data_range_ = self.data_max_ - self.data_min_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Using the fitted MinMaxScaler to transform a given set of features.

        Args:
            X (np.ndarray): the features array.

        Returns:
            X | X_scaled (np.ndarray): the new transformed features.
        """
        # transforming the features set
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = (
            X_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )

        # clipping the scaler's output
        if self.clip:
            X_scaled = np.clip(
                a=X_scaled, a_min=self.feature_range[0], a_max=self.feature_range[1]
            )

        if self.copy:
            X = X_scaled.copy()
            return X

        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation (converts a transformed
        set of features to its original values).

        Args:
            X (np.ndarray): the transformed features array.

        Returns:
            Xt (np.ndarray): the original features array.
        """
        X = convert_array_numpy(X)

        Xt = (X - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        Xt *= self.data_max_ - self.data_min_
        Xt += self.data_min_
        return Xt

    def fit_transform(self, X: np.ndarray, y: np.array = None) -> np.ndarray:
        """
        Fits the MinMaxScaler and then transforms the given set of features in sequence.

        Args:
            X (np.ndarray): the features array.
            y (np.array, optional): the targets array (will be ignored). Defaults to None.

        Returns:
            np.ndarray: the new transformed features.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X)


class StandardScaler(BaseScaler):
    """
    Creates a class (inherited from BaseScaler) for the StandardScaler.
    """

    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ) -> None:
        """
        Creates a StandardScaler's instance.

        Args:
            copy (bool, optional): whether to create a copy of the
                transformed values or not. Defaults to True.
            with_mean (bool, optional): whether to use the mean or not.
                Defaults to True.
            with_std (bool, optional): whether to use the standard
                deviation or not. Defaults to True.
        """
        self.copy = copy
        self.with_std = with_std
        self.with_mean = with_mean
        self.scale_ = None
        self.mean_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.std = None

    def fit(self, X: Union[np.ndarray, List], y: np.array = None) -> None:
        """
        Fits the StandardScaler.

        Args:
            X (Union[np.ndarray, List]): the features array.
            y (np.array, optional): the targets array (will be
                ignore). Defaults to None.
        """
        X = convert_array_numpy(X)

        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]

        if not self.with_mean and not self.with_std:
            self.var_ = None
            self.mean_ = None
        else:
            # calculating the variance and the mean of
            # the given features
            self.var_ = X.var(axis=0)
            self.mean_ = X.mean(axis=0)

        # calculating the mean of the given features
        if not self.with_mean:
            self.mean_ = None
        else:
            self.mean_ = X.mean(axis=0)

        if not self.with_std:
            self.scale_ = None
            self.var_ = None
        else:
            # calculating the standard deviation, the variance
            # and the scale of the given features
            self.std = np.std(X, axis=0)
            self.var_[self.var_ == 0] = 1
            self.scale_ = np.sqrt(self.var_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Using the fitted StandardScaler to transform a given set of features.

        Args:
            X (np.ndarray): the features array.

        Returns:
            X_ | X (np.ndarray): the new transformed features.
        """
        X = convert_array_numpy(X)

        if not self.with_std:
            self.std = 1

        if not self.with_mean:
            X_ = X / self.std
        else:
            X_ = (X - self.mean_) / self.std

        if self.copy:
            return X_

        X = X_.copy()
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation (converts a transformed
        set of features to its original values).

        Args:
            X (np.ndarray): the transformed features array.

        Returns:
            Xt (np.ndarray): the original features array.
        """
        X = convert_array_numpy(X)

        if not self.with_std:
            self.std = 1

        if not self.with_mean:
            Xt = X * self.std
        else:
            Xt = (X * self.std) + self.mean_

        return Xt

    def fit_transform(self, X: np.ndarray, y: np.array = None) -> np.ndarray:
        """
        Fits the StandardScaler and then transforms the given set of features in sequence.

        Args:
            X (np.ndarray): the features array.
            y (np.array, optional): the targets array (will be ignored). Defaults to None.

        Returns:
            np.ndarray: the new transformed features.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X)
