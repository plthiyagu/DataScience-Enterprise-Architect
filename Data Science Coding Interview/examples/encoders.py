from scipy.sparse import csr_matrix
from abc import ABC
from typing import List, Union, Any, Tuple
from .utils import convert_array_numpy
import numpy as np
import math


class BaseEncoder(ABC):
    """
    Encoders base class.
    """

    def __init__(self) -> None:
        pass

    def fit(self, *args: np.ndarray) -> None:
        """
        Abstract method to the fit the encoder.
        """

    def transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Abstract method to use the fitted encoder to transform the data.
        """

    def fit_transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Abstract method to fit the encoder and then used it to transform the data.
        """

    def inverse_transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Abstract method to use the fitted encoder to inverse transform the data
        (get the original value).
        """


class LabelEncoder(BaseEncoder):
    """
    Creates a class (inherited from BaseScaler) for the LabelEncoder.
    """

    def __init__(self) -> None:
        """
        Creates a LabelEncoder's instance.
        """
        self.classes_ = None
        self.classes_map_ = None

    def fit(self, *args: np.ndarray) -> None:
        """
        Fits the LabelEncoder.

        Args:
            y (np.array): the classes array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the classes array is expected.\n")

        y = args[0]

        if not isinstance(y, (List, np.ndarray)):
            raise TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        self.classes_ = np.sort(np.unique(y))
        self.classes_map_ = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Using the fitted LabelEncoder to encode the classes.

        Args:
            y (np.array): the classes array.

        Returns:
            y (np.ndarray): the encoded classes array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the classes array is expected.\n")

        y = args[0]

        if not isinstance(y, (List, np.ndarray)):
            raise TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        return np.array([self.classes_map_[v] for v in y])

    def fit_transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Fits the LabelEncoder and then transforms the given set of classes in sequence.

        Args:
            y (np.array): the classes array.

        Returns:
            np.ndarray: the encoded classes array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the classes array is expected.\n")

        y = args[0]

        if not isinstance(y, (List, np.ndarray)):
            raise TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation (converts a encoded
        set of classes to its original values).

        Args:
            y (np.ndarray): the encoded classes array.

        Returns:
            np.ndarray: the original classes array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the classes array is expected.\n")

        y = args[0]

        if not isinstance(y, (List, np.ndarray)):
            raise TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        inverse_classes_map = dict(map(reversed, self.classes_map_.items()))
        return np.array([inverse_classes_map[v] for v in y])


class OneHotEncoder(BaseEncoder):
    """
    Creates a class (inherited from BaseScaler) for the OneHotEncoder.
    """

    def __init__(
        self,
        categories: Union[str, List] = "auto",
        drop: str = None,
        sparse_output: bool = True,
        dtype: Any = np.float64,
        handle_unknown: str = "error",
        min_frequency: Union[int, float] = None,
        max_categories: int = None,
        feature_name_combiner: Union[str, callable] = "concat",
    ) -> None:
        """
        Creates a OneHotEncoder's instance.
        """
        self.categories_ = categories
        self.drop_ = drop
        self.sparse_output_ = sparse_output
        self.dtype_ = dtype
        self.handle_unknown_ = handle_unknown
        self.min_frequency_ = min_frequency
        self.max_categories_ = max_categories
        self.feature_name_combiner = feature_name_combiner
        self.n_features_in_ = None
        self.infrequent_categories_ = None
        self.drop_idx_ = None
        self.categories_map_ = None
        self.infrequents = None
        self._valid_handle_unknown = ["error", "ignore", "infrequent_if_exist"]
        self._valid_drop = ["first", "if_binary"]

    def fit(self, *args: np.ndarray) -> None:
        """
        Fits the OneHotEncoder.

        Args:
            X (np.array): the features array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the features array is expected.\n")

        X = convert_array_numpy(args[0])
        self.n_features_in_ = X.shape[1]
        self.drop_idx_ = np.empty(self.n_features_in_, dtype=object)

        _ordered_categories = []

        # detecting the unique categories within the features
        if self.categories_ == "auto":
            for feature in range(self.n_features_in_):
                _unique_values_feature = np.sort(np.unique(X[:, feature])).tolist()
                _unique_values_feature = np.asarray(
                    _unique_values_feature, dtype=object
                )
                _ordered_categories.append(_unique_values_feature)
        else:
            if isinstance(self.categories_, List):
                for category in self.categories_:
                    _unique_values_feature = np.sort(np.asarray(category)).tolist()
                    _unique_values_feature = np.asarray(
                        _unique_values_feature, dtype=object
                    )
                    _ordered_categories.append(_unique_values_feature)
            else:
                raise TypeError(
                    f"The categories value must be a list or 'auto', got {self.categories_}.\n"
                )

        self.categories_ = _ordered_categories.copy()

        # if the min frequency parameter is specified, then detect
        # the infrequent categories within the features. moreover, we also
        # need to "delete" the infrequent categories and add a "infrequent" category
        if self.min_frequency_ is not None or self.max_categories_ is not None:
            _ordered_categories = self._check_infrequents(X, _ordered_categories)

        self.categories_map_ = {}

        # mapping the categories into integers that will be used as an auxiliary
        # step to one hot encode the categories
        # e.g.: categories = ['male', 'female', 'other'] =>
        # {'female': 0, 'male': 1, 'other': 2'}.
        # note: if the infrequency class exists, then it will be mapped in the last position
        for i in range(self.n_features_in_):
            _features_map = {}

            for j, c in enumerate(_ordered_categories[i]):
                if self.infrequents is not None or not c in self.infrequents[i]:
                    _features_map[c] = j
                else:
                    if c in self.infrequents[i]:
                        _features_map[c] = len(_ordered_categories[i]) - 1

            self.categories_map_[i] = _features_map

    def _check_infrequents(
        self,
        X: np.ndarray,
        categories: List,
    ) -> List:
        """
        Auxiliary function to detect the infrequent categories by
        checking their frequency among the feature.

        Args:
            X (np.ndarray): the features array.
            categories (List): a list with the known categories
                for all the features and that is created during
                the fitting.

        Returns:
            categories (List): the list with the know categories
                filtered by the max categories and min frequency
                parameters (if desired).
        """
        min_frequency = 0
        self.infrequents = {}

        # validating the min frequency value
        if self.min_frequency_ is not None:
            if isinstance(self.min_frequency_, int):
                try:
                    assert self.min_frequency_ > 0
                    min_frequency = self.min_frequency_
                except AssertionError as error:
                    raise ValueError(
                        "The min frequency value must be greater than 0.\n"
                    ) from error
            elif isinstance(self.min_frequency_, float):
                try:
                    assert (self.min_frequency_ > 0) and (self.min_frequency_ < 1)
                    min_frequency = math.floor(
                        self.min_frequency_ * self.n_features_in_
                    )
                except AssertionError as error:
                    raise ValueError(
                        "The min frequency value must be between [0, 1].\n"
                    ) from error
            else:
                raise TypeError(
                    f"The min frequency value must be int or float, got {type(self.min_frequency_)}"
                )

            for feature in range(self.n_features_in_):
                # counting the occurencies of each category within the feature
                unique, counts = np.unique(X[:, feature], return_counts=True)
                counts = np.asarray((unique, counts)).T

                # filtering and mapping the infrequent categories (with number
                #  of occurencies lower than the min frequency)
                infrequent_categories = list(
                    filter(lambda x: x[1] < min_frequency, counts)
                )

                if len(infrequent_categories) > 0:
                    self.infrequents[feature] = [i[0] for i in infrequent_categories]
                else:
                    self.infrequents[feature] = []

            if self.max_categories_ is not None:
                for c, _ in enumerate(categories):
                    if len(self.infrequents[c]) > 0:
                        _temp = list(categories[c])
                        _temp.append("infrequent")
                        categories[c] = np.asarray(_temp, dtype="O")
                        _indexes = [
                            np.where(categories[c] == i) for i in self.infrequents[c]
                        ]
                        categories[c] = np.delete(categories[c], _indexes)

        if self.max_categories_ is not None:
            # validating the max categories value
            try:
                assert self.max_categories_ > 0
            except AssertionError as error:
                raise ValueError(
                    "The max categories value should be greater than 0.\n"
                ) from error

            if self.min_frequency_ is None:
                self.infrequents = {i: [] for i in range(self.n_features_in_)}

            for i in range(self.n_features_in_):
                # checking if the categories within the features exceeded the max value
                if len(categories[i]) > self.max_categories_:
                    # sorting the categories based on its counts
                    unique, counts = np.unique(X[:, i], return_counts=True)
                    counts = np.asarray((unique, counts)).T
                    counts = counts[np.flip(counts[:, 1].argsort())]

                    # selecting only the categories names, which is sorted by
                    # the name of occurencies, to filter
                    features = [f for f, _ in counts]

                    # selecting only the categories that were not mapped into
                    # the infrequent category
                    # e.g.: features = ['male', 'female', 'other'], infrequent = ['other']
                    #       => features - infrequent = ['male', 'female']
                    remaining_categories = np.setdiff1d(
                        features, self.infrequents[i], assume_unique=True
                    )

                    # getting the first max_categories_ - 1 categories
                    # excluding the categories that were mapped into the infrequent
                    # class and including the 'infrequent' category that will be added
                    remaining_categories = remaining_categories[
                        : self.max_categories_ - 1
                    ]

                    # getting the features that surpassed the max categories limits
                    # and adding them to the infrequent category
                    # e.g.: features = ['male', 'female', 'other'], infrequent = ['other']
                    #    => remaining_categories = ['male'] (max = 1)
                    #    => remaining_categories U infrequent = ['male', 'other']
                    #    => new_infrequents = features - remaining_categories U infrequent
                    #    => new_infrequents = ['male', 'female', 'other'] - ['male', 'other']
                    #    => new_infrequents = ['female']
                    new_infrequents = np.setdiff1d(
                        features,
                        np.union1d(remaining_categories, self.infrequents[i]),
                        assume_unique=True,
                    )

                    # updating the infrequents classes with the categories
                    # that were excluded
                    self.infrequents[i].extend(new_infrequents)

                    # sorting the remaining categories and adding the
                    # 'infrequent' category
                    remaining_categories = remaining_categories.tolist()
                    remaining_categories.sort()
                    remaining_categories.append("infrequent")
                    remaining_categories = np.asarray(remaining_categories, dtype="O")
                    categories[i] = remaining_categories

        return categories

    def transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Using the fitted OneHotEncoder to encode the features.

        Args:
            X (np.array): the features array.

        Returns:
            y (np.ndarray): the encoded features array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the features array is expected.\n")

        X = convert_array_numpy(args[0])

        # mapping and transforming the features
        new_X, features_mapping = self._feature_transformation(X)

        # dropping the desired indexes
        new_X = self._drop_indexes(new_X, features_mapping)

        # casting the new array to the desired type
        try:
            new_X.astype(self.dtype_)
        except Exception as e:
            raise e

        # transforming the new array into a sparse matrix
        if self.sparse_output_:
            return csr_matrix(new_X)

        return new_X

    def _feature_transformation(self, X: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Auxiliary function to encode the features and that is used
        during the fitting.

        Args:
            X (np.ndarray): the features array

        Returns:
            Tuple[np.ndarray, List[Tuple]]: the transformed features
                array and the features mapping, respectively.
        """
        new_X = []
        features_mapping = []
        count = 0

        for i in range(self.n_features_in_):
            # getting the features (keys) and its categories (values)
            features_values = list(self.categories_map_[i].values())
            features_keys = list(self.categories_map_[i].keys())

            # detecting the number of categories within the features
            n_values = np.max(features_values) + 1
            encoded_features = []
            is_binary = n_values == 2

            # mapping the features information (this will be used in the
            # drop indexes function)
            # e.g.: [(0, True, 'Male', 0), (0, True, 'Female', 1)]
            #         ^    ^      ^    ^
            #    feature binary cate-  position in the
            #     index  or not  gory   new encoded array
            for ck in features_keys:
                features_mapping.append((i, is_binary, ck, count))
                count += 1

            for v in X[:, i]:
                if self.handle_unknown_ == "ignore":
                    try:
                        _X = np.array([self.categories_map_[i][v]])
                        encoded_features.append(np.eye(n_values)[_X])
                    except KeyError:
                        # creating an array filled with zeros
                        encoded_features.append(np.zeros(n_values))
                elif self.handle_unknown_ == "infrequent_if_exist":
                    # creating an array full of zeros where
                    # the last position if filled with 1
                    if (v in self.infrequents[i]) or (
                        v not in self.categories_map_[i].keys()
                    ):
                        _X = np.array([self.categories_map_[i]["infrequent"]])
                    else:
                        _X = np.array([self.categories_map_[i][v]])

                    encoded_features.append(np.eye(n_values)[_X])
                elif self.handle_unknown_ == "error":
                    try:
                        _X = np.array([self.categories_map_[i][v]])
                        encoded_features.append(np.eye(n_values)[_X])
                    except KeyError as error:
                        raise error

            new_X.append(np.vstack(encoded_features))

        # stacking the encoded features matrixes
        new_X = np.hstack(new_X)
        return new_X, features_mapping

    def _drop_indexes(self, X: np.ndarray, features_mapping: List[Tuple]) -> np.ndarray:
        """
        Auxiliary function to delete the desired indexes.

        Args:
            X (np.ndarray): the features array.
            features_mapping (List[Tuple]): a list containing informations
                about each feature (such as the index, name, if it's binary
                or not, the new start and end indexes) created during the fitting.

        Returns:
            np.ndarray: the new features array.
        """
        # validating the drop parameter value
        if self.drop_ is not None:
            if isinstance(self.drop_, str):
                try:
                    assert self.drop_ in self._valid_drop
                except AssertionError as error:
                    raise ValueError(
                        f"Invalid drop option, must be {self._valid_drop}.\n"
                    ) from error
            else:
                if not isinstance(self.drop_, (List, np.ndarray)):
                    raise TypeError(
                        f"The drop type must be a list or np.ndarray, got {type(self.drop_)}.\n"
                    )

        indexes_to_drop = []
        res = []

        if self.drop_ in ["first", "if_binary"]:
            last_index = -1

            for i, b, n, p in features_mapping:
                if last_index != i:
                    # if the binary option was selected, then detecting whether
                    # the feature is binary or not
                    if (self.drop_ == "if_binary") and (not b):
                        continue

                    # if matches the criteria, adding the index to be dropped
                    indexes_to_drop.append(p)
                    last_index = i
                    res.append((i, b, n, p))
                else:
                    continue

        elif isinstance(self.drop_, (List, np.ndarray)):
            for i, b, n, p in features_mapping:
                # checking if the feature's name is present in the list
                # containing the features that will be dropped
                if n in self.drop_:
                    res.append((i, b, n, p))
                    indexes_to_drop.append(p)

        # deleting the features based on its indexes
        X = np.delete(X, indexes_to_drop, axis=-1)

        # if no features were dropped, then drop_idx_ must
        # be a singular None
        if len(res) == 0:
            self.drop_idx_ = None
        else:
            # otherwise, we must create a list with the same size as the number
            # of features and fill the indexes with the indexes of the categories
            # that were excluded
            # e.g.: [None, 1, None, 0, None] => the category 1 of the index 1 and
            # the category 0 of the feature 3 were deleted
            for i, _, n, _ in res:
                self.drop_idx_[i] = self.categories_map_[i][n]

        return X

    def fit_transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Fits the LabelEncoder and then transforms the given set of classes in sequence.

        Args:
            X (np.array): the features array.

        Returns:
            np.ndarray: the encoded features array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the features array is expected.\n")

        X = convert_array_numpy(args[0])

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, *args: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation (converts a encoded
        set of features to its original values).

        Args:
            X (np.ndarray): the encoded features array.

        Returns:
            np.ndarray: the original features array.
        """
        if len(args) > 1:
            raise RuntimeError("Only the features array is expected.\n")

        X = convert_array_numpy(args[0])

        # FIXME: Think in a better way for validation the shape of array.
        # The commented code bellow only works when drop == None.

        # checking the shape of the array
        # try:
        #     count = 0

        #     for (k, v) in self.categories_map_.items():
        #         for _ in self.categories_map_[k].keys():
        #             count += 1

        #     assert X.shape[1] == count

        # except AssertionError:
        #     raise ArithmeticError("Array's shape is different from what is expected.\n")

        inreverse_mapping = {}
        count = 0
        Xt = []

        # creating an inverse mapping of the features
        # and its categories
        for k, v in self.categories_map_.items():
            _features_categories = {}

            for j in v.keys():
                _features_categories[count] = j
                count += 1

            inreverse_mapping[k] = _features_categories

        for i in range(X.shape[0]):
            if self.handle_unknown_ == "ignore":
                converted = np.array([None] * self.n_features_in_)
            elif self.handle_unknown_ == "infrequent_if_exist":
                converted = np.array(["infrequent"] * self.n_features_in_)

            # checking which position the hot encoded is at for the given
            # feature and mapping it back to the original value
            hot_encoded = np.argwhere(X[i, :] == 1)
            hot_encoded = hot_encoded.reshape(-1)
            converted = []

            for k, v in inreverse_mapping.items():
                indexes = list(v.keys())
                encoded_index = list(set(indexes) & set(hot_encoded))

                if len(encoded_index) == 0:
                    if self.handle_unknown_ == "error":
                        raise KeyError("Value not found during the model training.\n")

                    continue

                converted[k] = inreverse_mapping[k][encoded_index[0]]

            Xt.append(converted)

        Xt = np.asarray(Xt, dtype="O")
        return Xt
