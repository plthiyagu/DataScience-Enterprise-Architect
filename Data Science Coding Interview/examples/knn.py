from scratchml.models.knn import KNNClassifier, KNNRegressor
from scratchml.utils import KFold
from sklearn.datasets import make_blobs, make_regression


def example_knn_classifier() -> None:
    """
    Practical example of how to use the KNNClassifier model.
    """
    # generating a dataset for the classfication task
    X, y = make_blobs(n_samples=2000, n_features=4, centers=3, shuffle=True)

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating a KNN Classifier instance
        knn = KNNClassifier(
            n_neighbors=5, weights="uniform", p=2, metric="minkowski", n_jobs=None
        )

        # fitting the model
        knn.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = knn.score(X=X_test, y=y_test, metric="accuracy")

        print(f"The model achieved an accuracy score of {score} on the fold {fold}.\n")


def example_knn_regressor() -> None:
    """
    Practical example of how to use the KNNRegressor model.
    """
    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=10000, n_features=5, n_targets=1, shuffle=True, noise=30
    )

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating a KNN Regressor instance
        knn = KNNRegressor(
            n_neighbors=5, weights="uniform", p=2, metric="minkowski", n_jobs=None
        )

        # fitting the model
        knn.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = knn.score(X=X_test, y=y_test, metric="r_squared")

        print(f"The model achieved a R² score of {score} on the fold {fold}.\n")


if __name__ == "__main__":
    example_knn_classifier()
    example_knn_regressor()
