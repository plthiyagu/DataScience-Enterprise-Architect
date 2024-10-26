from scratchml.models.random_forest import RandomForestClassifier, RandomForestRegressor
from scratchml.utils import KFold
from sklearn.datasets import make_regression, make_classification


def example_random_forest_classifier() -> None:
    """
    Practical example of how to use the RandomForestClassifier model.
    """
    # generating a dataset for the classfication task
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=2, shuffle=True)

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating a Random Forest Classifier instance
        rf = RandomForestClassifier(
            n_estimators=100,
            bootstrap=True,
            criterion="gini",
            max_depth=10,
            max_samples=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            verbose=0,
        )

        # fitting the model
        rf.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = rf.score(X=X_test, y=y_test, metric="accuracy")

        print(f"The model achieved an accuracy score of {score} on the fold {fold}.\n")


def example_random_forest_regressor() -> None:
    """
    Practical example of how to use the RandomForestRegressor model.
    """
    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=2000, n_features=4, n_targets=1, shuffle=True, noise=0, coef=False
    )

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating a Random Forest Classifier instance
        rf = RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            criterion="squared_error",
            max_depth=10,
            max_samples=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            verbose=0,
        )

        # fitting the model
        rf.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = rf.score(X=X_test, y=y_test, metric="r_squared")

        print(f"The model achieved a R² score of {score} on the fold {fold}.\n")


if __name__ == "__main__":
    example_random_forest_classifier()
    example_random_forest_regressor()
