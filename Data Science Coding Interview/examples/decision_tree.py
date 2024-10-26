from scratchml.models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from scratchml.utils import KFold
from sklearn.datasets import make_regression, make_classification


def example_decision_tree_classifier() -> None:
    """
    Practical example of how to use the DecisionTreeClassifier model.
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

        # creating a Decision Tree Classifier instance
        dt = DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            verbose=0,
        )

        # fitting the model
        dt.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = dt.score(X=X_test, y=y_test, metric="accuracy")

        print(f"The model achieved an accuracy score of {score} on the fold {fold}.\n")


def example_decision_tree_regressor() -> None:
    """
    Practical example of how to use the DecisionTreeRegressor model.
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

        # creating a Decision Tree Regressor instance
        dt = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            verbose=0,
        )

        # fitting the model
        dt.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = dt.score(X=X_test, y=y_test, metric="r_squared")

        print(f"The model achieved a R² score of {score} on the fold {fold}.\n")


if __name__ == "__main__":
    example_decision_tree_classifier()
    example_decision_tree_regressor()
