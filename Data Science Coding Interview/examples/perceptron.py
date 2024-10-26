from scratchml.models.perceptron import Perceptron
from scratchml.utils import KFold
from sklearn.datasets import make_classification


def example_perceptron() -> None:
    """
    Practical example of how to use the Perceptron model.
    """
    # generating a dataset for the classfication set
    X, y = make_classification(
        n_samples=10000,
        n_features=5,
        n_classes=2,
        n_clusters_per_class=1,
        n_informative=2,
        n_redundant=1,
        n_repeated=0,
        shuffle=True,
    )

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating a logistic regression model
        perceptron = Perceptron(
            penalty=None,
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        # fitting the model
        perceptron.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = perceptron.score(X=X_test, y=y_test, metric="accuracy")

        print(f"The model achieved an accuracy score of {score} on the fold {fold}.\n")


if __name__ == "__main__":
    example_perceptron()
