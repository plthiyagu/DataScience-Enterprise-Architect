from scratchml.models.naive_bayes import GaussianNB
from scratchml.utils import train_test_split
from sklearn.datasets import make_classification


def example_gaussian_naive_bayes() -> None:
    """
    Practical example of how to use the Gaussian Naive Bayes model.
    """
    # generating a dataset for the classfication task
    X, y = make_classification(
        n_samples=10000,
        n_features=10,
        n_classes=9,
        n_clusters_per_class=1,
        n_informative=7,
        n_redundant=2,
        n_repeated=0,
        shuffle=True,
    )

    # creating a Gaussian Naive Bayes model
    gnb = GaussianNB(priors=None, var_smoothing=1e-09)

    # splitting the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=True, stratify=False
    )

    # fitting the model
    gnb.fit(X_train, y_train)

    # assessing the model's performance
    score = gnb.score(X=X_test, y=y_test, metric="accuracy")

    print(f"The model achieved an accuracy score of {score}.\n")


if __name__ == "__main__":
    example_gaussian_naive_bayes()
