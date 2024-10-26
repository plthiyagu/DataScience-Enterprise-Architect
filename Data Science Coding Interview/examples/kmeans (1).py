from scratchml.models.kmeans import KMeans
from sklearn.datasets import make_blobs


def example_kmeans() -> None:
    """
    Practical example of how to use the KMeans model.
    """
    # generating a dataset for the classfication task
    X, y = make_blobs(n_samples=10000, n_features=8, centers=3, shuffle=True)

    # creating a KMeans model for classification
    kmeans = KMeans(
        n_init=5, n_clusters=3, max_iter=100, tol=0.0001, verbose=1, n_jobs=None
    )

    kmeans.fit(X, y)

    score = kmeans.score()

    print(f"The model achieved a negative inertia score of {score}.\n")


if __name__ == "__main__":
    example_kmeans()
