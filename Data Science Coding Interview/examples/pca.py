from scratchml.models.pca import PCA
from sklearn.datasets import make_blobs


def example_pca() -> None:
    """
    Practical example of how to use the Principal Component Analysis (PCA) model.
    """
    # generating a dataset for the classfication task
    X, _ = make_blobs(n_samples=10000, n_features=8, centers=3, shuffle=True)

    # creating a PCA instance
    pca = PCA(n_components=2)

    pca.fit(X)

    X_transformed = pca.transform(X)

    print(f"X transformed: {X_transformed}")

    X_original = pca.inverse_transform(X_transformed)

    print(f"Inverse of X transformed: {X_original}")

    print(f"Covariance: {pca.get_covariance()}")

    print(f"Precision: {pca.get_precision()}")


if __name__ == "__main__":
    example_pca()
