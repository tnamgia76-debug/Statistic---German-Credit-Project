from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


def build_knn(
    preprocessor,
    n_neighbors=5,
    weights="uniform"
):

    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        ))
    ])