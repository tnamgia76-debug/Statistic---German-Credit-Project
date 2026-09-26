from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_rbf_svm(
    preprocessor,
    C=1.0,
    gamma="scale"
):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
        ))
    ])