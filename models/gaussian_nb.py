from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB


def build_gaussian_nb(
    preprocessor,
    var_smoothing=1e-9
):

    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", GaussianNB(
            var_smoothing=var_smoothing
        ))
    ])