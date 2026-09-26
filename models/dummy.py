from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier


def build_dummy(preprocessor):

    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", DummyClassifier(
            strategy="prior"
        ))
    ])

