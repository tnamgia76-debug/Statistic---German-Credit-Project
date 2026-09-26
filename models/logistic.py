"""
Logistic Regression model builder.

Nested-CV search grid:
- C = {0.1, 1, 10}
- class_weight = {None, "balanced"}
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def build_logistic(
    preprocessor,
    C=1.0,
    class_weight=None
):

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        ))
    ])