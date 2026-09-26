from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


def build_gradient_boosting(
    preprocessor,
    learning_rate=0.05,
    max_depth=1
):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", GradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ))
    ])