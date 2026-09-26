"""
Random Forest model builder.

Current demonstration configuration:
- n_estimators = 200
- max_depth = 5
- min_samples_leaf = 3

Nested-CV search grid:
- max_depth = {5, None}
- min_samples_leaf = {3, 10}
"""
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    preprocessor,
    max_depth=5,
    min_samples_leaf=3
):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        ))
    ])

