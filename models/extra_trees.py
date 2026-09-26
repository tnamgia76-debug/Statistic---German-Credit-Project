from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier


def build_extra_trees(
    preprocessor,
    max_depth=5,
    min_samples_leaf=3
):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", ExtraTreesClassifier(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        ))
    ])