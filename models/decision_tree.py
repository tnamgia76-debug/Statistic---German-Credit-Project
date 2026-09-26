from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(
    preprocessor,
    max_depth=3,
    min_samples_leaf=5
):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),

        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
    ])