"""Exhaustively tune every configured hyperparameter combination for each model."""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import app


OUTPUT_PATH = Path(__file__).parent / "data" / "best_hyperparameters.csv"


SEARCHES = {
    "Random Forest": (
        RandomForestClassifier(random_state=app.RANDOM_STATE, n_jobs=1),
        {
            "model__n_estimators": [250, 400],
            "model__max_depth": [10, 16, None],
            "model__min_samples_split": [2, 4],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt"],
            "model__max_leaf_nodes": [None, 100],
            "smote__k_neighbors": [3, 5],
        },
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=app.RANDOM_STATE),
        {
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__criterion": ["gini", "entropy"],
            "model__max_leaf_nodes": [None, 50, 100],
            "smote__k_neighbors": [3, 5],
        },
    ),
    "Logistic Regression": (
        LogisticRegression(random_state=app.RANDOM_STATE),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__solver": ["lbfgs", "newton-cg"],
            "model__class_weight": [None, "balanced"],
            "model__tol": [0.0001, 0.001],
            "model__max_iter": [1000, 3000],
            "smote__k_neighbors": [3, 5],
        },
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
            "model__leaf_size": [20, 30, 40],
            "smote__k_neighbors": [3, 5],
        },
    ),
}


def tune_models() -> pd.DataFrame:
    data = app.load_data()
    features = app.get_model_features(data)
    target = data[app.TARGET]
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=app.RANDOM_STATE,
        stratify=target,
    )

    results = []
    for name, (model, parameter_grid) in SEARCHES.items():
        pipeline = Pipeline(
            [
                ("preprocess", app.make_preprocessor(features)),
                ("smote", SMOTE(random_state=app.RANDOM_STATE)),
                ("model", model),
            ]
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
            refit=False,
        )
        search.fit(x_train, y_train)
        result = {
            "Model": name,
            "Combinations tested": len(ParameterGrid(parameter_grid)),
            "Best CV macro F1": search.best_score_,
        }
        result.update(
            {
                parameter.removeprefix("model__").removeprefix("smote__"): value
                for parameter, value in search.best_params_.items()
            }
        )
        results.append(result)

    result_data = pd.DataFrame(results)
    result_data.to_csv(OUTPUT_PATH, index=False)
    print(result_data.to_string(index=False))
    print(f"Saved best parameters to {OUTPUT_PATH}")
    return result_data


if __name__ == "__main__":
    tune_models()
