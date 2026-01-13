import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    root_mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


def build_preprocessor(df, feature_cols):
    num_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    cat_cols = list(set(feature_cols) - set(num_cols))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor


def ml_agent(state):
    print("🤖 ML Agent started...")

    df = state["df"]
    target = state["target"]
    task_type = state["task_type"]
    feature_cols = state["feature_cols"]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
    )

    preprocessor = build_preprocessor(df, feature_cols)

    models = {}
    results = {}

    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced"
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            ),
            "GradientBoosting": GradientBoostingClassifier(random_state=42)
        }

    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted")
            }

            if len(np.unique(y_test)) == 2:
                try:
                    proba = pipeline.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_test, proba)
                except:
                    pass

        else:
            metrics = {
                "rmse": root_mean_squared_error(y_test, preds),
                "r2": r2_score(y_test, preds)
            }

        results[name] = metrics

    # Select best model
    if task_type == "classification":
        best_model = max(results, key=lambda x: results[x]["f1"])
    else:
        best_model = min(results, key=lambda x: results[x]["rmse"])

    state.update({
        "model_results": results,
        "best_model": best_model
    })

    print(" ML training completed")
    print(f" Best model: {best_model}")

    return state
