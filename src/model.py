"""Leakage-safe preprocessing, model fitting, evaluation, and prediction."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"


def build_pipeline(X: pd.DataFrame, model: Any | None = None) -> Pipeline:
    """Build preprocessing + estimator in one pipeline so CV cannot leak."""
    X = X.drop(columns=[ID_COLUMN], errors="ignore")
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor),
                     ("classifier", model or LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate target, then explicitly remove identifiers from model inputs."""
    if TARGET not in df:
        raise ValueError(f"Data must contain target column '{TARGET}'.")
    y_raw = df[TARGET].astype(str).str.strip()
    if not y_raw.isin(["Yes", "No"]).all():
        raise ValueError("Churn target must contain only 'Yes' and 'No'.")
    X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    if X.empty:
        raise ValueError("No usable feature columns remain after removing target and customerID.")
    return X, y_raw.map({"No": 0, "Yes": 1}).astype(int).rename(TARGET)


def train_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                   cross_validate: bool = True) -> tuple[Pipeline, dict[str, Any]]:
    """Stratified holdout evaluation; CV is run on training partition only."""
    X, y = split_features_target(df)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Training requires at least two examples of each Churn class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    roc_fpr, roc_tpr, _ = roc_curve(y_test, proba)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, proba)
    labels = [0, 1]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "test_rows": int(len(y_test)),
        "class_counts": {"No": int((y == 0).sum()), "Yes": int((y == 1).sum())},
        "positive_rate": float(y.mean()),
        "model": "Logistic Regression (class_weight=balanced)",
        "threshold": 0.5,
        # Store actual holdout curve points, not scores from the refit model.
        "curves": {
            "roc_fpr": roc_fpr.tolist(),
            "roc_tpr": roc_tpr.tolist(),
            "pr_recall": pr_recall.tolist(),
            "pr_precision": pr_precision.tolist(),
        },
    }
    min_class = int(y_train.value_counts().min())
    folds = min(5, min_class)
    metrics["cv_f1_mean"] = None
    metrics["cv_f1_std"] = None
    if cross_validate and folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(build_pipeline(X_train), X_train, y_train, cv=cv, scoring="f1", n_jobs=None)
        metrics["cv_f1_mean"] = float(scores.mean())
        metrics["cv_f1_std"] = float(scores.std())
        metrics["cv_folds"] = folds
    # Refit on all labeled rows for the saved production artifact after holdout evaluation.
    pipeline.fit(X, y)
    return pipeline, metrics


def predict_churn(pipeline: Pipeline, customer: pd.DataFrame) -> tuple[int, float]:
    """Return predicted label and estimated probability of churn (not certainty)."""
    if not isinstance(customer, pd.DataFrame) or customer.empty:
        raise ValueError("Provide one customer as a non-empty pandas DataFrame.")
    features = customer.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    probability = float(pipeline.predict_proba(features)[:, 1][0])
    prediction = int(pipeline.predict(features)[0])
    return prediction, probability


def risk_category(probability: float) -> str:
    """Coarse action bands for a model score; these are not certainty labels."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if probability < 0.30:
        return "Lower modeled risk"
    if probability < 0.60:
        return "Moderate modeled risk"
    return "Higher modeled risk"
