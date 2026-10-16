import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model import build_pipeline, predict_churn, risk_category, split_features_target, train_evaluate


def test_split_excludes_identifier_and_maps_target(synthetic_telco):
    X, y = split_features_target(synthetic_telco)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert set(y.unique()) == {0, 1}


def test_preprocessing_handles_missing_and_unseen_category(synthetic_telco):
    X, y = split_features_target(synthetic_telco)
    X.loc[0, "TotalCharges"] = np.nan
    pipe = build_pipeline(X).fit(X.iloc[1:], y.iloc[1:])
    customer = X.iloc[[0]].copy()
    customer["Contract"] = "Future contract"
    prediction, probability = predict_churn(pipe, customer)
    assert prediction in (0, 1)
    assert 0 <= probability <= 1


def test_training_evaluation_and_refit(synthetic_telco):
    pipeline, metrics = train_evaluate(synthetic_telco, cross_validate=True)
    assert isinstance(pipeline, Pipeline)
    assert 0 <= metrics["accuracy"] <= 1
    assert len(metrics["confusion_matrix"]) == 2
    assert metrics["roc_auc"] is not None
    assert metrics["cv_f1_mean"] is not None
    assert len(metrics["curves"]["roc_fpr"]) > 1
    assert len(metrics["curves"]["pr_recall"]) > 1
    assert "customerID" not in pipeline.feature_names_in_


def test_invalid_target_and_risk_probability(synthetic_telco):
    bad = synthetic_telco.copy()
    bad.loc[0, "Churn"] = "unknown"
    with pytest.raises(ValueError):
        split_features_target(bad)
    with pytest.raises(ValueError):
        risk_category(1.1)
    assert risk_category(.2) == "Lower modeled risk"
