import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_telco():
    """Small deterministic test-only fixture; never presented as real data."""
    rng = np.random.default_rng(7)
    n = 80
    churn = np.array(["No"] * 56 + ["Yes"] * 24)
    rng.shuffle(churn)
    return pd.DataFrame({
        "customerID": [f"test-{i}" for i in range(n)],
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(20, 8000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Churn": churn,
    })
