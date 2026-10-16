import pandas as pd
import pytest

from src.data_loader import DataFileMissingError, load_telco_data


def test_loader_converts_blank_total_charges_to_missing(tmp_path):
    path = tmp_path / "telco.csv"
    pd.DataFrame({"customerID": ["a", "b"], "TotalCharges": [" ", "42.5"],
                  "Churn": ["No", "Yes"]}).to_csv(path, index=False)
    df = load_telco_data(path)
    assert pd.isna(df.loc[0, "TotalCharges"])
    assert df.loc[1, "TotalCharges"] == 42.5


def test_missing_file_has_actionable_message(tmp_path):
    with pytest.raises(DataFileMissingError, match="Download the public IBM"):
        load_telco_data(tmp_path / "missing.csv")
