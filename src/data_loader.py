"""Loading and validating the public IBM Telco Customer Churn CSV."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REQUIRED_COLUMNS = {"customerID", "Churn", "TotalCharges"}


class DataFileMissingError(FileNotFoundError):
    """Raised when the user has not placed the public dataset in data/."""


def load_telco_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the dataset, preserving blank TotalCharges values as missing.

    Download the IBM Telco Customer Churn CSV and save it at ``path``. The
    canonical file name is WA_Fn-UseC_-Telco-Customer-Churn.csv.
    """
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise DataFileMissingError(
            f"Telco churn data was not found at: {csv_path}\n"
            "Download the public IBM Telco Customer Churn CSV (see README.md), "
            "then save it as data/WA_Fn-UseC_-Telco-Customer-Churn.csv, or "
            "pass its path with --data /path/to/file.csv."
        )
    try:
        df = pd.read_csv(csv_path, na_values=["", " ", "NA", "N/A"])
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read CSV at {csv_path}: {exc}") from exc
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].astype("string").str.strip()
    if df.empty:
        raise ValueError(f"CSV contains no customer rows: {csv_path}")
    if not df["Churn"].isin(["Yes", "No"]).all():
        unexpected = sorted(df.loc[~df["Churn"].isin(["Yes", "No"]), "Churn"].dropna().unique().tolist())
        raise ValueError(f"Churn must contain only 'Yes'/'No'; unexpected values: {unexpected}")
    return df
