"""Train/evaluate the churn baseline and save a deployable joblib artifact."""
import argparse
import json
from pathlib import Path

import joblib

from src.data_loader import DEFAULT_DATA_PATH, load_telco_data
from src.model import train_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to IBM Telco CSV")
    parser.add_argument("--output", type=Path, default=Path("models/churn_pipeline.joblib"), help="Artifact path")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()
    df = load_telco_data(args.data)
    pipeline, metrics = train_evaluate(df, cross_validate=not args.no_cv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics, "feature_columns": [
        c for c in df.columns if c not in {"Churn", "customerID"}]}, args.output)
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model bundle: {args.output.resolve()}")
    print(f"Saved metrics: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
