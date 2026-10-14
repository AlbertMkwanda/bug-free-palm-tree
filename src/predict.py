"""Score a CSV of mobile-money transactions with the trained classifier."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "mobile_money_fraud_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing the model feature columns")
    parser.add_argument("--output", type=Path, default=Path("scored_transactions.csv"))
    arguments = parser.parse_args()
    frame = pd.read_csv(arguments.input_csv)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        parser.error(f"input is missing required feature columns: {', '.join(missing)}")
    if not MODEL_PATH.exists():
        parser.error("model is missing; run `python -m src.train` first")
    model = joblib.load(MODEL_PATH)
    with (ROOT / "artifacts" / "metrics.json").open(encoding="utf-8") as metrics_file:
        threshold = json.load(metrics_file)["threshold"]
    result = frame.copy()
    result["fraud_probability"] = model.predict_proba(frame[FEATURES])[:, 1]
    result["predicted_label"] = result["fraud_probability"].ge(threshold).map({True: "fraud", False: "legitimate"})
    result.to_csv(arguments.output, index=False)
    print(f"Wrote {len(result)} scored rows to {arguments.output}")


if __name__ == "__main__":
    main()