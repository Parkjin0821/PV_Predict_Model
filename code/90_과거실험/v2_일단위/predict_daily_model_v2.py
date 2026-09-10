"""Run a saved UCUBE daily PV tabular model on a prepared feature CSV.

The input CSV must contain the feature columns listed in feature_schema.json.
The script never invents missing site facts; missing model features are rejected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))

import joblib
import numpy as np
import pandas as pd


MODEL_FILES = {
    "linear": "multiple_linear_regression.joblib",
    "lightgbm": "lightgbm_daily.joblib",
    "xgboost": "xgboost_daily.joblib",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day daily PV energy (kWh).")
    parser.add_argument("--site", choices=["gwangju", "gimje"], required=True)
    parser.add_argument("--model", choices=MODEL_FILES, default="linear")
    parser.add_argument("--input", type=Path, required=True, help="Prepared feature CSV")
    parser.add_argument("--output", type=Path, required=True, help="Prediction CSV")
    args = parser.parse_args()

    bundle_path = ROOT / "outputs" / "models" / args.site / MODEL_FILES[args.model]
    bundle = joblib.load(bundle_path)
    frame = pd.read_csv(args.input)
    missing = [name for name in bundle["features"] if name not in frame.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))

    x = frame[bundle["features"]].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(pd.Series(bundle["impute"]))
    if x.isna().any().any():
        bad = x.columns[x.isna().any()].tolist()
        raise ValueError("No training-time imputation value for: " + ", ".join(bad))

    predicted = np.maximum(bundle["model"].predict(x), 0.0)
    result = frame.copy()
    result["predicted_next_day_energy_kwh"] = predicted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
