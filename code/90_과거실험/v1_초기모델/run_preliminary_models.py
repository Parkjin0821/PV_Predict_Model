"""Preliminary one-hour-ahead PV baseline models without weather inputs.

This is deliberately labelled preliminary. It establishes data leakage-safe
chronological baselines before KMA weather observations/forecasts are joined.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


CAPACITY_KW = 240.0  # Report-stated Gwangju plant capacity; requires field confirmation.


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0, CAPACITY_KW)
    err = p - y
    valid = np.isfinite(y) & np.isfinite(p)
    y, p, err = y[valid], p[valid], err[valid]
    denom = np.abs(y) + np.abs(p)
    return {
        "n": int(len(y)),
        "ME_kW": float(err.mean()),
        "MAE_kW": float(mean_absolute_error(y, p)),
        "MSE_kW2": float(mean_squared_error(y, p)),
        "RMSE_kW": float(math.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "Pearson_r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else None,
        "NMAE_capacity_pct": float(mean_absolute_error(y, p) / CAPACITY_KW * 100),
        "NRMSE_capacity_pct": float(math.sqrt(mean_squared_error(y, p)) / CAPACITY_KW * 100),
        "WAPE_pct": float(np.abs(err).sum() / np.abs(y).sum() * 100) if np.abs(y).sum() else None,
        "sMAPE_pct": float(np.mean(np.divide(2 * np.abs(err), denom,
            out=np.zeros_like(err), where=denom > 1e-9)) * 100),
    }


def make_frame():
    df = pd.read_csv(ROOT / "outputs" / "gwangju_hourly_plant.csv", parse_dates=["hour"])
    df = df[df["complete_5_inverters"]].copy()
    ymap = df.set_index("hour")["plant_output_power_mean_sum"]
    f = pd.DataFrame(index=ymap.index)
    f["current_power"] = ymap
    f["lag_1h"] = ymap.reindex(f.index - pd.Timedelta(hours=1)).to_numpy()
    f["lag_2h"] = ymap.reindex(f.index - pd.Timedelta(hours=2)).to_numpy()
    f["lag_24h"] = ymap.reindex(f.index - pd.Timedelta(hours=24)).to_numpy()
    f["lag_48h"] = ymap.reindex(f.index - pd.Timedelta(hours=48)).to_numpy()
    f["target_1h"] = ymap.reindex(f.index + pd.Timedelta(hours=1)).to_numpy()
    hour = f.index.hour + f.index.minute / 60
    doy = f.index.dayofyear
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["month"] = f.index.month
    return f.dropna()


def main():
    out = ROOT / "outputs"
    preview = ROOT / "previews"
    preview.mkdir(exist_ok=True)
    frame = make_frame()
    features = ["current_power", "lag_1h", "lag_2h", "lag_24h", "lag_48h",
                "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]
    n = len(frame)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    train, valid, test = frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]
    X_train = pd.concat([train[features], valid[features]])
    y_train = pd.concat([train["target_1h"], valid["target_1h"]])
    X_test, y_test = test[features], test["target_1h"]

    models = {
        "Persistence_current": None,
        "Previous_day_same_hour": None,
        "Multiple_linear_regression": LinearRegression(),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                  random_state=42, verbosity=-1),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
                                random_state=42, n_jobs=4),
    }
    predictions = {}
    for name, model in models.items():
        if name == "Persistence_current":
            pred = X_test["current_power"].to_numpy()
        elif name == "Previous_day_same_hour":
            pred = X_test["lag_24h"].to_numpy()
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        predictions[name] = np.clip(pred, 0, CAPACITY_KW)

    results = {name: metrics(y_test, pred) for name, pred in predictions.items()}
    metadata = {
        "status": "preliminary_no_weather",
        "target": "next-hour plant AC output power (sum of 5 inverter hourly means)",
        "forecast_horizon": "1 hour",
        "capacity_kw_from_report_unconfirmed": CAPACITY_KW,
        "feature_availability": "only values at or before forecast issue time",
        "split": {
            "train_plus_validation_n": len(X_train), "test_n": len(X_test),
            "test_start": test.index.min().isoformat(), "test_end": test.index.max().isoformat(),
        },
        "limitations": [
            "KMA weather data not yet joined; these are autoregressive/calendar baselines.",
            "Only hours with all five inverters reporting are included.",
            "240 kW capacity is from the source HWP report and awaits field confirmation.",
            "This pilot target/horizon is not the final operational specification.",
        ],
        "metrics": results,
    }
    (out / "preliminary_model_metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_table = pd.DataFrame({"issue_timestamp": test.index,
                               "target_timestamp": test.index + pd.Timedelta(hours=1),
                               "actual_kw": y_test.to_numpy(), **predictions})
    pred_table.to_csv(out / "preliminary_test_predictions.csv", index=False, encoding="utf-8-sig")

    score = pd.DataFrame(results).T.sort_values("RMSE_kW")
    score.to_csv(out / "preliminary_model_scorecard.csv", encoding="utf-8-sig")

    sample_start = pred_table["target_timestamp"].max().floor("D") - pd.Timedelta(days=13)
    sample = pred_table[pred_table["target_timestamp"] >= sample_start].set_index("target_timestamp")
    sample = sample.reindex(pd.date_range(sample.index.min().floor("h"), sample.index.max().ceil("h"), freq="h"))
    plt.figure(figsize=(14, 6))
    plt.plot(sample.index, sample["actual_kw"], label="Actual", color="black", linewidth=1.8)
    for name in ["Persistence_current", "Multiple_linear_regression", "LightGBM", "XGBoost"]:
        plt.plot(sample.index, sample[name], label=name, linewidth=1.0, alpha=0.8)
    plt.title("Gwangju PV preliminary 1-hour-ahead forecast (no KMA weather)")
    plt.ylabel("Plant AC output power (kW)")
    plt.xlabel("Timestamp (KST assumed from source)")
    plt.legend(ncol=2)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(preview / "preliminary_forecast_comparison.png", dpi=160)
    plt.close()
    print(score[["MAE_kW", "RMSE_kW", "R2", "Pearson_r", "NMAE_capacity_pct"]].to_string())


if __name__ == "__main__":
    main()
