"""Preliminary ARIMAX one-step-ahead benchmark on the regular hourly grid."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from run_neural_models import regular_series

CAPACITY = 240.0


def metrics(y, p):
    p = np.clip(np.asarray(p, float), 0, CAPACITY)
    y = np.asarray(y, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    err = p - y
    denom = np.abs(y) + np.abs(p)
    return {
        "n": int(len(y)), "ME_kW": float(err.mean()),
        "MAE_kW": float(mean_absolute_error(y, p)),
        "MSE_kW2": float(mean_squared_error(y, p)),
        "RMSE_kW": float(math.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)), "Pearson_r": float(np.corrcoef(y, p)[0, 1]),
        "NMAE_capacity_pct": float(mean_absolute_error(y, p)/CAPACITY*100),
        "NRMSE_capacity_pct": float(math.sqrt(mean_squared_error(y, p))/CAPACITY*100),
        "WAPE_pct": float(np.abs(err).sum()/np.abs(y).sum()*100),
        "sMAPE_pct": float(np.mean(np.divide(2*np.abs(err), denom, out=np.zeros_like(err), where=denom>1e-9))*100),
    }


def main():
    grid = regular_series()
    comparison = pd.read_csv(ROOT / "outputs" / "preliminary_test_predictions.csv",
                             parse_dates=["target_timestamp"])
    target_times = pd.DatetimeIndex(comparison["target_timestamp"])
    target_start = target_times.min()
    y = grid["power"].copy()
    idx = grid.index
    exog = pd.DataFrame(index=idx)
    hour = idx.hour + 0.5
    doy = idx.dayofyear
    exog["hour_sin"] = np.sin(2*np.pi*hour/24)
    exog["hour_cos"] = np.cos(2*np.pi*hour/24)
    exog["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    exog["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    train_y, test_y = y[y.index < target_start], y[y.index >= target_start]
    train_x, test_x = exog.loc[train_y.index], exog.loc[test_y.index]

    model = SARIMAX(train_y, exog=train_x, order=(2, 0, 2), seasonal_order=(1, 0, 0, 24),
                    enforce_stationarity=False, enforce_invertibility=False,
                    initialization="approximate_diffuse")
    fit = model.fit(disp=False, maxiter=60)
    # append updates the state with each observed test value; predictions are therefore
    # rolling one-step-ahead rather than an unrealistic multi-month open-loop forecast.
    extended = fit.append(test_y, exog=test_x, refit=False)
    pred = extended.get_prediction(start=target_start, end=test_y.index.max(), dynamic=False).predicted_mean
    actual = pd.Series(comparison["actual_kw"].to_numpy(), index=target_times)
    forecast = pred.reindex(target_times)
    result = metrics(actual, forecast)
    payload = {
        "status": "preliminary_no_weather", "model": "SARIMAX",
        "order": [2, 0, 2], "seasonal_order": [1, 0, 0, 24],
        "exogenous": ["hour_sin", "hour_cos", "doy_sin", "doy_cos"],
        "evaluation": "rolling one-step predictions on observed complete five-inverter daytime targets",
        "limitations": ["No KMA weather inputs", "240 kW report capacity not field-confirmed",
                        "Structural night zeros included; daytime missing values retained as missing"],
        "aic": float(fit.aic), "metrics": result,
    }
    out = ROOT / "outputs"
    (out / "preliminary_arima_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"target_timestamp": actual.index, "actual_kw": actual.to_numpy(),
                  "SARIMAX": forecast.to_numpy()}).to_csv(
        out / "preliminary_arima_predictions.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame({"SARIMAX": result}).T[["MAE_kW", "RMSE_kW", "R2", "Pearson_r"]].to_string())


if __name__ == "__main__":
    main()
