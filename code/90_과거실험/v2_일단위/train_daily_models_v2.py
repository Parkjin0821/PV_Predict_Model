"""Train daily next-day PV production models for a UCUBE site.

Target: next day's plant daily generation (kWh), based on the sum of each
inverter's daily meter maximum. Optional environment/topography/equipment
columns are automatically used only when populated; missing facts stay blank.
"""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)


def evaluate(actual, predicted, capacity_kw):
    y = np.asarray(actual, float)
    p = np.clip(np.asarray(predicted, float), 0, capacity_kw * 24)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    e = y - p  # positive ME/MPE means the model under-predicted
    pct_mask = np.abs(y) > 1e-9
    pct = e[pct_mask] / y[pct_mask] * 100
    return {
        "n": int(len(y)),
        "ME_kWh": float(np.mean(e)),
        "MAE_kWh": float(np.mean(np.abs(e))),
        "MPE_pct": float(np.mean(pct)) if len(pct) else None,
        "MAPE_pct": float(np.mean(np.abs(pct))) if len(pct) else None,
        "MSE_kWh2": float(np.mean(e**2)),
        "RMSE_kWh": float(np.sqrt(np.mean(e**2))),
        "R2": float(r2_score(y, p)),
        "Pearson_r": float(np.corrcoef(y, p)[0, 1]),
        "NMAE_capacity_day_pct": float(np.mean(np.abs(e)) / (capacity_kw * 24) * 100),
        "NRMSE_capacity_day_pct": float(np.sqrt(np.mean(e**2)) / (capacity_kw * 24) * 100),
    }


def make_feature_frame(site):
    raw = pd.read_csv(ROOT / "outputs" / f"{site}_daily_plant.csv", parse_dates=["date"])
    raw = raw.sort_values("date").set_index("date")
    full = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="D"))
    y = raw["daily_energy_kwh"].where(raw["quality_ok"].astype(bool)).reindex(full.index)
    f = pd.DataFrame(index=full.index)
    f["current_energy"] = y
    for lag in [1, 2, 3, 7, 14, 28]:
        f[f"lag_{lag}d"] = y.shift(lag)
    f["roll_mean_7d"] = y.shift(1).rolling(7, min_periods=5).mean()
    f["roll_std_7d"] = y.shift(1).rolling(7, min_periods=5).std()
    f["roll_mean_14d"] = y.shift(1).rolling(14, min_periods=10).mean()
    f["roll_max_14d"] = y.shift(1).rolling(14, min_periods=10).max()
    doy, dow, month = f.index.dayofyear, f.index.dayofweek, f.index.month
    f["doy_sin"] = np.sin(2*np.pi*doy/365.25); f["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    f["dow_sin"] = np.sin(2*np.pi*dow/7); f["dow_cos"] = np.cos(2*np.pi*dow/7)
    f["month_sin"] = np.sin(2*np.pi*month/12); f["month_cos"] = np.cos(2*np.pi*month/12)

    optional_prefixes = ("weather_", "forecast_weather_", "terrain_", "module_", "inverter_")
    optional_names = ["plant_latitude", "plant_longitude", "elevation_m", "horizon_shading_index",
                      "installation_type", "structure_type", "tracking_type"]
    optional_used = []
    for col in full.columns:
        if (col.startswith(optional_prefixes) or col in optional_names) and full[col].notna().any():
            if pd.api.types.is_numeric_dtype(full[col]):
                f[col] = pd.to_numeric(full[col], errors="coerce")
                optional_used.append(col)
    f["target_next_day"] = y.shift(-1)
    f["target_date"] = f.index + pd.Timedelta(days=1)
    required = ["current_energy", "lag_1d", "lag_2d", "lag_3d", "lag_7d", "lag_14d",
                "roll_mean_7d", "roll_std_7d", "roll_mean_14d", "target_next_day"]
    model_frame = f.dropna(subset=required).copy()
    numeric_features = [c for c in f.columns if c not in ("target_next_day", "target_date") and
                        pd.api.types.is_numeric_dtype(f[c])]
    # Optional columns may have isolated gaps; fit imputation values on training only later.
    return raw, full, y, model_frame, numeric_features, optional_used


class DailySeqModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        cls = nn.RNN if kind == "RNN" else nn.LSTM
        self.seq = cls(6, 24, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1))
    def forward(self, x):
        out, _ = self.seq(x)
        return self.head(out[:, -1]).squeeze(-1)


def neural_sequences(y, dates, capacity):
    idx = pd.date_range(y.index.min(), y.index.max(), freq="D")
    vals = y.reindex(idx)
    observed = vals.notna().astype(float)
    power = vals.fillna(0).to_numpy(float) / (capacity * 8)  # stable scale near daily CF range
    doy, dow = idx.dayofyear, idx.dayofweek
    xall = np.column_stack([power, observed.to_numpy(), np.sin(2*np.pi*doy/365.25),
                            np.cos(2*np.pi*doy/365.25), np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)]).astype("float32")
    xs, ys = [], []
    for target_date in dates:
        pos = idx.get_loc(pd.Timestamp(target_date))
        if pos < 14 or not np.isfinite(vals.iloc[pos]):
            raise ValueError(f"Invalid neural target date: {target_date}")
        xs.append(xall[pos-14:pos]); ys.append(vals.iloc[pos] / (capacity * 8))
    return np.stack(xs), np.asarray(ys, dtype="float32")


def train_neural(kind, xtr, ytr, xv, yv, xt):
    model = DailySeqModel(kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=False)
    xv_t, yv_t = torch.from_numpy(xv), torch.from_numpy(yv)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(60):
        model.train(); losses=[]
        for xb, yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb), yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): vl=loss_fn(model(xv_t), yv_t).item()
        history.append({"epoch":epoch+1,"train_mse":float(np.mean(losses)),"valid_mse":vl})
        if vl < best-1e-6:
            best, state, stale = vl, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 8: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): pred=model(torch.from_numpy(xt)).numpy()
    return model, pred, history


def main(site):
    model_dir = ROOT / "outputs" / "models" / site
    model_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = ROOT / "previews"; preview_dir.mkdir(exist_ok=True)
    raw, full, y_series, frame, features, optional_used = make_feature_frame(site)
    capacity = float(raw["plant_capacity_kw_report"].dropna().iloc[0])
    n = len(frame); a, b = int(n*.70), int(n*.85)
    train, valid, test = frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]
    train_valid = pd.concat([train, valid])
    impute = train_valid[features].median(numeric_only=True)
    Xtv = train_valid[features].fillna(impute); ytv = train_valid["target_next_day"]
    Xt = test[features].fillna(impute); yt = test["target_next_day"]

    models = {
        "Persistence_previous_day": None,
        "Previous_week_same_day": None,
        "Multiple_linear_regression": LinearRegression(),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=.025, num_leaves=24,
                                  min_child_samples=15, random_state=SEED, verbosity=-1),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=.025, max_depth=5,
                                subsample=.85, colsample_bytree=.85, objective="reg:squarederror",
                                random_state=SEED, n_jobs=4),
    }
    preds, fitted = {}, {}
    for name, model in models.items():
        if name == "Persistence_previous_day": pred = Xt["current_energy"].to_numpy()
        elif name == "Previous_week_same_day": pred = Xt["lag_7d"].to_numpy()
        else:
            model.fit(Xtv, ytv); pred = model.predict(Xt); fitted[name] = model
        preds[name] = np.clip(pred, 0, capacity*24)

    # Rolling one-day-ahead SARIMAX on the same target dates.
    target_dates = pd.DatetimeIndex(test["target_date"])
    start = target_dates.min()
    arima_idx = pd.date_range(y_series.index.min(), y_series.index.max(), freq="D")
    arima_scale = capacity * 8
    ay = y_series.reindex(arima_idx) / arima_scale
    ax = pd.DataFrame({"doy_sin":np.sin(2*np.pi*arima_idx.dayofyear/365.25),
                       "doy_cos":np.cos(2*np.pi*arima_idx.dayofyear/365.25),
                       "dow_sin":np.sin(2*np.pi*arima_idx.dayofweek/7),
                       "dow_cos":np.cos(2*np.pi*arima_idx.dayofweek/7)}, index=arima_idx)
    arima = SARIMAX(ay[ay.index < start], exog=ax[ax.index < start], order=(2,0,1),
                    seasonal_order=(1,0,0,7), enforce_stationarity=False, enforce_invertibility=False,
                    trend="c", initialization="approximate_diffuse")
    arima_fit = arima.fit(disp=False, maxiter=80)
    test_y_full, test_x_full = ay[ay.index >= start], ax[ax.index >= start]
    appended = arima_fit.append(test_y_full, exog=test_x_full, refit=False)
    arima_pred = appended.get_prediction(start=start, end=test_y_full.index.max(), dynamic=False).predicted_mean
    preds["SARIMAX_ARIMA"] = np.clip(arima_pred.reindex(target_dates).to_numpy() * arima_scale, 0, capacity*24)

    # RNN/LSTM use the same test target dates and chronological train/validation partitions.
    xtr, ytr = neural_sequences(y_series, train["target_date"], capacity)
    xv, yv = neural_sequences(y_series, valid["target_date"], capacity)
    xte, yte = neural_sequences(y_series, test["target_date"], capacity)
    neural_hist = {}
    for kind in ["RNN", "LSTM"]:
        model, scaled_pred, hist = train_neural(kind, xtr, ytr, xv, yv, xte)
        preds[kind] = np.clip(scaled_pred * capacity * 8, 0, capacity*24)
        neural_hist[kind] = hist
        torch.save({"state_dict":model.state_dict(), "kind":kind, "sequence_days":14,
                    "input_features":["energy_scaled","observed_mask","doy_sin","doy_cos","dow_sin","dow_cos"],
                    "capacity_kw_report":capacity}, model_dir / f"{kind.lower()}_daily.pt")

    scores = {name:evaluate(yt, pred, capacity) for name,pred in preds.items()}
    base_rmse = scores["Persistence_previous_day"]["RMSE_kWh"]
    for value in scores.values(): value["RMSE_skill_vs_persistence_pct"] = (1-value["RMSE_kWh"]/base_rmse)*100

    # Save operational model files and metadata.
    joblib.dump({"model":fitted["Multiple_linear_regression"],"features":features,"impute":impute.to_dict(),
                 "target":"next_day_daily_energy_kwh"}, model_dir / "multiple_linear_regression.joblib")
    # joblib avoids native-library Unicode path limitations on Korean Windows paths.
    joblib.dump({"model":fitted["LightGBM"],"features":features,"impute":impute.to_dict(),
                 "target":"next_day_daily_energy_kwh"}, model_dir / "lightgbm_daily.joblib")
    joblib.dump({"model":fitted["XGBoost"],"features":features,"impute":impute.to_dict(),
                 "target":"next_day_daily_energy_kwh"}, model_dir / "xgboost_daily.joblib")
    arima_fit.save(model_dir / "sarimax_daily.pkl")
    bundle = {"site":site,"capacity_kw_report_unconfirmed":capacity,
              "target":"next-day plant daily generation (kWh)","features":features,
              "optional_static_or_external_features_used":optional_used,
              "split":{"train_n":len(train),"validation_n":len(valid),"test_n":len(test),
                       "test_target_start":target_dates.min().date().isoformat(),
                       "test_target_end":target_dates.max().date().isoformat()},
              "error_definition":"e = actual - predicted; positive ME/MPE means under-prediction",
              "metrics":scores,"neural_training_history":neural_hist}
    (model_dir / "model_metadata_and_metrics.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding="utf-8")
    (model_dir / "feature_schema.json").write_text(json.dumps({"required_features":features,
        "optional_supported_groups":{"environment":["forecast_weather_irradiance","forecast_weather_temperature",
            "forecast_weather_humidity","forecast_weather_cloud","forecast_weather_rain","forecast_weather_wind"],
        "topography":["plant_latitude","plant_longitude","elevation_m","terrain_slope_deg","terrain_aspect_deg","horizon_shading_index"],
        "equipment":["plant_capacity_kw_report","module_tilt_deg","module_azimuth_deg","module_efficiency_pct",
            "inverter_capacity_total_kw","inverter_efficiency_pct","installation_type","structure_type","tracking_type"]}},
        ensure_ascii=False,indent=2),encoding="utf-8")

    pred_table = pd.DataFrame({"issue_date":test.index,"target_date":target_dates,"actual_kwh":yt.to_numpy(),**preds})
    pred_table.to_csv(model_dir / "test_predictions.csv",index=False,encoding="utf-8-sig")
    score_table = pd.DataFrame(scores).T.sort_values("RMSE_kWh")
    score_table.to_csv(model_dir / "model_scorecard.csv",encoding="utf-8-sig")

    # One manager-facing performance capture, not a dashboard.
    plot = pred_table.tail(min(90,len(pred_table)))
    plt.figure(figsize=(14,7))
    plt.plot(plot["target_date"],plot["actual_kwh"],color="black",linewidth=2,label="Actual")
    for name,color in [("Persistence_previous_day","#4C78A8"),("Multiple_linear_regression","#F58518"),
                       ("LightGBM","#54A24B"),("XGBoost","#E45756")]:
        plt.plot(plot["target_date"],plot[name],linewidth=1.2,label=name,color=color,alpha=.9)
    plt.title(f"{site.upper()} daily PV next-day forecast performance")
    plt.ylabel("Daily generation (kWh)"); plt.xlabel("Target date")
    plt.grid(alpha=.2); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(preview_dir / f"{site}_daily_model_performance.png",dpi=180); plt.close()
    print(site, score_table[["ME_kWh","MAE_kWh","MPE_pct","MAPE_pct","MSE_kWh2","RMSE_kWh","R2","Pearson_r"]].to_string())


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("gwangju","gimje"):
        raise SystemExit("usage: train_daily_models_v2.py gwangju|gimje")
    main(sys.argv[1])
