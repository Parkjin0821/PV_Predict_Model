"""Build leakage-aware Gwangju PV datasets and multi-horizon Python models.

Targets are plant AC power at future valid times. 5/15-minute and hourly
models use weather forecasts that were archived 24 or 48 hours before the
valid time, ensuring the forecast information existed no later than issue time.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_packages"))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LAT, LON, ELEVATION_M = 35.1601, 126.8515, 21.0
TERRAIN_SLOPE_DEG = 0.0
TERRAIN_ASPECT_DEG = 0.0
CAPACITY_KW = 240.0
BASE_WEATHER = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
]
NUMERIC_RAW = ["입력전압", "입력전류", "입력전력", "출력전력", "주파수", "역률", "온도"]


def load_weather() -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = json.loads((ROOT / "external" / "gwangju_historical_forecast_hourly.json").read_text(encoding="utf-8"))
    prev = json.loads((ROOT / "external" / "gwangju_previous_runs_day1_day2_hourly.json").read_text(encoding="utf-8"))
    h = pd.DataFrame(hist["hourly"]); p = pd.DataFrame(prev["hourly"])
    h["time"] = pd.to_datetime(h["time"]); p["time"] = pd.to_datetime(p["time"])
    h = h.set_index("time").apply(pd.to_numeric, errors="coerce").add_prefix("observed_weather_")
    p = p.set_index("time").apply(pd.to_numeric, errors="coerce").add_prefix("forecast_")
    return h, p


def solar_position(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    doy = index.dayofyear.to_numpy(float)
    hour = index.hour.to_numpy(float) + index.minute.to_numpy(float) / 60
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    eqtime = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                     - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    true_solar_minutes = (hour*60 + eqtime + 4*LON - 60*9) % 1440
    ha = np.deg2rad(true_solar_minutes/4 - 180)
    lat = np.deg2rad(LAT)
    cosz = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha)
    zen = np.arccos(np.clip(cosz, -1, 1))
    elevation = 90 - np.rad2deg(zen)
    az = np.arctan2(np.sin(ha), np.cos(ha)*np.sin(lat)-np.tan(decl)*np.cos(lat))
    azimuth = (np.rad2deg(az) + 180) % 360
    return elevation, azimuth


def interpolate_weather(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    union = frame.index.union(index).sort_values()
    expanded = frame.reindex(union).interpolate(method="time").ffill().bfill()
    return expanded.reindex(index)


def build_regular(freq: str, hist: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    global TERRAIN_SLOPE_DEG, TERRAIN_ASPECT_DEG
    terrain_path = ROOT / "external" / "gwangju_terrain_grid.json"
    if terrain_path.exists():
        terrain = json.loads(terrain_path.read_text(encoding="utf-8"))
        TERRAIN_SLOPE_DEG = float(terrain["terrain_slope_deg_dem_approx"])
        TERRAIN_ASPECT_DEG = float(terrain["terrain_aspect_deg_dem_approx"])
    raw = pd.read_csv(ROOT / "input" / "gwangju_inverter_selected_raw.csv", low_memory=False)
    raw["생성일"] = pd.to_datetime(raw["생성일"])
    for c in NUMERIC_RAW:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw.loc[~raw["온도"].between(-50, 80), "온도"] = np.nan
    raw.loc[~raw["출력전력"].between(0, 80), "출력전력"] = np.nan
    raw["통신정상"] = raw["통신에러"].isna().astype(float)
    start = raw["생성일"].min().floor(freq); end = raw["생성일"].max().ceil(freq)
    idx = pd.date_range(start, end, freq=freq)
    wx = interpolate_weather(hist, idx)
    prev_wx = interpolate_weather(prev, idx)
    inv_frames = []
    for inv, part in raw.groupby("inverter", sort=True):
        part = part.set_index("생성일").sort_index()
        agg = part[NUMERIC_RAW + ["통신정상"]].resample(freq).mean().reindex(idx)
        agg["raw_observed"] = agg["출력전력"].notna().astype(int)
        night = wx["observed_weather_shortwave_radiation"] <= 3
        for power_col in ["출력전력", "입력전력", "입력전류"]:
            agg.loc[night & agg[power_col].isna(), power_col] = 0.0
        agg[NUMERIC_RAW + ["통신정상"]] = agg[NUMERIC_RAW + ["통신정상"]].interpolate(
            method="time", limit=2, limit_area="inside"
        )
        agg["inverter"] = int(inv); inv_frames.append(agg)
    panel = pd.concat(inv_frames).rename_axis("time").reset_index()
    grouped = panel.groupby("time")
    plant = pd.DataFrame(index=idx)
    plant["inverters_available"] = grouped["출력전력"].count().reindex(idx).fillna(0)
    plant["inverters_raw_observed"] = grouped["raw_observed"].sum().reindex(idx).fillna(0)
    plant["plant_output_kw"] = grouped["출력전력"].sum(min_count=5).reindex(idx)
    plant["plant_input_power_kw"] = grouped["입력전력"].sum(min_count=5).reindex(idx)
    plant["plant_input_current_a"] = grouped["입력전류"].sum(min_count=5).reindex(idx)
    for src, dest in [("입력전압", "mean_input_voltage_v"), ("주파수", "mean_frequency_hz"),
                      ("역률", "mean_power_factor"), ("온도", "mean_inverter_temperature_c"),
                      ("통신정상", "mean_communication_ok")]:
        plant[dest] = grouped[src].mean().reindex(idx)
    plant = pd.concat([plant, wx, prev_wx], axis=1)
    elevation, azimuth = solar_position(plant.index)
    plant["solar_elevation_deg"] = elevation; plant["solar_azimuth_deg"] = azimuth
    plant["is_daylight"] = (plant["observed_weather_shortwave_radiation"] > 5).astype(int)
    # Known static facts; unknown module/inverter nameplate details remain blank in metadata.
    plant["site_latitude"] = LAT; plant["site_longitude"] = LON
    plant["site_elevation_dem_m"] = ELEVATION_M; plant["reported_capacity_kw"] = CAPACITY_KW
    plant["reported_inverter_count"] = 5
    plant["terrain_slope_deg_dem_approx"] = TERRAIN_SLOPE_DEG
    plant["terrain_aspect_deg_dem_approx"] = TERRAIN_ASPECT_DEG
    plant["capacity_factor"] = plant["plant_output_kw"] / CAPACITY_KW
    plant.index.name = "time"
    out_name = {"5min":"gwangju_5min_model_dataset.csv", "15min":"gwangju_15min_model_dataset.csv", "1h":"gwangju_1hour_model_dataset.csv"}[freq]
    plant.to_csv(ROOT / "outputs" / out_name, encoding="utf-8-sig")
    return plant


def metrics(actual, predicted) -> dict:
    y = np.asarray(actual, float); p = np.clip(np.asarray(predicted, float), 0, CAPACITY_KW)
    mask = np.isfinite(y) & np.isfinite(p); y, p = y[mask], p[mask]
    e = y - p; nz = np.abs(y) > 1e-6
    pct = e[nz] / y[nz] * 100
    return {
        "n": int(len(y)), "ME_kW": float(e.mean()), "MAE_kW": float(np.abs(e).mean()),
        "MPE_pct": float(pct.mean()) if len(pct) else None,
        "MAPE_pct": float(np.abs(pct).mean()) if len(pct) else None,
        "MSE_kW2": float((e**2).mean()), "RMSE_kW": float(np.sqrt((e**2).mean())),
        "R2": float(r2_score(y, p)),
    }


def base_features(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["current_power_kw"] = df["plant_output_kw"]
    lags_min = sorted(set([step_minutes, step_minutes*2, step_minutes*3, 15, 30, 60, 120, 1440]))
    for minutes in lags_min:
        steps = max(1, round(minutes / step_minutes))
        f[f"power_lag_{minutes}min"] = df["plant_output_kw"].shift(steps)
    for minutes in [30, 60, 240]:
        window = max(2, round(minutes / step_minutes))
        f[f"power_roll_mean_{minutes}min"] = df["plant_output_kw"].shift(1).rolling(window, min_periods=max(2, window//2)).mean()
        f[f"power_roll_std_{minutes}min"] = df["plant_output_kw"].shift(1).rolling(window, min_periods=max(2, window//2)).std()
    f["power_ramp_1step"] = df["plant_output_kw"].diff()
    for c in ["plant_input_power_kw", "plant_input_current_a", "mean_input_voltage_v",
              "mean_frequency_hz", "mean_power_factor", "mean_inverter_temperature_c",
              "mean_communication_ok", "inverters_available"]:
        f["equipment_current_" + c] = df[c]
    for c in BASE_WEATHER:
        f["current_weather_" + c] = df["observed_weather_" + c]
    for c in ["site_latitude", "site_longitude", "site_elevation_dem_m",
              "terrain_slope_deg_dem_approx", "terrain_aspect_deg_dem_approx"]:
        f["topography_" + c] = df[c]
    for c in ["reported_capacity_kw", "reported_inverter_count"]:
        f["equipment_static_" + c] = df[c]
    return f


def target_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    minute = index.hour*60 + index.minute
    f["target_time_day_sin"] = np.sin(2*np.pi*minute/1440)
    f["target_time_day_cos"] = np.cos(2*np.pi*minute/1440)
    f["target_time_year_sin"] = np.sin(2*np.pi*index.dayofyear/365.25)
    f["target_time_year_cos"] = np.cos(2*np.pi*index.dayofyear/365.25)
    f["target_time_week_sin"] = np.sin(2*np.pi*index.dayofweek/7)
    f["target_time_week_cos"] = np.cos(2*np.pi*index.dayofweek/7)
    elev, az = solar_position(index)
    f["target_solar_elevation_deg"] = elev
    f["target_solar_azimuth_sin"] = np.sin(np.deg2rad(az))
    f["target_solar_azimuth_cos"] = np.cos(np.deg2rad(az))
    return f


def make_horizon_frame(df: pd.DataFrame, step_minutes: int, horizon_steps: int) -> tuple[pd.DataFrame, list[str]]:
    f = base_features(df, step_minutes)
    lead_hours = horizon_steps * step_minutes / 60
    previous_day = 1 if lead_hours <= 24 else 2
    target_idx = df.index + pd.to_timedelta(horizon_steps*step_minutes, unit="min")
    tf = target_time_features(target_idx); tf.index = df.index
    f = pd.concat([f, tf], axis=1)
    for c in BASE_WEATHER:
        source = f"forecast_{c}_previous_day{previous_day}"
        f["target_forecast_" + c] = df[source].shift(-horizon_steps)
    f["target_power_kw"] = df["plant_output_kw"].shift(-horizon_steps)
    f["target_daylight"] = (f["target_solar_elevation_deg"] > 3).astype(int)
    f["target_time"] = target_idx
    required = ["current_power_kw", "power_lag_60min", "target_power_kw", "target_forecast_shortwave_radiation"]
    f = f.dropna(subset=required)
    features = [c for c in f.columns if c not in ["target_power_kw", "target_daylight", "target_time"]]
    return f, features


def fit_horizon_models(name: str, df: pd.DataFrame, step_minutes: int, horizons: list[int]) -> dict:
    out_dir = ROOT / "models" / name; out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows, score_rows, metadata = [], [], {}
    for h in horizons:
        frame, features = make_horizon_frame(df, step_minutes, h)
        n = len(frame); a, b = int(n*.70), int(n*.85)
        train, valid, test = frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]
        impute = train[features].median(numeric_only=True)
        Xtr, Xv, Xt = train[features].fillna(impute), valid[features].fillna(impute), test[features].fillna(impute)
        ytr, yv, yt = train["target_power_kw"], valid["target_power_kw"], test["target_power_kw"]
        candidates = {
            "multiple_regression_ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
            "lightgbm": LGBMRegressor(n_estimators=240, learning_rate=.04, num_leaves=31,
                                       min_child_samples=30, subsample=.9, colsample_bytree=.9,
                                       random_state=SEED, verbosity=-1, n_jobs=4),
        }
        fitted, valid_scores = {}, {}
        for model_name, model in candidates.items():
            model.fit(Xtr, ytr); fitted[model_name] = model
            pv = np.clip(model.predict(Xv), 0, CAPACITY_KW)
            day = valid["target_daylight"].astype(bool).to_numpy()
            valid_scores[model_name] = metrics(yv[day], pv[day])["RMSE_kW"] if day.any() else metrics(yv, pv)["RMSE_kW"]
        selected = min(valid_scores, key=valid_scores.get)
        for model_name, model in fitted.items():
            pred = np.clip(model.predict(Xt), 0, CAPACITY_KW)
            all_m = metrics(yt, pred); day_mask = test["target_daylight"].astype(bool).to_numpy()
            day_m = metrics(yt[day_mask], pred[day_mask]) if day_mask.any() else all_m
            score_rows.append({"resolution":name, "horizon_steps":h,
                               "horizon_minutes":h*step_minutes, "model":model_name,
                               "selected_on_validation":model_name == selected,
                               **{"all_"+k:v for k,v in all_m.items()},
                               **{"daylight_"+k:v for k,v in day_m.items()}})
            if model_name == selected:
                for t, target_t, actual, predicted, daylight in zip(test.index, test["target_time"], yt, pred, test["target_daylight"]):
                    prediction_rows.append({"issue_time":t, "target_time":target_t, "horizon_steps":h,
                                            "horizon_minutes":h*step_minutes, "actual_kw":actual,
                                            "predicted_kw":predicted, "target_daylight":int(daylight),
                                            "selected_model":selected})
        joblib.dump({"models":fitted, "selected_model":selected, "features":features,
                     "impute":impute.to_dict(), "step_minutes":step_minutes,
                     "horizon_steps":h, "target":"plant_ac_power_kw"}, out_dir / f"horizon_{h:03d}.joblib")
        metadata[str(h)] = {"selected_model":selected, "validation_daylight_rmse_kw":valid_scores,
                            "n":n, "test_start":str(test.index.min()), "test_end":str(test.index.max())}
    scores = pd.DataFrame(score_rows); predictions = pd.DataFrame(prediction_rows)
    scores.to_csv(ROOT / "outputs" / f"{name}_model_scorecard.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(ROOT / "outputs" / f"{name}_selected_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"scorecard": scores, "predictions": predictions, "metadata": metadata}


def correlation_outputs(df15: pd.DataFrame) -> None:
    cols = {
        "발전출력": "plant_output_kw", "입력전력": "plant_input_power_kw",
        "인버터온도": "mean_inverter_temperature_c", "역률": "mean_power_factor",
        "일사량": "observed_weather_shortwave_radiation", "DNI": "observed_weather_direct_normal_irradiance",
        "기온": "observed_weather_temperature_2m", "습도": "observed_weather_relative_humidity_2m",
        "운량": "observed_weather_cloud_cover", "강수": "observed_weather_precipitation",
        "풍속": "observed_weather_wind_speed_10m", "기압": "observed_weather_surface_pressure",
        "태양고도": "solar_elevation_deg",
    }
    daylight = df15[df15["is_daylight"].eq(1)][list(cols.values())].rename(columns={v:k for k,v in cols.items()})
    daylight.corr(method="pearson").to_csv(ROOT / "outputs" / "correlation_pearson_daylight.csv", encoding="utf-8-sig")
    daylight.corr(method="spearman").to_csv(ROOT / "outputs" / "correlation_spearman_daylight.csv", encoding="utf-8-sig")


def daily_model(df1h: pd.DataFrame) -> dict:
    energy = df1h["plant_output_kw"].resample("1D").sum(min_count=12)  # 1h mean kW -> kWh
    wx_cols = {}
    for c in BASE_WEATHER:
        s = df1h[f"forecast_{c}_previous_day1"]
        wx_cols[f"forecast_{c}_mean"] = s.resample("1D").mean()
        if c in ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "precipitation"]:
            wx_cols[f"forecast_{c}_sum"] = s.resample("1D").sum()
    daily = pd.DataFrame({"daily_energy_kwh":energy, **wx_cols})
    daily["lag_1d"] = daily["daily_energy_kwh"].shift(1); daily["lag_7d"] = daily["daily_energy_kwh"].shift(7)
    daily["roll_mean_7d"] = daily["daily_energy_kwh"].shift(1).rolling(7, min_periods=5).mean()
    daily["doy_sin"] = np.sin(2*np.pi*daily.index.dayofyear/365.25)
    daily["doy_cos"] = np.cos(2*np.pi*daily.index.dayofyear/365.25)
    frame = daily.dropna().copy(); features = [c for c in frame.columns if c != "daily_energy_kwh"]
    n=len(frame); a,b=int(n*.7),int(n*.85); tr,va,te=frame.iloc[:a],frame.iloc[a:b],frame.iloc[b:]
    models={"multiple_regression_ridge":Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=10.0))]),
            "lightgbm":LGBMRegressor(n_estimators=240,learning_rate=.04,num_leaves=20,min_child_samples=12,random_state=SEED,verbosity=-1)}
    valid_rmse={}; fitted={}
    for k,m in models.items():
        m.fit(tr[features],tr["daily_energy_kwh"]); fitted[k]=m
        valid_rmse[k]=float(np.sqrt(np.mean((va["daily_energy_kwh"]-np.maximum(m.predict(va[features]),0))**2)))
    selected=min(valid_rmse,key=valid_rmse.get); pred=np.maximum(fitted[selected].predict(te[features]),0)
    # Daily energy metrics use their natural kWh units.
    y=te["daily_energy_kwh"].to_numpy(); e=y-pred; pct=e[y>0]/y[y>0]*100
    result={"selected_model":selected,"validation_rmse_kwh":valid_rmse,"test_n":len(te),
            "test_ME_kWh":float(e.mean()),"test_MAE_kWh":float(np.abs(e).mean()),
            "test_MPE_pct":float(pct.mean()),"test_MAPE_pct":float(np.abs(pct).mean()),
            "test_MSE_kWh2":float((e**2).mean()),"test_RMSE_kWh":float(np.sqrt((e**2).mean())),
            "test_R2":float(r2_score(y,pred)),"test_start":str(te.index.min()),"test_end":str(te.index.max())}
    daily.to_csv(ROOT/"outputs"/"gwangju_daily_weather_model_dataset.csv",encoding="utf-8-sig")
    pd.DataFrame({"date":te.index,"actual_kwh":y,"predicted_kwh":pred}).to_csv(ROOT/"outputs"/"daily_selected_predictions.csv",index=False,encoding="utf-8-sig")
    joblib.dump({"models":fitted,"selected_model":selected,"features":features},ROOT/"models"/"daily_next_day.joblib")
    return result


def aggregation_feasibility(df1h: pd.DataFrame) -> dict:
    energy = df1h["plant_output_kw"].resample("1D").sum(min_count=12)
    monthly = energy.resample("MS").sum(min_count=20); annual = energy.resample("YS").sum(min_count=300)
    pd.DataFrame({"monthly_energy_kwh":monthly}).to_csv(ROOT/"outputs"/"gwangju_monthly_energy.csv",encoding="utf-8-sig")
    pd.DataFrame({"annual_energy_kwh":annual}).to_csv(ROOT/"outputs"/"gwangju_annual_energy.csv",encoding="utf-8-sig")
    return {"monthly_complete_or_partial_points":int(monthly.notna().sum()),
            "annual_complete_or_partial_points":int(annual.notna().sum()),
            "monthly_model_status":"prototype only; fewer than 24 monthly observations",
            "annual_model_status":"not fitted; fewer than 3 annual observations"}


def main() -> None:
    hist, prev = load_weather()
    datasets = {freq:build_regular(freq,hist,prev) for freq in ["5min","15min","1h"]}
    correlation_outputs(datasets["15min"])
    results = {
        "five_minute": fit_horizon_models("5min", datasets["5min"], 5, [12,24,36,48]),
        "fifteen_minute": fit_horizon_models("15min", datasets["15min"], 15, [4,8,12,16]),
        "hourly": fit_horizon_models("1hour", datasets["1h"], 60, list(range(24,49))),
    }
    daily = daily_model(datasets["1h"]); feasibility = aggregation_feasibility(datasets["1h"])
    summary = {
        "site":"광주광역시청","python_model":True,
        "known_factors":{"environment":BASE_WEATHER,"topography":{"latitude":LAT,"longitude":LON,"elevation_m":ELEVATION_M,
                         "terrain_slope_deg_dem_approx":TERRAIN_SLOPE_DEG,"terrain_aspect_deg_dem_approx":TERRAIN_ASPECT_DEG},
                         "time":["minute","hour","weekday","day_of_year","solar_elevation","solar_azimuth"],
                         "equipment":{"reported_capacity_kw":CAPACITY_KW,"inverter_count":5,
                                      "installation_locations":["의회 주차장","야외 음악당"],
                                      "telemetry":["input voltage/current/power","output power","frequency","power factor","temperature","communication availability"]}},
        "unknown_not_inferred":["module maker/model/quantity","module tilt/azimuth","inverter maker/model/nameplate capacity","horizon shading"],
        "daily":daily,"long_period_feasibility":feasibility,
        "weather_leakage_control":"previous_day1 used for target horizons <=24h; previous_day2 for >24h",
    }
    (ROOT/"outputs"/"model_run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"daily":daily,"feasibility":feasibility},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
