"""UCUBE 광주 3시간 격자예보 650일 잠정 검증.

원본 파일은 수정하지 않는다. 두 수집 CSV를 메모리에서 통합하고, 3시간 예보를
15분으로 정렬한 뒤 초단기 15분 및 1시간 단위 익일 운영 시나리오를 검증한다.

주의:
- TMP/REH는 일별 선형보간, 범주형 SKY는 일별 전방채움한다.
- 일사량 대용치는 학습구간의 관측 일사량으로만 월×SKY 보정계수를 학습한다.
- 1시간 익일 모델은 05시 발표 시점 이후 값을 피하기 위해 최소 48시간 lag만 쓴다.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전")
BASE_CSV_1 = PROJECT / "광주_격자예보_3시간단위.csv"
BASE_CSV_2 = Path(r"C:\Users\u-cube\JIN\광주_격자예보_3시간단위.csv")
TRAIN_PARQUET = PROJECT / "train_standardized.parquet"
TEST_PARQUET = PROJECT / "test_standardized.parquet"
STATS_JSON = PROJECT / "normalization_stats.json"

OUT = Path(r"C:\Users\u-cube\Documents\ChatGPT\유큐브\결과물\예측모델\광주\grid_forecast_650d_interim_v2_2026-08-18")
OUT.mkdir(parents=True, exist_ok=True)

FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
TEST_START = pd.Timestamp("2026-03-16 00:00:00")
TEST_END = pd.Timestamp("2026-06-06 23:59:59")
RANDOM_STATE = 42


def solar_elevation_approx(index: pd.DatetimeIndex) -> np.ndarray:
    """광주 ASOS 좌표 기준 근사 태양고도(KST)."""
    lat = math.radians(35.17294)
    n = index.dayofyear.to_numpy()
    local_hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    gamma = 2 * math.pi / 365 * (n - 1 + (local_hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    time_offset = eqtime + 4 * 126.89156 - 60 * 9
    true_solar_min = local_hour * 60 + time_offset
    hour_angle = np.radians(true_solar_min / 4 - 180)
    elev = np.arcsin(
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    return np.degrees(elev)


def build_forecast_15min() -> tuple[pd.DataFrame, dict]:
    frames = [pd.read_csv(path, dtype={"발표일": str}) for path in (BASE_CSV_1, BASE_CSV_2)]
    daily = pd.concat(frames, ignore_index=True)
    daily["발표일"] = daily["발표일"].astype(str).str.zfill(8)
    duplicate_count = int(daily.duplicated("발표일").sum())
    daily = daily.drop_duplicates("발표일", keep="last").sort_values("발표일")

    records = []
    for row in daily.to_dict("records"):
        issue_day = datetime.strptime(row["발표일"], "%Y%m%d")
        target_day = issue_day + timedelta(days=1)
        for hour in FCST_HOURS:
            records.append(
                {
                    "timestamp": target_day + timedelta(hours=hour),
                    "issue_day": issue_day.date().isoformat(),
                    "TMP_fcst": float(row[f"TMP_{hour:02d}h"]),
                    "SKY_fcst": float(row[f"SKY_{hour:02d}h"]),
                    "REH_fcst": float(row[f"REH_{hour:02d}h"]),
                }
            )
    three_hour = pd.DataFrame(records).sort_values("timestamp")

    pieces = []
    for target_date, group in three_hour.groupby(three_hour["timestamp"].dt.date):
        idx = pd.date_range(pd.Timestamp(target_date), periods=96, freq="15min")
        part = group.set_index("timestamp").reindex(idx)
        part["TMP_fcst"] = part["TMP_fcst"].interpolate(method="time").ffill().bfill()
        part["REH_fcst"] = part["REH_fcst"].interpolate(method="time").ffill().bfill()
        part["SKY_fcst"] = part["SKY_fcst"].ffill().bfill()
        part["issue_day"] = group["issue_day"].iloc[0]
        part.index.name = "생성일"
        pieces.append(part.reset_index())

    forecast = pd.concat(pieces, ignore_index=True).sort_values("생성일")
    forecast["solar_elevation"] = solar_elevation_approx(pd.DatetimeIndex(forecast["생성일"]))
    forecast["solar_base"] = np.maximum(
        0.0, np.sin(np.radians(forecast["solar_elevation"].to_numpy()))
    )
    sky_to_cloud = {1.0: 0.0, 2.0: 2.5, 3.0: 5.0, 4.0: 10.0}
    forecast["cloud10_fcst"] = forecast["SKY_fcst"].map(sky_to_cloud)

    expected_days = (
        pd.to_datetime(daily["발표일"], format="%Y%m%d").max()
        - pd.to_datetime(daily["발표일"], format="%Y%m%d").min()
    ).days + 1
    audit = {
        "issue_rows": int(len(daily)),
        "issue_start": daily["발표일"].min(),
        "issue_end": daily["발표일"].max(),
        "expected_continuous_issue_days": int(expected_days),
        "duplicate_issue_days_before_dedup": duplicate_count,
        "forecast_15min_rows": int(len(forecast)),
        "target_start": str(forecast["생성일"].min()),
        "target_end": str(forecast["생성일"].max()),
        "missing_values": int(forecast.isna().sum().sum()),
        "interpolation": {"TMP": "linear within target day", "REH": "linear within target day", "SKY": "forward-fill within target day"},
    }
    return forecast, audit


def inverse_standardized(series: pd.Series, stats: dict, name: str) -> pd.Series:
    item = stats["standardization"][name]
    return series.astype(float) * float(item["std"]) + float(item["mean"])


def load_model_rows(forecast: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    train = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    train["source_split"] = "train"
    test["source_split"] = "test"
    data = pd.concat([train, test], ignore_index=True)
    data["생성일"] = pd.to_datetime(data["생성일"])
    data["인버터번호"] = data["인버터번호"].astype(str)
    data = data.merge(forecast, on="생성일", how="inner", validate="many_to_one")

    for lag in ("발전량_lag1", "발전량_lag2", "발전량_lag4"):
        data[f"{lag}_raw"] = inverse_standardized(data[lag], stats, lag)
    for name in ("일사량", "일조", "습도", "지면온도", "전운량", "기온", "풍속", "강수량"):
        data[f"{name}_obs"] = inverse_standardized(data[name], stats, name)

    target = stats["target"]
    # 시간순으로 연결한 뒤 동시간 전일 기준모델을 만든다.
    data = data.sort_values(["인버터번호", "생성일"]).reset_index(drop=True)
    data["same_time_yesterday"] = data.groupby("인버터번호")[target].shift(96)
    return data, stats, stats["features"]


def fit_irradiance_mapping(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    unique_weather = data.drop_duplicates("생성일").copy()
    train_weather = unique_weather[
        (unique_weather["source_split"] == "train")
        & (unique_weather["solar_base"] > 0.05)
        & (unique_weather["일사량_obs"] >= 0)
    ].copy()
    train_weather["month"] = train_weather["생성일"].dt.month
    train_weather["sky_code"] = train_weather["SKY_fcst"].round().astype(int)
    train_weather["irr_ratio"] = (
        train_weather["일사량_obs"] / train_weather["solar_base"].clip(lower=0.05)
    ).clip(lower=0, upper=10)

    monthly = train_weather.groupby(["month", "sky_code"])["irr_ratio"].median().to_dict()
    by_sky = train_weather.groupby("sky_code")["irr_ratio"].median().to_dict()
    global_ratio = float(train_weather["irr_ratio"].median())

    out = data.copy()
    month = out["생성일"].dt.month.to_numpy()
    sky = out["SKY_fcst"].round().astype(int).to_numpy()
    ratios = np.array(
        [monthly.get((int(m), int(s)), by_sky.get(int(s), global_ratio)) for m, s in zip(month, sky)],
        dtype=float,
    )
    out["irradiance_est"] = out["solar_base"].to_numpy() * ratios
    mapping = {
        "method": "training-only median(observed irradiance / solar_base) by month and forecast SKY",
        "training_rows_unique_timestamp": int(len(train_weather)),
        "global_ratio": global_ratio,
        "monthly_sky_ratio": {f"{m:02d}_SKY{s}": float(v) for (m, s), v in monthly.items()},
        "fallback_sky_ratio": {f"SKY{s}": float(v) for s, v in by_sky.items()},
    }
    return out, mapping


def metrics(y: np.ndarray, pred: np.ndarray, reference: float, threshold: float) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = y - pred
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred, err = y[mask], pred[mask], err[mask]
    pct = y > threshold
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "ME": float(np.mean(err)),
        "MAE": float(np.mean(np.abs(err))),
        "MSE": float(np.mean(err**2)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(np.mean(np.abs(err[pct] / y[pct])) * 100) if pct.any() else None,
        "NMAE_pct_reference": float(np.mean(np.abs(err)) / reference * 100),
        "NRMSE_pct_reference": float(np.sqrt(np.mean(err**2)) / reference * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "reference_output": float(reference),
        "mape_target_threshold": float(threshold),
    }


def model_features_15min(data: pd.DataFrame) -> list[str]:
    data["hour_sin_fcst"] = np.sin(2 * np.pi * (data["생성일"].dt.hour + data["생성일"].dt.minute / 60) / 24)
    data["hour_cos_fcst"] = np.cos(2 * np.pi * (data["생성일"].dt.hour + data["생성일"].dt.minute / 60) / 24)
    data["doy_sin_fcst"] = np.sin(2 * np.pi * data["생성일"].dt.dayofyear / 365.25)
    data["doy_cos_fcst"] = np.cos(2 * np.pi * data["생성일"].dt.dayofyear / 365.25)
    for inv in sorted(data["인버터번호"].unique()):
        data[f"inv_{inv}"] = (data["인버터번호"] == inv).astype(int)
    return [
        "발전량_lag1_raw", "발전량_lag2_raw", "발전량_lag4_raw",
        "TMP_fcst", "REH_fcst", "SKY_fcst", "cloud10_fcst",
        "solar_elevation", "solar_base", "irradiance_est",
        "hour_sin_fcst", "hour_cos_fcst", "doy_sin_fcst", "doy_cos_fcst",
        *[f"inv_{inv}" for inv in sorted(data["인버터번호"].unique())],
    ]


def fit_models(x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=3.0))])
    ridge.fit(x_train, y_train)
    lightgbm = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.2,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=-1,
    )
    lightgbm.fit(x_train, y_train)
    return {"Ridge": ridge, "LightGBM": lightgbm}


def evaluate_15min(data: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = stats["target"]
    forecast_features = model_features_15min(data)
    base_features = [
        "발전량_lag1_raw", "발전량_lag2_raw", "발전량_lag4_raw",
        "hour_sin_fcst", "hour_cos_fcst", "doy_sin_fcst", "doy_cos_fcst",
        *[f"inv_{inv}" for inv in sorted(data["인버터번호"].unique())],
    ]
    observed_features = base_features + [
        "일사량_obs", "일조_obs", "습도_obs", "지면온도_obs",
        "전운량_obs", "기온_obs", "풍속_obs", "강수량_obs",
    ]
    required = sorted(set(forecast_features + observed_features + [target]))
    usable = data.dropna(subset=required).copy()
    train = usable[(usable["source_split"] == "train") & (usable["생성일"] < TEST_START)]
    test = usable[
        (usable["source_split"] == "test")
        & (usable["생성일"] >= TEST_START)
        & (usable["생성일"] <= TEST_END)
    ].copy()

    forecast_models = fit_models(train[forecast_features], train[target])
    no_weather_model = fit_models(train[base_features], train[target])["LightGBM"]
    observed_model = fit_models(train[observed_features], train[target])["LightGBM"]
    preds = {
        "Persistence_lag1": test["발전량_lag1_raw"].to_numpy(),
        "Same_time_yesterday": test["same_time_yesterday"].to_numpy(),
        "LightGBM_no_weather": np.clip(no_weather_model.predict(test[base_features]), 0, 11),
        "Ridge_grid3h": np.clip(forecast_models["Ridge"].predict(test[forecast_features]), 0, 11),
        "LightGBM_grid3h": np.clip(forecast_models["LightGBM"].predict(test[forecast_features]), 0, 11),
        "LightGBM_observed_upper": np.clip(observed_model.predict(test[observed_features]), 0, 11),
    }
    joblib.dump({"model": no_weather_model, "features": base_features}, OUT / "15min_lightgbm_no_weather_model.joblib")
    joblib.dump({"model": forecast_models["Ridge"], "features": forecast_features}, OUT / "15min_ridge_grid3h_model.joblib")
    joblib.dump({"model": forecast_models["LightGBM"], "features": forecast_features}, OUT / "15min_lightgbm_grid3h_model.joblib")
    joblib.dump({"model": observed_model, "features": observed_features}, OUT / "15min_lightgbm_observed_upper_model.joblib")

    pred_out = test[["생성일", "인버터번호", target, "solar_elevation", "TMP_fcst", "SKY_fcst", "REH_fcst", "irradiance_est"]].copy()
    score_rows = []
    for name, pred in preds.items():
        pred_out[name] = pred
        for scope, mask in {
            "all": np.ones(len(test), dtype=bool),
            "daylight": test["solar_elevation"].to_numpy() > 0,
        }.items():
            valid = mask & np.isfinite(pred)
            row = {"horizon": "15min", "scope": scope, "model": name}
            row.update(metrics(test.loc[valid, target].to_numpy(), np.asarray(pred)[valid], 11.0, 0.1))
            score_rows.append(row)

    importance = pd.DataFrame({
        "feature": forecast_features,
        "importance_gain": forecast_models["LightGBM"].booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(OUT / "15min_lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(score_rows), pred_out


def build_hourly(data: pd.DataFrame, target: str) -> pd.DataFrame:
    work = data.copy()
    work["hour"] = work["생성일"].dt.floor("h")
    agg = {
        target: "sum",
        "TMP_fcst": "mean",
        "REH_fcst": "mean",
        "SKY_fcst": "first",
        "cloud10_fcst": "first",
        "solar_elevation": "mean",
        "solar_base": "mean",
        "irradiance_est": "mean",
        "일사량_obs": "sum",
        "일조_obs": "sum",
        "습도_obs": "mean",
        "지면온도_obs": "mean",
        "전운량_obs": "mean",
        "기온_obs": "mean",
        "풍속_obs": "mean",
        "강수량_obs": "sum",
        "source_split": "first",
        "생성일": "count",
    }
    hourly = work.groupby(["인버터번호", "hour"], as_index=False).agg(agg)
    hourly = hourly.rename(columns={"생성일": "quarter_count"})
    hourly = hourly[hourly["quarter_count"] == 4].copy()
    hourly = hourly.sort_values(["인버터번호", "hour"]).reset_index(drop=True)

    group = hourly.groupby("인버터번호")[target]
    for lag in (48, 72, 96, 120, 144, 168):
        hourly[f"generation_lag{lag}h"] = group.shift(lag)
    hourly["available_same_hour_mean"] = hourly[
        [f"generation_lag{lag}h" for lag in (48, 72, 96, 120, 144, 168)]
    ].mean(axis=1)
    hourly["hour_sin_fcst"] = np.sin(2 * np.pi * hourly["hour"].dt.hour / 24)
    hourly["hour_cos_fcst"] = np.cos(2 * np.pi * hourly["hour"].dt.hour / 24)
    hourly["doy_sin_fcst"] = np.sin(2 * np.pi * hourly["hour"].dt.dayofyear / 365.25)
    hourly["doy_cos_fcst"] = np.cos(2 * np.pi * hourly["hour"].dt.dayofyear / 365.25)
    for inv in sorted(hourly["인버터번호"].unique()):
        hourly[f"inv_{inv}"] = (hourly["인버터번호"] == inv).astype(int)
    return hourly


def evaluate_hourly(data: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = stats["target"]
    hourly = build_hourly(data, target)
    forecast_features = [
        "generation_lag48h", "generation_lag72h", "generation_lag96h",
        "generation_lag120h", "generation_lag144h", "generation_lag168h",
        "available_same_hour_mean",
        "TMP_fcst", "REH_fcst", "SKY_fcst", "cloud10_fcst",
        "solar_elevation", "solar_base", "irradiance_est",
        "hour_sin_fcst", "hour_cos_fcst", "doy_sin_fcst", "doy_cos_fcst",
        *[f"inv_{inv}" for inv in sorted(hourly["인버터번호"].unique())],
    ]
    base_features = [
        "generation_lag48h", "generation_lag72h", "generation_lag96h",
        "generation_lag120h", "generation_lag144h", "generation_lag168h",
        "available_same_hour_mean",
        "hour_sin_fcst", "hour_cos_fcst", "doy_sin_fcst", "doy_cos_fcst",
        *[f"inv_{inv}" for inv in sorted(hourly["인버터번호"].unique())],
    ]
    observed_features = base_features + [
        "일사량_obs", "일조_obs", "습도_obs", "지면온도_obs",
        "전운량_obs", "기온_obs", "풍속_obs", "강수량_obs",
    ]
    required = sorted(set(forecast_features + observed_features + [target]))
    usable = hourly.dropna(subset=required).copy()
    train = usable[(usable["source_split"] == "train") & (usable["hour"] < TEST_START)]
    test = usable[
        (usable["source_split"] == "test")
        & (usable["hour"] >= TEST_START)
        & (usable["hour"] <= TEST_END)
    ].copy()

    forecast_models = fit_models(train[forecast_features], train[target])
    no_weather_model = fit_models(train[base_features], train[target])["LightGBM"]
    observed_model = fit_models(train[observed_features], train[target])["LightGBM"]
    preds = {
        "Persistence_48h": test["generation_lag48h"].to_numpy(),
        "Available_same_hour_mean": test["available_same_hour_mean"].to_numpy(),
        "LightGBM_no_weather": np.clip(no_weather_model.predict(test[base_features]), 0, 44),
        "Ridge_grid3h": np.clip(forecast_models["Ridge"].predict(test[forecast_features]), 0, 44),
        "LightGBM_grid3h": np.clip(forecast_models["LightGBM"].predict(test[forecast_features]), 0, 44),
        "LightGBM_observed_upper": np.clip(observed_model.predict(test[observed_features]), 0, 44),
    }
    joblib.dump({"model": no_weather_model, "features": base_features}, OUT / "1hour_lightgbm_no_weather_model.joblib")
    joblib.dump({"model": forecast_models["Ridge"], "features": forecast_features}, OUT / "1hour_ridge_grid3h_model.joblib")
    joblib.dump({"model": forecast_models["LightGBM"], "features": forecast_features}, OUT / "1hour_lightgbm_grid3h_model.joblib")
    joblib.dump({"model": observed_model, "features": observed_features}, OUT / "1hour_lightgbm_observed_upper_model.joblib")

    pred_out = test[["hour", "인버터번호", target, "solar_elevation", "TMP_fcst", "SKY_fcst", "REH_fcst", "irradiance_est"]].copy()
    score_rows = []
    for name, pred in preds.items():
        pred_out[name] = pred
        for scope, mask in {
            "all": np.ones(len(test), dtype=bool),
            "daylight": test["solar_elevation"].to_numpy() > 0,
        }.items():
            valid = mask & np.isfinite(pred)
            row = {"horizon": "1hour_day_ahead", "scope": scope, "model": name}
            row.update(metrics(test.loc[valid, target].to_numpy(), np.asarray(pred)[valid], 44.0, 0.4))
            score_rows.append(row)

    importance = pd.DataFrame({
        "feature": forecast_features,
        "importance_gain": forecast_models["LightGBM"].booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(OUT / "1hour_lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(score_rows), pred_out


def weather_alignment(data: pd.DataFrame) -> pd.DataFrame:
    unique = data.drop_duplicates("생성일")
    test = unique[(unique["생성일"] >= TEST_START) & (unique["생성일"] <= TEST_END)]
    rows = []
    for name, observed, forecast in (
        ("temperature_C", "기온_obs", "TMP_fcst"),
        ("humidity_pct", "습도_obs", "REH_fcst"),
        ("cloud_0_10", "전운량_obs", "cloud10_fcst"),
        ("irradiance", "일사량_obs", "irradiance_est"),
    ):
        row = {"variable": name}
        row.update(metrics(test[observed].to_numpy(), test[forecast].to_numpy(), 1.0, 0.01))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(score15: pd.DataFrame, score1h: pd.DataFrame, weather: pd.DataFrame, audit: dict) -> None:
    def best_line(score: pd.DataFrame, horizon: str) -> str:
        day = score[(score["scope"] == "daylight") & (score["model"] == "LightGBM_grid3h")]
        best = day.sort_values("RMSE").iloc[0]
        return (
            f"- {horizon}: {best['model']} — RMSE {best['RMSE']:.4f}, "
            f"MAE {best['MAE']:.4f}, MAPE {best['MAPE_pct']:.1f}%, R² {best['R2']:.4f}"
        )

    def score_table(score: pd.DataFrame) -> str:
        day = score[score["scope"] == "daylight"]
        lines = ["| 모델 | 표본수 | MAE | RMSE | MAPE | R² |", "|---|---:|---:|---:|---:|---:|"]
        for row in day.itertuples():
            lines.append(
                f"| {row.model} | {int(row.n):,} | {row.MAE:.4f} | {row.RMSE:.4f} | "
                f"{row.MAPE_pct:.1f}% | {row.R2:.4f} |"
            )
        return "\n".join(lines)

    def value(score: pd.DataFrame, model: str, metric_name: str) -> float:
        row = score[(score["scope"] == "daylight") & (score["model"] == model)].iloc[0]
        return float(row[metric_name])

    rmse15_no = value(score15, "LightGBM_no_weather", "RMSE")
    rmse15_fc = value(score15, "LightGBM_grid3h", "RMSE")
    rmse15_obs = value(score15, "LightGBM_observed_upper", "RMSE")
    rmse1_no = value(score1h, "LightGBM_no_weather", "RMSE")
    rmse1_fc = value(score1h, "LightGBM_grid3h", "RMSE")
    rmse1_obs = value(score1h, "LightGBM_observed_upper", "RMSE")

    temp_mae = float(weather.loc[weather["variable"] == "temperature_C", "MAE"].iloc[0])
    reh_mae = float(weather.loc[weather["variable"] == "humidity_pct", "MAE"].iloc[0])
    cloud_mae = float(weather.loc[weather["variable"] == "cloud_0_10", "MAE"].iloc[0])

    report = f"""# 광주 3시간 격자예보 650일 잠정 검증

> **상태: 진단용·공식 기준 불충족.** 이 보고서의 15분 결과는 `+15분` 단일수평이며,
> 공식 기준인 15분 출력단위 `+1h/+2h/+3h/+4h`를 검증하지 않았다. 1시간 결과도
> 오전 10~11시 배치 기준 `+24h~+48h` 수평별 운영을 명시적으로 재현하지 않았다.
> 따라서 아래 수치는 공식 성능표나 최종 운영성능으로 사용하지 않는다.

## 결론

현재 확보된 650일 TMP·SKY·REH 예보만 사용한 잠정 검증이다. 원본 710일 중
2026-06-06~2026-08-04 발표분 60일과 WSD·POP는 미수집 상태이므로 최종 성능이 아니다.

{best_line(score15, '초단기 15분')}
{best_line(score1h, '1시간 단위 익일')}

- 15분: 3시간 예보 추가 시 기상 없는 LightGBM 대비 RMSE {(rmse15_no-rmse15_fc)/rmse15_no*100:.1f}% 개선
- 1시간 익일: 3시간 예보 추가 시 기상 없는 LightGBM 대비 RMSE {(rmse1_no-rmse1_fc)/rmse1_no*100:.1f}% 개선
- 관측기상 상한선 대비 RMSE 격차: 15분 {(rmse15_fc/rmse15_obs-1)*100:.1f}%, 1시간 익일 {(rmse1_fc/rmse1_obs-1)*100:.1f}%

## 낮시간 성능표

### 초단기 15분

{score_table(score15)}

### 1시간 단위 익일

{score_table(score1h)}

## 예보 입력 자체의 관측 대비 오차

- 기온 MAE: {temp_mae:.2f}°C
- 습도 MAE: {reh_mae:.2f}%p
- 운량 MAE: {cloud_mae:.2f}/10

## 실험 범위

- 예보 발표일: {audit['issue_start']}~{audit['issue_end']} ({audit['issue_rows']}일)
- 예측 대상 예보범위: {audit['target_start']}~{audit['target_end']}
- 고정 시험기간: {TEST_START}~{TEST_END}
- 예보요소: TMP, SKY, REH
- TMP·REH: 같은 목표일 안에서 3시간→15분 선형보간
- SKY: 같은 목표일 안에서 전방채움
- 일사량 대용치: 학습구간에서만 월×SKY 보정계수를 학습하고 태양고도와 결합

## 해석 주의

- `LightGBM_grid3h`와 `Ridge_grid3h`에는 관측 일사량·일조·지면온도·풍속·강수량을 사용하지 않았다.
- `LightGBM_observed_upper`만 비교용 상한선으로 시험구간 관측기상을 사용했다.
- 15분 모델은 예측 직전 발전량 lag를 사용한다.
- 1시간 익일 모델은 05시 발표 이후 정보 누출을 피하기 위해 최소 48시간 이전 발전량만 사용한다.
- NMAE·NRMSE 기준값은 기존 전처리 범위에 맞춰 인버터별 15분 11, 1시간 44를 사용했다.
- MAPE는 야간 및 극소 발전량 왜곡을 줄이기 위해 각각 0.1, 0.4 초과 표본에서 계산했다.
- 기존 실측기상 3단계 실험 및 v3 다중시간수평 실험과 별도 계보로 관리한다.
- 12시간 예보 원본 파일은 현재 로컬 검색에서 확인되지 않아 이번 동일기간 표에는 포함하지 않았다.

## 생성 파일

- `forecast_650d_combined.csv`: 두 원본 예보 CSV를 합친 작업 사본
- `weather_alignment_metrics.csv`: 관측기상 대비 예보·추정 입력 오차
- `15min_scorecard.csv`, `15min_predictions.csv`
- `1hour_scorecard.csv`, `1hour_predictions.csv`
- LightGBM·Ridge 모델 및 특징중요도
- `experiment_metadata.json`, `data_quality_audit.json`, `irradiance_mapping.json`
"""
    (OUT / "잠정검증_분석보고서.md").write_text(report, encoding="utf-8")


def main() -> None:
    forecast, audit = build_forecast_15min()
    # 원본 보존: 통합본은 결과 폴더에만 새로 저장한다.
    issue_copy = pd.concat(
        [pd.read_csv(BASE_CSV_1, dtype={"발표일": str}), pd.read_csv(BASE_CSV_2, dtype={"발표일": str})],
        ignore_index=True,
    ).drop_duplicates("발표일", keep="last").sort_values("발표일")
    issue_copy.to_csv(OUT / "forecast_650d_combined.csv", index=False, encoding="utf-8-sig")
    (OUT / "data_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    data, stats, _ = load_model_rows(forecast)
    data, mapping = fit_irradiance_mapping(data)
    (OUT / "irradiance_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    weather = weather_alignment(data)
    score15, pred15 = evaluate_15min(data, stats)
    score1h, pred1h = evaluate_hourly(data, stats)

    weather.to_csv(OUT / "weather_alignment_metrics.csv", index=False, encoding="utf-8-sig")
    score15.to_csv(OUT / "15min_scorecard.csv", index=False, encoding="utf-8-sig")
    pred15.to_csv(OUT / "15min_predictions.csv", index=False, encoding="utf-8-sig")
    score1h.to_csv(OUT / "1hour_scorecard.csv", index=False, encoding="utf-8-sig")
    pred1h.to_csv(OUT / "1hour_predictions.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "experiment_id": "GWANGJU_GRID_FCST_650D_INTERIM_20260818",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lineage": "3-stage forecast-based provisional validation; separate from v3 multihorizon and observed-weather experiments",
        "forecast_issue_period": [audit["issue_start"], audit["issue_end"]],
        "target_forecast_period": [audit["target_start"], audit["target_end"]],
        "test_period": [str(TEST_START), str(TEST_END)],
        "forecast_variables": ["TMP", "SKY", "REH"],
        "missing_variables": ["WSD", "POP"],
        "preprocessing_version": "grid650_interim_v1",
        "software": {"python": "3.14", "lightgbm": lgb.__version__},
        "units": {"target": "original 주기별발전량_최종 units per inverter", "TMP": "C", "REH": "%", "SKY": "KMA category 1-4"},
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(score15, score1h, weather, audit)

    print(f"완료: {OUT}")
    print("[15분 낮시간]")
    print(score15[score15["scope"] == "daylight"][["model", "n", "MAE", "RMSE", "MAPE_pct", "R2"]].to_string(index=False))
    print("[1시간 익일 낮시간]")
    print(score1h[score1h["scope"] == "daylight"][["model", "n", "MAE", "RMSE", "MAPE_pct", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
