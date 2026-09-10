"""익일 일간 총량 직접모델을 학습하고 10시 발행 1시간 예측과 계층 조정한다."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from model_common import expanding_folds, optimize_nonnegative_weights, regression_metrics, write_json
from pv_pipeline import ROOT, load_config


OUT = ROOT / "outputs" / "일간_계층조정"
HOUR_ENSEMBLE = ROOT / "outputs" / "1시간_앙상블"
KMA_MODEL_DATA = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\gwangju_1hour_model_dataset_kma_observed.csv")  # 08-20 경로 정비: 원래 존재하지 않던 pv_environment_data/processed 경로를 실제 위치로 교정. 주의: 이 파일은 여전히 open_meteo_forecast_* 컬럼(비공식)에 의존하므로 공식 결과로 쓰려면 이 스크립트의 특성 로직을 build_official_hourly_dataset_v1.py 산출물(같은 폴더의 processed/gwangju_1hour_model_dataset_official_v1_2026-08-20.csv, Open-Meteo 없음) 기준으로 다시 짜야 한다 — 5/6번 단계 작업.
FORECAST_NAMES = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
]
ACTUAL = "실제_익일발전량_kWh"


def energy_metrics(actual, predicted) -> dict:
    return {key.replace("_kW", "_kWh"): value for key, value in regression_metrics(actual, predicted).items()}


def load_hourly() -> pd.DataFrame:
    return pd.read_csv(KMA_MODEL_DATA, parse_dates=["time"], low_memory=False).set_index("time").sort_index()


def build_daily_dataset(hourly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    daily_actual = pd.read_parquet(ROOT / "outputs" / "집계_일간_실제발전량.parquet").copy()
    daily_actual.index = pd.to_datetime(daily_actual.index)
    data = pd.DataFrame(index=daily_actual.index)
    data[ACTUAL] = daily_actual["일간발전량_kWh"]
    data["7일전_일간발전량_kWh"] = data[ACTUAL].shift(7)
    data["2일전_일간발전량_kWh"] = data[ACTUAL].shift(2)
    data["2일전기준_7일이동평균_kWh"] = data[ACTUAL].shift(2).rolling(7, min_periods=4).mean()
    data["2일전기준_30일이동평균_kWh"] = data[ACTUAL].shift(2).rolling(30, min_periods=15).mean()
    data["2일전기준_30일표준편차_kWh"] = data[ACTUAL].shift(2).rolling(30, min_periods=15).std()

    # 매일 10시 발행을 기준으로 목표일 00~10시는 1일 전 예보,
    # 11~23시는 2일 전 예보 칸에서 해당 목표시각 예보를 선택한다.
    chosen = pd.DataFrame(index=hourly.index)
    earlier = hourly.index.hour <= 10
    for name in FORECAST_NAMES:
        chosen[name] = np.where(
            earlier,
            hourly[f"open_meteo_forecast_{name}_previous_day1"],
            hourly[f"open_meteo_forecast_{name}_previous_day2"],
        )
    chosen["날짜"] = chosen.index.normalize()
    aggregation = {
        "temperature_2m": ["mean", "min", "max"],
        "relative_humidity_2m": ["mean", "min", "max"],
        "precipitation": ["sum", "max"],
        "cloud_cover": ["mean", "max"],
        "shortwave_radiation": ["sum", "max"],
        "direct_normal_irradiance": ["sum", "max"],
        "diffuse_radiation": ["sum", "max"],
        "wind_speed_10m": ["mean", "max"],
        "wind_direction_10m": ["mean"],
        "surface_pressure": ["mean", "min", "max"],
    }
    forecast_daily = chosen.groupby("날짜").agg(aggregation)
    forecast_daily.columns = [f"목표일예보_{name}_{stat}" for name, stat in forecast_daily.columns]
    data = data.join(forecast_daily, how="left")

    equipment = hourly[[
        "plant_input_power_kw", "mean_input_voltage_v", "mean_frequency_hz",
        "mean_power_factor", "mean_inverter_temperature_c", "mean_communication_ok",
        "inverters_available", "기상청관측_기온_C", "기상청관측_상대습도_pct",
        "기상청관측_강수량_mm", "기상청관측_전운량_pct", "기상청관측_일사량_W_m2",
        "기상청관측_일조시간_hr", "기상청관측_풍속_m_s", "기상청관측_풍향_deg",
        "기상청관측_현지기압_hPa", "기상청관측_해면기압_hPa",
        "기상청관측_적설_cm", "기상청관측_지면온도_C",
    ]].copy()
    equipment["날짜"] = equipment.index.normalize()
    previous = equipment.groupby("날짜").mean(numeric_only=True).shift(2)
    previous.columns = [f"2일전평균_{name}" for name in previous.columns]
    data = data.join(previous, how="left")

    day = data.index.dayofyear
    data["목표일_연주기_sin"] = np.sin(2 * np.pi * day / 365.25)
    data["목표일_연주기_cos"] = np.cos(2 * np.pi * day / 365.25)
    data["목표일_월"] = data.index.month
    data["목표일_요일"] = data.index.dayofweek
    data["해발고도_m"] = float(hourly["site_elevation_dem_m"].dropna().median())
    data["설비용량_kW"] = float(hourly["reported_capacity_kw"].dropna().median())
    data["지형경사_deg"] = float(hourly["terrain_slope_deg_dem_approx"].dropna().median())
    data["지형방위_deg"] = float(hourly["terrain_aspect_deg_dem_approx"].dropna().median())
    data["7일전지속성예측_kWh"] = data["7일전_일간발전량_kWh"]
    features = [name for name in data.columns if name not in {ACTUAL, "7일전지속성예측_kWh"}]
    data = data.dropna(subset=[ACTUAL, "7일전지속성예측_kWh"])
    data.index.name = "예측대상일"
    return data, features


def new_models(seed: int):
    return {
        "LightGBM": lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=500, learning_rate=0.025,
            num_leaves=15, max_depth=6, min_child_samples=14,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbosity=-1, n_jobs=4,
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:absoluteerror", n_estimators=500, learning_rate=0.025,
            max_depth=4, min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, random_state=seed, n_jobs=4,
        ),
    }


def train_direct(data: pd.DataFrame, features: list[str], config: dict):
    seed = int(config["random_seed"])
    capacity_daily = float(config["site"]["capacity_kw"]) * 24
    oof_parts = []
    for train_mask, valid_mask, fold_name in expanding_folds(pd.Series(data.index), config):
        train, valid = data.iloc[train_mask].copy(), data.iloc[valid_mask].copy()
        medians = train[features].median(numeric_only=True)
        part = pd.DataFrame(index=valid.index)
        part[ACTUAL] = valid[ACTUAL]
        part["7일전지속성예측_kWh"] = valid["7일전지속성예측_kWh"]
        for name, model in new_models(seed).items():
            model.fit(train[features].fillna(medians), train[ACTUAL])
            part[f"{name}예측_kWh"] = np.clip(model.predict(valid[features].fillna(medians)), 0, capacity_daily)
        part["검증구간"] = fold_name
        oof_parts.append(part.reset_index())
    oof = pd.concat(oof_parts, ignore_index=True)
    direct_candidates = ["7일전지속성예측_kWh", "LightGBM예측_kWh", "XGBoost예측_kWh"]
    result = optimize_nonnegative_weights(oof[ACTUAL], oof[direct_candidates])
    weights = np.asarray(result["가중치"], dtype=float)
    oof["직접모델_자동가중예측_kWh"] = np.clip(oof[direct_candidates].to_numpy() @ weights, 0, capacity_daily)

    test_start = pd.Timestamp(config["final_test"]["start"])
    test_end = pd.Timestamp(config["final_test"]["end"])
    train = data[data.index < test_start]
    test = data[(data.index >= test_start) & (data.index <= test_end)].copy()
    medians = train[features].median(numeric_only=True)
    result_test = pd.DataFrame(index=test.index)
    result_test[ACTUAL] = test[ACTUAL]
    result_test["7일전지속성예측_kWh"] = test["7일전지속성예측_kWh"]
    final_models = {}
    for name, model in new_models(seed).items():
        model.fit(train[features].fillna(medians), train[ACTUAL])
        result_test[f"{name}예측_kWh"] = np.clip(model.predict(test[features].fillna(medians)), 0, capacity_daily)
        final_models[name] = model
    result_test["직접모델_자동가중예측_kWh"] = np.clip(result_test[direct_candidates].to_numpy() @ weights, 0, capacity_daily)
    payload = {**result, "모델별가중치": {name.replace("예측_kWh", ""): float(value) for name, value in zip(direct_candidates, weights)}}
    joblib.dump({"models": final_models, "features": features, "medians": medians.to_dict(), "weights": payload}, OUT / "일간_직접모델.joblib")
    return oof, result_test.reset_index(), payload


def hourly_daily_totals(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path).copy()
    frame["예측발행시각"] = pd.to_datetime(frame["예측발행시각"])
    frame["예측대상시각"] = pd.to_datetime(frame["예측대상시각"])
    frame = frame[
        frame["예측발행시각"].dt.hour.eq(10)
        & frame["예측수평_시간"].between(14, 37)
        & frame["예측대상시각"].dt.normalize().eq(frame["예측발행시각"].dt.normalize() + pd.Timedelta(days=1))
    ].copy()
    frame["예측대상일"] = frame["예측대상시각"].dt.normalize()
    totals = frame.groupby("예측대상일", as_index=False).agg(
        시간모델합계예측_kWh=("자동가중앙상블예측_kW", "sum"),
        예측된낮시간수=("자동가중앙상블예측_kW", "size"),
    )
    return totals, frame


def reconcile(oof: pd.DataFrame, test: pd.DataFrame, config: dict):
    hourly_oof, _ = hourly_daily_totals(HOUR_ENSEMBLE / "교차검증_자동가중예측.parquet")
    hourly_test, detail_test = hourly_daily_totals(HOUR_ENSEMBLE / "최종시험_공통표본_예측.parquet")
    oof_join = oof.merge(hourly_oof, on="예측대상일", how="inner")
    test_join = test.merge(hourly_test, on="예측대상일", how="inner")
    candidates = ["직접모델_자동가중예측_kWh", "시간모델합계예측_kWh"]
    result = optimize_nonnegative_weights(oof_join[ACTUAL], oof_join[candidates])
    weights = np.asarray(result["가중치"], dtype=float)
    for frame in [oof_join, test_join]:
        frame["계층조정_일간예측_kWh"] = np.clip(
            frame[candidates].to_numpy() @ weights,
            0,
            float(config["site"]["capacity_kw"]) * 24,
        )
    payload = {**result, "구성별가중치": {name.replace("예측_kWh", ""): float(value) for name, value in zip(candidates, weights)}}

    detail_test = detail_test.merge(test_join[["예측대상일", "계층조정_일간예측_kWh"]], on="예측대상일", how="inner")
    sums = detail_test.groupby("예측대상일")["자동가중앙상블예측_kW"].transform("sum")
    detail_test["계층조정배율"] = np.where(sums > 0, detail_test["계층조정_일간예측_kWh"] / sums, 0)
    detail_test["계층조정_1시간예측_kW"] = np.clip(
        detail_test["자동가중앙상블예측_kW"] * detail_test["계층조정배율"],
        0,
        float(config["site"]["capacity_kw"]),
    )
    return oof_join, test_join, detail_test, payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    hourly = load_hourly()
    data, features = build_daily_dataset(hourly)
    oof, test, direct_weights = train_direct(data, features, config)
    oof_final, test_final, detail, hierarchy_weights = reconcile(oof, test, config)
    score_rows = []
    for name in ["7일전지속성예측_kWh", "LightGBM예측_kWh", "XGBoost예측_kWh", "직접모델_자동가중예측_kWh", "시간모델합계예측_kWh", "계층조정_일간예측_kWh"]:
        score_rows.append({"모델": name.replace("예측_kWh", ""), **energy_metrics(test_final[ACTUAL], test_final[name])})
    scores = pd.DataFrame(score_rows)
    oof_final.to_parquet(OUT / "교차검증_계층조정예측.parquet")
    test_final.to_parquet(OUT / "최종시험_일간예측.parquet")
    detail.to_parquet(OUT / "최종시험_계층조정_1시간상세.parquet")
    scores.to_csv(OUT / "최종시험_일간성능표.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "일간_직접모델_자동가중치.json", direct_weights)
    write_json(OUT / "일간_시간계층_자동가중치.json", hierarchy_weights)
    write_json(OUT / "자료정보.json", {"일간전체유효표본": len(data), "특성수": len(features), "계층조정시험표본": len(test_final)})
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
