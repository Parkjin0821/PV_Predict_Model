"""1시간마다 +1~48시간을 직접 예측하는 LightGBM 기준모델."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model_common import expanding_folds, regression_metrics, write_json
from pv_pipeline import ROOT, load_config, solar_position


OUT = ROOT / "outputs" / "1시간_기준모델"
KMA_MODEL_DATA = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\gwangju_1hour_model_dataset_kma_observed.csv")  # 08-20 경로 정비: 원래 존재하지 않던 pv_environment_data/processed 경로를 실제 위치로 교정. 주의: 이 파일은 여전히 open_meteo_forecast_* 컬럼(비공식)에 의존하므로 공식 결과로 쓰려면 이 스크립트의 특성 로직을 build_official_hourly_dataset_v1.py 산출물(같은 폴더의 processed/gwangju_1hour_model_dataset_official_v1_2026-08-20.csv, Open-Meteo 없음) 기준으로 다시 짜야 한다 — 5/6번 단계 작업.
FORECAST_NAMES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


def load_hourly() -> pd.DataFrame:
    return pd.read_csv(KMA_MODEL_DATA, parse_dates=["time"], low_memory=False).set_index("time").sort_index()


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    power = frame["plant_output_kw"]
    out = pd.DataFrame(index=frame.index)
    for hours in [1, 2, 3, 6, 12, 24, 48, 168]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours)
    for hours in [6, 24, 168]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.shift(1).rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.shift(1).rolling(hours, min_periods=max(3, hours // 2)).std()
    for source in [
        "plant_input_power_kw",
        "plant_input_current_a",
        "mean_input_voltage_v",
        "mean_frequency_hz",
        "mean_power_factor",
        "mean_inverter_temperature_c",
        "mean_communication_ok",
        "inverters_available",
    ]:
        out[f"직전_{source}"] = frame[source].shift(1)
    for source in [
        "기상청관측_기온_C",
        "기상청관측_상대습도_pct",
        "기상청관측_강수량_mm",
        "기상청관측_전운량_pct",
        "기상청관측_일사량_W_m2",
        "기상청관측_일조시간_hr",
        "기상청관측_풍속_m_s",
        "기상청관측_풍향_deg",
        "기상청관측_현지기압_hPa",
        "기상청관측_해면기압_hPa",
        "기상청관측_적설_cm",
        "기상청관측_지면온도_C",
    ]:
        out[f"직전_{source}"] = frame[source].shift(1)
    minute = out.index.hour * 60 + out.index.minute
    out["발행시각_일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["발행시각_일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["발행시각_연주기_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["발행시각_연주기_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    return out


def horizon_frame(frame: pd.DataFrame, base: pd.DataFrame, horizon: int, config: dict) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy()
    target_time = frame.index + pd.Timedelta(hours=horizon)
    out["예측발행시각"] = out.index
    out["예측대상시각"] = target_time
    out["실제_1시간평균출력_kW"] = frame["plant_output_kw"].shift(-horizon)
    forecast_day = 1 if horizon <= 24 else 2
    for name in FORECAST_NAMES:
        out[f"목표예보_{name}"] = frame[f"open_meteo_forecast_{name}_previous_day{forecast_day}"].shift(-horizon)
    elevation, azimuth = solar_position(
        target_time,
        float(config["site"]["latitude"]),
        float(config["site"]["longitude"]),
    )
    out["목표_태양고도_deg"] = elevation
    out["목표_태양방위_sin"] = np.sin(np.deg2rad(azimuth))
    out["목표_태양방위_cos"] = np.cos(np.deg2rad(azimuth))
    target_minute = target_time.hour * 60 + target_time.minute
    out["목표시각_일주기_sin"] = np.sin(2 * np.pi * target_minute / 1440)
    out["목표시각_일주기_cos"] = np.cos(2 * np.pi * target_minute / 1440)
    out["목표시각_연주기_sin"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    out["목표시각_연주기_cos"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)
    out["예측수평_시간"] = horizon
    out["직전출력지속성예측_kW"] = out["발전출력_1시간전_kW"]
    # +48시간 이내에는 목표시각의 1주 전 실제값이 항상 발행시각 이전이다.
    out["동시간1주전예측_kW"] = frame["plant_output_kw"].shift(168 - horizon)
    out["목표_물리적낮"] = (out["목표_태양고도_deg"] > 0).astype("int8")
    features = [
        name
        for name in out.columns
        if name
        not in {
            "예측발행시각",
            "예측대상시각",
            "실제_1시간평균출력_kW",
            "직전출력지속성예측_kW",
            "동시간1주전예측_kW",
            "목표_물리적낮",
        }
    ]
    out = out.dropna(subset=features + ["실제_1시간평균출력_kW", "동시간1주전예측_kW"])
    return out, features


def new_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.3,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    frame = load_hourly()
    base = base_features(frame)
    capacity = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    test_start, test_end = pd.Timestamp(config["final_test"]["start"]), pd.Timestamp(config["final_test"]["end"])
    oof_parts, test_parts, scores, metadata = [], [], [], {}
    for horizon in range(1, 49):
        print(f"1시간 기준모델: +{horizon}시간")
        data, features = horizon_frame(frame, base, horizon, config)
        daylight = data[data["목표_물리적낮"].eq(1)].copy()
        folds = []
        for train_mask, valid_mask, fold_name in expanding_folds(daylight["예측대상시각"], config):
            train, valid = daylight.iloc[train_mask], daylight.iloc[valid_mask]
            medians = train[features].median(numeric_only=True)
            model = new_model(seed)
            model.fit(train[features].fillna(medians), train["실제_1시간평균출력_kW"])
            part = valid[["예측발행시각", "예측대상시각", "실제_1시간평균출력_kW", "직전출력지속성예측_kW", "동시간1주전예측_kW"]].copy()
            part["LightGBM예측_kW"] = np.clip(model.predict(valid[features].fillna(medians)), 0, capacity)
            part["예측수평_시간"] = horizon
            part["검증구간"] = fold_name
            folds.append(part)
        if folds:
            oof_parts.append(pd.concat(folds, ignore_index=True))
        train = daylight[daylight["예측대상시각"] < test_start]
        test = daylight[(daylight["예측대상시각"] >= test_start) & (daylight["예측대상시각"] <= test_end)].copy()
        medians = train[features].median(numeric_only=True)
        model = new_model(seed)
        model.fit(train[features].fillna(medians), train["실제_1시간평균출력_kW"])
        test["LightGBM예측_kW"] = np.clip(model.predict(test[features].fillna(medians)), 0, capacity)
        test["예측수평_시간"] = horizon
        test_parts.append(test[["예측발행시각", "예측대상시각", "예측수평_시간", "실제_1시간평균출력_kW", "직전출력지속성예측_kW", "동시간1주전예측_kW", "LightGBM예측_kW", "목표_태양고도_deg"]])
        joblib.dump(
            {"model": model, "features": features, "medians": medians.to_dict(), "horizon_hours": horizon},
            OUT / f"LightGBM_{horizon:02d}시간.joblib",
        )
        if 24 <= horizon <= 48:
            for model_name, column in [("동시간1주전", "동시간1주전예측_kW"), ("LightGBM", "LightGBM예측_kW")]:
                scores.append({"예측수평_시간": horizon, "모델": model_name, **regression_metrics(test["실제_1시간평균출력_kW"], test[column])})
        metadata[str(horizon)] = {"학습표본수": int(len(train)), "시험표본수": int(len(test)), "특성수": len(features)}
    pd.concat(oof_parts, ignore_index=True).to_parquet(OUT / "교차검증_비표본예측.parquet")
    pd.concat(test_parts, ignore_index=True).to_parquet(OUT / "최종시험_예측.parquet")
    pd.DataFrame(scores).to_csv(OUT / "공식24_48시간_성능표.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "학습정보.json", metadata)
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()
