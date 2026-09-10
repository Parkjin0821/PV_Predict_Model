"""5분마다 발행하는 15분 출력단위 +1~4시간 초단기 기준모델."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model_common import expanding_folds, regression_metrics, write_json
from pv_pipeline import OUTPUT as DATA_OUTPUT
from pv_pipeline import ROOT, load_config, solar_position


MODEL_OUTPUT = ROOT / "outputs" / "초단기_기준모델"


def base_issue_features(five: pd.DataFrame) -> pd.DataFrame:
    power = five["발전출력_kW"]
    features = pd.DataFrame(index=five.index)
    for minutes in [5, 15, 30, 60, 120, 240, 1440]:
        features[f"발전출력_{minutes}분전_kW"] = power.shift(minutes // 5)
    for minutes in [30, 60, 240]:
        window = minutes // 5
        features[f"발전출력_{minutes}분이동평균_kW"] = power.shift(1).rolling(window, min_periods=window).mean()
        features[f"발전출력_{minutes}분이동표준편차_kW"] = power.shift(1).rolling(window, min_periods=window).std()
    for column in [
        "입력전력_kW",
        "입력전류_A",
        "입력전압평균_V",
        "주파수평균_Hz",
        "역률평균",
        "인버터평균온도_C",
        "통신정상비율",
        "가용인버터수",
    ]:
        features[f"직전_{column}"] = five[column].shift(1)
    minute = features.index.hour * 60 + features.index.minute
    features["발행시각_일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    features["발행시각_일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    features["발행시각_연주기_sin"] = np.sin(2 * np.pi * features.index.dayofyear / 365.25)
    features["발행시각_연주기_cos"] = np.cos(2 * np.pi * features.index.dayofyear / 365.25)
    return features


def horizon_frame(
    five: pd.DataFrame,
    forecast: pd.DataFrame,
    base: pd.DataFrame,
    horizon_minutes: int,
    config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    steps = horizon_minutes // 5
    target_time = five.index + pd.Timedelta(minutes=horizon_minutes)
    target_interval_power = five["발전출력_kW"].rolling(3, min_periods=3).mean().shift(-2)
    frame = base.copy()
    frame["예측발행시각"] = frame.index
    frame["예측대상시각"] = target_time
    frame["실제_15분평균출력_kW"] = target_interval_power.shift(-steps)

    weather = forecast.reindex(target_time)
    for column in ["기온예보_C", "하늘상태예보", "습도예보_pct", "예보운량_10분율"]:
        frame[f"목표_{column}"] = weather[column].to_numpy()
    frame["목표_예보발표시각"] = weather["예보발표시각"].to_numpy()
    frame["예보시점유효"] = (
        pd.to_datetime(frame["목표_예보발표시각"]) <= pd.to_datetime(frame["예측발행시각"])
    ).astype("int8")

    elevation, azimuth = solar_position(
        target_time,
        float(config["site"]["latitude"]),
        float(config["site"]["longitude"]),
    )
    frame["목표_태양고도_deg"] = elevation
    frame["목표_태양방위_sin"] = np.sin(np.deg2rad(azimuth))
    frame["목표_태양방위_cos"] = np.cos(np.deg2rad(azimuth))
    target_minute = target_time.hour * 60 + target_time.minute
    frame["목표시각_일주기_sin"] = np.sin(2 * np.pi * target_minute / 1440)
    frame["목표시각_일주기_cos"] = np.cos(2 * np.pi * target_minute / 1440)
    frame["목표시각_연주기_sin"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    frame["목표시각_연주기_cos"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)
    frame["예측수평_분"] = horizon_minutes
    frame["지속성예측_kW"] = frame["발전출력_5분전_kW"]
    frame["목표_물리적낮"] = (frame["목표_태양고도_deg"] > 0).astype("int8")
    frame["목표_핵심낮시간"] = (frame["목표_태양고도_deg"] > 3).astype("int8")
    features = [
        column
        for column in frame.columns
        if column
        not in {
            "예측발행시각",
            "예측대상시각",
            "실제_15분평균출력_kW",
            "목표_예보발표시각",
            "지속성예측_kW",
            "목표_물리적낮",
            "목표_핵심낮시간",
        }
    ]
    required = features + ["실제_15분평균출력_kW", "지속성예측_kW"]
    frame = frame[frame["예보시점유효"].eq(1)].dropna(subset=required).copy()
    return frame, features


def new_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=280,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.3,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def main() -> None:
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    five = pd.read_parquet(DATA_OUTPUT / "정제_5분_기준자료.parquet")
    forecast = pd.read_parquet(DATA_OUTPUT / "기상청_650일_5분확장예보.parquet")
    base = base_issue_features(five)
    test_start = pd.Timestamp(config["final_test"]["start"])
    test_end = pd.Timestamp(config["final_test"]["end"])
    capacity = float(config["site"]["capacity_kw"])

    all_oof, all_test, scores, metadata = [], [], [], {}
    for horizon in config["ultra_short"]["horizon_minutes"]:
        print(f"초단기 기준모델: +{horizon}분")
        frame, features = horizon_frame(five, forecast, base, int(horizon), config)
        daylight = frame["목표_물리적낮"].eq(1)
        model_rows = frame[daylight].copy()
        fold_rows = []
        for train_mask, valid_mask, fold_name in expanding_folds(model_rows["예측대상시각"], config):
            train, valid = model_rows.iloc[train_mask], model_rows.iloc[valid_mask]
            medians = train[features].median(numeric_only=True)
            model = new_model(int(config["random_seed"]))
            model.fit(train[features].fillna(medians), train["실제_15분평균출력_kW"])
            prediction = np.clip(model.predict(valid[features].fillna(medians)), 0, capacity)
            part = valid[["예측발행시각", "예측대상시각", "실제_15분평균출력_kW", "지속성예측_kW"]].copy()
            part["LightGBM예측_kW"] = prediction
            part["예측수평_분"] = int(horizon)
            part["검증구간"] = fold_name
            fold_rows.append(part)
        if fold_rows:
            all_oof.append(pd.concat(fold_rows, ignore_index=True))

        train = model_rows[model_rows["예측대상시각"] < test_start]
        test = model_rows[
            (model_rows["예측대상시각"] >= test_start)
            & (model_rows["예측대상시각"] <= test_end)
        ].copy()
        medians = train[features].median(numeric_only=True)
        model = new_model(int(config["random_seed"]))
        model.fit(train[features].fillna(medians), train["실제_15분평균출력_kW"])
        test["LightGBM예측_kW"] = np.clip(model.predict(test[features].fillna(medians)), 0, capacity)
        test["예측수평_분"] = int(horizon)
        all_test.append(
            test[[
                "예측발행시각",
                "예측대상시각",
                "예측수평_분",
                "실제_15분평균출력_kW",
                "지속성예측_kW",
                "LightGBM예측_kW",
                "목표_태양고도_deg",
            ]]
        )
        joblib.dump(
            {"model": model, "features": features, "medians": medians.to_dict(), "horizon_minutes": int(horizon)},
            MODEL_OUTPUT / f"LightGBM_{int(horizon):03d}분.joblib",
        )
        for name, column in [("지속성", "지속성예측_kW"), ("LightGBM", "LightGBM예측_kW")]:
            score = {
                "예측수평_분": int(horizon),
                "모델": name,
                **regression_metrics(test["실제_15분평균출력_kW"], test[column]),
            }
            score["목표통과"] = bool(
                score.get("가중절대비율오차_pct", 999) <= config["acceptance"]["maximum_wape_pct"]
                and score.get("결정계수", -999) >= config["acceptance"]["minimum_r2"]
            )
            scores.append(score)
        metadata[str(horizon)] = {
            "학습표본수": int(len(train)),
            "시험표본수": int(len(test)),
            "특성수": len(features),
            "시험시작": str(test["예측대상시각"].min()),
            "시험종료": str(test["예측대상시각"].max()),
        }

    pd.concat(all_oof, ignore_index=True).to_parquet(MODEL_OUTPUT / "교차검증_비표본예측.parquet")
    pd.concat(all_test, ignore_index=True).to_parquet(MODEL_OUTPUT / "최종시험_예측.parquet")
    pd.DataFrame(scores).to_csv(MODEL_OUTPUT / "최종시험_성능표.csv", index=False, encoding="utf-8-sig")
    write_json(MODEL_OUTPUT / "학습정보.json", metadata)
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()
