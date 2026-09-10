# -*- coding: utf-8 -*-
"""부안 총출력 개선판(작업순서 5번, 기상특성 포함) - expanding-window
walk-forward(발행일 기준). config/부안_기상포함_walkforward_v1_2026-08-28.json
에 사전동결된 규칙을 그대로 따른다. API 호출 없음, Codex의 결합
산출물(부안_과거발전_ASOS_NWP_결합.parquet)만 읽기전용으로 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "config" / "부안_기상포함_walkforward_v1_2026-08-28.json"
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_v1_2026-08-28"
    r"\부안_과거발전_ASOS_NWP_결합.parquet"
)
OUT_DIR = HERE / "outputs" / "총출력개선판_2026-08-28"
SEED = 42


def load_cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def build_dataset(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(JOIN_PARQUET)
    df["prediction_issue_time_kst"] = pd.to_datetime(df["prediction_issue_time_kst"])
    df["target_time_kst"] = pd.to_datetime(df["target_time_kst"])
    df["issue_day"] = df["prediction_issue_time_kst"].dt.normalize()

    # quality_status==valid_ge9of12 아니면 타깃 자체를 안 믿는다(임의보간 없음).
    df.loc[df["quality_status"] != "valid_ge9of12", cfg["target"]] = np.nan

    # LightGBM은 object dtype을 못 받는다 - bool/문자 컬럼을 명시적으로 캐스팅.
    df["physical_daylight"] = pd.to_numeric(df["physical_daylight"], errors="coerce")

    hour = df["target_time_kst"].dt.hour
    doy = df["target_time_kst"].dt.dayofyear
    df["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 1일전 같은 대상시각 lag - 실제 달력일 기준(행 위치 아님).
    key = df[["target_time_kst", cfg["target"]]].copy()
    key["lag_target_time"] = key["target_time_kst"] - pd.Timedelta(days=1)
    lookup = df.set_index("target_time_kst")[cfg["target"]]
    df["lag_1day_same_slot_kw"] = key["lag_target_time"].map(lookup)

    return df


def feature_columns(cfg: dict) -> list[str]:
    f = cfg["features"]
    return (f["forecast_weather"] + f["issue_time_asos"] + f["physical"] +
           f["cyclical"] + f["lag"])


def expanding_window_folds(issue_days: pd.DatetimeIndex, initial_train: int,
                           test_block: int) -> list[tuple]:
    folds = []
    n = len(issue_days)
    end = initial_train
    while end + test_block <= n:
        folds.append((issue_days[:end], issue_days[end:end + test_block]))
        end += test_block
    return folds


def run() -> dict:
    cfg = load_cfg()
    df = build_dataset(cfg)
    feats = feature_columns(cfg)
    target = cfg["target"]

    issue_days = np.sort(df["issue_day"].unique())
    issue_days = pd.DatetimeIndex(issue_days)
    folds = expanding_window_folds(issue_days, cfg["initial_train_issue_days"],
                                   cfg["test_block_issue_days"])

    fold_results = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤에 있음"
        train = df[df["issue_day"].isin(train_days)].dropna(subset=feats + [target])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=feats + [target])
        if len(train) < 50 or len(test) < 5:
            continue

        model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                              max_depth=4, min_child_samples=15, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[feats], train[target])
        pred = np.clip(model.predict(test[feats]), 0, None)
        pers = test["lag_1day_same_slot_kw"].to_numpy()
        y_true = test[target].to_numpy()

        mae_model = mean_absolute_error(y_true, pred)
        rmse_model = float(np.sqrt(mean_squared_error(y_true, pred)))
        mae_pers = mean_absolute_error(y_true, pers)
        rmse_pers = float(np.sqrt(mean_squared_error(y_true, pers)))

        fold_results.append({
            "폴드": i,
            "학습구간": f"{train_days.min().date()}~{train_days.max().date()}({len(train)}행)",
            "시험구간": f"{test_days.min().date()}~{test_days.max().date()}({len(test)}행)",
            "모델_MAE_kW": round(mae_model, 2), "모델_RMSE_kW": round(rmse_model, 2),
            "지속성_MAE_kW": round(mae_pers, 2), "지속성_RMSE_kW": round(rmse_pers, 2),
            "MAE_개선율_pct": round((1 - mae_model / mae_pers) * 100, 1) if mae_pers else None,
            "RMSE_개선율_pct": round((1 - rmse_model / rmse_pers) * 100, 1) if rmse_pers else None,
        })

    if not fold_results:
        return {"오류": "생성된 폴드가 없음", "issue_days": len(issue_days)}

    avg_mae_model = float(np.mean([r["모델_MAE_kW"] for r in fold_results]))
    avg_mae_pers = float(np.mean([r["지속성_MAE_kW"] for r in fold_results]))
    avg_rmse_model = float(np.mean([r["모델_RMSE_kW"] for r in fold_results]))
    avg_rmse_pers = float(np.mean([r["지속성_RMSE_kW"] for r in fold_results]))

    return {
        "issue_days_total": int(len(issue_days)),
        "폴드수": len(fold_results),
        "특성목록": feats,
        "폴드별_결과": fold_results,
        "전체평균_모델_MAE_kW": round(avg_mae_model, 2),
        "전체평균_지속성_MAE_kW": round(avg_mae_pers, 2),
        "전체평균_모델_RMSE_kW": round(avg_rmse_model, 2),
        "전체평균_지속성_RMSE_kW": round(avg_rmse_pers, 2),
        "_주의": "기상특성 포함 개선판(3시간 단위, kW) - 08-28 예비모델(기상특성 없음, "
              "일간 kWh 단위)과 단위·grain이 달라 숫자를 직접 비교하지 말고 각자의 "
              "지속성 대비 개선율로 비교할 것. historical_archive_join_candidate "
              "한계 상속(NWP 수신시각 미검증).",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "부안_총출력_기상포함개선판_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
