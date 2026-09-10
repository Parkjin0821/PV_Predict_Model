# -*- coding: utf-8 -*-
"""부안 총출력 예비모델(작업순서 5번) - expanding-window walk-forward.

기상특성 없이(ASOS/NWP 백필 전) 발전량 자체의 시계열 특성만으로 만드는
예비모델이다. config/부안_예비모델_walkforward_v1_2026-08-28.json에
사전동결된 폴드 규칙을 그대로 따른다 - 결과를 본 뒤 규칙을 바꾸지 않는다.

API 호출 없음. Codex가 만든 일간 공식후보(부안_발전소_일간_공식후보.csv)
와 부안_전처리_규칙_v1_2026-08-28.json(결함구간)만 읽기전용으로 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "config" / "부안_예비모델_walkforward_v1_2026-08-28.json"
PREP_CFG_PATH = HERE / "config" / "부안_전처리_규칙_v1_2026-08-28.json"
DAILY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_발전소_일간_공식후보.csv"
)
OUT_DIR = HERE / "outputs" / "총출력예비모델_2026-08-28"

SEED = 42


def load_cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def load_prep_cfg() -> dict:
    return json.loads(PREP_CFG_PATH.read_text(encoding="utf-8"))


def build_dataset(prep_cfg: dict) -> pd.DataFrame:
    """일간 공식후보를 읽어 결함구간 제외 + 유효일만 남긴 연속 달력 프레임을
    만든다. 결측일은 행 자체를 지우지 않고 daily_energy_kwh만 NaN으로
    남겨(reindex) lag가 실제 달력일 기준으로 정직하게 계산되게 한다."""
    df = pd.read_csv(DAILY_CSV, encoding="utf-8-sig")
    df["date_kst"] = pd.to_datetime(df["date_kst"])
    df["valid"] = df["quality_status"] == "valid_daylight_ge90pct"

    defect = prep_cfg["defect_period"]
    d_start = pd.Timestamp(defect["start_inclusive"])
    d_end_ex = pd.Timestamp(defect["end_exclusive"])

    full_index = pd.date_range(df["date_kst"].min(), df["date_kst"].max(), freq="D")
    frame = pd.DataFrame(index=full_index)
    frame.index.name = "date_kst"

    daily = df.set_index("date_kst")["daily_energy_kwh"].reindex(full_index)
    valid_mask = df.set_index("date_kst")["valid"].reindex(full_index).fillna(False)
    defect_mask = pd.Series((full_index >= d_start) & (full_index < d_end_ex), index=full_index)

    frame["daily_energy_kwh_raw"] = daily
    # 결함구간·비유효일은 lag 계산의 "실제 값"에서도 제외한다(가짜값으로
    # lag를 오염시키지 않기 위해 NaN 유지) - 학습/시험 타깃도 이 컬럼 기준.
    frame["daily_energy_kwh"] = daily.where(valid_mask & ~defect_mask)
    frame["is_usable"] = (valid_mask & ~defect_mask)
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    doy = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    y = f["daily_energy_kwh"]
    f["lag_1day_kwh"] = y.shift(1)
    f["lag_7day_kwh"] = y.shift(7)
    f["rolling_mean_3day_kwh"] = y.shift(1).rolling(3, min_periods=3).mean()
    f["rolling_mean_7day_kwh"] = y.shift(1).rolling(7, min_periods=7).mean()
    f["rolling_std_7day_kwh"] = y.shift(1).rolling(7, min_periods=7).std()
    return f


FEATURE_COLS = ["doy_sin", "doy_cos", "lag_1day_kwh", "lag_7day_kwh",
                "rolling_mean_3day_kwh", "rolling_mean_7day_kwh", "rolling_std_7day_kwh"]


def expanding_window_folds(usable_dates: pd.DatetimeIndex, initial_train: int,
                           test_block: int) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """usable_dates(실제 사용가능한 날짜만, 시간순 정렬)를 initial_train개로
    처음 학습하고 test_block개씩 시험하며 학습창을 계속 넓힌다."""
    folds = []
    n = len(usable_dates)
    train_end = initial_train
    while train_end + test_block <= n:
        train_dates = usable_dates[:train_end]
        test_dates = usable_dates[train_end:train_end + test_block]
        folds.append((train_dates, test_dates))
        train_end += test_block
    return folds


def run() -> dict:
    cfg = load_cfg()
    prep_cfg = load_prep_cfg()
    raw = build_dataset(prep_cfg)
    feat = build_features(raw)

    usable_dates = feat.index[feat["is_usable"]]
    folds = expanding_window_folds(
        usable_dates, cfg["initial_train_days"], cfg["test_block_days"])

    fold_results = []
    for i, (train_dates, test_dates) in enumerate(folds, start=1):
        train = feat.loc[train_dates].dropna(subset=FEATURE_COLS + ["daily_energy_kwh"])
        test = feat.loc[test_dates].dropna(subset=FEATURE_COLS + ["daily_energy_kwh"])
        if len(train) < 20 or len(test) < 3:
            continue

        assert train_dates.max() < test_dates.min(), "시간누출: 학습구간이 시험구간보다 뒤에 있음"

        model = LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=7,
                              max_depth=3, min_child_samples=10, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[FEATURE_COLS], train["daily_energy_kwh"])
        pred = model.predict(test[FEATURE_COLS])
        pers = test["lag_1day_kwh"].to_numpy()  # 지속성 베이스라인(전일값)

        y_true = test["daily_energy_kwh"].to_numpy()
        mae_model = mean_absolute_error(y_true, pred)
        rmse_model = float(np.sqrt(mean_squared_error(y_true, pred)))
        mae_pers = mean_absolute_error(y_true, pers)
        rmse_pers = float(np.sqrt(mean_squared_error(y_true, pers)))

        fold_results.append({
            "폴드": i,
            "학습구간": f"{train_dates.min().date()}~{train_dates.max().date()}({len(train)}행)",
            "시험구간": f"{test_dates.min().date()}~{test_dates.max().date()}({len(test)}행)",
            "모델_MAE_kWh": round(mae_model, 1), "모델_RMSE_kWh": round(rmse_model, 1),
            "지속성_MAE_kWh": round(mae_pers, 1), "지속성_RMSE_kWh": round(rmse_pers, 1),
            "MAE_개선율_pct": round((1 - mae_model / mae_pers) * 100, 1) if mae_pers else None,
            "RMSE_개선율_pct": round((1 - rmse_model / rmse_pers) * 100, 1) if rmse_pers else None,
        })

    if not fold_results:
        return {"오류": "생성된 폴드가 없음(데이터 부족)", "usable_days": len(usable_dates)}

    all_mae_model = np.mean([r["모델_MAE_kWh"] for r in fold_results])
    all_mae_pers = np.mean([r["지속성_MAE_kWh"] for r in fold_results])
    all_rmse_model = np.mean([r["모델_RMSE_kWh"] for r in fold_results])
    all_rmse_pers = np.mean([r["지속성_RMSE_kWh"] for r in fold_results])

    return {
        "usable_days_total": int(len(usable_dates)),
        "폴드수": len(fold_results),
        "폴드별_결과": fold_results,
        "전체평균_모델_MAE_kWh": round(float(all_mae_model), 1),
        "전체평균_지속성_MAE_kWh": round(float(all_mae_pers), 1),
        "전체평균_모델_RMSE_kWh": round(float(all_rmse_model), 1),
        "전체평균_지속성_RMSE_kWh": round(float(all_rmse_pers), 1),
        "_주의": "예비모델(기상특성 미포함) - ASOS/NWP 백필 후 개선판 별도 예정. "
              "지속성보다 나쁘게 나와도 그대로 보고함(사전동결 기준 - 채택판정 아님).",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "부안_총출력_예비모델_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
