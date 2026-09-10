# -*- coding: utf-8 -*-
"""부안 총출력모델 v5(08-31, Codex 독립감사 지적 4건 전부 반영).

## 정정한 4가지(전부 실측 재확인 완료, AGENTS.md 참고)
1. 특성선택(상관/VIF)이 전체 843행(시험기간 포함)으로 계산돼 있었다 -
   폴드별 "그 폴드의 학습구간 데이터만"으로 다시 계산해서 ASOS 4종
   저상관 결론이 매 폴드에서도 유지되는지 확인한다(leakage 제거).
2. 결함구간 필터가 issue_day 기준이라 실제로는 target_time_kst 기준
   04-16~05-23이 빠지는 하루 밀림 오류가 있었다 - target_time_kst의
   날짜로 다시 필터링한다.
3. "총출력모델 전부 결함구간 미제외"는 과장이었다 - 예비모델(day-level,
   total_output_preliminary_model_v1_2026-08-28.py)은 이미 제외
   적용돼 있었다. 이 v5는 기상결합(NWP/GRID join) 경로에 한정된 정정이다.
4. 폴드별 지표를 단순평균해 전체지표로 썼다 - 행수가 다른 폴드를
   동일가중 평균한 것도, 특히 RMSE를 폴드별로 평균한 것도 통계적으로
   틀렸다(sqrt(mean(se)) != mean(sqrt(mse_fold))). 이번엔 모든 시험행의
   (실측,예측) 쌍을 전부 모아 한 번에 pooled MAE/RMSE를 계산한다.
   마지막 18개 발행일이 어느 폴드 시험구간에도 안 들어가 평가에서
   빠졌던 것도, 마지막 폴드가 남은 날을 전부 흡수하도록 고쳐 해결한다.

## 판정 톤
사용자(+Codex) 지시대로 "잠정치"로만 보고한다 - v4를 최종기준으로
확정하지 않는다. 이 v5도 아직 promote_to_official=false다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_v1_2026-08-28"
    r"\부안_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"
)
OUT_DIR = HERE / "outputs" / "총출력개선판v5_방법론정정_2026-08-31"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20

# ★정정2★ target_time_kst 날짜 기준(실제 발전량 날짜) - issue_day 기준 아님.
DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")

ASOS4 = ["issue_asos_기온_C", "issue_asos_풍속_m_s", "issue_asos_상대습도_pct", "issue_asos_전운량_pct"]
WEATHER8 = ["forecast_DSWRF", "forecast_TCDC", "forecast_LCDC", "forecast_MCDC", "forecast_HCDC",
            "forecast_REH", "forecast_POP", "forecast_SKY"]
PHYSICAL = ["solar_elevation_deg", "physical_daylight"]
CYCLICAL = ["target_hour_sin", "target_hour_cos", "doy_sin", "doy_cos"]
LAG = ["lag_1day_same_slot_kw"]
FEATURES_15 = WEATHER8 + PHYSICAL + CYCLICAL + LAG           # ASOS4 제외(v4 후보)
FEATURES_19 = WEATHER8 + ASOS4 + PHYSICAL + CYCLICAL + LAG   # ASOS4 포함(v2 대응, 대조군)


def build_dataset() -> pd.DataFrame:
    df = pd.read_parquet(JOIN_PARQUET)
    df["prediction_issue_time_kst"] = pd.to_datetime(df["prediction_issue_time_kst"])
    df["target_time_kst"] = pd.to_datetime(df["target_time_kst"])
    df["issue_day"] = df["prediction_issue_time_kst"].dt.normalize()
    df["target_day"] = df["target_time_kst"].dt.normalize()

    in_defect = (df["target_day"] >= DEFECT_START) & (df["target_day"] < DEFECT_END_EXCLUSIVE)
    removed_targets = sorted(df.loc[in_defect, "target_day"].dt.strftime("%Y-%m-%d").unique().tolist())
    df = df.loc[~in_defect].copy()

    df.loc[df["quality_status"] != "valid_ge9of12", TARGET] = np.nan
    df["physical_daylight"] = pd.to_numeric(df["physical_daylight"], errors="coerce")

    hour = df["target_time_kst"].dt.hour
    doy = df["target_time_kst"].dt.dayofyear
    df["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    key = df[["target_time_kst", TARGET]].copy()
    key["lag_target_time"] = key["target_time_kst"] - pd.Timedelta(days=1)
    lookup = df.set_index("target_time_kst")[TARGET]
    df["lag_1day_same_slot_kw"] = key["lag_target_time"].map(lookup)

    return df, removed_targets


def expanding_folds_full_coverage(issue_days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    """★정정4★ 마지막 폴드가 남은 날을 전부 흡수 - 어떤 발행일도 평가에서 빠지지 않는다."""
    n = len(issue_days)
    folds = []
    end = initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((issue_days[:end], issue_days[end:test_end]))
        end = test_end
    return folds


def fold_internal_correlation_check(df: pd.DataFrame, folds: list[tuple]) -> list[dict]:
    """★정정1★ 각 폴드의 "그 폴드 학습구간"만으로 ASOS4 vs 타깃 상관을 다시 계산."""
    daylight = df[df["physical_daylight"] == 1]
    out = []
    for i, (train_days, _test_days) in enumerate(folds, start=1):
        train = daylight[daylight["issue_day"].isin(train_days)]
        row = {"폴드": i, "학습발행일수": len(train_days), "학습행수(낮시간)": len(train)}
        for c in ASOS4:
            sub = train[[c, TARGET]].dropna()
            row[c] = round(float(sub[c].corr(sub[TARGET])), 3) if len(sub) >= 10 else None
        out.append(row)
    return out


def pooled_score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = y_true - pred
    return {
        "n": int(len(y_true)),
        "MAE_kW": round(float(np.mean(np.abs(err))), 2),
        "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2),
    }


def run_walkforward(df: pd.DataFrame, folds: list[tuple], features: list[str], model_kind: str) -> dict:
    all_true, all_pred, all_pers = [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤에 있음"
        train = df[df["issue_day"].isin(train_days)].dropna(subset=features + [TARGET])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=features + [TARGET])
        if len(train) < 50 or len(test) < 1:
            continue

        if model_kind == "lightgbm":
            model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                                  max_depth=4, min_child_samples=15, subsample=0.9,
                                  colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=SEED, n_jobs=-1, verbosity=-1)
        elif model_kind == "linear":
            model = LinearRegression()
        else:
            raise ValueError(model_kind)
        model.fit(train[features], train[TARGET])
        pred = np.clip(model.predict(test[features]), 0, None)
        y_true = test[TARGET].to_numpy()
        pers = test["lag_1day_same_slot_kw"].to_numpy()

        all_true.append(y_true); all_pred.append(pred); all_pers.append(pers)
        fold_rows.append({
            "폴드": i, "시험발행일수": len(test_days), "시험행수": len(test),
            **{f"모델_{k}": v for k, v in pooled_score(y_true, pred).items() if k != "n"},
        })

    y_true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    pers_all = np.concatenate(all_pers)

    return {
        "폴드수": len(fold_rows),
        "폴드별": fold_rows,
        "pooled_모델": pooled_score(y_true_all, pred_all),
        "pooled_지속성": pooled_score(y_true_all, pers_all),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, removed_targets = build_dataset()
    issue_days = pd.DatetimeIndex(np.sort(df["issue_day"].unique()))
    folds = expanding_folds_full_coverage(issue_days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    corr_check = fold_internal_correlation_check(df, folds)

    lgbm15 = run_walkforward(df, folds, FEATURES_15, "lightgbm")
    lgbm19 = run_walkforward(df, folds, FEATURES_19, "lightgbm")
    lin15 = run_walkforward(df, folds, FEATURES_15, "linear")

    def improve(model_mae, pers_mae):
        return round((1 - model_mae / pers_mae) * 100, 1) if pers_mae else None

    result = {
        "정정사항": {
            "결함구간_target_time_kst기준_제외된_날짜수": len(removed_targets),
            "결함구간_첫날_끝날": [removed_targets[0], removed_targets[-1]] if removed_targets else None,
        },
        "issue_days_total": int(len(issue_days)),
        "폴드구성": [{"폴드": i + 1, "학습일수": len(tr), "시험일수": len(te)}
                  for i, (tr, te) in enumerate(folds)],
        "정정1_폴드내부_ASOS4_상관재검증": corr_check,
        "LightGBM_15특성(ASOS4제외)": {
            **lgbm15,
            "pooled_MAE_개선율_pct": improve(lgbm15["pooled_모델"]["MAE_kW"], lgbm15["pooled_지속성"]["MAE_kW"]),
            "pooled_RMSE_개선율_pct": improve(lgbm15["pooled_모델"]["RMSE_kW"], lgbm15["pooled_지속성"]["RMSE_kW"]),
        },
        "LightGBM_19특성(ASOS4포함_대조군)": {
            **lgbm19,
            "pooled_MAE_개선율_pct": improve(lgbm19["pooled_모델"]["MAE_kW"], lgbm19["pooled_지속성"]["MAE_kW"]),
            "pooled_RMSE_개선율_pct": improve(lgbm19["pooled_모델"]["RMSE_kW"], lgbm19["pooled_지속성"]["RMSE_kW"]),
        },
        "선형회귀_15특성(대조군)": {
            **lin15,
            "pooled_MAE_개선율_pct": improve(lin15["pooled_모델"]["MAE_kW"], lin15["pooled_지속성"]["MAE_kW"]),
            "pooled_RMSE_개선율_pct": improve(lin15["pooled_모델"]["RMSE_kW"], lin15["pooled_지속성"]["RMSE_kW"]),
        },
        "_판정": "잠정치(사용자 지시) - promote_to_official 대상 아님. v4를 대체하는 최종판정이 아니라 "
               "방법론 정정판. 15특성 vs 19특성 pooled 성능차가 이번 결론(ASOS4 제외 타당성)의 핵심 근거.",
    }

    (OUT_DIR / "부안_총출력_v5_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
