# -*- coding: utf-8 -*-
"""영광 중장기 - 일간 총발전량(kWh) 예측(09-07) - 김제 방법론 그대로 재사용.

김제 `medium_term_daily_v1_gimje_2026-09-01.py`와 완전히 동일한 로직
(8시각 예보 요약→일간 특성, 전일지속성·계절climatology 기준모델,
walk-forward). 데이터는 전부 영광 자체.

영광은 김제와 달리 `is_defect_period` 컬럼 자체가 결합parquet에 없다
(알려진 결함구간이 없어서 - 09-01 시간정렬 감사에서 완전가용률
99.947%로 이미 확인됨) - 그래서 그 제외단계는 생략한다. 일간 CSV도
709일 전부 이미 `valid_daylight_ge90pct`로 확인돼 있어 추가 필터링이
사실상 no-op이지만, 규약은 동일하게 유지(코드 일관성).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_라이브연계_v1_2026-09-14"
    r"\영광_과거발전_ASOS_NWP_GRID_결합_라이브연계_v1_2026-09-14.parquet"
)
# ★09-14 변경★: 09-03 정적본 → 코덱스 라이브연계본(2026-09-14까지
# 연장, 컬럼 동일·중복 0건 확인)으로 교체. 부안 사례 교훈대로 아래
# DAILY_ENERGY_CSV도 반드시 같이 바꿈. build_medium() 배포 재실행은
# 안 함(코덱스 보고: 가을 완전표본 아직 0건, 영광은 발행 전 수신
# NWP 부족이 원인).
DAILY_ENERGY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_라이브연계_v1_2026-09-14"
    r"\영광_발전소_일간_라이브연계_D1검증용.csv"
)
OUT_DIR = HERE / "outputs" / "영광_중장기_일간_v1_2026-09-07"

TARGET = "daily_energy_kwh"
SEED = 42
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
VIF_CORR_MIN_ROWS = 10

# ★09-15 추가(Claude, 정확도개선 파일럿 #2 - 부안/김제 검증 완료, 부작용
# 0건 확인해 실채택)★: 영광 설비용량 634kW(AGENTS.md 09-08 인버터분해
# 항목, plant_id 7912 정원13대 - 명판 미검증 잠정치).
CAPACITY_KW = 639.94  # ★09-16 통일★ 발전소 API 정격(AC 계통연계)으로 4지역 기준 통일. 인버터 등록용량 합계(부안1000/김제1100/영광634)는 DC·명판측 값이라 clip 상한·nMAE 분모로 부적합 - Blockdata 구성감시용으로만 남긴다.
DAILY_CAPACITY_KWH = CAPACITY_KW * 24.0

WEATHER8 = ["forecast_DSWRF", "forecast_TCDC", "forecast_LCDC", "forecast_MCDC", "forecast_HCDC",
            "forecast_REH", "forecast_POP", "forecast_SKY"]
CANDIDATE_FEATURES = (
    [f"{v}_mean" for v in WEATHER8] + [f"{v}_max" for v in WEATHER8]
    + ["daily_energy_lag1_kwh", "doy_sin", "doy_cos"]
)


def build_daily_dataset() -> tuple[pd.DataFrame, dict]:
    hourly = pd.read_parquet(JOIN_PARQUET)
    hourly["prediction_issue_time_kst"] = pd.to_datetime(hourly["prediction_issue_time_kst"])
    hourly["issue_day"] = hourly["prediction_issue_time_kst"].dt.normalize()
    hourly["target_day"] = hourly["issue_day"] + pd.Timedelta(days=1)

    agg = hourly.groupby("issue_day").agg(
        **{f"{v}_mean": (v, "mean") for v in WEATHER8},
        **{f"{v}_max": (v, "max") for v in WEATHER8},
        target_day=("target_day", "first"),
        n_hours=("target_time_kst", "count"),
    ).reset_index()
    before = len(agg)
    agg = agg[agg["n_hours"] == 8].copy()  # 8시각 전부 확보된 발행일만

    daily = pd.read_csv(DAILY_ENERGY_CSV)
    daily["date_kst"] = pd.to_datetime(daily["date_kst"])
    daily.loc[daily["quality_status"] != "valid_daylight_ge90pct", TARGET] = np.nan
    daily = daily[["date_kst", TARGET]].rename(columns={"date_kst": "target_day"})

    df = agg.merge(daily, on="target_day", how="left")

    lookup = daily.set_index("target_day")[TARGET]
    df["daily_energy_lag1_kwh"] = (df["target_day"] - pd.Timedelta(days=1)).map(lookup)

    doy = df["target_day"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df, {"8시각_미충족_제외_발행일수": before - len(agg),
               "비고": "영광은 알려진 결함구간 없음(defect_period 컬럼 자체가 없음)"}


def expanding_folds_full_coverage(days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    n = len(days)
    folds, end = [], initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((days[:end], days[end:test_end]))
        end = test_end
    return folds


def fold_internal_correlation(df: pd.DataFrame, folds: list[tuple]) -> list[dict]:
    rows = []
    for i, (train_days, _test_days) in enumerate(folds, start=1):
        train = df[df["issue_day"].isin(train_days)]
        x = train[CANDIDATE_FEATURES + [TARGET]].dropna()
        row = {"폴드": i, "학습일수": len(train_days), "학습행수": len(x)}
        if len(x) >= VIF_CORR_MIN_ROWS:
            row["타깃상관"] = {c: round(float(x[c].corr(x[TARGET])), 3) for c in CANDIDATE_FEATURES}
        rows.append(row)
    return rows


def pooled_score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = y_true - pred
    return {"n": int(len(y_true)), "MAE_kWh": round(float(np.mean(np.abs(err))), 1),
            "RMSE_kWh": round(float(np.sqrt(np.mean(err ** 2))), 1)}


def run_walkforward(df: pd.DataFrame, folds: list[tuple]) -> dict:
    all_true, all_model, all_pers, all_clim = [], [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출"
        train = df[df["issue_day"].isin(train_days)].dropna(subset=CANDIDATE_FEATURES + [TARGET])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=CANDIDATE_FEATURES + [TARGET])
        if len(train) < MIN_ROWS_PER_FOLD or len(test) < 1:
            continue

        # ★09-15 변경(Claude, 규제튜닝 그리드 실측 - 부안 스크립트와 동일 근거)★
        model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, max_depth=3,
                              min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[CANDIDATE_FEATURES], train[TARGET])
        pred_model = np.clip(model.predict(test[CANDIDATE_FEATURES]), 0, DAILY_CAPACITY_KWH)

        y_true = test[TARGET].to_numpy()
        pred_pers = test["daily_energy_lag1_kwh"].to_numpy()

        train_doy = train["target_day"].dt.dayofyear.to_numpy()
        train_y = train[TARGET].to_numpy()
        overall_mean = float(np.nanmean(train_y))
        test_doy = test["target_day"].dt.dayofyear.to_numpy()
        pred_clim = np.full(len(test), overall_mean)
        for j, d in enumerate(test_doy):
            circ_dist = np.minimum(np.abs(train_doy - d), 365 - np.abs(train_doy - d))
            window = train_y[circ_dist <= 10]
            if len(window) >= 3:
                pred_clim[j] = float(np.mean(window))

        all_true.append(y_true); all_model.append(pred_model)
        all_pers.append(pred_pers); all_clim.append(pred_clim)
        fold_rows.append({"폴드": i, "시험일수": len(test_days), "시험행수": len(test),
                          "모델_MAE": pooled_score(y_true, pred_model)["MAE_kWh"],
                          "전일지속성_MAE": pooled_score(y_true, pred_pers)["MAE_kWh"],
                          "계절climatology_MAE": pooled_score(y_true, pred_clim)["MAE_kWh"]})

    y_true_all = np.concatenate(all_true)
    return {
        "폴드수": len(fold_rows), "폴드별": fold_rows,
        "pooled_모델": pooled_score(y_true_all, np.concatenate(all_model)),
        "pooled_전일지속성": pooled_score(y_true_all, np.concatenate(all_pers)),
        "pooled_계절climatology": pooled_score(y_true_all, np.concatenate(all_clim)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = build_daily_dataset()
    days = pd.DatetimeIndex(np.sort(df["issue_day"].unique()))
    folds = expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    corr = fold_internal_correlation(df, folds)
    perf = run_walkforward(df, folds)

    def improve(model_mae, base_mae):
        return round((1 - model_mae / base_mae) * 100, 1) if base_mae else None

    result = {
        **meta, "issue_days_total": int(len(days)),
        "폴드내부_상관분석(leakage없음)": corr,
        "성능": {
            **perf,
            "pooled_MAE_개선율_vs전일지속성_pct": improve(perf["pooled_모델"]["MAE_kWh"], perf["pooled_전일지속성"]["MAE_kWh"]),
            "pooled_MAE_개선율_vs계절climatology_pct": improve(perf["pooled_모델"]["MAE_kWh"], perf["pooled_계절climatology"]["MAE_kWh"]),
        },
        "_판정": "잠정치 - promote_to_official 대상 아님(historical_archive_join_candidate 한계 상속).",
    }

    (OUT_DIR / "영광_중장기_일간_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "폴드내부_상관분석(leakage없음)"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
