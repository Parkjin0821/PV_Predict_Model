# -*- coding: utf-8 -*-
"""부안 중장기 - 일간 총발전량(kWh) D+1 예측(09-14 신규).

## 배경
09-08에 "부안 중장기 전부 미구축"으로 확정했던 이유는 가을 표본이
없어서가 아니라 **애초에 이 파일 자체가 없었다** - 김제·영광의
`medium_term_daily_v1_*.py`를 부안용으로 아직 안 만들었을 뿐,
`build_regional_ultrashort_mediumterm_issue_safe_v1_2026-09-08.py`의
`build_medium()`은 이미 지역에 무관하게 동작하도록 만들어져 있다
(주석: "부안은 09-08 확정대로 미구축, 임의생성 금지" - 이건 "만들지
말라"가 아니라 "이 파일이 없어서 아직 안 됐다"는 상태 기록이었음).

## 방법 (재구현 없음 - 김제 스크립트 구조 그대로, 부안 데이터만 연결)
`medium_term_daily_v1_gimje_2026-09-01.py`와 피처·검증 로직 완전
동일(WEATHER8 8종 평균/최대 + 전일지속성 lag + doy_sin/cos, expanding
walk-forward). 딱 하나 다른 점: 김제 데이터엔 `is_defect_period`
컬럼이 있어 그걸로 결함구간을 걸렀는데, 부안 결합parquet엔 그 컬럼이
없어서 대신 기존에 이미 확정된 부안 설비결함 기간(DEFECT_START~
DEFECT_END_EXCLUSIVE, `ultra_short_historical_backtest_v1_2026-08-31.py`
와 동일 값)으로 직접 걸러낸다.

## 가을(9~11월) 관련 솔직한 한계
지금 연결된 결합parquet·일간CSV가 2026-08-05까지만 있어(코덱스 전처리
갱신 대기 중, AGENTS.md 09-14 요청 참고) **이번 실행으로는 가을철
walk-forward 검증이 아직 안 된다** - 8월까지 데이터로 구조·정합성만
확인. 코덱스가 부안 전처리(이 결합parquet 포함)를 갱신하면 재실행해서
가을 포함 검증을 마저 해야 한다. 다만 특성 자체가 그날그날 실제
날씨예보(WEATHER8)를 쓰므로, doy_sin/cos + lag만 쓰는 것보다 계절
공백에 상대적으로 강건할 것으로 기대(가을 앞뒤(8월·12월) 데이터가
있어 극단적 외삽이 아니라 보간에 가까움).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_라이브연계_v1_2026-09-14"
    r"\부안_과거발전_ASOS_NWP_GRID_결합_라이브연계_v1_2026-09-14.parquet"
)
# ★09-14 변경★: 08-31 정적 결합본(08-05 캡) → 코덱스 신규 라이브연계
# 결합본(2026-09-14까지 연장, 컬럼 스키마 동일·중복 0건 확인)으로
# 교체. AGENTS.md 09-14 항목 참고. 수정 전 백업:
# medium_term_daily_v1_buan_2026-09-14.py.backup_before_livebridge_20260914.
# ★09-15 재수정★: 위 09-14 시점에 "1단계 중간산출물"(시간집계_라이브연계의
# 부안_발전소_일간_라이브연계.csv)을 잘못 가리키고 있었음이 09-15
# 재검증에서 드러남 - 이 파일은 daylight_expected_slots를 "그날 지금까지
# 들어온 슬롯 수"로 계산해서 진행 중인 당일(예: 09-15 오전)을 100%
# 완료로 오판정(부분값 587kWh를 완전값처럼 노출). 2단계 최종산출물
# (build_buan_weather_power_live_bridge가 원시 5분데이터로 직접
# daylight_expected_slots를 물리적 전체 일광슬롯 수로 재계산한 D1검증용.csv)
# 로 교체 - 이쪽이 진행 중인 날을 올바르게 invalid 처리함(09-15 확인됨).
DAILY_ENERGY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_라이브연계_v1_2026-09-14"
    r"\부안_발전소_일간_라이브연계_D1검증용.csv"
)
# ★09-14 수정★: JOIN_PARQUET만 라이브연계로 바꾸고 이건 옛 경로 그대로
# 둬서 TARGET(daily_energy_kwh)이 08-03 이후 전부 NaN으로 빠지는 실수를
# 함(clean이 09-14까지 늘어난 issue_day 중 08-03에서 끊김). 발전량
# 라이브연계 산출물(코덱스, AGENTS.md 09-14 항목)의 일간 버전으로 같이
# 교체.
OUT_DIR = HERE.parent / "outputs" / "부안_중장기_일간_v1_2026-09-14"

# ultra_short_historical_backtest_v1_2026-08-31.py와 동일값(재확정 아님, 그대로 재사용)
DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")

# ★09-15 추가(Claude, 정확도개선 파일럿 #2 - kt_capclip_candidate 검증 후
# 부작용 없음 확인해 실채택)★: 8x125kW=1,000kW(AGENTS.md 09-07 인버터
# 분해 항목, 명판 미검증 잠정치) - 하한(0)만 있던 clip에 물리적 일간
# 상한(용량×24h)도 추가. 실측 최대치가 이보다 훨씬 낮아 평소엔 안
# 걸리지만, 예보 극단치·모델 이상예측 시 최후 방어선.
CAPACITY_KW = 998.715  # ★09-16 통일★ 발전소 API 정격(AC 계통연계)으로 4지역 기준 통일. 인버터 등록용량 합계(부안1000/김제1100/영광634)는 DC·명판측 값이라 clip 상한·nMAE 분모로 부적합 - Blockdata 구성감시용으로만 남긴다.
DAILY_CAPACITY_KWH = CAPACITY_KW * 24.0

TARGET = "daily_energy_kwh"
SEED = 42
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
VIF_CORR_MIN_ROWS = 10

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
    # 부안엔 is_defect_period 컬럼이 없어 기존 확정 결함기간으로 직접 표시(위 docstring 참고).
    hourly["is_defect_period"] = (
        (hourly["issue_day"] >= DEFECT_START) & (hourly["issue_day"] < DEFECT_END_EXCLUSIVE)
    ).astype(int)

    agg = hourly.groupby("issue_day").agg(
        **{f"{v}_mean": (v, "mean") for v in WEATHER8},
        **{f"{v}_max": (v, "max") for v in WEATHER8},
        target_day=("target_day", "first"),
        defect_rows=("is_defect_period", "sum"),
        n_hours=("target_time_kst", "count"),
    ).reset_index()
    before = len(agg)
    agg = agg[agg["defect_rows"] == 0].copy()  # 그 날 중 하나라도 결함구간이면 전체 제외
    removed_defect_days = before - len(agg)
    agg = agg[agg["n_hours"] == 8].copy()  # 8시각 전부 확보된 발행일만(부분 발행일 제외)

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

    return df, {"결함구간_제외_발행일수": removed_defect_days,
                "8시각_미충족_제외_발행일수": before - removed_defect_days - len(agg)}


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

        # ★09-15 변경(Claude, 규제튜닝 그리드 실측)★: max_depth 4→3.
        # 부안·김제·영광 1D스윕+결합테스트에서 3지역 전부 동시개선(부안
        # -2.5%·김제-0.8%·영광-0.5%)된 유일한 방향, 다른 규제값과 조합해도
        # 더 안 좋아져 단독채택(hyperparam_grid_search_medium_daily_v1_
        # 2026-09-15.py 결과, AGENTS.md 참고).
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

    if not all_true:
        return {"폴드수": 0, "폴드별": [], "경고": "유효 폴드 없음"}
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
        **meta,
        "issue_days_total": int(len(days)),
        "issue_day_range": [str(days.min()), str(days.max())] if len(days) else None,
        "폴드구성": [{"폴드": i + 1, "학습일수": len(tr), "시험일수": len(te)} for i, (tr, te) in enumerate(folds)],
        "폴드내부_상관분석(leakage없음)": corr,
        "성능": ({
            **perf,
            "pooled_MAE_개선율_vs전일지속성_pct": improve(perf["pooled_모델"]["MAE_kWh"], perf["pooled_전일지속성"]["MAE_kWh"]),
            "pooled_MAE_개선율_vs계절climatology_pct": improve(perf["pooled_모델"]["MAE_kWh"], perf["pooled_계절climatology"]["MAE_kWh"]),
        } if perf.get("폴드수", 0) > 0 else perf),
        "데이터_현황": (
            f"issue_day_range가 {days.min().date()}~{days.max().date()}까지 "
            "라이브연계 데이터 반영됨(09-14 코덱스 브릿지). 위 성능 수치는 "
            "실제 가을(9월) 표본 포함 결과."
        ),
        "_판정": "잠정치 - promote_to_official 대상 아님(가을 표본 아직 소수, 계속 누적 필요).",
    }

    (OUT_DIR / "부안_중장기_일간_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
