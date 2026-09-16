# -*- coding: utf-8 -*-
"""부안 중장기 - 일간 총발전량(kWh) D+1 예측 - 09-15 후보(Claude, 사용자
정확도개선 요청 1·2번 파일럿).

## 이 후보에서 바뀐 것(medium_term_daily_v1_buan_2026-09-14.py 대비)
① **청천지수(Kt) 피처 추가**: `ultra_short_term_v1_buan_2026-09-08.py`의
`haurwitz_clearsky_ghi_wm2()`와 부안 자체 시간집계 모듈의
`solar_elevation_deg()`를 그대로 재사용(재구현 없음)해, 시각별
forecast_DSWRF(예보 일사)를 같은 시각 이론 청천GHI로 나눈 Kt를 계산하고
WEATHER8과 동일하게 발행일 단위 mean/max로 집계해 후보 피처에 추가.
doy_sin/cos(달력 근사)보다 직접적인 물리량이라 계절 전환기 반영에
도움될 것으로 기대(2턴 전 대화에서 사용자에게 제안한 우선순위 1번).
② **정격용량 상한 클리핑**: 기존엔 `np.clip(pred, 0, None)`로 하한만
막고 있었음(상한 무제한) - 8x125kW=1,000kW(AGENTS.md 09-07 인버터분해
항목 재사용, 명판 미검증 잠정치)의 일간 물리적 상한(용량×24h=24,000kWh)
으로 상한도 clip.

**원본(09-14) 보존** - 이 파일은 별도 후보이며 A/B 비교 후에만 원본 교체
여부를 판단한다. 공식모델·라이브서빙·스케줄러는 전혀 안 건드림.

--- 아래는 09-14 원본 docstring 그대로 ---

부안 중장기 - 일간 총발전량(kWh) D+1 예측(09-14 신규).

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

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 재구현 없음 - 기존 검증된 함수 그대로 재사용.
_agg = _load_module("buan_time_aggregate_for_kt", HERE.parent.parent.parent
                     / "02_전처리" / "부안" / "build_buan_time_aggregates_v1_2026-08-28.py")
_ultra = _load_module("buan_ultra_for_haurwitz", HERE / "ultra_short_term_v1_buan_2026-09-08.py")
solar_elevation_deg = _agg.solar_elevation_deg
haurwitz_clearsky_ghi_wm2 = _ultra.haurwitz_clearsky_ghi_wm2

# 8x125kW(AGENTS.md 09-07 인버터분해 항목, 명판 미검증 잠정치) - 재사용.
CAPACITY_KW = 1000.0
DAILY_CAPACITY_KWH = CAPACITY_KW * 24.0
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
OUT_DIR = HERE.parent / "outputs" / "부안_중장기_일간_v1_2026-09-15_kt_capclip_candidate"

# ultra_short_historical_backtest_v1_2026-08-31.py와 동일값(재확정 아님, 그대로 재사용)
DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")

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
    + ["forecast_kt_mean", "forecast_kt_max"]
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

    # ★09-15 신규(Kt 청천지수 피처, 재구현 없음)★: target_time_kst 기준
    # 태양고도 → Haurwitz 이론청천GHI → forecast_DSWRF/청천GHI = Kt.
    # 고도<=0(청천GHI=0)인 시각은 정의상 Kt 없음(NaN) - 원래도 야간엔
    # 예보발행 대상이 아니라 실질적으로 8시각 전부 주간이라 영향 적음.
    elev = solar_elevation_deg(pd.DatetimeIndex(hourly["target_time_kst"]))
    clearsky_ghi = haurwitz_clearsky_ghi_wm2(elev)
    hourly["forecast_kt"] = hourly["forecast_DSWRF"] / np.where(clearsky_ghi > 1e-6, clearsky_ghi, np.nan)
    hourly["forecast_kt"] = hourly["forecast_kt"].clip(lower=0, upper=1.5)  # 물리적 이상치 방지(청천지수 통상 0~1.2대)

    agg = hourly.groupby("issue_day").agg(
        **{f"{v}_mean": (v, "mean") for v in WEATHER8},
        **{f"{v}_max": (v, "max") for v in WEATHER8},
        forecast_kt_mean=("forecast_kt", "mean"),
        forecast_kt_max=("forecast_kt", "max"),
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

        model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, max_depth=4,
                              min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[CANDIDATE_FEATURES], train[TARGET])
        # ★09-15 신규★: 하한(0)만 막던 것을 정격용량×24h(물리적 일간 상한)까지 clip.
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
