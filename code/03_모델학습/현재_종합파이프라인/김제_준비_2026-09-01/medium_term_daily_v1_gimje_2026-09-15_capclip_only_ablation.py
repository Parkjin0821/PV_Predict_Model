# -*- coding: utf-8 -*-
"""김제 중장기 - 일간 총발전량(kWh) 예측(09-01).

프로젝트 확정 티어 정의("일/월/연 주기, 일간·월간 누적")의 "일간" 부분.
지금까지 만든 초단기(+1h~4h, 순간출력 kW)·단기(D+1, 8시각 순간출력 kW)와
다르게, **하루 총량(kWh)을 직접 타깃**으로 한다(광주 "일간 D+1" 티어와
동일 스코프) - 8시각 순간출력을 사다리꼴적분해서 추정하지 않고 별도
모델로 직접 학습(광주도 이 방식).

## 데이터
- 발전량 타깃: 김제_발전소_일간_공식후보.csv(daily_energy_kwh,
  quality_status=valid_daylight_ge90pct만 신뢰).
- 기상 특성: 결합 parquet의 8시각 예보를 발행일(issue_day) 단위로
  요약(평균·최대)해서 일간 특성으로 변환 - 원본 시각별 값이 아니라
  "그 날 하루 예보가 전반적으로 어땠는지"를 나타내는 요약통계.
- 결함구간: is_defect_period(hourly) 다수결로 그 날 전체 제외.
- lag: 전일 실적(persistence baseline이자 특성 후보).

## 방법론
D+1(단기) 스크립트와 동일 원칙 - 폴드내부 leakage없는 상관재검증,
pooled MAE/RMSE, expanding-window 풀커버리지 폴드.
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


# 재구현 없음 - 김제 자체 태양고도 함수 + 부안 Haurwitz(범용, 위치무관) 재사용.
_agg = _load_module("gimje_time_aggregate_for_kt", HERE.parent.parent.parent
                     / "02_전처리" / "김제" / "build_gimje_time_aggregates_v1_2026-08-31.py")
_ultra = _load_module("buan_ultra_for_haurwitz_gimje", HERE.parent / "부안_준비_2026-08-28"
                       / "ultra_short_term_v1_buan_2026-09-08.py")
solar_elevation_deg = _agg.solar_elevation_deg
haurwitz_clearsky_ghi_wm2 = _ultra.haurwitz_clearsky_ghi_wm2

# 김제 설비용량 1,100kW(AGENTS.md 인버터 pro-rata 재조정 항목, 명판 미검증 잠정치) - 재사용.
CAPACITY_KW = 1100.0
DAILY_CAPACITY_KWH = CAPACITY_KW * 24.0
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_라이브연계_v1_2026-09-14"
    r"\김제_과거발전_ASOS_NWP_GRID_결합_라이브연계_v1_2026-09-14.parquet"
)
# ★09-14 변경★: 08-31 정적본 → 코덱스 라이브연계본(2026-09-14까지 연장,
# 컬럼 동일·중복 0건 확인, AGENTS.md 참고)으로 교체. 부안 사례에서
# JOIN_PARQUET만 바꾸고 DAILY_ENERGY_CSV를 안 바꿔서 TARGET이 끊기는
# 실수가 있었음 - 아래 DAILY_ENERGY_CSV도 반드시 같이 바꿀 것.
# 이 파일의 build_medium() 호출(별도 배포 스크립트)은 공식 경로를
# 그대로 덮어쓰므로, 현재는 이 파일을 직접 실행한 walk-forward
# 재검증까지만 하고 배포 재실행은 하지 않는다(코덱스 보고: 가을
# 완전표본 아직 0건).
DAILY_ENERGY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_라이브연계_v1_2026-09-14"
    r"\김제_발전소_일간_라이브연계_D1검증용.csv"
)
OUT_DIR = HERE / "outputs" / "김제_중장기_일간_v1_2026-09-15_capclip_only_ablation"

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
)  # ablation: Kt 피처 제외, 용량클리핑만 격리 테스트


def build_daily_dataset() -> tuple[pd.DataFrame, dict]:
    hourly = pd.read_parquet(JOIN_PARQUET)
    hourly["prediction_issue_time_kst"] = pd.to_datetime(hourly["prediction_issue_time_kst"])
    hourly["issue_day"] = hourly["prediction_issue_time_kst"].dt.normalize()
    hourly["target_day"] = hourly["issue_day"] + pd.Timedelta(days=1)

    # ★09-15 신규(Kt 청천지수 피처, 재구현 없음 - 부안 파일럿과 동일 로직)★
    elev = solar_elevation_deg(pd.DatetimeIndex(hourly["target_time_kst"]))
    clearsky_ghi = haurwitz_clearsky_ghi_wm2(elev)
    hourly["forecast_kt"] = hourly["forecast_DSWRF"] / np.where(clearsky_ghi > 1e-6, clearsky_ghi, np.nan)
    hourly["forecast_kt"] = hourly["forecast_kt"].clip(lower=0, upper=1.5)

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
    agg = agg[agg["defect_rows"] == 0].copy()  # 그 날 8시각 중 하나라도 결함구간이면 전체 제외
    removed_defect_days = before - len(agg)
    agg = agg[agg["n_hours"] == 8].copy()  # 8시각 전부 확보된 발행일만(부분 발행일 제외)

    daily = pd.read_csv(DAILY_ENERGY_CSV)
    daily["date_kst"] = pd.to_datetime(daily["date_kst"])
    daily.loc[daily["quality_status"] != "valid_daylight_ge90pct", TARGET] = np.nan
    daily = daily[["date_kst", TARGET]].rename(columns={"date_kst": "target_day"})

    df = agg.merge(daily, on="target_day", how="left")

    # lag: 전일(target_day - 1) 실적 - persistence 기준이자 특성.
    lookup = daily.set_index("target_day")[TARGET]
    df["daily_energy_lag1_kwh"] = (df["target_day"] - pd.Timedelta(days=1)).map(lookup)

    doy = df["target_day"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df, {"결함구간_제외_발행일수": removed_defect_days, "8시각_미충족_제외_발행일수": before - removed_defect_days - len(agg)}


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
        pred_model = np.clip(model.predict(test[CANDIDATE_FEATURES]), 0, DAILY_CAPACITY_KWH)

        y_true = test[TARGET].to_numpy()
        pred_pers = test["daily_energy_lag1_kwh"].to_numpy()  # 전일 실적 지속성
        # 계절climatology: ±10일 원형창 평균(학습기간이 짧은 초반 폴드는 특정
        # day-of-year가 학습구간에 아예 없을 수 있어 정확히 같은 날만 보면
        # 결측이 남음 - 원형창으로 완화, 그래도 못 채우면 학습구간 전체평균).
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
        **meta,
        "issue_days_total": int(len(days)),
        "폴드구성": [{"폴드": i + 1, "학습일수": len(tr), "시험일수": len(te)} for i, (tr, te) in enumerate(folds)],
        "폴드내부_상관분석(leakage없음)": corr,
        "성능": {
            **perf,
            "pooled_MAE_개선율_vs전일지속성_pct": improve(perf["pooled_모델"]["MAE_kWh"], perf["pooled_전일지속성"]["MAE_kWh"]),
            "pooled_MAE_개선율_vs계절climatology_pct": improve(perf["pooled_모델"]["MAE_kWh"], perf["pooled_계절climatology"]["MAE_kWh"]),
        },
        "_판정": "잠정치 - promote_to_official 대상 아님(historical_archive_join_candidate 한계 상속).",
    }

    (OUT_DIR / "김제_중장기_일간_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
