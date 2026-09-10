# -*- coding: utf-8 -*-
"""김제 초단기(+1h) 총출력모델(09-01) - 광주 06번(2026-08-20밤) 검증결론 재사용
+ 김제 자체 데이터로 독립 재검증.

## 광주에서 이미 검증되고 채택된 것(재실행 없이 결론만 재사용)
- **타깃은 raw(kW), 청천지수(kt) 정규화 타깃 아님** - 광주가 5폴드로 직접
  비교해서 raw가 일관되게 나음을 확인(arXiv 2608.02088의 "정규화가
  유리하다"는 결론과 반대로 나왔음, 임의로 남의 결론을 안 가져다 씀).
- **스마트지속성**(직전 청천지수를 유지한다 가정 - kt_T × 미래 청천전력)이
  단순지속성보다 훨씬 강한 기준모델.
- **청천전력 = 설비용량 × Haurwitz 청천GHI / 1000**(Haurwitz 1945 근사식).

## 김제 자체로 독립 검증하는 것(광주 결론을 그대로 안 가져다 씀)
- 특성별 상관분석(leakage 없이 폴드내부) - 어떤 lag·ASOS 실측 특성이
  실제로 유효한지 김제 데이터로 직접 확인.
- LightGBM vs 두 기준모델(단순지속성/스마트지속성) 성능 비교.

## 데이터 소스(전부 이미 백필 완료된 것만 사용, API 호출 없음)
- 5분 발전량: 김제_발전소_5분_공식후보.parquet(09-01 동료대조 재분류 반영본)
- ASOS243 실측: 기상청_ASOS243_시간환경_20240825_20260804.csv(발행 아닌
  "실제 관측"값 그대로 사용 - +1h 예측은 리드타임이 짧아 최신 실측을
  merge_asof backward로 붙여도 미래누출 아님, T 시점까지 관측된 값만 사용)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
PLANT_5MIN = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_발전소_5분_공식후보.parquet"
)
ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31\ASOS"
    r"\기상청_ASOS243_시간환경_20240825_20260804.csv"
)
DEFECT_CONFIG = HERE.parent.parent.parent / "02_전처리" / "김제" / "config" / "김제_전처리_규칙_v1_2026-08-31.json"
OUT_DIR = HERE / "outputs" / "김제_초단기1h_v1_2026-09-01"

CAPACITY_KW = 1100.0  # ★AC 후보, 명판 미검증★ - 110kW×10, 기존 산출물과 동일 정의 재사용
LEAD_SLOTS = 12  # 5분×12 = 60분(+1h)
SEED = 42
TARGET = "target_power_t+1h_kw"
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
VIF_CORR_MIN_ROWS = 10

ASOS_COLS = {
    "시각": "observation_time_kst", "지점번호": "station",
    "기온_C": "obs_temp_c", "강수량_mm": "obs_rain_mm", "풍속_m_s": "obs_wind_ms",
    "상대습도_pct": "obs_rh_pct", "전운량_10분위": "obs_cloud_tenths", "전운량_pct": "obs_cloud_pct",
    "일사량_W_m2": "obs_ghi_wm2",
}
CANDIDATE_FEATURES = [
    "power_lag_0min", "power_lag_30min", "power_lag_60min",
    "solar_elevation_now", "solar_elevation_t1h",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
# ASOS243(부안·김제 공유 관측소)은 일사량계가 없어 일사량_W_m2/MJ_m2가
# 100% 결측(09-01 실측 확인) - obs_ghi_wm2는 후보에서 제외.


def haurwitz_clearsky_ghi_wm2(elevation_deg: np.ndarray) -> np.ndarray:
    """Haurwitz(1945) 청천 GHI 근사식. elevation<=0이면 0."""
    elev = np.clip(elevation_deg, 0, 90)
    zenith_rad = np.deg2rad(90.0 - elev)
    cosz = np.cos(zenith_rad)
    cosz_safe = np.where(cosz > 1e-3, cosz, np.nan)
    ghi = 1098.0 * cosz_safe * np.exp(-0.059 / cosz_safe)
    return np.where(elevation_deg <= 0, 0.0, np.nan_to_num(ghi, nan=0.0))


def load_dataset() -> tuple[pd.DataFrame, dict]:
    plant = pd.read_parquet(PLANT_5MIN)
    plant["grid_time_kst"] = pd.to_datetime(plant["grid_time_kst"])
    plant = plant.sort_values("grid_time_kst").reset_index(drop=True)

    # 그리드가 정확히 5분 간격 연속이어야 shift(정수 행이동)가 시간이동과 같다.
    diffs = plant["grid_time_kst"].diff().dropna().unique()
    if not (len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=5)):
        raise RuntimeError(f"5분 연속그리드가 아님 - shift 기반 lag/타깃 계산 불가: {diffs[:5]}")

    defect_cfg = json.loads(DEFECT_CONFIG.read_text(encoding="utf-8"))["defect_period"]
    defect_start = pd.Timestamp(defect_cfg["start_inclusive"])
    defect_end_excl = pd.Timestamp(defect_cfg["end_exclusive"])
    before = len(plant)
    plant = plant.loc[~((plant["grid_time_kst"] >= defect_start) & (plant["grid_time_kst"] < defect_end_excl))].copy()
    removed_defect_rows = before - len(plant)

    # 공식 신뢰 가능한 값만 타깃/lag 후보로 인정(동료대조 재분류 반영된 quality_status 게이팅).
    valid_quality = {"complete_observed", "complete_with_short_interpolation",
                     "complete_with_night_zero", "complete_with_idle_zero"}
    plant.loc[~plant["quality_status"].isin(valid_quality), "plant_ac_power_kw"] = np.nan

    asos = pd.read_csv(ASOS_CSV)
    asos = asos.rename(columns=ASOS_COLS)
    asos["observation_time_kst"] = pd.to_datetime(asos["observation_time_kst"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    asos = asos[["observation_time_kst"] + [v for v in ASOS_COLS.values() if v not in ("observation_time_kst", "station")]]
    asos = asos.sort_values("observation_time_kst").drop_duplicates("observation_time_kst")

    df = pd.merge_asof(
        plant, asos, left_on="grid_time_kst", right_on="observation_time_kst",
        direction="backward", tolerance=pd.Timedelta("3h"),
    )

    # lag(과거 실측) - 전부 T 시점까지 알 수 있는 값만.
    df["power_lag_0min"] = df["plant_ac_power_kw"]
    df["power_lag_30min"] = df["plant_ac_power_kw"].shift(6)
    df["power_lag_60min"] = df["plant_ac_power_kw"].shift(12)

    # 타깃(T+1h) - 미래값, 특성으로 절대 안 씀.
    df[TARGET] = df["plant_ac_power_kw"].shift(-LEAD_SLOTS)
    df["target_time_kst"] = df["grid_time_kst"] + pd.Timedelta(minutes=60)

    df["solar_elevation_now"] = df["solar_elevation_deg"]
    df["solar_elevation_t1h"] = df["solar_elevation_deg"].shift(-LEAD_SLOTS)

    df["clearsky_ghi_now_wm2"] = haurwitz_clearsky_ghi_wm2(df["solar_elevation_now"].to_numpy())
    df["clearsky_power_now_kw"] = CAPACITY_KW * df["clearsky_ghi_now_wm2"] / 1000.0
    df["clearsky_ghi_t1h_wm2"] = haurwitz_clearsky_ghi_wm2(df["solar_elevation_t1h"].to_numpy())
    df["clearsky_power_t1h_kw"] = CAPACITY_KW * df["clearsky_ghi_t1h_wm2"] / 1000.0

    kt_valid = df["clearsky_power_now_kw"] > 1.0  # 너무 작은 분모(여명/황혼) 배제
    df["kt_now"] = np.where(kt_valid, df["power_lag_0min"] / df["clearsky_power_now_kw"], np.nan)
    df["kt_now"] = df["kt_now"].clip(0, 1.5)  # 물리적 상한 근처로 완화(센서·구름반사 이상치 방지)

    hour = df["grid_time_kst"].dt.hour + df["grid_time_kst"].dt.minute / 60.0
    doy = df["grid_time_kst"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["day"] = df["grid_time_kst"].dt.normalize()

    # 평가 대상: 대상시각(T+1h)이 물리적 낮(태양고도>0)인 행만 - 야간 항상0 예측은
    # 정보가 없어 모델 성능을 인위적으로 부풀린다(광주도 낮시간만 평가).
    daylight_target = df["solar_elevation_t1h"] > 0
    df = df[daylight_target].copy()

    return df, {"결함구간_제외행수(5분)": removed_defect_rows}


def pooled_score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = y_true - pred
    return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
            "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}


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
        train = df[df["day"].isin(train_days)]
        x = train[CANDIDATE_FEATURES + [TARGET]].dropna()
        row = {"폴드": i, "학습일수": len(train_days), "학습행수": len(x)}
        if len(x) >= VIF_CORR_MIN_ROWS:
            row["타깃상관"] = {c: round(float(x[c].corr(x[TARGET])), 3) for c in CANDIDATE_FEATURES}
        rows.append(row)
    return rows


def run_walkforward(df: pd.DataFrame, folds: list[tuple], features: list[str]) -> dict:
    all_true, all_model, all_simple, all_smart = [], [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤에 있음"
        need = features + [TARGET, "power_lag_0min", "kt_now", "clearsky_power_t1h_kw"]
        train = df[df["day"].isin(train_days)].dropna(subset=features + [TARGET])
        test = df[df["day"].isin(test_days)].dropna(subset=need)
        if len(train) < MIN_ROWS_PER_FOLD or len(test) < 1:
            continue

        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[features], train[TARGET])
        pred_model = np.clip(model.predict(test[features]), 0, None)

        y_true = test[TARGET].to_numpy()
        pred_simple = test["power_lag_0min"].to_numpy()  # 단순지속성
        pred_smart = np.clip(test["kt_now"].to_numpy() * test["clearsky_power_t1h_kw"].to_numpy(), 0, None)

        all_true.append(y_true); all_model.append(pred_model)
        all_simple.append(pred_simple); all_smart.append(pred_smart)
        fold_rows.append({
            "폴드": i, "시험일수": len(test_days), "시험행수": len(test),
            "모델_MAE": pooled_score(y_true, pred_model)["MAE_kW"],
            "스마트지속성_MAE": pooled_score(y_true, pred_smart)["MAE_kW"],
        })

    y_true_all = np.concatenate(all_true)
    return {
        "폴드수": len(fold_rows), "폴드별": fold_rows,
        "pooled_모델": pooled_score(y_true_all, np.concatenate(all_model)),
        "pooled_단순지속성": pooled_score(y_true_all, np.concatenate(all_simple)),
        "pooled_스마트지속성": pooled_score(y_true_all, np.concatenate(all_smart)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = load_dataset()
    days = pd.DatetimeIndex(np.sort(df["day"].unique()))
    folds = expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    corr = fold_internal_correlation(df, folds)
    perf = run_walkforward(df, folds, CANDIDATE_FEATURES)

    def improve(model_mae, base_mae):
        return round((1 - model_mae / base_mae) * 100, 1) if base_mae else None

    result = {
        **meta,
        "capacity_kw_사용값": CAPACITY_KW,
        "capacity_주의": "AC 후보(110kW×10), Blockdata 인버터별 명판 미검증 - 청천전력 계산에만 사용",
        "리드타임": "+1h(12슬롯×5분)",
        "평가대상": "대상시각(T+1h) 태양고도>0인 행만(야간 자명한 0 제외)",
        "일수_total": int(len(days)),
        "폴드구성": [{"폴드": i + 1, "학습일수": len(tr), "시험일수": len(te)} for i, (tr, te) in enumerate(folds)],
        "폴드내부_상관분석(leakage없음)": corr,
        "성능": {
            **perf,
            "pooled_MAE_개선율_vs단순지속성_pct": improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_단순지속성"]["MAE_kW"]),
            "pooled_MAE_개선율_vs스마트지속성_pct": improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_스마트지속성"]["MAE_kW"]),
            "pooled_RMSE_개선율_vs스마트지속성_pct": improve(perf["pooled_모델"]["RMSE_kW"], perf["pooled_스마트지속성"]["RMSE_kW"]),
        },
        "_방법론출처": "타깃(raw kW)·스마트지속성·Haurwitz청천전력 정의는 광주 2026-08-20밤 "
                     "검증결론 재사용(재실행 안 함). 특성상관·모델성능은 김제 자체 데이터로 독립검증.",
        "_판정": "잠정치 - promote_to_official 대상 아님(historical_archive_join_candidate 한계 상속, "
               "라이브 shadow 검증 전).",
    }

    (OUT_DIR / "김제_초단기1h_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
