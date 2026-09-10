# -*- coding: utf-8 -*-
"""김제 초단기(+1h~+4h) 총출력모델(09-01, v2 - 리드타임 파라미터화).

v1(ultra_short_term_v1_gimje_2026-09-01.py, +1h 전용)을 일반화 - 프로젝트
확정 티어 정의("5~15분 주기, 15분 단위, 추후 1~4시간")에 맞춰 +1h~+4h
전부 계산. 방법론(raw kW 타깃·스마트지속성·Haurwitz청천전력)은 v1과 동일,
광주 2026-08-20밤 검증결론 재사용. 리드타임별 특성상관·성능은 김제 자체
데이터로 각각 독립 검증(리드타임이 길어질수록 lag 특성 가치가 어떻게
바뀌는지 보는 게 핵심 - 08-20밤 결과처럼 "긴 리드에서 다른 특성이 필요"
할 수 있음을 미리 가정하지 않고 실측으로 확인).
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
OUT_DIR = HERE / "outputs" / "김제_초단기멀티호라이즌_v2_2026-09-01"

CAPACITY_KW = 1100.0  # ★AC 후보, 명판 미검증★
SEED = 42
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
VIF_CORR_MIN_ROWS = 10
LEAD_HOURS = [1, 2, 3, 4]

ASOS_COLS = {
    "시각": "observation_time_kst", "지점번호": "station",
    "기온_C": "obs_temp_c", "강수량_mm": "obs_rain_mm", "풍속_m_s": "obs_wind_ms",
    "상대습도_pct": "obs_rh_pct", "전운량_10분위": "obs_cloud_tenths", "전운량_pct": "obs_cloud_pct",
}
BASE_FEATURES = [
    "solar_elevation_now", "solar_elevation_target",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_MINUTES = [0, 15, 30, 60]  # 직전~1시간 전까지 4개 lag(리드타임과 무관하게 항상 "지금까지 안 값")


def haurwitz_clearsky_ghi_wm2(elevation_deg: np.ndarray) -> np.ndarray:
    elev = np.clip(elevation_deg, 0, 90)
    zenith_rad = np.deg2rad(90.0 - elev)
    cosz = np.cos(zenith_rad)
    cosz_safe = np.where(cosz > 1e-3, cosz, np.nan)
    ghi = 1098.0 * cosz_safe * np.exp(-0.059 / cosz_safe)
    return np.where(elevation_deg <= 0, 0.0, np.nan_to_num(ghi, nan=0.0))


def load_base() -> tuple[pd.DataFrame, dict]:
    plant = pd.read_parquet(PLANT_5MIN)
    plant["grid_time_kst"] = pd.to_datetime(plant["grid_time_kst"])
    plant = plant.sort_values("grid_time_kst").reset_index(drop=True)

    diffs = plant["grid_time_kst"].diff().dropna().unique()
    if not (len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=5)):
        raise RuntimeError(f"5분 연속그리드가 아님: {diffs[:5]}")

    defect_cfg = json.loads(DEFECT_CONFIG.read_text(encoding="utf-8"))["defect_period"]
    defect_start = pd.Timestamp(defect_cfg["start_inclusive"])
    defect_end_excl = pd.Timestamp(defect_cfg["end_exclusive"])
    before = len(plant)
    plant = plant.loc[~((plant["grid_time_kst"] >= defect_start) & (plant["grid_time_kst"] < defect_end_excl))].copy()
    removed_defect_rows = before - len(plant)

    valid_quality = {"complete_observed", "complete_with_short_interpolation",
                     "complete_with_night_zero", "complete_with_idle_zero"}
    plant.loc[~plant["quality_status"].isin(valid_quality), "plant_ac_power_kw"] = np.nan

    asos = pd.read_csv(ASOS_CSV).rename(columns=ASOS_COLS)
    asos["observation_time_kst"] = pd.to_datetime(asos["observation_time_kst"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    keep = ["observation_time_kst"] + [v for v in ASOS_COLS.values() if v not in ("observation_time_kst", "station")]
    asos = asos[keep].sort_values("observation_time_kst").drop_duplicates("observation_time_kst")

    df = pd.merge_asof(plant, asos, left_on="grid_time_kst", right_on="observation_time_kst",
                       direction="backward", tolerance=pd.Timedelta("3h"))

    for m in LAG_MINUTES:
        slots = m // 5
        df[f"power_lag_{m}min"] = df["plant_ac_power_kw"].shift(slots)

    df["solar_elevation_now"] = df["solar_elevation_deg"]
    df["clearsky_ghi_now_wm2"] = haurwitz_clearsky_ghi_wm2(df["solar_elevation_now"].to_numpy())
    df["clearsky_power_now_kw"] = CAPACITY_KW * df["clearsky_ghi_now_wm2"] / 1000.0
    kt_valid = df["clearsky_power_now_kw"] > 1.0
    df["kt_now"] = np.where(kt_valid, df["power_lag_0min"] / df["clearsky_power_now_kw"], np.nan).clip(0, 1.5)

    hour = df["grid_time_kst"].dt.hour + df["grid_time_kst"].dt.minute / 60.0
    doy = df["grid_time_kst"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day"] = df["grid_time_kst"].dt.normalize()

    return df, {"결함구간_제외행수(5분)": removed_defect_rows}


def build_horizon_frame(df: pd.DataFrame, lead_hours: int) -> tuple[pd.DataFrame, str, list[str]]:
    lead_slots = int(lead_hours * 60 / 5)
    target_col = f"target_power_t+{lead_hours}h_kw"
    d = df.copy()
    d[target_col] = d["plant_ac_power_kw"].shift(-lead_slots)
    d["solar_elevation_target"] = d["solar_elevation_deg"].shift(-lead_slots)
    d["clearsky_ghi_target_wm2"] = haurwitz_clearsky_ghi_wm2(d["solar_elevation_target"].to_numpy())
    d["clearsky_power_target_kw"] = CAPACITY_KW * d["clearsky_ghi_target_wm2"] / 1000.0
    d = d[d["solar_elevation_target"] > 0].copy()  # 대상시각 낮만 평가
    features = [f"power_lag_{m}min" for m in LAG_MINUTES] + BASE_FEATURES
    return d, target_col, features


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


def fold_internal_correlation(d: pd.DataFrame, target_col: str, features: list[str], folds: list[tuple]) -> list[dict]:
    rows = []
    for i, (train_days, _test_days) in enumerate(folds, start=1):
        train = d[d["day"].isin(train_days)]
        x = train[features + [target_col]].dropna()
        row = {"폴드": i, "학습일수": len(train_days), "학습행수": len(x)}
        if len(x) >= VIF_CORR_MIN_ROWS:
            row["타깃상관"] = {c: round(float(x[c].corr(x[target_col])), 3) for c in features}
        rows.append(row)
    return rows


def run_walkforward(d: pd.DataFrame, target_col: str, features: list[str], folds: list[tuple]) -> dict:
    all_true, all_model, all_simple, all_smart = [], [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출"
        need = features + [target_col, "power_lag_0min", "kt_now", "clearsky_power_target_kw"]
        train = d[d["day"].isin(train_days)].dropna(subset=features + [target_col])
        test = d[d["day"].isin(test_days)].dropna(subset=need)
        if len(train) < MIN_ROWS_PER_FOLD or len(test) < 1:
            continue
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[features], train[target_col])
        pred_model = np.clip(model.predict(test[features]), 0, None)

        y_true = test[target_col].to_numpy()
        pred_simple = test["power_lag_0min"].to_numpy()
        pred_smart = np.clip(test["kt_now"].to_numpy() * test["clearsky_power_target_kw"].to_numpy(), 0, None)

        all_true.append(y_true); all_model.append(pred_model)
        all_simple.append(pred_simple); all_smart.append(pred_smart)
        fold_rows.append({"폴드": i, "시험일수": len(test_days), "시험행수": len(test),
                          "모델_MAE": pooled_score(y_true, pred_model)["MAE_kW"],
                          "스마트지속성_MAE": pooled_score(y_true, pred_smart)["MAE_kW"]})

    if not all_true:
        return {"폴드수": 0, "폴드별": [], "오류": "유효 폴드 없음"}
    y_true_all = np.concatenate(all_true)
    return {
        "폴드수": len(fold_rows), "폴드별": fold_rows,
        "pooled_모델": pooled_score(y_true_all, np.concatenate(all_model)),
        "pooled_단순지속성": pooled_score(y_true_all, np.concatenate(all_simple)),
        "pooled_스마트지속성": pooled_score(y_true_all, np.concatenate(all_smart)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, meta = load_base()

    def improve(model_mae, base_mae):
        return round((1 - model_mae / base_mae) * 100, 1) if base_mae else None

    horizon_results = {}
    for h in LEAD_HOURS:
        d, target_col, features = build_horizon_frame(base, h)
        days = pd.DatetimeIndex(np.sort(d["day"].unique()))
        folds = expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
        corr = fold_internal_correlation(d, target_col, features, folds)
        perf = run_walkforward(d, target_col, features, folds)
        entry = {
            "리드타임": f"+{h}h", "일수_total": int(len(days)),
            "폴드내부_상관분석_1번폴드예시": corr[0] if corr else None,
            "성능": perf,
        }
        if perf.get("폴드수", 0) > 0:
            entry["성능"]["pooled_MAE_개선율_vs단순지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_단순지속성"]["MAE_kW"])
            entry["성능"]["pooled_MAE_개선율_vs스마트지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_스마트지속성"]["MAE_kW"])
            entry["성능"]["pooled_RMSE_개선율_vs스마트지속성_pct"] = improve(perf["pooled_모델"]["RMSE_kW"], perf["pooled_스마트지속성"]["RMSE_kW"])
        horizon_results[f"+{h}h"] = entry
        print(f"[+{h}h] 완료 - pooled MAE(모델)={perf.get('pooled_모델',{}).get('MAE_kW')}, "
              f"스마트지속성={perf.get('pooled_스마트지속성',{}).get('MAE_kW')}")

    result = {
        **meta,
        "capacity_kw_사용값": CAPACITY_KW,
        "capacity_주의": "AC 후보(110kW×10), 명판 미검증 - 청천전력 계산에만 사용",
        "평가대상": "각 리드타임의 대상시각 태양고도>0인 행만",
        "리드타임별_결과": horizon_results,
        "_방법론출처": "타깃(raw kW)·스마트지속성·Haurwitz청천전력 정의는 광주 2026-08-20밤 검증결론 재사용. "
                     "특성상관·성능은 리드타임별로 김제 자체 데이터 독립검증.",
        "_판정": "잠정치 - promote_to_official 대상 아님(historical_archive_join_candidate 한계, 라이브 shadow 검증 전).",
    }

    (OUT_DIR / "김제_초단기_1to4h_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "리드타임별_결과"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
