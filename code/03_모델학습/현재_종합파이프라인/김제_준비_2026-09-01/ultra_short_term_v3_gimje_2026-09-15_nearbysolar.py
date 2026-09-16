# -*- coding: utf-8 -*-
"""김제 초단기(+1h~+4h) - 인근관측소(전주146) 실측 일사량 후보 피처판(09-15).

09-15 사용자 지시("정식 피처 추가하고 재학습해서 MAE 비교해줘")에 따른
후보 모델. `ultra_short_term_v2_gimje_multihorizon_2026-09-01.py`(현재
라이브 배포판, `공식_초단기_issue_safe_v1_2026-09-08`)를 그대로 재사용
(재구현 안 함)하되 BASE_FEATURES에 `nearby_ghi_wm2` 하나만 추가한다.

## 리크 방지
09-01 상관분석(`check_correlation_nearby_solar_gimje_v1_2026-09-01.py`)은
탐색용으로 target_time(미래) 기준 join이었다. 이 스크립트는 부안 09-15
후보판과 동일하게 merge_asof(backward, 3h tolerance)로 issue_time
(grid_time_kst) 시점에 알려진 값만 사용 - 기존 obs_temp_c 등과 동일한
안전조건, 미래 참조 없음.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLANT_5MIN = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_발전소_5분_공식후보.parquet"
)
ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31\ASOS"
    r"\기상청_ASOS243_시간환경_20240825_20260804.csv"
)
NEARBY_GHI_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
    r"\전주146\기상청_ASOS146_시간환경_20240825_20260804.csv"
)
DEFECT_CONFIG = HERE.parent.parent.parent / "02_전처리" / "김제" / "config" / "김제_전처리_규칙_v1_2026-08-31.json"
OUT_DIR = HERE / "outputs" / "김제_초단기_v3_인근일사량_후보_2026-09-15"

CAPACITY_KW = 1100.0
SEED = 42
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
LEAD_HOURS = [1, 2, 3, 4]

ASOS_COLS = {
    "시각": "observation_time_kst", "지점번호": "station",
    "기온_C": "obs_temp_c", "강수량_mm": "obs_rain_mm", "풍속_m_s": "obs_wind_ms",
    "상대습도_pct": "obs_rh_pct", "전운량_10분위": "obs_cloud_tenths", "전운량_pct": "obs_cloud_pct",
}
# ★09-15 신규★ nearby_ghi_wm2 추가 - 그 외 v2_2026-09-01과 완전 동일.
BASE_FEATURES = [
    "solar_elevation_now", "solar_elevation_target",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms",
    "nearby_ghi_wm2",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_MINUTES = [0, 15, 30, 60]

# 김제 v2 원본엔 STRUCTURES 속성이 없어 build_regional_ultrashort_
# mediumterm_issue_safe_v1_2026-09-08.py가 GIMJE_FALLBACK_STRUCTURES로
# 폴백한다(단일 "raw" 구조) - 이 후보판도 프로덕션과 동일 조건 비교를
# 위해 그 값을 그대로 명시해둔다(재구현 아님, 값만 동일하게 복제).
STRUCTURES = {
    "raw": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=1.0),
}


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

    # ★09-15 신규★ 전주146 인근일사량(부안 후보판과 동일 안전조건).
    nearby = pd.read_csv(NEARBY_GHI_CSV)
    nearby["observation_time_kst"] = pd.to_datetime(nearby["시각"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    nearby = nearby[["observation_time_kst", "일사량_W_m2"]].rename(columns={"일사량_W_m2": "nearby_ghi_wm2"})
    nearby = nearby.sort_values("observation_time_kst").drop_duplicates("observation_time_kst")
    df = pd.merge_asof(df.sort_values("grid_time_kst"), nearby, left_on="grid_time_kst",
                       right_on="observation_time_kst", direction="backward",
                       tolerance=pd.Timedelta("3h"), suffixes=("", "_nearby"))

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
    d = d[d["solar_elevation_target"] > 0].copy()
    features = [f"power_lag_{m}min" for m in LAG_MINUTES] + BASE_FEATURES
    return d, target_col, features
