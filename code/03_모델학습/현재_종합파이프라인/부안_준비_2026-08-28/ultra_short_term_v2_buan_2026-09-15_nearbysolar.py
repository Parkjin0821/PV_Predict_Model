# -*- coding: utf-8 -*-
"""부안 초단기(+1h~+4h) - 인근관측소(전주146) 실측 일사량 후보 피처판(09-15).

## 배경
09-15 사용자 지시("정식 피처 추가하고 재학습해서 MAE 비교해줘")에 따른
후보 모델. `ultra_short_term_v1_buan_2026-09-08.py`(현재 라이브 배포판,
`공식_초단기_v2_날씨피처_2026-09-08`)를 그대로 재사용(재구현 안 함)하되
BASE_FEATURES에 `nearby_ghi_wm2` 하나만 추가한다 - 그 외 로직·구조 후보·
lag·시간피처는 전혀 안 건드림.

## 근거(09-15 상관분석, AGENTS.md 2026-09-15(6) 참고)
부안 station 243은 일사계가 없어 obs_ghi_wm2를 못 씀. 대신 전주146
(약 38km)의 실측 일사량을 부안 발전량과 폴드내부 재검증(leakage 없이)
했더니 평균|r|=0.798(0.787~0.813, 부호일관, 5폴드) - 기존 solar_elevation
(0.691)·forecast_DSWRF(0.514)보다 뚜렷이 강함. **이 스크립트는 그 상관을
실제 모델 성능(MAE) 개선으로 이어지는지 검증하는 다음 단계**.

## 리크 방지(중요)
09-15 상관분석(`check_correlation_nearby_solar_buan.py`)은 탐색용이라
전주146 관측을 **target_time(미래) 기준**으로 join했다(같은 시각 실측
대 같은 시각 발전량 - 물리적 신호 강도 확인용). 하지만 실제 서빙 모델은
issue_time(현재)에서 h시간 뒤를 예측해야 하므로, 미래 관측값을 피처로
쓸 수 없다(누출). 이 스크립트는 광주·영광의 obs_ghi_wm2와 완전히 동일한
방식으로 **merge_asof(..., direction="backward")로 issue_time(grid_time_kst)
시점에 알려진 가장 최근 전주146 관측값만** 사용한다 - 기존 obs_temp_c 등
자체 ASOS 피처와 정확히 같은 안전조건.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLANT_5MIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_인버터별_5분_야간0포함.parquet"
)
ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\기상과거백필_v1_2026-08-28"
    r"\ASOS\기상청_ASOS243_시간환경_20251212_20260804.csv"
)
NEARBY_GHI_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
    r"\전주146\기상청_ASOS146_시간환경_20240825_20260804.csv"
)
OUT_DIR = HERE / "outputs" / "부안_초단기_v2_인근일사량_후보_2026-09-15"

CAPACITY_KW = 1000.0
LATITUDE, LONGITUDE = 35.7874617462152, 126.73000042548799
EXPECTED_INVERTERS = 8
DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")
VALID_INVERTER_QUALITY = ["observed", "physical_zero_night", "night_zero_physical"]

INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20
MIN_ROWS_PER_FOLD = 30
LEAD_HOURS = [1, 2, 3, 4]

ASOS_COLS = {
    "시각": "observation_time_kst", "지점번호": "station",
    "기온_C": "obs_temp_c", "강수량_mm": "obs_rain_mm", "풍속_m_s": "obs_wind_ms",
    "상대습도_pct": "obs_rh_pct", "전운량_10분위": "obs_cloud_tenths", "전운량_pct": "obs_cloud_pct",
}
# ★09-15 신규★ nearby_ghi_wm2 추가 - 그 외 v1_2026-09-08과 완전 동일.
BASE_FEATURES = [
    "solar_elevation_now", "solar_elevation_target",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms",
    "nearby_ghi_wm2",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_MINUTES = [0, 15, 30, 60]

STRUCTURES = {
    "raw": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=1.0),
    "shallow_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=7, max_depth=3,
                        min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                        reg_alpha=0.3, reg_lambda=2.0),
    "leaf_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=31, max_depth=-1,
                     min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
                     reg_alpha=0.5, reg_lambda=3.0),
}


def solar_elevation_deg(ts: pd.DatetimeIndex) -> np.ndarray:
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    time_offset = eqtime + 4 * LONGITUDE - 60 * 9
    tst = hour * 60 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)
    lat_r = np.deg2rad(LATITUDE)
    cos_zenith = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    return 90.0 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))


def haurwitz_clearsky_ghi_wm2(elevation_deg: np.ndarray) -> np.ndarray:
    elev = np.clip(elevation_deg, 0, 90)
    zenith_rad = np.deg2rad(90.0 - elev)
    cosz = np.cos(zenith_rad)
    cosz_safe = np.where(cosz > 1e-3, cosz, np.nan)
    ghi = 1098.0 * cosz_safe * np.exp(-0.059 / cosz_safe)
    return np.where(elevation_deg <= 0, 0.0, np.nan_to_num(ghi, nan=0.0))


def load_plant_total_5min() -> pd.Series:
    df = pd.read_parquet(PLANT_5MIN_PARQUET, columns=[
        "grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"])
    df["ok"] = df["quality_status_after_night"].isin(VALID_INVERTER_QUALITY)
    ok = df[df["ok"]]
    count_full = ok.groupby("grid_time_kst")["inverter_number"].nunique()
    valid_times = count_full[count_full == EXPECTED_INVERTERS].index
    total = ok[ok["grid_time_kst"].isin(valid_times)].groupby("grid_time_kst")["ac_power_kw"].sum()
    total = total[(total.index.normalize() < DEFECT_START) | (total.index.normalize() >= DEFECT_END_EXCLUSIVE)]

    start, end = total.index.min().floor("5min"), total.index.max().ceil("5min")
    grid = pd.date_range(start, end, freq="5min")
    return total.reindex(grid)


def load_base() -> tuple[pd.DataFrame, dict]:
    power = load_plant_total_5min()
    plant = pd.DataFrame({"grid_time_kst": power.index, "plant_ac_power_kw": power.to_numpy()})

    diffs = plant["grid_time_kst"].diff().dropna().unique()
    if not (len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=5)):
        raise RuntimeError(f"5분 연속그리드가 아님: {diffs[:5]}")

    asos = pd.read_csv(ASOS_CSV).rename(columns=ASOS_COLS)
    asos["observation_time_kst"] = pd.to_datetime(asos["observation_time_kst"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    keep = ["observation_time_kst"] + [v for v in ASOS_COLS.values() if v not in ("observation_time_kst", "station")]
    asos = asos[keep].sort_values("observation_time_kst").drop_duplicates("observation_time_kst")

    df = pd.merge_asof(plant, asos, left_on="grid_time_kst", right_on="observation_time_kst",
                       direction="backward", tolerance=pd.Timedelta("3h"))

    # ★09-15 신규★ 전주146 인근일사량 - 로컬 ASOS와 완전히 동일한 안전조건
    # (backward, 3h tolerance)으로만 join. 미래시각 참조 없음.
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

    df["solar_elevation_deg"] = solar_elevation_deg(pd.DatetimeIndex(df["grid_time_kst"]))
    df["solar_elevation_now"] = df["solar_elevation_deg"]
    df["clearsky_ghi_now_wm2"] = haurwitz_clearsky_ghi_wm2(df["solar_elevation_now"].to_numpy())
    df["clearsky_power_now_kw"] = CAPACITY_KW * df["clearsky_ghi_now_wm2"] / 1000.0
    kt_valid = df["clearsky_power_now_kw"] > 1.0
    df["kt_now"] = np.clip(
        np.where(kt_valid, df["power_lag_0min"] / df["clearsky_power_now_kw"], np.nan), 0, 1.5
    )

    hour = df["grid_time_kst"].dt.hour + df["grid_time_kst"].dt.minute / 60.0
    doy = df["grid_time_kst"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day"] = df["grid_time_kst"].dt.normalize()

    return df, {"8대완전가용_5분표본수": int(power.notna().sum())}


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
