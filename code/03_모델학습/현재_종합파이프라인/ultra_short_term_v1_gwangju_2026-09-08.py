# -*- coding: utf-8 -*-
"""광주 초단기(+1h~+4h) 총출력모델(09-08) - 영광 v1 방법론 그대로 재사용.

## 배경
09-08 사용자 지적("광주 기존 초단기가 사실 NWP(DSWRF/TCDC/LCDC)에
의존해서 지금 결측 게이트에 똑같이 막혀있다 - 그럼 부안/김제/영광처럼
NWP 무관 버전을 광주도 만들자")으로 착수. 광주의 기존 08-26 초단기
파이프라인(`live_feature_assembler_초단기_v1_2026-08-26.py`)은 재사용
안 하고 완전히 새로 만든다 - 방법론(타깃 raw kW·스마트지속성·Haurwitz
청천전력·구조튜닝 3후보)은 영광 v1(`ultra_short_term_v1_yeonggwang_
2026-09-07.py`)과 **완전히 동일**하게 재사용(재구현 안 함), 데이터
로딩만 광주 형식(csv, parquet 아님)에 맞게 새로 작성한다.

## 광주가 영광과 같은 점(그래서 영광 v1을 그대로 재사용 가능)
광주 ASOS156도 실제 일사계를 보유한다(라이브 실측 확인: 09-08 15시대
`일사량_W_m2` 747~836 W/m2 - 부안·김제 ASOS243과 달리 유효). 그래서
`obs_ghi_wm2`를 영광과 동일하게 후보특성에 포함한다.

## 데이터 소스(전부 저장자료, API 호출 없음)
- 발전량: `v3_multihorizon_2026-08-14/03_분석데이터/gwangju_5min_
  model_dataset.csv`의 `plant_output_kw` - 이미 5대 인버터 전부 유효한
  시각만 값이 채워져 있고(inverters_available==5, 나머지는 자연
  NaN) 나머지 컬럼(구식 NWP/OpenMeteo 파생 특성 등)은 이번 설계에서
  전혀 쓰지 않는다(순수 발전량·시각만 재사용).
- 기상: `환경요인_기상청_ASOS/기상청_ASOS156_시간환경_모델용.csv`
  (710일 표준기간과 동일 범위, 실측 확인: 17,040행=710일×24h 정확히
  일치).
- 좌표: config.json 확정값(위도35.14428133·경도126.84058771,
  치평동 대표좌표) - `02_전처리/clean_nwp_direct_diffuse_bsrn_qc_v1.py`
  의 LATITUDE/LONGITUDE와 동일값, NOAA 근사식도 그 모듈과 동일 계열
  (부안·김제·영광 전처리 스크립트가 쓰는 것과 동일 공식, 좌표만 광주용).
- 용량: config.json 활성 프로필 240.58kW(09-02 밤 사용자 확정,
  blockdata_live_inverter_sum_240_58).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLANT_5MIN_CSV = (
    Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\v3_multihorizon_2026-08-14")
    / "03_분석데이터" / "gwangju_5min_model_dataset.csv"
)
ASOS_CSV = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\기상청_ASOS156_시간환경_모델용.csv")
OUT_DIR = HERE / "outputs" / "광주_초단기_v1_2026-09-08"

CAPACITY_KW = 240.58  # config.json 활성 프로필(09-02 밤 확정)
LATITUDE, LONGITUDE = 35.14428133, 126.84058771  # 치평동 대표좌표(config.json과 동일)

INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
LEAD_HOURS = [1, 2, 3, 4]

ASOS_COLS = {
    "시각": "observation_time_kst",
    "기온_C": "obs_temp_c", "상대습도_pct": "obs_rh_pct",
    "전운량_pct": "obs_cloud_pct", "풍속_m_s": "obs_wind_ms",
    "일사량_W_m2": "obs_ghi_wm2",
}
BASE_FEATURES = [
    "solar_elevation_now", "solar_elevation_target",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms", "obs_ghi_wm2",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_MINUTES = [0, 15, 30, 60]

# 영광 v1과 동일 구조 후보 3종(그대로 재사용, 재구현 안 함).
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
    """광주/부안/김제/영광과 동일한 NOAA 근사식(치평동 좌표) -
    `02_전처리/clean_nwp_direct_diffuse_bsrn_qc_v1.py`의 solar_position과
    같은 계열, 벡터화 형태만 다른 지역 전처리 스크립트들과 통일."""
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
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
    true_solar_min = hour * 60 + time_offset
    hour_angle = np.deg2rad(true_solar_min / 4 - 180)
    lat_r = np.deg2rad(LATITUDE)
    sin_elev = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    return np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))


def haurwitz_clearsky_ghi_wm2(elevation_deg: np.ndarray) -> np.ndarray:
    """영광 v1과 동일 공식(Haurwitz 1945) 그대로 재사용."""
    elev = np.clip(elevation_deg, 0, 90)
    zenith_rad = np.deg2rad(90.0 - elev)
    cosz = np.cos(zenith_rad)
    cosz_safe = np.where(cosz > 1e-3, cosz, np.nan)
    ghi = 1098.0 * cosz_safe * np.exp(-0.059 / cosz_safe)
    return np.where(elevation_deg <= 0, 0.0, np.nan_to_num(ghi, nan=0.0))


def load_base() -> tuple[pd.DataFrame, dict]:
    plant = pd.read_csv(PLANT_5MIN_CSV, usecols=["time", "plant_output_kw"])
    plant["grid_time_kst"] = pd.to_datetime(plant["time"])
    plant = plant.rename(columns={"plant_output_kw": "plant_ac_power_kw"})
    plant = plant.sort_values("grid_time_kst").reset_index(drop=True)[
        ["grid_time_kst", "plant_ac_power_kw"]
    ]
    # plant_output_kw는 원본 csv 자체가 5대 인버터 전부 유효한 시각만
    # 값을 채워뒀음을 실측 확인(inverters_available==5 행수와 정확히
    # 일치) - 부분합계·임의보간 없음. 별도 완전가용 필터 불필요.

    diffs = plant["grid_time_kst"].diff().dropna().unique()
    if not (len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=5)):
        raise RuntimeError(f"5분 연속그리드가 아님: {diffs[:5]}")

    asos = pd.read_csv(ASOS_CSV).rename(columns=ASOS_COLS)
    asos["observation_time_kst"] = pd.to_datetime(asos["observation_time_kst"])
    keep = ["observation_time_kst"] + [v for v in ASOS_COLS.values() if v != "observation_time_kst"]
    asos = asos[keep].sort_values("observation_time_kst").drop_duplicates("observation_time_kst")

    df = pd.merge_asof(plant, asos, left_on="grid_time_kst", right_on="observation_time_kst",
                       direction="backward", tolerance=pd.Timedelta("3h"))

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

    return df, {"결측구간_제외행수(5분)": 0, "비고": "gwangju_5min_model_dataset.csv가 이미 완전가용 시각만 채워둠"}


def build_horizon_frame(df: pd.DataFrame, lead_hours: int) -> tuple[pd.DataFrame, str, list[str]]:
    lead_slots = int(lead_hours * 60 / 5)
    target_col = f"target_power_t+{lead_hours}h_kw"
    d = df.copy()
    d[target_col] = d["plant_ac_power_kw"].shift(-lead_slots)
    d["solar_elevation_target"] = d["solar_elevation_deg"].shift(-lead_slots)
    d["clearsky_ghi_target_wm2"] = haurwitz_clearsky_ghi_wm2(d["solar_elevation_target"].to_numpy())
    d["clearsky_power_target_kw"] = CAPACITY_KW * d["clearsky_ghi_target_wm2"] / 1000.0
    d = d[d["solar_elevation_target"] > 0].copy()  # 대상시각 낮만 평가(영광 v1과 동일)
    features = [f"power_lag_{m}min" for m in LAG_MINUTES] + BASE_FEATURES
    return d, target_col, features
