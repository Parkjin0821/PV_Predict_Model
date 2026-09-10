# -*- coding: utf-8 -*-
"""부안 초단기(+1h~+4h) 총출력모델(09-08, 날씨피처 추가판) - 김제/영광/광주와
동일 방법론(타깃 raw kW·스마트지속성·Haurwitz청천전력·구조튜닝 3후보) 재사용.

## 배경
09-08 사용자 질의("부안만 유독 오차가 크다") → 감사 결과 부안 기존
초단기(08-28, `outputs/공식_초단기_issue_safe_v1_2026-09-08/`)는 발전량
자체의 lag/이동평균만 쓰고(`live_feature_assembler_ultrashort_v1_2026-09-08.py`
REGIONS["부안"]가 `mode="power_only", asos_db=None`으로 하드코딩돼있었음)
날씨 관측치가 전혀 없다는 걸 확인. 그런데 부안용 **라이브 ASOS 수집
자체는 이미 정상 동작 중**이었다(`kma_live_inputs_v1_2026-08-28\\
kma_live_inputs.sqlite3`의 asos_hourly, 관측소 243, 20:21 최신행 확인) -
"새로 수집을 시작"할 필요 없이 조립기 연결만 빠져있던 것. 이 스크립트는
김제 v2(`ultra_short_term_v2_gimje_multihorizon_2026-09-01.py`)와 완전히
동일한 레시피를 그대로 재사용해(재구현 안 함) 부안에도 날씨 피처를 태운
새 모델을 만든다.

## 부안이 김제와 같은 점(그래서 obs_ghi_wm2 제외 등 김제 레시피를 그대로 씀)
- **같은 관측소(243)** - 부안도 라이브·과거 ASOS 전부 station=243. 이
  관측소는 일사계가 없어(실측 확인: 일사량_W_m2 전부 결측) obs_ghi_wm2를
  피처에서 제외한다(김제와 동일 이유·동일 처리).
- 과거 ASOS csv도 부안 자체 파일(`기상과거백필_v1_2026-08-28/ASOS/
  기상청_ASOS243_시간환경_20251212_20260804.csv`)을 쓴다 - 부안 발전량
  기간(2025-12-12~2026-08-05)과 정확히 겹쳐서 굳이 김제 파일을 빌릴
  필요가 없었다(실측 확인).

## 부안이 김제와 다른 점(그래서 별도 스크립트로 새로 작성)
- 원본 parquet(`부안_인버터별_5분_야간0포함.parquet`)이 인버터별 원시
  행이라 총출력 집계가 안 돼있고 태양고도도 없다 - 08-31 기존 백테스트
  스크립트(`ultra_short_historical_backtest_v1_2026-08-31.py`)의 8대
  완전가용 집계·결함구간(04-15~05-22) 제외 로직과, 02_전처리(`build_
  buan_time_aggregates_v1_2026-08-28.py`)의 NOAA태양고도식(부안 좌표)을
  그대로 재사용해 이 스크립트 안에서 직접 만든다.
- 데이터 기간이 짧다(약 236일, 다른 지역은 600~700일+) - 기존 부안
  백테스트와 동일하게 INITIAL_TRAIN_DAYS=60/TEST_BLOCK_DAYS=20을 쓴다
  (다른 지역의 90/30을 그대로 쓰면 초기학습창이 짧은 이력 대비 과하게
  큼 - 근거 있는 조정, 임의 축소 아님).
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
OUT_DIR = HERE / "outputs" / "부안_초단기_v1_2026-09-08"

CAPACITY_KW = 1000.0  # AC 후보, 기존 부안 파이프라인과 동일값
LATITUDE, LONGITUDE = 35.7874617462152, 126.73000042548799  # 02_전처리/부안와 동일 좌표
EXPECTED_INVERTERS = 8
DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")
VALID_INVERTER_QUALITY = ["observed", "physical_zero_night", "night_zero_physical"]

INITIAL_TRAIN_DAYS = 60  # 부안 자체 이력이 짧아(약 236일) 기존 08-31 백테스트값 재사용
TEST_BLOCK_DAYS = 20
MIN_ROWS_PER_FOLD = 30
LEAD_HOURS = [1, 2, 3, 4]

ASOS_COLS = {
    "시각": "observation_time_kst", "지점번호": "station",
    "기온_C": "obs_temp_c", "강수량_mm": "obs_rain_mm", "풍속_m_s": "obs_wind_ms",
    "상대습도_pct": "obs_rh_pct", "전운량_10분위": "obs_cloud_tenths", "전운량_pct": "obs_cloud_pct",
}
# ★obs_ghi_wm2 제외★ - 관측소 243은 일사계 없음(김제와 동일 이유, 실측 확인).
BASE_FEATURES = [
    "solar_elevation_now", "solar_elevation_target",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_MINUTES = [0, 15, 30, 60]

# 영광 v1·광주 v1과 동일 구조 후보 3종(그대로 재사용, 재구현 안 함).
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
    """`02_전처리/부안/build_buan_time_aggregates_v1_2026-08-28.py`의
    NOAA 근사식을 그대로 재사용(부안 좌표, 벡터화 형태만 통일)."""
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
    """Haurwitz(1945) 공식 - 영광 v1·광주 v1과 동일, 그대로 재사용."""
    elev = np.clip(elevation_deg, 0, 90)
    zenith_rad = np.deg2rad(90.0 - elev)
    cosz = np.cos(zenith_rad)
    cosz_safe = np.where(cosz > 1e-3, cosz, np.nan)
    ghi = 1098.0 * cosz_safe * np.exp(-0.059 / cosz_safe)
    return np.where(elevation_deg <= 0, 0.0, np.nan_to_num(ghi, nan=0.0))


def load_plant_total_5min() -> pd.Series:
    """`ultra_short_historical_backtest_v1_2026-08-31.py`의 load_plant_5min_series()
    와 동일 로직(8대 완전가용 시각만 합산, 결함구간 제외, 격자 재색인) -
    재구현 안 하고 그대로 가져옴."""
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
    d = d[d["solar_elevation_target"] > 0].copy()  # 대상시각 낮만 평가(다른 지역과 동일)
    features = [f"power_lag_{m}min" for m in LAG_MINUTES] + BASE_FEATURES
    return d, target_col, features
