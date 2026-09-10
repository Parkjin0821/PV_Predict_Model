# -*- coding: utf-8 -*-
"""공식 시간단위 데이터셋 v3 — 고정-tm(리드타임 15~36h, 누출 없음) 재구성.

v2(누출 tm)와의 차이는 NWP 소스 파일 하나뿐이다. ASOS 관측·장비텔레메트리
(08-20 밤 재정의 반영됨)·격자예보는 애초에 누출이 없었으므로 그대로 쓴다.
일조시간·모듈온도도 이 파일 안에서 같이 재산출한다(청천모양 재구성이
끝난 1시간 해상도 DSWRFLX/DSWRF를 그대로 재사용).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS"
    r"\gwangju_1hour_model_dataset_kma_observed.csv"
)
NWP_V3_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\nwp_hourly_interpolation_v3_fixed_tm_2026-08-21"
    r"\광주_예보_1시간재구성_고정tm_710일.csv"
)
INVERTER_REDEFINED_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\inverter_equipment_feature_redefinition_v1_2026-08-20"
    r"\광주_인버터_주파수온도_재정의_1시간_710일.csv"
)
OUT_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed")
OUT_CSV = OUT_DIR / "gwangju_1hour_model_dataset_official_v3_fixed_tm_2026-08-21.csv"

U0, U1 = 25.0, 6.84       # Faiman(2008)
DELTA_T_CND = 3.0          # Sandia PVPMC
THRESHOLD_W_M2 = 120.0
DAYLIGHT_COVERAGE_MIN = 0.90


def derive_sunshine(nwp: pd.DataFrame) -> pd.Series:
    day = nwp.index.normalize()
    tmp = nwp.copy()
    tmp["날짜"] = day
    daylight = tmp[tmp["태양고도_deg"] > 0]
    rows = {}
    for date, g in daylight.groupby("날짜"):
        n_total = len(g)
        n_valid = g["DSWRFLX_bsrn정제"].notna().sum()
        coverage = n_valid / n_total if n_total else 0.0
        if coverage >= DAYLIGHT_COVERAGE_MIN:
            rows[date] = (g["DSWRFLX_bsrn정제"].dropna() >= THRESHOLD_W_M2).sum()
        else:
            rows[date] = None
    return pd.Series(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    open_meteo_cols = [c for c in base.columns if c.startswith("open_meteo_forecast_")]
    base = base.drop(columns=open_meteo_cols)
    print(f"[1] Open-Meteo 예보 컬럼 {len(open_meteo_cols)}개 제거")

    inverter_redef = pd.read_csv(INVERTER_REDEFINED_CSV, parse_dates=["time"], low_memory=False).set_index("time")
    base = base.drop(columns=["mean_frequency_hz", "mean_inverter_temperature_c"])
    base["mean_frequency_hz"] = inverter_redef["재정의_주파수평균_Hz_정상만"]
    base["mean_inverter_temperature_c"] = inverter_redef["재정의_인버터평균온도_C_정상만"]
    print("[2] 장비특성 재정의값 반영")

    nwp = pd.read_csv(NWP_V3_CSV, parse_dates=["time"], low_memory=False).set_index("time")
    print(f"[3] 고정-tm 1시간 재구성 NWP 로드: {len(nwp)}행, {len(nwp.columns)}컬럼")

    t_module = nwp["TMP"] + nwp["DSWRF"] / (U0 + U1 * nwp["WSD"])
    t_cell = t_module + (nwp["DSWRF"] / 1000.0) * DELTA_T_CND
    nwp = nwp.copy()
    nwp["추정_모듈표면온도"] = t_module
    nwp["추정_출력온도"] = t_cell
    print("[4] 모듈표면온도·출력온도 재계산")

    sunshine = derive_sunshine(nwp)
    n_valid = sunshine.notna().sum()
    print(f"[5] 일조시간 산출: {n_valid}/{len(sunshine)}일 성공 ({n_valid/len(sunshine)*100:.1f}%)")

    merged = base.join(nwp, how="left")
    merged["추정_일조시간_hr"] = merged.index.normalize().map(sunshine)

    merged.index.name = "time"
    merged.to_csv(OUT_CSV, encoding="utf-8-sig")

    print(f"\n행수: {len(merged)}, 컬럼수: {len(merged.columns)}")
    check_cols = list(nwp.columns) + ["추정_일조시간_hr"]
    for c in check_cols:
        v = merged[c].notna().sum()
        print(f"  {c:20s} 유효 {v}/{len(merged)} ({v/len(merged)*100:.1f}%)")
    print(f"\n저장 완료: {OUT_CSV}")


if __name__ == "__main__":
    main()
