# -*- coding: utf-8 -*-
"""08-20 밤 전면 재작업판 — 공식 시간단위 학습 데이터셋 v2.

v1(`build_official_hourly_dataset_v1.py`)과의 차이:
1. NWP·격자예보를 "3시간 값을 그대로 시각별 컬럼에 얹는" 방식에서
   `interpolate_nwp_3h_to_1h_v1.py`가 만든 **진짜 1시간 해상도 재구성값**으로
   교체(DSWRF가 순간값이 아니라 직전3시간평균이라는 신규 발견 반영,
   청천모양 보존적 재분배 사용). 결과적으로 NWP 계열 컬럼의 시간 커버리지가
   기존 33%(3시간에 1번)에서 대부분 시각으로 대폭 확대된다.
2. 격자예보(TMP·SKY·REH·WSD·POP)가 v1에서는 아예 누락돼 있었다(08-20 감사
   발견) — 이번엔 포함.
3. 모듈표면온도·출력온도도 이 1시간 자료로 다시 계산해 넣는다(입력이 전부
   갖춰진 시각이면 매 시각 값이 나옴 — 기존엔 8개 시점/일 뿐이었음).
4. 일조시간은 `derive_gwangju_future_sunshine_v1.py`(08-20 밤 재작업판,
   1시간 임계값 판정 + 낮시간 유효비율>=90% 방식)의 결과를 쓴다.
5. 이번 버전에서는 시간정렬(shift) 문제를 원천 차단한다: NWP 계열은
   "그 시각에 실제로 유효한 예보값"으로 이미 자리잡혀 있으므로, 이후
   ablation 스크립트에서 예보 컬럼에 shift(1)을 적용하면 안 된다(그게
   바로 저녁 감사에서 발견된 치명적 버그였다) — 이 스크립트 자체는 값을
   시각에 맞게 정렬해서 내보내는 역할만 하고, "몇 시간 전 정보까지
   알았는가"(발행시각 대비 리드타임)는 5번 재실행 스크립트에서 별도로
   다룬다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS"
    r"\gwangju_1hour_model_dataset_kma_observed.csv"
)
NWP_V2_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\nwp_hourly_interpolation_v2_2026-08-20밤"
    r"\광주_예보_1시간재구성_710일.csv"
)
SUNSHINE_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\nwp_hourly_interpolation_v2_2026-08-20밤"
    r"\광주_추정_일조시간_1시간기반_710일.csv"
)
INVERTER_REDEFINED_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\inverter_equipment_feature_redefinition_v1_2026-08-20"
    r"\광주_인버터_주파수온도_재정의_1시간_710일.csv"
)
OUT_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed")
OUT_CSV = OUT_DIR / "gwangju_1hour_model_dataset_official_v2_2026-08-20밤.csv"

U0, U1 = 25.0, 6.84  # Faiman(2008) 개방형 거치 기본계수
DELTA_T_CND = 3.0  # Sandia PVPMC 개방형 거치 기본계수


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

    nwp = pd.read_csv(NWP_V2_CSV, parse_dates=["time"], low_memory=False).set_index("time")
    print(f"[3] 1시간 재구성 NWP·격자예보 로드: {len(nwp)}행, {len(nwp.columns)}컬럼")

    # 모듈표면온도·출력온도 재계산 (TMP·WSD·DSWRF 전부 1시간 해상도로 갖춰짐)
    t_module = nwp["TMP"] + nwp["DSWRF"] / (U0 + U1 * nwp["WSD"])
    t_cell = t_module + (nwp["DSWRF"] / 1000.0) * DELTA_T_CND
    nwp = nwp.copy()
    nwp["추정_모듈표면온도"] = t_module
    nwp["추정_출력온도"] = t_cell
    print("[4] 모듈표면온도·출력온도 1시간 해상도로 재계산")

    merged = base.join(nwp, how="left")

    sun = pd.read_csv(SUNSHINE_CSV, parse_dates=["날짜"])
    sun_map = sun.set_index("날짜")["추정_일조시간_hr"]
    merged["추정_일조시간_hr"] = merged.index.normalize().map(sun_map)
    print("[5] 일조시간(08-20 밤 재작업판) 날짜 기준 병합")

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
