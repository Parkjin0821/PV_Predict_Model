# -*- coding: utf-8 -*-
"""단기·초단기 라이브 조립기에 input_quality_gate를 배선하기 위한 공용
lag_spec 생성기(사용자 지시 ②, 좁은 범위: 발전출력 lag/이동통계 특성의
야간 결측만 예외처리, 나머지는 기존처럼 차단).

## 시각 오프셋 정의 출처(재구현 아님 - 그대로 옮겨적었을 뿐)
- 단기: `backtest_harness_v1_2026-08-20밤.py`의 `build_frame()`
  - `power = df["plant_output_kw"].shift(1)`
  - `발전출력_{h}시간전_kW = power.shift(h-1)` → 원천시각 = issue_time - h시간
  - `발전출력_{h}시간이동평균/표준편차_kW = power.rolling(h,...)` → 윈도우
    = issue_time-1시간 부터 issue_time-h시간까지 h개 시각
  - h in [1,2,3,6,24](점), [6,24](이동통계)
- 초단기: `train_ultra_short_official_v1_2026-08-21.py`의
  `build_ultra_short_frame()`
  - `power = quarter["발전출력_kW"].shift(1)`
  - `발전출력_{m}분전_kW = power.shift(steps-1)`(steps*15=m분) → 원천시각
    = issue_time - m분
  - `발전출력_{m}분이동평균/표준편차_kW = power.rolling(steps,...)` →
    윈도우 = issue_time-15분 부터 issue_time-m분까지 steps개 시각
  - steps in [1,2,4,8,16](점=15/30/60/120/240분), [4,16](이동통계=1시간/4시간)

이 두 시각 오프셋이 원본 학습 코드와 어긋나면 게이트가 "야간"이라고
잘못 승인할 원천시각이 틀린 시각을 가리키게 되므로, 원본 함수의 shift/
rolling 라인과 반드시 대조 후 수정할 것 - 주석에 라인 인용을 남겨둔 이유.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("iqg", HERE / "input_quality_gate_v1_2026-08-28.py")
iqg = _ilu.module_from_spec(_spec)
sys.modules["iqg"] = iqg
_spec.loader.exec_module(iqg)

HOURLY_POINT_LAG_HOURS = (1, 2, 3, 6, 24)
HOURLY_ROLLING_HOURS = (6, 24)
QUARTER_POINT_LAG_STEPS = (1, 2, 4, 8, 16)   # *15분
QUARTER_ROLLING_STEPS = (4, 16)              # *15분


def hourly_lag_specs(issue_time: pd.Timestamp) -> list:
    """단기(1시간 격자) 조립기용. issue_time은 tz 유무 무관하게 그대로
    빼기만 하므로, 호출부의 solar_elevation_deg 조회 시리즈와 tz가 같아야
    한다(호출부에서 맞출 것)."""
    specs = []
    for h in HOURLY_POINT_LAG_HOURS:
        specs.append(iqg.LagSpec(f"발전출력_{h}시간전_kW", [issue_time - pd.Timedelta(hours=h)]))
    for h in HOURLY_ROLLING_HOURS:
        window = [issue_time - pd.Timedelta(hours=k) for k in range(1, h + 1)]
        specs.append(iqg.LagSpec(f"발전출력_{h}시간이동평균_kW", window))
        specs.append(iqg.LagSpec(f"발전출력_{h}시간이동표준편차_kW", window))
    return specs


def quarter_lag_specs(issue_time: pd.Timestamp) -> list:
    """초단기(15분 격자) 조립기용."""
    specs = []
    for steps in QUARTER_POINT_LAG_STEPS:
        minutes = steps * 15
        specs.append(iqg.LagSpec(f"발전출력_{minutes}분전_kW",
                                 [issue_time - pd.Timedelta(minutes=minutes)]))
    for steps in QUARTER_ROLLING_STEPS:
        minutes = steps * 15
        window = [issue_time - pd.Timedelta(minutes=15 * k) for k in range(1, steps + 1)]
        specs.append(iqg.LagSpec(f"발전출력_{minutes}분이동평균_kW", window))
        specs.append(iqg.LagSpec(f"발전출력_{minutes}분이동표준편차_kW", window))
    return specs


def apply_gate(row_dict: dict, required_features: list[str], cfg: dict,
               lag_specs: list, elevation_at) -> dict:
    """조립기가 호출할 단일 진입점. 게이트 판정 + (승인된 경우) row_dict의
    해당 특성을 실제로 0.0으로 채우기까지 한 번에 한다(게이트 자체는
    부작용 없는 순수함수라 이 함수가 그 다음 단계를 대신 해준다).
    반환값에 "row"(채워진 뒤의 값)를 포함해 호출부가 그대로 predict_kw에
    넘길 수 있게 한다."""
    claims = iqg.build_night_zero_fill_claims(row_dict, required_features, lag_specs, elevation_at)
    result = iqg.evaluate(row_dict, required_features, cfg, zero_fill_claims=claims)
    filled = dict(row_dict)
    for feat in result["explained_features"]:
        v = filled.get(feat)
        if v is None or (isinstance(v, float) and v != v):
            filled[feat] = 0.0
    result["row"] = filled
    result["night_zero_filled_features"] = [
        s.feature for s in lag_specs
        if s.feature in result["explained_features"]
    ]
    return result
