# -*- coding: utf-8 -*-
"""v6(08-05~24 구멍보정) 데이터셋을 안전하게 불러오는 래퍼.

## 왜 필요한가(08-26 실제 실행 중 발견한 결함)
`daily_direct_final_audit_v1_2026-08-25.py::corrected_dataset()`을
`e2e.V5_DIR`만 v6 폴더로 몽키패치해서 그대로 호출해봤더니, 추정 20일이
**lag/rolling 입력**(원하는 동작)뿐 아니라 **학습 타겟(`실제_
일간발전량_kWh`)**으로도 그대로 들어가는 걸 확인했다.
`e2e.build_daily_dataset_v5()`가 `data[ACTUAL] = daily_actual["일간
발전량_kWh"]`를 raw 파일의 모든 행에서 무조건 가져오고, `부분가용
여부<1`로만 걸러내기 때문이다 — 우리가 lag용으로 쓰이게 하려고
추정 20일의 `부분가용일`을 일부러 0으로 넣었는데(안 그러면 lag에서도
빠짐), 그 결과 같은 조건으로 타겟에도 같이 들어가버렸다.

**추정치를 정답(학습 타겟)으로 쓰는 건 안 된다** — 모델이 우리가 만든
추정값을 재현하도록 학습되는 꼴이라, 애초에 "임의로 채우지 않는다"
원칙을 정면으로 어긴다. 그래서 이 래퍼가 그 20일을 **타겟에서만**
사후 제외한다(lag/rolling 계산은 `corrected_dataset()` 안에서 이미
끝난 뒤이므로, 그 결과에 영향 없이 행만 뺄 수 있다).

## 재구현 없음
`corrected_dataset()` 자체는 전혀 안 고친다 — 그대로 호출하고, 반환된
`data`에서 추정일 20행만 뒤늦게 제거한다. lag/rolling 특성값은 이미
계산 완료된 상태라 그대로 유지된다(추정 20일은 "다른 날짜의 30일
평균을 계산하는 재료"로는 계속 쓰이고, "그 자신이 학습·평가 대상"
으로는 안 쓰임 — 원하는 동작 그대로).

## 사용법(이 프로젝트 전역 관례와 동일 — importlib로 로드, 파일명에
## 하이픈이 있어 일반 import 문으로는 못 불러온다)
```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "v6loader", "daily_dataset_v6_구멍보정_로더_2026-08-26.py")
mod = importlib.util.module_from_spec(spec); sys.modules["v6loader"] = mod
spec.loader.exec_module(mod)
data, features, n_partial, n_estimated_excluded = mod.load_v6_dataset(capacity_kw)
# data에는 이미 추정 20일이 타겟에서 빠져있음 — 곧바로 train/predict_oof에 사용 가능
```
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
V6_DIR = ROOT / "outputs" / "v6_일간구멍보정_2026-08-26"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_mod = _load("v6loader_daily", "daily_direct_final_audit_v1_2026-08-25.py")
e2e = daily_mod.e2e


def estimated_dates() -> pd.DatetimeIndex:
    raw = pd.read_parquet(V6_DIR / "집계_일간_실제발전량_v5.parquet")
    raw.index = pd.to_datetime(raw.index)
    if "추정치_여부" not in raw.columns:
        raise RuntimeError("v6 parquet에 '추정치_여부' 컬럼이 없다 — "
                          "build_daily_gap_estimate_v1_2026-08-26.py로 다시 만들 것.")
    return raw.index[raw["추정치_여부"] == 1]


def load_v6_dataset(capacity_kw: float):
    """corrected_dataset()을 v6 기준으로 호출하되, 추정 20일을 **타겟
    후보에서만** 제외한다. lag/rolling 특성 계산은 원본 그대로(재구현 아님)."""
    est_dates = estimated_dates()

    old_dir = e2e.V5_DIR
    e2e.V5_DIR = V6_DIR
    try:
        data, features, n_partial = daily_mod.corrected_dataset(capacity_kw)
    finally:
        e2e.V5_DIR = old_dir

    # 추정 20일이 lag/rolling 창 안에 얼마나 섞여있는지 감사용 컬럼(정보용,
    # 모델 입력 특성으로 강제하지 않음 — 필요하면 호출부에서 features에
    # 추가해서 쓸 수 있게 남겨둔다).
    est_flag_series = pd.Series(0, index=data.index)
    est_flag_series.loc[data.index.isin(est_dates)] = 1
    # 원본 raw 인덱스 기준 rolling 계산과 동일한 정렬로 "이 목표일의 30일
    # 창에 추정치가 며칠 들어있는지"를 재현한다(shift(2)+rolling(30) 동일 규약).
    raw_est = pd.read_parquet(V6_DIR / "집계_일간_실제발전량_v5.parquet")
    raw_est.index = pd.to_datetime(raw_est.index)
    est_col = raw_est["추정치_여부"].reindex(
        pd.date_range(raw_est.index.min(), raw_est.index.max(), freq="D")).fillna(0)
    window_est_count = est_col.shift(2).rolling(30, min_periods=1).sum().reindex(data.index)
    data["2일전기준_30일창_추정일수"] = window_est_count.fillna(0)

    before = len(data)
    data_clean = data[~data.index.isin(est_dates)].copy()
    n_excluded = before - len(data_clean)

    return data_clean, features, n_partial, n_excluded


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data, features, n_partial, n_excluded = load_v6_dataset(219.0)
    print(f"타겟행수={len(data):,} (추정일 {n_excluded}일 타겟에서 제외됨) "
          f"부분가용역사일={n_partial} 특성수={len(features)}")
    print(f"학습기간: {data.index.min().date()} ~ {data.index.max().date()}")
    tail = data[["2일전기준_30일이동평균_kWh", "2일전기준_30일표준편차_kWh",
                "2일전기준_30일창_추정일수"]].tail(8)
    print(tail.to_string())
