# -*- coding: utf-8 -*-
"""v7(v6 + 라이브 Blockdata 실측 이어붙임) 데이터셋을 안전하게 불러오는 래퍼.

## v6 로더와의 관계 — 재구현 아님, 폴더만 교체
`daily_dataset_v6_구멍보정_로더_2026-08-26.py`와 완전히 동일한 로직이다.
바뀐 건 `e2e.V5_DIR`이 가리키는 폴더가 v6에서 v7로 바뀐 것뿐이다. v7은
`build_daily_from_blockdata_live_v1_2026-08-26.py`가 매 실행마다 v6를
기준으로 다시 만드는 산출물이라, 라이브 Blockdata에 새 완결일이 쌓일
때마다 그 스크립트를 재실행하면 이 로더가 자동으로 최신 v7을 읽는다
(이 파일 자체는 그대로 — 코드 수정 불필요).

## 추정치_여부 처리
v7의 `추정치_여부` 컬럼은 v6에서 그대로 이어받는다 — 08-05~24 20일은
여전히 1(카운터뺄셈+ASOS배분 추정), 그 외(원본 v5 + 라이브 실측 신규일)는
0이다. v6 로더와 동일하게 **추정 20일만 학습 타겟에서 제외**하고
lag/rolling 입력으로는 그대로 쓴다. 라이브 실측 신규일(0)은 애초에
추정이 아니므로 별도 제외 없이 타겟으로도 정상 사용된다.

## 사용법(v6 로더와 동일 관례)
```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "v7loader", "daily_dataset_v7_라이브연계_로더_2026-08-26.py")
mod = importlib.util.module_from_spec(spec); sys.modules["v7loader"] = mod
spec.loader.exec_module(mod)
data, features, n_partial, n_estimated_excluded = mod.load_v7_dataset(capacity_kw)
```
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
V7_DIR = ROOT / "outputs" / "v7_라이브연계_2026-08-26"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_mod = _load("v7loader_daily", "daily_direct_final_audit_v1_2026-08-25.py")
e2e = daily_mod.e2e


def estimated_dates() -> pd.DatetimeIndex:
    raw = pd.read_parquet(V7_DIR / "집계_일간_실제발전량_v5.parquet")
    raw.index = pd.to_datetime(raw.index)
    if "추정치_여부" not in raw.columns:
        raise RuntimeError("v7 parquet에 '추정치_여부' 컬럼이 없다 — "
                          "build_daily_from_blockdata_live_v1_2026-08-26.py로 다시 만들 것.")
    return raw.index[raw["추정치_여부"] == 1]


def latest_real_date() -> pd.Timestamp:
    """감사·readiness 게이트용 — v7에서 실제(추정 아님) 값이 있는 가장 최근
    날짜. v6 마지막(2026-08-24)보다 나중이면 라이브 실측이 실제로 반영된
    것이다."""
    raw = pd.read_parquet(V7_DIR / "집계_일간_실제발전량_v5.parquet")
    raw.index = pd.to_datetime(raw.index)
    real = raw[(raw["추정치_여부"] == 0) & raw["일간발전량_kWh"].notna()]
    if real.empty:
        raise RuntimeError("v7에 유효한 실측일이 하나도 없다.")
    return real.index.max()


def load_v7_dataset(capacity_kw: float):
    """corrected_dataset()을 v7 기준으로 호출하되, 추정 20일을 **타겟
    후보에서만** 제외한다. lag/rolling 특성 계산은 원본 그대로(재구현 아님)."""
    est_dates = estimated_dates()

    old_dir = e2e.V5_DIR
    e2e.V5_DIR = V7_DIR
    try:
        data, features, n_partial = daily_mod.corrected_dataset(capacity_kw)
    finally:
        e2e.V5_DIR = old_dir

    raw_est = pd.read_parquet(V7_DIR / "집계_일간_실제발전량_v5.parquet")
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
    data, features, n_partial, n_excluded = load_v7_dataset(219.0)
    print(f"타겟행수={len(data):,} (추정일 {n_excluded}일 타겟에서 제외됨) "
          f"부분가용역사일={n_partial} 특성수={len(features)}")
    print(f"학습기간: {data.index.min().date()} ~ {data.index.max().date()}")
    print(f"v7 최신 실측일: {latest_real_date().date()}")
