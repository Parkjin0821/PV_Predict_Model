# -*- coding: utf-8 -*-
"""live_gate_wiring_v1_2026-08-28.py 오프라인 테스트.

핵심: lag_specs의 시각 오프셋이 원본 학습코드(backtest_harness_v1_2026-
08-20밤.py, train_ultra_short_official_v1_2026-08-21.py)의 shift/rolling
정의와 정확히 일치하는지 - 여기서 어긋나면 게이트가 틀린 시각의 태양고도로
"야간"을 잘못 승인하게 된다. 합성 데이터만 사용.
"""
from __future__ import annotations

import importlib.util as ilu
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("lgw", PARENT / "live_gate_wiring_v1_2026-08-28.py")
lgw = ilu.module_from_spec(spec)
sys.modules["lgw"] = lgw
spec.loader.exec_module(lgw)
iqg = lgw.iqg

CFG = iqg.load_config()


def test_1_hourly_point_lag_offsets_match_source():
    """backtest_harness의 power=df.shift(1); power.shift(h-1) 조합은
    발전출력_h시간전_kW.iloc[i] = df.iloc[i-h]와 동치 - 즉 원천시각은
    issue_time - h시간이어야 한다."""
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.hourly_lag_specs(issue)
    by_name = {s.feature: s.source_times for s in specs}
    assert by_name["발전출력_1시간전_kW"] == [issue - pd.Timedelta(hours=1)]
    assert by_name["발전출력_24시간전_kW"] == [issue - pd.Timedelta(hours=24)]
    assert by_name["발전출력_3시간전_kW"] == [issue - pd.Timedelta(hours=3)]
    return True


def test_2_hourly_rolling_window_matches_source():
    """power.rolling(h,...) at row i covers df.iloc[i-h:i] (h개, i-1 포함
    i-h까지) - 원천시각은 issue-1시간 ~ issue-h시간, h개."""
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.hourly_lag_specs(issue)
    by_name = {s.feature: s.source_times for s in specs}
    window6 = by_name["발전출력_6시간이동평균_kW"]
    assert len(window6) == 6, window6
    assert window6[0] == issue - pd.Timedelta(hours=1)
    assert window6[-1] == issue - pd.Timedelta(hours=6)
    window24_std = by_name["발전출력_24시간이동표준편차_kW"]
    assert len(window24_std) == 24, len(window24_std)
    return True


def test_3_quarter_point_lag_offsets_match_source():
    """build_ultra_short_frame: power=quarter.shift(1); power.shift(steps-1)
    → 발전출력_{steps*15}분전_kW.iloc[i] = quarter.iloc[i-steps] - 원천시각
    = issue - steps*15분."""
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.quarter_lag_specs(issue)
    by_name = {s.feature: s.source_times for s in specs}
    assert by_name["발전출력_15분전_kW"] == [issue - pd.Timedelta(minutes=15)]
    assert by_name["발전출력_240분전_kW"] == [issue - pd.Timedelta(minutes=240)]
    assert by_name["발전출력_60분전_kW"] == [issue - pd.Timedelta(minutes=60)]
    return True


def test_4_quarter_rolling_window_matches_source():
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.quarter_lag_specs(issue)
    by_name = {s.feature: s.source_times for s in specs}
    window_1h = by_name["발전출력_60분이동평균_kW"]  # steps=4
    assert len(window_1h) == 4, window_1h
    assert window_1h[0] == issue - pd.Timedelta(minutes=15)
    assert window_1h[-1] == issue - pd.Timedelta(minutes=60)
    window_4h_std = by_name["발전출력_240분이동표준편차_kW"]  # steps=16
    assert len(window_4h_std) == 16, len(window_4h_std)
    return True


def test_5_apply_gate_fills_confirmed_night_feature():
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.hourly_lag_specs(issue)
    elev = {issue - pd.Timedelta(hours=1): -8.0}
    row = {"발전출력_1시간전_kW": math.nan}
    result = lgw.apply_gate(row, ["발전출력_1시간전_kW"], CFG, specs, elev.get)
    assert result["status"] == iqg.opstatus.DEGRADED, result
    assert result["row"]["발전출력_1시간전_kW"] == 0.0, result["row"]
    assert "발전출력_1시간전_kW" in result["night_zero_filled_features"]
    return True


def test_6_apply_gate_leaves_unresolved_missing_blocked():
    """야간 확인이 안 되는 다른 결측 특성이 섞여 있으면 blocked, 그
    특성의 값은 채워지지 않는다(NaN 그대로)."""
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.hourly_lag_specs(issue)
    elev = {issue - pd.Timedelta(hours=1): -8.0}
    row = {"발전출력_1시간전_kW": math.nan, "DSWRF": math.nan}
    result = lgw.apply_gate(row, ["발전출력_1시간전_kW", "DSWRF"], CFG, specs, elev.get)
    assert result["status"] == iqg.opstatus.BLOCKED, result
    assert result["row"]["발전출력_1시간전_kW"] == 0.0, result["row"]  # 이건 채워짐
    dswrf = result["row"]["DSWRF"]
    assert dswrf is None or (isinstance(dswrf, float) and dswrf != dswrf), result["row"]
    return True


def test_7_apply_gate_normal_row_unaffected():
    """이미 값이 다 있는 정상 행은 게이트를 거쳐도 값이 하나도 안 바뀌어야
    한다 - 회귀검사의 핵심 전제."""
    issue = pd.Timestamp("2026-08-28 21:00:00")
    specs = lgw.hourly_lag_specs(issue)
    row = {"발전출력_1시간전_kW": 3.4567, "DSWRF": 120.5}
    result = lgw.apply_gate(row, ["발전출력_1시간전_kW", "DSWRF"], CFG, specs, lambda t: None)
    assert result["status"] == iqg.opstatus.NORMAL, result
    assert result["row"] == row, result["row"]
    return True


TESTS = [
    test_1_hourly_point_lag_offsets_match_source,
    test_2_hourly_rolling_window_matches_source,
    test_3_quarter_point_lag_offsets_match_source,
    test_4_quarter_rolling_window_matches_source,
    test_5_apply_gate_fills_confirmed_night_feature,
    test_6_apply_gate_leaves_unresolved_missing_blocked,
    test_7_apply_gate_normal_row_unaffected,
]


def main() -> int:
    results = []
    for t in TESTS:
        try:
            ok = t()
            results.append((t.__name__, "PASS" if ok else "FAIL", ""))
        except AssertionError as e:
            results.append((t.__name__, "FAIL", str(e)))
        except Exception as e:
            results.append((t.__name__, "ERROR", "%s: %s" % (type(e).__name__, e)))
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    for name, status, msg in results:
        print("[%s] %s %s" % (status, name, ("- " + msg) if msg else ""))
    print("\n%d/%d PASS" % (n_pass, len(results)))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
