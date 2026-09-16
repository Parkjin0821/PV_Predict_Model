from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER_PATH = ROOT / "live_feature_assembler_ultrashort_v1_2026-09-08.py"


def load_assembler():
    spec = importlib.util.spec_from_file_location("ultrashort_assembler_lag_tolerance_test", ASSEMBLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


assembler = load_assembler()


def test_gwangju_accepts_one_five_minute_source_slot() -> None:
    target = pd.Timestamp("2026-09-16 13:14:40")
    series = pd.Series([123.0], index=[pd.Timestamp("2026-09-16 13:19:39")])

    assert np.isnan(assembler.lookup_near(series, target, tol_minutes=4.0))
    assert assembler.lookup_near(series, target, tol_minutes=5.0) == 123.0


def test_gap_over_five_minutes_remains_missing() -> None:
    target = pd.Timestamp("2026-09-16 13:14:40")
    series = pd.Series([123.0], index=[pd.Timestamp("2026-09-16 13:19:47")])

    assert np.isnan(assembler.lookup_near(series, target, tol_minutes=5.0))


def test_tolerance_change_is_scoped_to_gwangju() -> None:
    assert assembler.REGIONS["광주"]["power_lag_tolerance_minutes"] == 5.0
    for region in ("부안", "김제", "영광"):
        assert assembler.REGIONS[region].get("power_lag_tolerance_minutes", 4.0) == 4.0


def test_gwangju_capacity_roles_are_separated() -> None:
    assert assembler.REGIONS["광주"]["capacity"] == 240.0
    assert assembler.REGIONS["광주"]["clip_capacity"] == 241.58


if __name__ == "__main__":
    test_gwangju_accepts_one_five_minute_source_slot()
    test_gap_over_five_minutes_remains_missing()
    test_tolerance_change_is_scoped_to_gwangju()
    test_gwangju_capacity_roles_are_separated()
    print("PASS: Gwangju power-lag tolerance regression tests")
