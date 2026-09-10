"""집계 quality_status pass-through 무API 회귀검사."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PREPROCESS = HERE.parent
TARGETS = (
    (PREPROCESS / "부안" / "build_buan_time_aggregates_v1_2026-08-28.py", 8),
    (PREPROCESS / "김제" / "build_gimje_time_aggregates_v1_2026-08-31.py", 10),
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈 로드 실패: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_aligned(n_inv: int) -> pd.DataFrame:
    times = pd.date_range("2026-06-01 12:00", periods=12, freq="5min", tz="Asia/Seoul")
    rows = []
    for slot, ts in enumerate(times):
        for inv in range(1, n_inv + 1):
            status = "observed"
            if slot == 1 and inv == 1:
                status = "interpolated_le10min"
            elif slot == 2 and inv == 1:
                status = "physical_zero_idle_supported"
            elif slot == 3 and inv == 1:
                # 값이 남아 있더라도 이 상태는 공식 합계에서 차단돼야 한다.
                status = "communication_outage_missing"
            rows.append(
                {
                    "grid_time_kst": ts,
                    "inverter_number": inv,
                    "ac_power_kw": 10.0,
                    "dc_power_kw": 11.0,
                    "quality_status_after_night": status,
                }
            )
    return pd.DataFrame(rows)


def run_one(path: Path, n_inv: int) -> None:
    module = load_module(path)
    plant = module.build_plant_five(synthetic_aligned(n_inv))

    assert plant.loc[0, "quality_status"] == "complete_observed"
    assert plant.loc[1, "quality_status"] == "complete_with_short_interpolation"
    assert plant.loc[2, "quality_status"] == "complete_with_idle_zero"
    assert plant.loc[3, "quality_status"] == "incomplete_no_official_target"
    assert pd.isna(plant.loc[3, "plant_ac_power_kw"])
    assert plant.loc[3, "invalid_quality_inverter_count"] == 1
    assert plant.loc[3, "inverter_1_quality_status"] == "communication_outage_missing"
    assert np.isclose(plant.loc[0, "plant_ac_power_kw"], 10.0 * n_inv)

    hourly = module.build_hourly(plant)
    assert hourly.loc[0, "source_complete_observed_slots"] == 9
    assert hourly.loc[0, "source_complete_with_short_interpolation_slots"] == 1
    assert hourly.loc[0, "source_complete_with_idle_zero_slots"] == 1
    assert hourly.loc[0, "source_incomplete_no_official_target_slots"] == 1


def main() -> None:
    for path, n_inv in TARGETS:
        run_one(path, n_inv)
        print(f"[PASS] {path.name}: {n_inv}대 상태 보존·비유효값 차단·시간 집계")
    print("[PASS] API 호출 0건")


if __name__ == "__main__":
    main()
