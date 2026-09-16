"""부안 정적 Excel 이력과 Blockdata 라이브 이력을 연결해 재학습용 집계를 만든다.

API 호출 없이 기존 bridge 검증 규칙과 부안 시간집계 함수를 재사용한다.
정적 이력 이후의 라이브 구간만 추가하며 경계 중복·부분 인버터 합산·교차
경계 보간을 허용하지 않는다. 결과는 기존 공식 산출물과 별도 폴더에 저장한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
BRIDGE = ROOT / r"02_전처리\부안\bridge_buan_excel_blockdata_live_v1_2026-08-28.py"
AGG = ROOT / r"02_전처리\부안\build_buan_time_aggregates_v1_2026-08-28.py"
DEFAULT_HISTORY = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간정렬_v1_2026-08-28\부안_인버터별_5분정렬.parquet")
DEFAULT_LIVE_DB = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3")
DEFAULT_OUTPUT = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_라이브연계_v1_2026-09-14")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈 로드 실패: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="부안 Excel-Blockdata 라이브 연계 집계")
    p.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    p.add_argument("--live-db", type=Path, default=DEFAULT_LIVE_DB)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bridge = load_module("buan_excel_live_bridge", BRIDGE)
    agg = load_module("buan_time_aggregate", AGG)

    history = pd.read_parquet(args.history)
    history["grid_time_kst"] = pd.to_datetime(history["grid_time_kst"], errors="raise")
    history["data_source"] = "excel_history"
    live_raw, plant_snapshots = bridge.load_live(args.live_db)
    live, snapshot_audit = bridge.validate_and_align_live(live_raw)
    # bridge 모듈은 출처를 구분해 `observed_blockdata_response_snapshot`으로
    # 표시하지만, 시간집계기의 공식 유효목록은 `observed`다. 완전 8대·정격
    # 검증을 통과한 라이브 응답만 들어오므로 집계 단계에서 observed로 정규화한다.
    live["quality_status"] = "observed"

    common = sorted(set(history.columns) | set(live.columns))
    history = history.reindex(columns=common)
    live = live.reindex(columns=common)
    overlap = history.merge(live, on=["grid_time_kst", "inverter_number"], how="inner")
    if not overlap.empty:
        raise RuntimeError(f"Excel/API 경계 중복 {len(overlap)}행")

    combined = pd.concat([history, live], ignore_index=True).sort_values(
        ["grid_time_kst", "inverter_number"]
    )
    if combined.duplicated(["grid_time_kst", "inverter_number"]).any():
        raise RuntimeError("결합 후 시각·인버터 중복")
    inv5 = agg.apply_night_zero(combined)
    plant5 = agg.build_plant_five(inv5)
    hourly = agg.build_hourly(plant5)
    daily = agg.build_daily(plant5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output_dir / "부안_인버터별_5분_Excel_API연결.parquet", index=False)
    inv5.to_parquet(args.output_dir / "부안_인버터별_5분_야간0포함_라이브연계.parquet", index=False)
    plant5.to_parquet(args.output_dir / "부안_발전소_5분_라이브연계.parquet", index=False)
    hourly.to_parquet(args.output_dir / "부안_발전소_1시간_라이브연계.parquet", index=False)
    daily.to_csv(args.output_dir / "부안_발전소_일간_라이브연계.csv", index=False, encoding="utf-8-sig")
    snapshot_audit.to_csv(args.output_dir / "부안_API응답단위_감사.csv", index=False, encoding="utf-8-sig")

    history_end = history["grid_time_kst"].max()
    live_start = live["grid_time_kst"].min()
    gap_minutes = max(0.0, float((live_start - history_end).total_seconds() / 60.0 - 5.0))
    summary = {
        "status": "live_bridge_candidate",
        "source": {"history": str(args.history), "live_db": str(args.live_db)},
        "history_end_kst": str(history_end),
        "live_start_kst": str(live_start),
        "unfilled_boundary_gap_minutes": gap_minutes,
        "history_rows": int(len(history)),
        "accepted_live_rows": int(len(live)),
        "accepted_live_snapshots": int(live["grid_time_kst"].nunique()),
        "rejected_live_responses": int((~snapshot_audit["accepted"]).sum()),
        "boundary_overlap_rows": int(len(overlap)),
        "plant_snapshot_rows_in_db": int(len(plant_snapshots)),
        "combined_rows": int(len(combined)),
        "hourly_rows": int(len(hourly)),
        "hourly_valid_rows": int(hourly["plant_ac_power_kw"].notna().sum()),
        "daily_rows": int(len(daily)),
        "daily_valid_rows": int(daily["daily_energy_kwh"].notna().sum()),
        "rules": {
            "api_calls": 0,
            "cross_boundary_interpolation": False,
            "partial_scaling": False,
            "daily_gate": "physical_daylight_valid_ratio>=0.90",
            "official_model_or_scheduler_changed": False,
        },
    }
    (args.output_dir / "부안_라이브연계_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
