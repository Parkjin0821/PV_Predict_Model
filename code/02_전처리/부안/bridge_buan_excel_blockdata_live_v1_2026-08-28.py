"""부안 과거 Excel 5분정렬 자료와 저장된 Blockdata 라이브 DB를 연결한다.

API 호출은 하지 않는다. 라이브 API 한 응답에 포함된 8대 인버터는 개별
measurement_time이 달라도 동일 response_hash의 source_latest_measurement_at을
공통 snapshot_time으로 사용한다. 이 규칙은 한 응답의 8대가 서로 다른 5분
슬롯으로 갈리는 문제를 방지한다.

공백은 소급 보간하지 않으며, 8대 완전수신 스냅샷만 공식 발전소 총출력으로
채택한다. Excel/API 경계의 중복, KST, 인버터 번호·정격, 단위를 검사한다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PLANT_ID = 16783
EXPECTED_INVERTERS = tuple(range(1, 9))
EXPECTED_INVERTER_CAPACITY_KW = 125.0
EXPECTED_PLANT_CAPACITY_KW = 1000.0

DEFAULT_HISTORY = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\시간정렬_v1_2026-08-28\부안_인버터별_5분정렬.parquet"
)
DEFAULT_LIVE_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\Excel_API_연결_v1_2026-08-28"
)

LIVE_MAP = {
    "dc_volt": "dc_voltage_v", "dc_current": "dc_current_a",
    "dc_power": "dc_power_kw", "ac_volt_r": "ac_voltage_r_v",
    "ac_volt_s": "ac_voltage_s_v", "ac_volt_t": "ac_voltage_t_v",
    "ac_current_r": "ac_current_r_a", "ac_current_s": "ac_current_s_a",
    "ac_current_t": "ac_current_t_a", "ac_power": "ac_power_kw",
    "freq": "frequency_hz", "pf": "power_factor_pct",
    "daily_energy": "daily_energy_kwh", "total_energy": "total_energy_kwh",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="부안 Excel-Blockdata 라이브 이력 연결")
    p.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    p.add_argument("--live-db", type=Path, default=DEFAULT_LIVE_DB)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def parse_kst(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise RuntimeError(f"{name} 시각 파싱 실패: {int(parsed.isna().sum())}행")
    return parsed.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)


def load_live(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        inv = pd.read_sql_query(
            """
            SELECT i.*, r.source_latest_measurement_at AS response_snapshot_time
            FROM inverter_measurements i
            JOIN raw_snapshots r USING(response_hash)
            WHERE i.plant_id=?
            """, conn, params=(PLANT_ID,),
        )
        plant = pd.read_sql_query(
            "SELECT * FROM plant_snapshots WHERE plant_id=?", conn, params=(PLANT_ID,)
        )
    finally:
        conn.close()
    if inv.empty:
        raise RuntimeError("부안 라이브 DB에 인버터 측정행이 없음")
    return inv, plant


def validate_and_align_live(inv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv["measurement_time_kst"] = parse_kst(inv["measurement_time"], "measurement_time")
    inv["response_snapshot_time_kst"] = parse_kst(
        inv["response_snapshot_time"], "response_snapshot_time"
    )
    inv["grid_time_kst"] = inv["response_snapshot_time_kst"].dt.round("5min")
    inv["inverter_number"] = pd.to_numeric(inv["inverter_number"], errors="raise").astype(int)
    inv["capacity"] = pd.to_numeric(inv["capacity"], errors="coerce")

    bad_numbers = sorted(set(inv["inverter_number"]) - set(EXPECTED_INVERTERS))
    if bad_numbers:
        raise RuntimeError(f"예상 밖 인버터 번호: {bad_numbers}")

    snapshot_audit = []
    accepted = []
    for response_hash, g in inv.groupby("response_hash", sort=True):
        numbers = sorted(g["inverter_number"].unique().tolist())
        duplicate_numbers = int(g.duplicated("inverter_number", keep=False).sum())
        capacities = pd.to_numeric(g["capacity"], errors="coerce")
        capacity_sum = float(capacities.sum(min_count=1))
        complete = numbers == list(EXPECTED_INVERTERS) and duplicate_numbers == 0
        capacity_ok = bool(
            capacities.notna().all()
            and np.allclose(capacities.to_numpy(), EXPECTED_INVERTER_CAPACITY_KW, atol=1e-6)
            and abs(capacity_sum - EXPECTED_PLANT_CAPACITY_KW) <= 1e-6
        )
        spread = float(
            (g["measurement_time_kst"].max() - g["measurement_time_kst"].min()).total_seconds()
        )
        snapshot_audit.append({
            "response_hash": response_hash,
            "snapshot_time_kst": str(g["response_snapshot_time_kst"].iloc[0]),
            "grid_time_kst": str(g["grid_time_kst"].iloc[0]),
            "row_count": int(len(g)), "inverter_numbers": numbers,
            "duplicate_inverter_rows": duplicate_numbers,
            "capacity_sum_kw": capacity_sum,
            "measurement_spread_seconds": spread,
            "complete_8": complete, "capacity_ok": capacity_ok,
            "accepted": bool(complete and capacity_ok),
        })
        if complete and capacity_ok:
            accepted.append(g)
    audit = pd.DataFrame(snapshot_audit)
    if not accepted:
        raise RuntimeError("8대·정격 검증을 통과한 라이브 응답이 없음")
    live = pd.concat(accepted, ignore_index=True)

    # 같은 5분격자에 여러 API 응답이 있으면 최신 응답 전체를 선택한다.
    chosen_hash = (
        live[["grid_time_kst", "response_hash", "response_snapshot_time_kst"]]
        .drop_duplicates()
        .sort_values(["grid_time_kst", "response_snapshot_time_kst"])
        .drop_duplicates("grid_time_kst", keep="last")
    )
    live = live.merge(chosen_hash[["grid_time_kst", "response_hash"]],
                      on=["grid_time_kst", "response_hash"], how="inner")
    if live.duplicated(["grid_time_kst", "inverter_number"]).any():
        raise RuntimeError("라이브 정렬 후 시각·인버터 중복 잔존")

    for old, new in LIVE_MAP.items():
        live[new] = pd.to_numeric(live[old], errors="coerce")
    live["plant_id"] = PLANT_ID
    live["inverter_capacity_kw"] = live["capacity"]
    live["was_observed"] = True
    live["was_interpolated"] = False
    live["interpolated_feature_count"] = 0
    live["alignment_offset_sec"] = (
        live["measurement_time_kst"] - live["grid_time_kst"]
    ).dt.total_seconds()
    live["alignment_abs_sec"] = live["alignment_offset_sec"].abs()
    live["long_or_edge_gap"] = False
    live["quality_status"] = "observed_blockdata_response_snapshot"
    live["source_file"] = "blockdata_history.sqlite3"
    live["source_excel_row"] = pd.NA
    live["data_source"] = "blockdata_live"
    return live, audit


def build_plant(x: pd.DataFrame) -> pd.DataFrame:
    ac = x.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    dc = x.pivot(index="grid_time_kst", columns="inverter_number", values="dc_power_kw")
    out = pd.DataFrame(index=ac.index)
    out["plant_id"] = PLANT_ID
    out["available_inverter_count"] = ac.notna().sum(axis=1).astype("int8")
    out["complete_8_inverters"] = out["available_inverter_count"].eq(8)
    out["plant_ac_power_kw"] = ac.sum(axis=1, min_count=8)
    out["plant_dc_power_kw"] = dc.sum(axis=1, min_count=8)
    out["partial_ac_sum_reference_kw"] = ac.sum(axis=1, min_count=1)
    out["data_source"] = x.groupby("grid_time_kst")["data_source"].first()
    return out.reset_index()


def main() -> None:
    args = parse_args()
    history = pd.read_parquet(args.history)
    history["grid_time_kst"] = pd.to_datetime(history["grid_time_kst"], errors="raise")
    history["data_source"] = "excel_history"
    live_raw, plant_snapshots = load_live(args.live_db)
    live, snapshot_audit = validate_and_align_live(live_raw)

    common = sorted(set(history.columns) | set(live.columns))
    history = history.reindex(columns=common)
    live = live.reindex(columns=common)
    overlap = history.merge(live, on=["grid_time_kst", "inverter_number"], how="inner")
    if not overlap.empty:
        raise RuntimeError(f"Excel/API 경계 중복 {len(overlap)}행 — 자동 덮어쓰기 금지")
    combined = pd.concat([history, live], ignore_index=True).sort_values(
        ["grid_time_kst", "inverter_number"]
    )
    if combined.duplicated(["grid_time_kst", "inverter_number"]).any():
        raise RuntimeError("결합 후 시각·인버터 중복")

    hist_end = history["grid_time_kst"].max()
    live_start = live["grid_time_kst"].min()
    gap_minutes = float((live_start - hist_end).total_seconds() / 60.0 - 5.0)
    plant = build_plant(combined)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output_dir / "부안_인버터별_5분_Excel_API연결.parquet", index=False)
    plant.to_parquet(args.output_dir / "부안_발전소_5분_Excel_API연결.parquet", index=False)
    snapshot_audit.to_csv(args.output_dir / "부안_API응답단위_감사.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "connected_with_explicit_gap" if gap_minutes > 0 else "connected_continuously",
        "rules": {
            "timezone": "Asia/Seoul",
            "live_grid_attribution": "one_response_hash_one_common_snapshot_then_nearest_5min",
            "required_inverters": list(EXPECTED_INVERTERS),
            "required_capacity_each_kw": EXPECTED_INVERTER_CAPACITY_KW,
            "partial_scaling": "forbidden", "cross_boundary_interpolation": "forbidden",
        },
        "history_end_kst": str(hist_end), "live_start_kst": str(live_start),
        "unfilled_boundary_gap_minutes": max(0.0, gap_minutes),
        "history_rows": int(len(history)), "accepted_live_rows": int(len(live)),
        "accepted_live_snapshots": int(live["grid_time_kst"].nunique()),
        "rejected_live_responses": int((~snapshot_audit["accepted"]).sum()),
        "boundary_overlap_rows": int(len(overlap)),
        "plant_snapshot_rows_in_db": int(len(plant_snapshots)),
        "unit_checks": {
            "capacity_sum_kw": EXPECTED_PLANT_CAPACITY_KW,
            "ac_power_over_150kw_per_inverter_rows": int((pd.to_numeric(live["ac_power_kw"], errors="coerce") > 150).sum()),
            "negative_ac_power_rows": int((pd.to_numeric(live["ac_power_kw"], errors="coerce") < 0).sum()),
        },
    }
    (args.output_dir / "부안_Excel_API_경계검증_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
