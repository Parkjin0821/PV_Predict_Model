# -*- coding: utf-8 -*-
"""Build an isolated Gwangju v6 peer-quality candidate from the v5 recipe.

The v5 partial-sum recovery policy is preserved.  Only the 150 audited 5-minute
timestamps (145 peer-audit rejects plus five pending non-zero anomalies) are
invalidated before the unchanged 15-minute/hourly/daily aggregation rules run.
No API is called and the official v5 directory is never written.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
PIPE_DIR = PROJECT / "03_모델학습" / "현재_종합파이프라인"
SOURCE_SCRIPT = PROJECT / "02_전처리" / "rebuild_plant_v5_recovered_2026-08-21.py"
AUDIT_DIR = Path(r"C:\Users\u-cube\Documents\ChatGPT\유큐브")
CHANGED_CSV = AUDIT_DIR / "gwangju_peer_quality_changed_rows_20260901.csv"
PENDING_CSV = AUDIT_DIR / "gwangju_peer_quality_flagged_nonzero_rows_20260901.csv"
OUT = PIPE_DIR / "outputs" / "v6_동료대조_후보_2026-09-01"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_rejections() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    changed = pd.read_csv(CHANGED_CSV, parse_dates=["grid_time_kst"])
    pending = pd.read_csv(PENDING_CSV, parse_dates=["grid_time_kst"])
    changed["v6_exclusion_reason"] = changed["quality_status_after_peer"]
    pending["v6_exclusion_reason"] = "equipment_anomaly_candidate"
    log = pd.concat([changed, pending], ignore_index=True, sort=False)
    if len(changed) != 145 or changed["grid_time_kst"].nunique() != 145:
        raise RuntimeError("peer-audit exclusion set is not exactly 145 unique timestamps")
    if len(pending) != 5 or pending["grid_time_kst"].nunique() != 5:
        raise RuntimeError("pending-review set is not exactly five unique timestamps")
    if log["grid_time_kst"].nunique() != 150:
        raise RuntimeError("combined v6 exclusion set is not exactly 150 unique timestamps")
    return log.sort_values(["grid_time_kst", "inverter_number"]), pd.DatetimeIndex(log["grid_time_kst"])


def main() -> None:
    source = load_module("gwangju_v5_source", SOURCE_SCRIPT)
    pv = source.pv_pipeline
    config = pv.load_config()
    capacity = float(config["site"]["capacity_kw"])
    log, rejected_times = load_rejections()
    OUT.mkdir(parents=True, exist_ok=True)

    probe = source.load_inverter_raw(1)
    for n in range(2, source.N_INVERTERS + 1):
        probe = probe.combine_first(source.load_inverter_raw(n))
    grid = pd.date_range(probe.index.min().floor("5min"), probe.index.max().ceil("5min"), freq="5min")
    missing_from_grid = rejected_times.difference(grid)
    if len(missing_from_grid):
        raise RuntimeError(f"audited timestamps absent from v5 grid: {missing_from_grid.tolist()}")

    elevation, azimuth = pv.solar_position(
        grid, float(config["site"]["latitude"]), float(config["site"]["longitude"])
    )
    night = pd.Series(elevation <= 0.0, index=grid)
    per_inv = {n: source.process_inverter_5min(n, grid, night) for n in range(1, source.N_INVERTERS + 1)}

    five = pd.DataFrame(index=grid)
    for src, dst in source.POWER_COLS.items():
        values = pd.concat([per_inv[n][f"{src}__inv{n}"] for n in range(1, source.N_INVERTERS + 1)], axis=1)
        five[dst] = values.sum(axis=1, skipna=True, min_count=1)
    for src, dst in source.MEAN_COLS.items():
        values = pd.concat([per_inv[n][f"{src}__inv{n}"] for n in range(1, source.N_INVERTERS + 1)], axis=1)
        five[dst] = values.mean(axis=1, skipna=True)

    raw_obs = pd.concat([per_inv[n]["_raw_observed"] for n in range(1, source.N_INVERTERS + 1)], axis=1)
    available = pd.concat(
        [per_inv[n][f"출력전력__inv{n}"].notna() for n in range(1, source.N_INVERTERS + 1)], axis=1
    )
    five["가용인버터수"] = available.sum(axis=1).astype("int8")
    five["원시관측인버터수"] = raw_obs.sum(axis=1).astype("int8")

    # Minimal v6 difference: invalidate exactly the audited timestamps.  This
    # does not impose a global five-inverter gate and therefore preserves v5's
    # 94-day inverter-5 partial-sum recovery policy everywhere else.
    before = five.loc[rejected_times, ["발전출력_kW", "입력전력_kW", "입력전류_A", "가용인버터수"]].copy()
    five.loc[rejected_times, ["발전출력_kW", "입력전력_kW", "입력전류_A"]] = np.nan
    five.loc[rejected_times, "가용인버터수"] = np.minimum(
        five.loc[rejected_times, "가용인버터수"].to_numpy(), source.N_INVERTERS - 1
    )
    five["v6_peer_excluded"] = five.index.isin(rejected_times).astype("int8")
    five["부분가용여부"] = ((five["가용인버터수"] > 0) & (five["가용인버터수"] < source.N_INVERTERS)).astype("int8")
    five["완전가용"] = (five["가용인버터수"] == source.N_INVERTERS).astype("int8")
    five.loc[~five["발전출력_kW"].between(0, capacity), "발전출력_kW"] = np.nan
    five["태양고도_deg"] = elevation
    five["태양방위각_deg"] = azimuth
    five["물리적낮"] = (elevation > 0).astype("int8")
    five["핵심낮시간"] = (elevation > 10).astype("int8")

    f15 = pv.aggregate_power(five, "15min", required_fraction=1.0)
    f1h = pv.aggregate_power(five, "1h", required_fraction=0.75)
    for frame, freq in ((f15, "15min"), (f1h, "1h")):
        grouped = five["가용인버터수"].resample(freq)
        frame["가용인버터수_최소"] = grouped.min()
        frame["가용인버터수_평균"] = grouped.mean().round(2)
        frame["완전가용비율"] = five["완전가용"].resample(freq).mean().round(3)
        frame["부분가용여부"] = five["부분가용여부"].resample(freq).max().astype("int8")
        frame["v6_peer_excluded_count"] = five["v6_peer_excluded"].resample(freq).sum().astype("int16")

    daily = pv.build_daily_actual(five)
    day_only = five[five["물리적낮"] == 1]
    day_key = day_only.index.normalize()
    daily["가용인버터수_낮시간최소"] = day_only.groupby(day_key)["가용인버터수"].min()
    daily["가용인버터수_낮시간평균"] = day_only.groupby(day_key)["가용인버터수"].mean().round(2)
    daily["부분가용일"] = (daily["가용인버터수_낮시간평균"] < source.N_INVERTERS).astype("int8")
    daily["v6_peer_excluded_count"] = five["v6_peer_excluded"].groupby(five.index.normalize()).sum().astype("int16")

    f15.to_parquet(OUT / "집계_15분_자료_v5.parquet")
    f1h.to_parquet(OUT / "집계_1시간_자료_v5.parquet")
    daily.to_parquet(OUT / "집계_일간_실제발전량_v5.parquet")
    log.to_csv(OUT / "v6_제외_원본행_150.csv", index=False, encoding="utf-8-sig")
    log[log["v6_exclusion_reason"] == "equipment_anomaly_candidate"].to_csv(
        OUT / "pending_review_5행_별도로그.csv", index=False, encoding="utf-8-sig"
    )
    before.reset_index(names="grid_time_kst").to_csv(
        OUT / "v6_제외전_플랜트값_150.csv", index=False, encoding="utf-8-sig"
    )

    v5_dir = PIPE_DIR / "outputs" / "v5_복구_2026-08-21"
    comparisons = []
    for name in ("집계_15분_자료_v5.parquet", "집계_1시간_자료_v5.parquet", "집계_일간_실제발전량_v5.parquet"):
        old, new = pd.read_parquet(v5_dir / name), pd.read_parquet(OUT / name)
        common = old.index.intersection(new.index)
        cols = sorted(set(old.columns).intersection(new.columns))
        changed_cells = 0
        changed_rows = pd.Series(False, index=common)
        for col in cols:
            a, b = old.loc[common, col], new.loc[common, col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                diff = ~np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
            else:
                diff = ~((a.eq(b)) | (a.isna() & b.isna())).to_numpy()
            changed_cells += int(diff.sum())
            changed_rows |= diff
        comparisons.append({"파일": name, "행수_v5": len(old), "행수_v6": len(new), "공통열수": len(cols),
                            "변경행수": int(changed_rows.sum()), "변경셀수": changed_cells})
    pd.DataFrame(comparisons).to_csv(OUT / "v5대비_집계회귀감사.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "candidate_built",
        "official_v5_modified": False,
        "api_called": False,
        "v5_partial_sum_policy_preserved": True,
        "peer_audit_excluded_timestamps": 145,
        "equipment_anomaly_candidate_timestamps": 5,
        "combined_excluded_timestamps": int(five["v6_peer_excluded"].sum()),
        "affected_dates": sorted({str(x.date()) for x in rejected_times}),
        "outputs": str(OUT),
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
