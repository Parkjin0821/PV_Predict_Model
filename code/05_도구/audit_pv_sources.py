"""현재 보유한 광주 태양광 모델 원천자료의 구조를 읽기 전용으로 감사한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\u-cube\Documents\ChatGPT\유큐브")
V3 = ROOT / "gwangju_pv_multihorizon_v3_workspace"
GRID = ROOT / "결과물" / "예측모델" / "광주" / "grid_forecast_650d_interim_v2_2026-08-18"
PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전")


def frame_audit(path: Path, time_candidates: list[str]) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    result: dict = {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items() if int(v) > 0},
    }
    for name in time_candidates:
        if name not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[name], errors="coerce")
        result["time_column"] = name
        result["time_start"] = str(parsed.min())
        result["time_end"] = str(parsed.max())
        result["invalid_times"] = int(parsed.isna().sum())
        result["duplicate_times"] = int(parsed.duplicated().sum())
        break
    for name in ["inverter", "인버터번호"]:
        if name in frame.columns:
            result["inverter_counts"] = {
                str(k): int(v) for k, v in frame[name].value_counts(dropna=False).sort_index().items()
            }
            if "time_column" in result:
                pair = pd.DataFrame(
                    {
                        "time": pd.to_datetime(frame[result["time_column"]], errors="coerce"),
                        "inverter": frame[name],
                    }
                )
                result["duplicate_time_inverter_pairs"] = int(pair.duplicated().sum())
    numeric_summary = {}
    for name in [
        "출력전력",
        "plant_output_kw",
        "actual_kw",
        "TMP_fcst",
        "SKY_fcst",
        "REH_fcst",
    ]:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        numeric_summary[name] = {
            "count": int(values.notna().sum()),
            "min": float(values.min()) if values.notna().any() else None,
            "median": float(values.median()) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
        }
    result["numeric_summary"] = numeric_summary
    return result


def parquet_audit(path: Path) -> dict:
    frame = pd.read_parquet(path)
    parsed = pd.to_datetime(frame["생성일"], errors="coerce")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "time_start": str(parsed.min()),
        "time_end": str(parsed.max()),
        "invalid_times": int(parsed.isna().sum()),
        "inverter_counts": {
            str(k): int(v)
            for k, v in frame["인버터번호"].value_counts(dropna=False).sort_index().items()
        },
    }


def main() -> None:
    paths = {
        "인버터_5분_원본": V3 / "input" / "gwangju_inverter_selected_raw.csv",
        "발전소_5분_모델자료": V3 / "outputs" / "gwangju_5min_model_dataset.csv",
        "발전소_15분_모델자료": V3 / "outputs" / "gwangju_15min_model_dataset.csv",
        "발전소_1시간_모델자료": V3 / "outputs" / "gwangju_1hour_model_dataset.csv",
        "650일_예보_결합자료": GRID / "forecast_650d_combined.csv",
    }
    audit = {
        key: frame_audit(path, ["생성일", "time", "timestamp", "issue_time", "hour", "발표일"])
        for key, path in paths.items()
    }
    audit["표준화_학습자료"] = parquet_audit(PROJECT / "train_standardized.parquet")
    audit["표준화_시험자료"] = parquet_audit(PROJECT / "test_standardized.parquet")
    stats = json.loads((PROJECT / "normalization_stats.json").read_text(encoding="utf-8"))
    audit["표준화_설정"] = {
        "target": stats.get("target"),
        "features": stats.get("features"),
        "top_level_keys": list(stats),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
