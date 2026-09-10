"""Extract read-only inverter XLSX logs into auditable 5-minute/hourly CSVs.

The source workbooks are never modified. OOXML is streamed directly because
the files are large and contain only cached/static values.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF_RE = re.compile(r"([A-Z]+)")


def col_index(ref: str) -> int:
    match = CELL_REF_RE.match(ref)
    value = 0
    for char in match.group(1) if match else "A":
        value = value * 26 + ord(char) - 64
    return value - 1


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    result = []
    with zf.open("xl/sharedStrings.xml") as stream:
        for _, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag == f"{{{NS_MAIN}}}si":
                result.append("".join(n.text or "" for n in elem.iter(f"{{{NS_MAIN}}}t")))
                elem.clear()
    return result


def first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{NS_MAIN}}}sheet")
    rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall(f"{{{NS_REL}}}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("Worksheet relationship not found")


def cell_value(cell: ET.Element, strings: list[str]):
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(n.text or "" for n in cell.iter(f"{{{NS_MAIN}}}t"))
    node = cell.find(f"{{{NS_MAIN}}}v")
    if node is None or node.text is None:
        return None
    raw = node.text
    if kind == "s":
        return strings[int(raw)]
    if kind in ("str", "e"):
        return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_timestamp(value) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp((float(value) - 25569) * 86400)
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def read_selected(path: Path, inverter: int) -> pd.DataFrame:
    wanted = ["생성일", "통신에러", "입력전압", "입력전류", "입력전력", "출력전력",
              "주파수", "역률", "온도", "일일", "누적", "상태", "주기별발전량"]
    records = []
    with zipfile.ZipFile(path) as zf:
        strings = shared_strings(zf)
        sheet_path = first_sheet_path(zf)
        headers = None
        indices = None
        with zf.open(sheet_path) as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != f"{{{NS_MAIN}}}row":
                    continue
                values = {col_index(c.attrib.get("r", "A1")): cell_value(c, strings)
                          for c in elem.findall(f"{{{NS_MAIN}}}c")}
                if headers is None:
                    width = max(values, default=-1) + 1
                    headers = [str(values.get(i) or "") for i in range(width)]
                    indices = {name: headers.index(name) for name in wanted}
                else:
                    row = {name: values.get(idx) for name, idx in indices.items()}
                    row["inverter"] = inverter
                    records.append(row)
                elem.clear()
    df = pd.DataFrame.from_records(records)
    df["생성일"] = df["생성일"].map(parse_timestamp)
    for col in ["입력전압", "입력전류", "입력전력", "출력전력", "주파수", "역률", "온도", "일일", "누적", "주기별발전량"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("생성일").reset_index(drop=True)


def longest_gaps(df: pd.DataFrame, limit: int = 12) -> list[dict]:
    times = df["생성일"].dropna().drop_duplicates().sort_values()
    gaps = times.diff().dt.total_seconds()
    positions = np.argsort(gaps.to_numpy(dtype=float, na_value=np.nan))[::-1]
    result = []
    for pos in positions:
        if pos == 0 or not math.isfinite(gaps.iloc[pos]):
            continue
        result.append({
            "gap_start": times.iloc[pos - 1].isoformat(sep=" "),
            "gap_end": times.iloc[pos].isoformat(sep=" "),
            "gap_hours": round(gaps.iloc[pos] / 3600, 3),
        })
        if len(result) >= limit:
            break
    return result


def robust_outlier_summary(df: pd.DataFrame, col: str) -> dict:
    x = df[col].dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    largest = df.nlargest(10, col)[["생성일", col, "누적", "출력전력"]]
    return {
        "q1": float(q1), "q3": float(q3), "iqr_3x_upper": float(upper),
        "count_above": int((x > upper).sum()),
        "largest": [{k: (v.isoformat(sep=" ") if isinstance(v, datetime) else None if pd.isna(v) else float(v))
                     for k, v in row.items()} for row in largest.to_dict("records")],
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    input_dir, output_dir = root / "input", root / "outputs"
    output_dir.mkdir(exist_ok=True)
    frames = []
    audit = {"inverters": {}}
    for inverter in range(1, 6):
        df = read_selected(input_dir / f"inverter_{inverter}.xlsx", inverter)
        frames.append(df)
        cumulative_delta = df["누적"].diff()
        audit["inverters"][str(inverter)] = {
            "rows": len(df),
            "longest_gaps": longest_gaps(df),
            "cumulative_negative_steps": int((cumulative_delta < 0).sum()),
            "largest_cumulative_drop": float(cumulative_delta.min()) if cumulative_delta.notna().any() else None,
            "period_generation_outliers": robust_outlier_summary(df, "주기별발전량"),
        }

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(output_dir / "gwangju_inverter_selected_raw.csv", index=False, encoding="utf-8-sig")

    # Align irregular ~5-minute device timestamps to a common hourly grid.
    # Power is averaged within the hour; energy-like precomputed increments are summed
    # only after excluding negative values and device-specific extreme errors.
    clean = raw.copy()
    clean["hour"] = clean["생성일"].dt.floor("h")
    clean["period_generation_clean"] = clean["주기별발전량"].where(clean["주기별발전량"] >= 0)
    for inverter, group in clean.groupby("inverter"):
        q1, q3 = group["period_generation_clean"].quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        clean.loc[(clean["inverter"] == inverter) & (clean["period_generation_clean"] > upper),
                  "period_generation_clean"] = np.nan

    hourly_device = (clean.groupby(["hour", "inverter"], as_index=False)
        .agg(samples=("출력전력", "count"), output_power_mean=("출력전력", "mean"),
             output_power_max=("출력전력", "max"), period_generation_sum=("period_generation_clean", "sum"),
             input_power_mean=("입력전력", "mean"), temperature_mean=("온도", "mean"),
             status_nonempty=("상태", "count"), comm_error_nonempty=("통신에러", "count")))
    hourly_device.to_csv(output_dir / "gwangju_hourly_by_inverter.csv", index=False, encoding="utf-8-sig")

    plant = (hourly_device.groupby("hour", as_index=False)
        .agg(inverters_reporting=("inverter", "nunique"), samples=("samples", "sum"),
             plant_output_power_mean_sum=("output_power_mean", "sum"),
             plant_output_power_max_sum=("output_power_max", "sum"),
             plant_period_generation_sum=("period_generation_sum", "sum")))
    plant["complete_5_inverters"] = plant["inverters_reporting"].eq(5)
    plant.to_csv(output_dir / "gwangju_hourly_plant.csv", index=False, encoding="utf-8-sig")

    audit["plant_hourly"] = {
        "rows": len(plant),
        "min": plant["hour"].min().isoformat(sep=" "),
        "max": plant["hour"].max().isoformat(sep=" "),
        "complete_5_inverters_hours": int(plant["complete_5_inverters"].sum()),
        "incomplete_hours": int((~plant["complete_5_inverters"]).sum()),
        "reporting_counts": {str(k): int(v) for k, v in plant["inverters_reporting"].value_counts().sort_index().items()},
    }
    (output_dir / "data_quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
