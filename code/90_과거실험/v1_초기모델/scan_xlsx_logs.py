"""Read-only streaming scanner for the five inverter XLSX logs.

The scanner parses OOXML directly with the Python standard library so large
workbooks can be profiled without recalculation or modification.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF_RE = re.compile(r"([A-Z]+)")


def col_index(cell_ref: str) -> int:
    match = CELL_REF_RE.match(cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return []
    values: list[str] = []
    with zf.open(name) as stream:
        for event, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag == f"{{{NS_MAIN}}}si":
                parts = [node.text or "" for node in elem.iter(f"{{{NS_MAIN}}}t")]
                values.append("".join(parts))
                elem.clear()
    return values


def first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{NS_MAIN}}}sheet")
    if sheet is None:
        raise ValueError("Workbook has no worksheet")
    rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall(f"{{{NS_REL}}}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"].lstrip("/")
            if target.startswith("xl/"):
                return target
            return f"xl/{target}"
    raise ValueError("Worksheet relationship not found")


def cell_value(cell: ET.Element, strings: list[str]):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{NS_MAIN}}}t"))
    value_node = cell.find(f"{{{NS_MAIN}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        idx = int(raw)
        return strings[idx] if 0 <= idx < len(strings) else raw
    if cell_type in ("str", "e"):
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        number = float(raw)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return raw


def parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Excel serial date, 1900 date system.
        return datetime.fromtimestamp((float(value) - 25569) * 86400)
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def percentile(sorted_values: list[float], q: float):
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def scan(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        strings = shared_strings(zf)
        sheet_path = first_sheet_path(zf)
        headers: list[str] = []
        missing: Counter[int] = Counter()
        numeric_count: Counter[int] = Counter()
        numeric_sum: Counter[int] = Counter()
        numeric_min: dict[int, float] = {}
        numeric_max: dict[int, float] = {}
        status_counts: Counter[str] = Counter()
        timestamps: list[datetime] = []
        timestamp_parse_fail = 0
        formulas = 0
        formula_samples: list[str] = []
        rows = 0

        with zf.open(sheet_path) as stream:
            for event, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != f"{{{NS_MAIN}}}row":
                    continue
                values: dict[int, object] = {}
                for cell in elem.findall(f"{{{NS_MAIN}}}c"):
                    idx = col_index(cell.attrib.get("r", "A1"))
                    values[idx] = cell_value(cell, strings)
                    formula_node = cell.find(f"{{{NS_MAIN}}}f")
                    if formula_node is not None:
                        formulas += 1
                        if len(formula_samples) < 10:
                            formula_samples.append(
                                f"{cell.attrib.get('r')}={formula_node.text or ''}"
                            )

                if rows == 0:
                    width = max(values.keys(), default=-1) + 1
                    headers = [str(values.get(i) or "") for i in range(width)]
                    rows += 1
                    elem.clear()
                    continue

                rows += 1
                for idx in range(len(headers)):
                    value = values.get(idx)
                    if value is None or value == "":
                        missing[idx] += 1
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        number = float(value)
                        numeric_count[idx] += 1
                        numeric_sum[idx] += number
                        numeric_min[idx] = min(numeric_min.get(idx, number), number)
                        numeric_max[idx] = max(numeric_max.get(idx, number), number)

                if headers:
                    ts = parse_timestamp(values.get(0))
                    if ts is None:
                        timestamp_parse_fail += 1
                    else:
                        timestamps.append(ts)
                if "상태" in headers:
                    status_idx = headers.index("상태")
                    status_value = values.get(status_idx)
                    if status_value not in (None, ""):
                        status_counts[str(status_value)] += 1
                elem.clear()

    data_rows = max(0, rows - 1)
    timestamps.sort()
    timestamp_counts = Counter(timestamps)
    duplicate_timestamp_rows = sum(count - 1 for count in timestamp_counts.values() if count > 1)
    unique_timestamps = sorted(timestamp_counts)
    gaps = [
        (b - a).total_seconds()
        for a, b in zip(unique_timestamps, unique_timestamps[1:])
        if b >= a
    ]
    gaps_sorted = sorted(gaps)

    column_profiles = []
    for idx, header in enumerate(headers):
        count = numeric_count[idx]
        column_profiles.append(
            {
                "index": idx + 1,
                "name": header,
                "missing": missing[idx],
                "missing_rate": round(missing[idx] / data_rows, 6) if data_rows else None,
                "numeric_count": count,
                "min": numeric_min.get(idx),
                "max": numeric_max.get(idx),
                "mean": round(numeric_sum[idx] / count, 6) if count else None,
            }
        )

    gap_buckets = Counter()
    for gap in gaps:
        if gap < 0:
            gap_buckets["negative"] += 1
        elif gap < 240:
            gap_buckets["under_4min"] += 1
        elif gap <= 360:
            gap_buckets["4_to_6min"] += 1
        elif gap <= 600:
            gap_buckets["over_6_to_10min"] += 1
        elif gap <= 3600:
            gap_buckets["over_10_to_60min"] += 1
        else:
            gap_buckets["over_60min"] += 1

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sheet_path": sheet_path,
        "rows_including_header": rows,
        "data_rows": data_rows,
        "columns": len(headers),
        "headers": headers,
        "formula_count": formulas,
        "formula_samples": formula_samples,
        "timestamp": {
            "parsed": len(timestamps),
            "parse_fail": timestamp_parse_fail,
            "min": timestamps[0].isoformat(sep=" ") if timestamps else None,
            "max": timestamps[-1].isoformat(sep=" ") if timestamps else None,
            "duplicate_rows": duplicate_timestamp_rows,
        },
        "gaps_seconds": {
            "count": len(gaps),
            "min": gaps_sorted[0] if gaps_sorted else None,
            "p25": percentile(gaps_sorted, 0.25),
            "median": statistics.median(gaps_sorted) if gaps_sorted else None,
            "p75": percentile(gaps_sorted, 0.75),
            "p95": percentile(gaps_sorted, 0.95),
            "max": gaps_sorted[-1] if gaps_sorted else None,
            "buckets": dict(gap_buckets),
        },
        "column_profiles": column_profiles,
        "top_statuses": status_counts.most_common(12),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    input_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "input")
    results = [scan(path) for path in sorted(input_dir.glob("inverter_*.xlsx"))]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

