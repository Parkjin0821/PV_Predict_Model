"""Profile site inverter workbooks without modifying their contents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LEGACY_TOOLS = Path(r"C:\Users\u-cube\Documents\ChatGPT\유큐브\광주_pv_model_workspace")
sys.path.insert(0, str(LEGACY_TOOLS))
from scan_xlsx_logs import scan


def main() -> int:
    root = Path(__file__).resolve().parent
    site = sys.argv[1] if len(sys.argv) > 1 else "gimje"
    files = sorted((root / "input" / site).glob("inverter_*.xlsx"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    results = [scan(p) for p in files]
    out = root / "outputs" / f"{site}_source_profile.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = [{
        "file": r["file"], "rows": r["data_rows"],
        "start": r["timestamp"]["min"], "end": r["timestamp"]["max"],
        "max_gap_h": round((r["gaps_seconds"]["max"] or 0) / 3600, 2),
        "formulas": r["formula_count"],
    } for r in results]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
