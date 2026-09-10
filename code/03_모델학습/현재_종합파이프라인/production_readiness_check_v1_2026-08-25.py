# -*- coding: utf-8 -*-
"""운영 자동화 전 수동 준비상태 감사.

외부 API를 호출하거나 스케줄을 등록하지 않는다. Blockdata·KMA SQLite와
운영모델 8개 번들을 읽어 현재 수동 추론을 시작할 수 있는지 판정한다.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "outputs" / "운영모델_v2_2026-08-25"
BLOCK_DB = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25\blockdata_history.sqlite3")
KMA_DB = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3")
OUT = ROOT / "outputs" / "운영준비상태_v1_2026-08-25"


def load_infer():
    path = ROOT / "production_inference_utils_v1_2026-08-25.py"
    spec = importlib.util.spec_from_file_location("ucube_prod_infer_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"추론 유틸을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


infer = load_infer()


def parse_iso(value: str | None) -> datetime | None:
    return None if not value else datetime.fromisoformat(value)


def blockdata_status() -> dict:
    if not BLOCK_DB.is_file():
        return {"존재": False, "상태": "DB없음"}
    conn = sqlite3.connect(BLOCK_DB)
    try:
        count, first, last = conn.execute(
            "SELECT COUNT(*), MIN(snapshot_time), MAX(snapshot_time) FROM plant_snapshots"
        ).fetchone()
        success = conn.execute("SELECT COUNT(*) FROM poll_attempts WHERE status='success'").fetchone()[0]
        failure = conn.execute("SELECT COUNT(*) FROM poll_attempts WHERE status='failure'").fetchone()[0]
    finally:
        conn.close()
    first_dt, last_dt = parse_iso(first), parse_iso(last)
    span_hours = 0.0 if not first_dt or not last_dt else (last_dt - first_dt).total_seconds() / 3600
    return {
        "존재": True, "발전소스냅샷행수": count, "최초시각": first,
        "최종시각": last, "이력시간_h": round(span_hours, 3),
        "성공호출": success, "실패호출": failure,
        "초단기4시간이력충족": span_hours >= 4,
        "단기24시간이력충족": span_hours >= 24,
        "일간32일이력충족": span_hours >= 32 * 24,
    }


def kma_status() -> dict:
    if not KMA_DB.is_file():
        return {"존재": False, "상태": "DB없음"}
    conn = sqlite3.connect(KMA_DB)
    try:
        asos = conn.execute("SELECT COUNT(*),MIN(observation_time),MAX(observation_time) FROM asos_hourly").fetchone()
        latest_grid = conn.execute("SELECT MAX(issue_date) FROM grid_forecast").fetchone()[0]
        grid = (0, 0) if latest_grid is None else conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(is_missing),0) FROM grid_forecast WHERE issue_date=?", (latest_grid,)
        ).fetchone()
        latest_nwp = conn.execute("SELECT MAX(issue_date) FROM nwp_run_status").fetchone()[0]
        nwp = None if latest_nwp is None else conn.execute(
            "SELECT stored_value_count,missing_value_count,complete,complete_before_issue,status_message FROM nwp_run_status WHERE issue_date=?",
            (latest_nwp,),
        ).fetchone()
        missing_vars = [] if latest_nwp is None else [
            {"변수": row[0], "결측수": row[1]}
            for row in conn.execute(
                "SELECT variable,SUM(is_missing) FROM nwp_values WHERE issue_date=? GROUP BY variable HAVING SUM(is_missing)>0 ORDER BY variable",
                (latest_nwp,),
            ).fetchall()
        ]
    finally:
        conn.close()
    return {
        "존재": True,
        "ASOS": {"행수": asos[0], "최초시각": asos[1], "최종시각": asos[2], "통과": asos[0] > 0},
        "격자예보": {"최신발표일": latest_grid, "행수": grid[0], "결측수": grid[1], "통과": grid[0] == 40},
        "NWP": None if nwp is None else {
            "최신발표일": latest_nwp, "저장값": nwp[0], "결측값": nwp[1],
            "전체응답완료": bool(nwp[2]), "D10이전입수": bool(nwp[3]),
            "상태": nwp[4], "결측변수": missing_vars,
        },
    }


def bundle_status() -> dict:
    files = sorted(BUNDLE_DIR.glob("*.joblib"))
    audits = {}
    for path in files:
        bundle = joblib.load(path)
        audits[path.name] = infer.audit_bundle(bundle)
    return {
        "번들수": len(files), "8종존재": len(files) == 8,
        "전체audit통과": len(files) == 8 and all(item["전체통과"] for item in audits.values()),
        "상세": audits,
    }


def main() -> int:
    block = blockdata_status()
    kma = kma_status()
    bundles = bundle_status()
    nwp = kma.get("NWP") or {}
    common_weather = bool(
        kma.get("ASOS", {}).get("통과")
        and kma.get("격자예보", {}).get("통과")
        and nwp.get("전체응답완료")
    )
    tiers = {
        "초단기": "ready" if bundles["전체audit통과"] and common_weather and block.get("초단기4시간이력충족") else "warming_up",
        "단기": "ready" if bundles["전체audit통과"] and common_weather and block.get("단기24시간이력충족") else "warming_up",
        "일간": "ready" if bundles["전체audit통과"] and common_weather and block.get("일간32일이력충족") else "warming_up",
    }
    summary = {
        "검사시각": datetime.now(tz=KST).isoformat(timespec="seconds"),
        "자동화등록": False,
        "Blockdata": block, "기상청": kma, "운영모델": bundles,
        "티어별준비상태": tiers,
        "판정": "manual_inference_ready" if all(v == "ready" for v in tiers.values()) else "expected_history_warmup",
        "주의": "NWP 결측은 LightGBM/XGBoost native missing 경로로 전달하며 임의 보간하지 않는다. D10 이후 받은 발표분은 해당 D10 예측에 소급 사용하지 않는다.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "운영준비상태.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"판정": summary["판정"], "티어별준비상태": tiers, "산출물": str(path)}, ensure_ascii=False))
    # 이력 워밍업은 예상 상태이므로 실패 exit로 처리하지 않는다.
    return 0 if bundles["전체audit통과"] and common_weather else 1


if __name__ == "__main__":
    raise SystemExit(main())
