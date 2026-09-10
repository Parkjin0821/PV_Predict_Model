# -*- coding: utf-8 -*-
"""광주 운영용 KMA 03 KST 런(전날 18 UTC) NWP 수집기.

기존 09시런 수집기의 DB·HTTP·파싱·재시도 구현을 그대로 호출하고,
동일 issue_date의 run_tm만 6시간 앞당긴다. nwp_run_status는 issue_date
단일 PK라 09시런 상태를 덮을 수 있으므로 이 수집기는 그 표를 쓰지 않고
nwp_values와 별도 상태 JSON만 신규 기록한다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE09_PATH = HERE / "collect_kma_nwp_live_v1_2026-08-25.py"
SAFETY_PATH = (
    HERE.parents[1] / "03_모델학습" / "현재_종합파이프라인"
    / "운영안전장치_2026-08-28" / "operational_status_v1_2026-08-28.py"
)
MANAGER_PATH = SAFETY_PATH.with_name("nwp_dual_run_manager_v1_2026-08-28.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈 로드 실패: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


live09 = _load("ucube_live_nwp_09_shared", LIVE09_PATH)
opstatus = _load("ucube_operational_status_shared", SAFETY_PATH)
manager = _load("ucube_nwp_dual_run_manager_shared", MANAGER_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="광주 KMA NWP 03시런 수집")
    parser.add_argument("--issue-date", help="발표일 YYYY-MM-DD, 기본 오늘 KST")
    parser.add_argument("--output-dir", type=Path, default=live09.DEFAULT_OUT)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.3)
    parser.add_argument("--force", action="store_true", help="기존 03시런 행도 재조회")
    args = parser.parse_args()
    if args.max_retries < 1:
        parser.error("--max-retries는 1 이상이어야 합니다.")
    return args


def _readonly_count(db_path: Path, issue_date: str, run_tm_text: str) -> dict:
    with opstatus.connect_readonly(db_path) as conn:
        values = conn.execute(
            "SELECT COUNT(1), COALESCE(SUM(is_missing),0) FROM nwp_values "
            "WHERE issue_date=? AND requested_tm_utc=?",
            (issue_date, run_tm_text),
        ).fetchone()
        requests = conn.execute(
            "SELECT COUNT(1), COALESCE(SUM(status='success'),0), "
            "COALESCE(SUM(status='failure'),0) FROM nwp_requests "
            "WHERE issue_date=? AND requested_tm_utc=?",
            (issue_date, run_tm_text),
        ).fetchone()
    return {
        "DB_nwp_values_COUNT": int(values[0]),
        "DB_missing_COUNT": int(values[1]),
        "DB_nwp_requests_COUNT": int(requests[0]),
        "DB_HTTP성공_COUNT": int(requests[1]),
        "DB_HTTP실패_COUNT": int(requests[2]),
    }


def main() -> int:
    args = parse_args()
    issue_day = live09.parse_issue_date(args.issue_date)
    issue_text = issue_day.strftime("%Y%m%d")
    run09_tm = live09.src.fixed_run_for_issue_day(issue_text)
    run03_tm = run09_tm - timedelta(hours=6)
    run_tm_text = run03_tm.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        db_path = args.output_dir / "kma_live_inputs.sqlite3"
        if db_path.is_file() and not args.force:
            target_day = datetime.combine(issue_day + timedelta(days=1), datetime.min.time())
            ready = manager.check_run_completeness(
                db_path, target_day, opstatus.RUN_03, manager.load_config()
            )
            if ready["완결"]:
                counts = _readonly_count(db_path, issue_day.isoformat(), run_tm_text)
                print(json.dumps({
                    "status": "already_complete",
                    "issue_date": issue_day.isoformat(),
                    "requested_tm_utc": run_tm_text,
                    "api_calls": 0,
                    **counts,
                    "run_kind": "03run_prev18utc",
                }, ensure_ascii=False))
                return 0
        summary = live09.collect_nwp(
            args,
            run03_tm,
            update_run_status=False,
            status_filename="nwp_collector_status_03run.json",
        )
        counts = _readonly_count(db_path, issue_day.isoformat(), run_tm_text)
        result = {**summary, **counts, "run_kind": "03run_prev18utc"}
        print(json.dumps(result, ensure_ascii=False))
        return 0 if summary.get("complete", False) else 1
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, live09.LiveNwpAuthError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
