# -*- coding: utf-8 -*-
"""nwp_dual_run_manager_v1_2026-08-28.py 오프라인 테스트.

운영 DB(kma_live_inputs.sqlite3)는 전혀 건드리지 않는다. 이 스크립트가
그때그때 만드는 임시 SQLite(테스트 전용)와 임시 매니페스트 디렉터리만
사용한다. 사용자 지시 1번의 "오프라인 테스트" 6개 항목을 그대로 구현.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
KST = ZoneInfo("Asia/Seoul")

spec = ilu.spec_from_file_location("nwpmgr", PARENT / "nwp_dual_run_manager_v1_2026-08-28.py")
nwpmgr = ilu.module_from_spec(spec)
sys.modules["nwpmgr"] = nwpmgr
spec.loader.exec_module(nwpmgr)
opstatus = nwpmgr.opstatus

REQUIRED_VARS = ["DSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]
REQUIRED_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

BASE_CFG = {
    "promotion_enabled": True,  # 테스트에서는 승격 로직 자체를 검증해야 하므로 활성화
    "run09_promotion_deadline_kst": "23:59",
    "run09_secondary_deadline_kst": "23:59",
    "required_vars": REQUIRED_VARS,
    "required_target_hours": REQUIRED_HOURS,
    "max_missing_ratio": 0.0,
    "max_retries": 4,
    "allow_secondary_fallback": False,
}


def make_test_db(path: Path, target_day_str: str, run03_complete: bool,
                 run09_complete: bool, run09_missing_vars: list | None = None) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE nwp_values (
          requested_tm_utc TEXT, variable TEXT, target_time_kst TEXT,
          value REAL, is_missing INTEGER, first_received_at TEXT
        )
    """)
    target_day = datetime.strptime(target_day_str, "%Y-%m-%d")
    run03_tm = nwpmgr._run_tm_utc(target_day, opstatus.RUN_03).strftime("%Y-%m-%dT%H:%M:%SZ")
    run09_tm = nwpmgr._run_tm_utc(target_day, opstatus.RUN_09).strftime("%Y-%m-%dT%H:%M:%SZ")
    missing_vars = set(run09_missing_vars or [])

    def insert_run(run_tm: str, complete: bool, skip_vars: set):
        for h in REQUIRED_HOURS:
            for v in REQUIRED_VARS:
                target_time_kst = "%sT%02d:00:00+09:00" % (target_day_str, h)
                if not complete and v == REQUIRED_VARS[0] and h == REQUIRED_HOURS[0]:
                    is_missing, val = 1, None
                elif v in skip_vars:
                    is_missing, val = 1, None
                else:
                    is_missing, val = 0, 100.0
                conn.execute(
                    "INSERT INTO nwp_values VALUES (?,?,?,?,?,?)",
                    (run_tm, v, target_time_kst, val, is_missing,
                     "2026-08-27T09:00:00+09:00"))

    insert_run(run03_tm, run03_complete, set())
    insert_run(run09_tm, run09_complete, missing_vars)
    conn.commit()
    conn.close()


def run_case(name: str, run03_complete: bool, run09_complete: bool,
            run09_missing_vars: list | None = None) -> dict:
    tmpdir = Path(tempfile.mkdtemp(prefix="nwp_test_"))
    try:
        db_path = tmpdir / "test.sqlite3"
        make_test_db(db_path, "2026-09-01", run03_complete, run09_complete, run09_missing_vars)
        manifest_dir = tmpdir / "manifest"
        cfg = dict(BASE_CFG)
        target_day = datetime(2026, 9, 1)
        now_kst = datetime(2026, 8, 31, 12, 0, tzinfo=KST)

        decision = nwpmgr.decide(target_day, cfg, db_path, now_kst)
        outcome = nwpmgr.promote(decision, manifest_dir)
        return {"name": name, "decision": decision, "outcome": outcome, "tmpdir": str(tmpdir)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_1_both_ok_promotes_09():
    r = run_case("1_둘다정상_09승격", run03_complete=True, run09_complete=True)
    assert r["decision"]["상태"] == opstatus.NORMAL, r["decision"]
    assert r["decision"]["사용런"] == opstatus.RUN_09
    assert r["outcome"]["승격"] is True
    return True


def test_2_run09_fail_keeps_03():
    r = run_case("2_09실패_03유지", run03_complete=True, run09_complete=False)
    assert r["decision"]["상태"] == opstatus.FALLBACK_03RUN, r["decision"]
    assert r["decision"]["사용런"] == opstatus.RUN_03
    return True


def test_3_run09_partial_missing_keeps_03():
    r = run_case("3_09일부결측_03유지", run03_complete=True, run09_complete=True,
                 run09_missing_vars=["TCDC"])
    assert r["decision"]["상태"] == opstatus.FALLBACK_03RUN, r["decision"]
    assert "TCDC" in r["decision"]["run09_검사"]["부족변수"]
    return True


def test_4_both_fail_blocked():
    r = run_case("4_둘다실패_blocked", run03_complete=False, run09_complete=False)
    assert r["decision"]["상태"] == opstatus.BLOCKED, r["decision"]
    assert r["decision"]["run03_검사"]["완결"] is False
    assert r["decision"]["run09_검사"]["완결"] is False
    return True


def test_5_no_overwrite_of_good_with_bad():
    """기존 정상자료(normal)가 이후 불완전 재판정(blocked)으로 덮이지 않아야 한다."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nwp_test_"))
    try:
        manifest_dir = tmpdir / "manifest"
        good = {"목표일": "2026-09-02", "상태": opstatus.NORMAL,
                "사용런": opstatus.RUN_09, "사유": "정상"}
        r1 = nwpmgr.promote(good, manifest_dir)
        assert r1["승격"] is True

        bad = {"목표일": "2026-09-02", "상태": opstatus.BLOCKED,
              "사용런": opstatus.RUN_NONE, "사유": "재판정 실패"}
        r2 = nwpmgr.promote(bad, manifest_dir)
        assert r2["승격"] is False, "불완전 판정이 승격되면 안 된다"
        final = json.loads((manifest_dir / "nwp_2026-09-02.json").read_text(encoding="utf-8"))
        assert final["상태"] == opstatus.NORMAL, "기존 정상 매니페스트가 훼손됐다"
        assert (manifest_dir / "nwp_2026-09-02.rejected.json").is_file()
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_6_idempotent_rerun():
    tmpdir = Path(tempfile.mkdtemp(prefix="nwp_test_"))
    try:
        db_path = tmpdir / "test.sqlite3"
        make_test_db(db_path, "2026-09-03", True, True)
        manifest_dir = tmpdir / "manifest"
        cfg = dict(BASE_CFG)
        target_day = datetime(2026, 9, 3)
        now_kst = datetime(2026, 9, 2, 12, 0, tzinfo=KST)

        d1 = nwpmgr.decide(target_day, cfg, db_path, now_kst)
        o1 = nwpmgr.promote(d1, manifest_dir)
        d2 = nwpmgr.decide(target_day, cfg, db_path, now_kst)
        o2 = nwpmgr.promote(d2, manifest_dir)
        assert o1["승격"] is True
        assert o2["승격"] is False and "unchanged" in o2["결과"], o2
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_7_exception_during_promote_no_mixed_state():
    """승격 도중 예외가 나도 staging 잔여물이 안 남고 기존 매니페스트가 그대로여야 한다."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nwp_test_"))
    try:
        manifest_dir = tmpdir / "manifest"
        manifest_dir.mkdir(parents=True)
        good = {"목표일": "2026-09-04", "상태": opstatus.FALLBACK_03RUN,
                "사용런": opstatus.RUN_03, "사유": "정상(03)"}
        nwpmgr.promote(good, manifest_dir)

        # 상태값이 정의되지 않은 잘못된 decision -> staging 검증에서 예외 유발
        broken = {"목표일": "2026-09-05", "상태": "존재하지않는상태",
                  "사용런": opstatus.RUN_03, "사유": "고의 오류"}
        outcome = nwpmgr.promote(broken, manifest_dir)
        assert outcome["승격"] is False
        assert not (manifest_dir / "nwp_2026-09-05.staging.json").exists(), "staging 잔여물 남음"
        assert not (manifest_dir / "nwp_2026-09-05.json").exists(), "잘못된 값이 승격됨"
        # 기존(09-04) 정상 매니페스트는 그대로
        prev = json.loads((manifest_dir / "nwp_2026-09-04.json").read_text(encoding="utf-8"))
        assert prev["상태"] == opstatus.FALLBACK_03RUN
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_8_promotion_disabled_forces_fallback():
    """promotion_enabled=false면 09시런이 완결이어도 03시런을 쓴다(현재 운영 기본값)."""
    r = run_case("8_승격비활성_03유지", run03_complete=True, run09_complete=True)
    cfg = dict(BASE_CFG)
    cfg["promotion_enabled"] = False
    tmpdir = Path(tempfile.mkdtemp(prefix="nwp_test_"))
    try:
        db_path = tmpdir / "test.sqlite3"
        make_test_db(db_path, "2026-09-06", True, True)
        target_day = datetime(2026, 9, 6)
        now_kst = datetime(2026, 9, 5, 12, 0, tzinfo=KST)
        decision = nwpmgr.decide(target_day, cfg, db_path, now_kst)
        assert decision["상태"] == opstatus.FALLBACK_03RUN, decision
        assert "promotion_enabled=false" in decision["사유"]
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


TESTS = [
    test_1_both_ok_promotes_09,
    test_2_run09_fail_keeps_03,
    test_3_run09_partial_missing_keeps_03,
    test_4_both_fail_blocked,
    test_5_no_overwrite_of_good_with_bad,
    test_6_idempotent_rerun,
    test_7_exception_during_promote_no_mixed_state,
    test_8_promotion_disabled_forces_fallback,
]


def main() -> int:
    results = []
    for t in TESTS:
        try:
            ok = t()
            results.append((t.__name__, "PASS" if ok else "FAIL", ""))
        except AssertionError as e:
            results.append((t.__name__, "FAIL", str(e)))
        except Exception as e:
            results.append((t.__name__, "ERROR", "%s: %s" % (type(e).__name__, e)))

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    for name, status, msg in results:
        print("[%s] %s %s" % (status, name, ("- " + msg) if msg else ""))
    print("\n%d/%d PASS" % (n_pass, len(results)))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
