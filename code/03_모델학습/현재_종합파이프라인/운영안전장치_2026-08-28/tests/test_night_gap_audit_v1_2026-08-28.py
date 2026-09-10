# -*- coding: utf-8 -*-
"""night_gap_audit_v1_2026-08-28.py 오프라인 테스트.

운영 DB(blockdata_history.sqlite3 등)는 읽지 않는다. 합성 데이터(임시
SQLite, 합성 poll_attempts/scheduler_events 딕셔너리)만 사용한다.

08-28 실측에서 실제로 벌어진 "poll_attempts 근거로 분류가 뒤집히는"
상황(작업은 정상 기동·API도 200 성공인데 new_inverter_rows=0이 반복되는
패턴)을 정확히 재현해 classify_gap_via_poll_attempts()가 이를
"정상수집_서버측_데이터정체_의심"으로 잡아내는지, 그리고 poll_attempts
근거가 아예 없는 구간에서는 None을 반환해 기존 Task-Scheduler 기반
classify_gap()으로 안전하게 폴백하는지 검증한다.
"""
from __future__ import annotations

import importlib.util as ilu
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("nga", PARENT / "night_gap_audit_v1_2026-08-28.py")
nga = ilu.module_from_spec(spec)
sys.modules["nga"] = nga
spec.loader.exec_module(nga)


def _mk_poll_row(t: str, status="success", http=200, new_rows=0, rhash="H1"):
    return {"시각": datetime.fromisoformat(t), "status": status, "http_status": http,
            "new_rows": new_rows, "dup_rows": 0, "response_hash": rhash,
            "quality_status": "ok", "error_type": None}


def test_0_classify_gap_distinguishes_query_failure_from_no_log():
    """08-28 3회차 실행에서 발견: 감사기간이 넓어 PowerShell 질의가
    타임아웃나면 '로그 없음'과 다른 문구로 표시돼야 한다(원인 오인 방지)."""
    gap = {"공백시작": "2026-08-27T20:00:00", "공백종료": "2026-08-27T21:00:00"}
    c1 = nga.classify_gap(gap, [], query_failed_reason=None)
    assert "로그 없음" in c1, c1
    c2 = nga.classify_gap(gap, [], query_failed_reason="타임아웃(120초 초과)")
    assert "질의 자체 실패" in c2, c2
    assert "로그 없음" not in c2, c2
    return True


def test_1_parse_event_time_dotnet_iso_ok():
    dt = nga._parse_event_time("2026-08-27T11:59:15.5660000Z")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 8 and dt.day == 27
    return True


def test_2_parse_event_time_none_on_garbage():
    assert nga._parse_event_time("not-a-date") is None
    assert nga._parse_event_time(None) is None
    return True


def test_3_classify_gap_via_poll_attempts_stall_pattern():
    """08-28 실측 재현: 96건 전부 success/200, new_rows=0이 95건,
    response_hash 고유값 1개 -> 서버측 데이터정체로 판정돼야 한다."""
    gap = {"공백시작": "2026-08-27T20:59:44", "공백종료": "2026-08-28T05:02:38"}
    rows = [_mk_poll_row("2026-08-27T21:%02d:00" % (m % 60), new_rows=0, rhash="SAME")
           for m in range(0, 96)]
    rows[0] = _mk_poll_row("2026-08-27T21:00:00", new_rows=3, rhash="SAME")  # 1건만 예외
    cls = nga.classify_gap_via_poll_attempts(gap, rows)
    assert cls is not None
    assert "서버측_데이터정체" in cls, cls
    return True


def test_4_classify_gap_via_poll_attempts_all_new_rows_normal():
    """전부 success/200이고 신규행도 매번 들어오면 - 다른 원인 재검토 필요로만 표시."""
    gap = {"공백시작": "2026-08-27T20:00:00", "공백종료": "2026-08-27T21:00:00"}
    rows = [_mk_poll_row("2026-08-27T20:%02d:00" % m, new_rows=5, rhash="H%d" % m)
           for m in range(0, 12)]
    cls = nga.classify_gap_via_poll_attempts(gap, rows)
    assert cls is not None
    assert "정상수집시도" in cls, cls
    assert "정체" not in cls
    return True


def test_5_classify_gap_via_poll_attempts_all_fail():
    gap = {"공백시작": "2026-08-27T20:00:00", "공백종료": "2026-08-27T21:00:00"}
    rows = [_mk_poll_row("2026-08-27T20:%02d:00" % m, status="error", http=500)
           for m in range(0, 5)]
    cls = nga.classify_gap_via_poll_attempts(gap, rows)
    assert cls is not None
    assert "기동후_API실패" in cls, cls
    return True


def test_6_classify_gap_via_poll_attempts_partial_fail():
    gap = {"공백시작": "2026-08-27T20:00:00", "공백종료": "2026-08-27T21:00:00"}
    rows = [_mk_poll_row("2026-08-27T20:%02d:00" % m, status="error", http=500)
           for m in range(0, 3)]
    rows += [_mk_poll_row("2026-08-27T20:%02d:00" % m, new_rows=2, rhash="H%d" % m)
            for m in range(3, 6)]
    cls = nga.classify_gap_via_poll_attempts(gap, rows)
    assert cls is not None
    assert "일부API실패" in cls, cls
    return True


def test_7_classify_gap_via_poll_attempts_no_evidence_falls_back_to_none():
    """이 구간에 poll_attempts 기록이 없으면 None(호출부가 TaskScheduler 기반으로 폴백)."""
    gap = {"공백시작": "2026-01-01T00:00:00", "공백종료": "2026-01-01T01:00:00"}
    rows = [_mk_poll_row("2026-08-27T20:00:00", new_rows=0)]  # 공백 구간 밖
    cls = nga.classify_gap_via_poll_attempts(gap, rows)
    assert cls is None
    return True


def _make_inverter_measurements_table(conn, with_power_cols: bool):
    if with_power_cols:
        conn.execute("CREATE TABLE inverter_measurements "
                    "(measurement_time TEXT, dc_power REAL, ac_power REAL)")
    else:
        conn.execute("CREATE TABLE inverter_measurements (measurement_time TEXT)")


def test_8_audit_source_prefers_poll_attempts_over_task_scheduler():
    """audit_source()가 poll_attempts 근거를 최종 '분류'로 채택하고,
    TaskScheduler 기준 분류는 별도 필드로 남겨 근거를 숨기지 않는지 확인.
    이 케이스는 발전량 컬럼이 없는(구버전 스키마 흉내) DB라 발전량0확인은
    None이 되고, 08-28 2차수정 이전처럼 보수적인 "_의심" 라벨로 남아야
    한다(발전량 실측 없이 함부로 "정상"이라 단정하지 않는다).
    임시 SQLite만 사용 - 운영 DB 미접근."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nga_test_"))
    try:
        db = tmpdir / "test.sqlite3"
        conn = sqlite3.connect(str(db))
        _make_inverter_measurements_table(conn, with_power_cols=False)
        conn.execute("CREATE TABLE poll_attempts (collected_at TEXT, status TEXT, "
                    "http_status INTEGER, new_inverter_rows INTEGER, "
                    "duplicate_inverter_rows INTEGER, response_hash TEXT, "
                    "quality_status TEXT, error_type TEXT)")
        # 활동기록: 20:00, 그다음 05:00(그 사이 공백)
        conn.execute("INSERT INTO inverter_measurements VALUES ('2026-08-27T20:00:00')")
        conn.execute("INSERT INTO inverter_measurements VALUES ('2026-08-28T05:00:00')")
        for m in range(0, 96):
            hh = 21 + (m * 5) // 60
            mm = (m * 5) % 60
            if hh >= 24:
                continue
            conn.execute(
                "INSERT INTO poll_attempts VALUES "
                "('2026-08-27T%02d:%02d:00', 'success', 200, 0, 0, 'SAME', 'ok', NULL)"
                % (hh, mm))
        conn.commit()
        conn.close()

        cfg = {
            "task_name": None,  # Task Scheduler 조회 생략(오프라인 테스트 - 실제 이벤트로그 미사용)
            "db": db, "table": "inverter_measurements", "time_col": "measurement_time",
            "poll_attempts_db": db,
            "expected_interval_min": 5,
            "backfill_possible": False, "backfill_reason": "테스트",
        }
        since = datetime(2026, 8, 27, 19, 0)
        until = datetime(2026, 8, 28, 6, 0)
        r = nga.audit_source("테스트소스", cfg, since, until)
        assert r["공백건수"] == 1, r
        g = r["공백목록"][0]
        assert g["분류_근거소스"] == "poll_attempts", g
        assert g["발전량0확인"] is None, g  # 컬럼 자체가 없어 확인 불가
        assert "서버측_데이터정체_의심" in g["분류"], g
        # TaskScheduler 기준 분류(task_name=None이라 이벤트 자체가 없음 -> 판정불가)도
        # 근거로 남아있어야 한다(숨기지 않음).
        assert "판정불가" in g["분류_TaskScheduler기준"], g
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_9_check_generation_near_zero_true_when_power_is_zero():
    """08-28 2차 수정 검증: 공백 전후 dc_power/ac_power가 전부 0이면
    check_generation_near_zero()가 True를 반환해야 한다(사용자 지적:
    야간엔 일사량이 없어 인버터가 동작할 필요가 없다는 설명의 근거)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nga_test_"))
    try:
        db = tmpdir / "test.sqlite3"
        conn = sqlite3.connect(str(db))
        _make_inverter_measurements_table(conn, with_power_cols=True)
        for t in ["2026-08-27T20:55:00", "2026-08-27T20:58:00",
                 "2026-08-28T05:03:00", "2026-08-28T05:05:00"]:
            conn.execute("INSERT INTO inverter_measurements VALUES (?, 0.0, 0.0)", (t,))
        conn.commit()
        conn.close()
        gap = {"공백시작": "2026-08-27T20:59:00", "공백종료": "2026-08-28T05:02:00"}
        result = nga.check_generation_near_zero(db, gap, margin_minutes=15)
        assert result is True, result
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_10_check_generation_near_zero_false_when_power_nonzero():
    tmpdir = Path(tempfile.mkdtemp(prefix="nga_test_"))
    try:
        db = tmpdir / "test.sqlite3"
        conn = sqlite3.connect(str(db))
        _make_inverter_measurements_table(conn, with_power_cols=True)
        conn.execute("INSERT INTO inverter_measurements VALUES (?, ?, ?)",
                    ("2026-08-27T20:58:00", 3.2, 3.0))  # 아직 발전 중
        conn.commit()
        conn.close()
        gap = {"공백시작": "2026-08-27T20:59:00", "공백종료": "2026-08-28T05:02:00"}
        result = nga.check_generation_near_zero(db, gap, margin_minutes=15)
        assert result is False, result
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_11_classify_gap_via_poll_attempts_with_confirmed_zero_generation():
    """generation_near_zero=True가 전달되면 "정상_야간유휴_확인"으로 분류되고,
    더 이상 Blockdata 문의를 암시하는 "_의심" 표현이 남지 않아야 한다."""
    gap = {"공백시작": "2026-08-27T20:59:44", "공백종료": "2026-08-28T05:02:38"}
    rows = [_mk_poll_row("2026-08-27T21:%02d:00" % (m % 60), new_rows=0, rhash="SAME")
           for m in range(0, 96)]
    cls = nga.classify_gap_via_poll_attempts(gap, rows, generation_near_zero=True)
    assert cls is not None
    assert "정상_야간유휴_확인" in cls, cls
    assert "의심" not in cls, cls
    return True


def test_12_audit_source_labels_confirmed_night_idle_when_power_data_present():
    """실제 파이프라인(audit_source)이 dc_power/ac_power 0을 확인하면
    최종 분류가 "정상_야간유휴_확인"이 되는지 종단 확인."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nga_test_"))
    try:
        db = tmpdir / "test.sqlite3"
        conn = sqlite3.connect(str(db))
        _make_inverter_measurements_table(conn, with_power_cols=True)
        conn.execute("INSERT INTO inverter_measurements VALUES "
                    "('2026-08-27T20:00:00', 0.0, 0.0)")
        conn.execute("INSERT INTO inverter_measurements VALUES "
                    "('2026-08-28T05:00:00', 0.0, 0.0)")
        conn.execute("CREATE TABLE poll_attempts (collected_at TEXT, status TEXT, "
                    "http_status INTEGER, new_inverter_rows INTEGER, "
                    "duplicate_inverter_rows INTEGER, response_hash TEXT, "
                    "quality_status TEXT, error_type NULL)")
        for m in range(0, 96):
            hh = 21 + (m * 5) // 60
            mm = (m * 5) % 60
            if hh >= 24:
                continue
            conn.execute(
                "INSERT INTO poll_attempts VALUES "
                "('2026-08-27T%02d:%02d:00', 'success', 200, 0, 0, 'SAME', 'ok', NULL)"
                % (hh, mm))
        conn.commit()
        conn.close()

        cfg = {
            "task_name": None,
            "db": db, "table": "inverter_measurements", "time_col": "measurement_time",
            "poll_attempts_db": db,
            "expected_interval_min": 5,
            "backfill_possible": False, "backfill_reason": "테스트",
        }
        since = datetime(2026, 8, 27, 19, 0)
        until = datetime(2026, 8, 28, 6, 0)
        r = nga.audit_source("테스트소스2", cfg, since, until)
        assert r["공백건수"] == 1, r
        g = r["공백목록"][0]
        assert g["발전량0확인"] is True, g
        assert "정상_야간유휴_확인" in g["분류"], g
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


TESTS = [
    test_0_classify_gap_distinguishes_query_failure_from_no_log,
    test_1_parse_event_time_dotnet_iso_ok,
    test_2_parse_event_time_none_on_garbage,
    test_3_classify_gap_via_poll_attempts_stall_pattern,
    test_4_classify_gap_via_poll_attempts_all_new_rows_normal,
    test_5_classify_gap_via_poll_attempts_all_fail,
    test_6_classify_gap_via_poll_attempts_partial_fail,
    test_7_classify_gap_via_poll_attempts_no_evidence_falls_back_to_none,
    test_8_audit_source_prefers_poll_attempts_over_task_scheduler,
    test_9_check_generation_near_zero_true_when_power_is_zero,
    test_10_check_generation_near_zero_false_when_power_nonzero,
    test_11_classify_gap_via_poll_attempts_with_confirmed_zero_generation,
    test_12_audit_source_labels_confirmed_night_idle_when_power_data_present,
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
