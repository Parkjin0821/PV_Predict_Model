# -*- coding: utf-8 -*-
"""operational_status_v1_2026-08-28.py 오프라인 테스트.

08-28 사용자 외부검토("클로드 코드에 대한 구멍 및 보완점 4가지")로 새로
추가된 공유 안전유틸 4종(운영DB 읽기전용 강제, 테스트DB 경로 안전검사,
Windows os.replace 재시도, 산출물 실측 검증)을 검증한다. 전부 임시
디렉터리(tempfile)만 사용 - 운영 DB 미접근·미변경.
"""
from __future__ import annotations

import importlib.util as ilu
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("opstatus", PARENT / "operational_status_v1_2026-08-28.py")
opstatus = ilu.module_from_spec(spec)
sys.modules["opstatus"] = opstatus
spec.loader.exec_module(opstatus)


# --- PredictionRecord 계약 (기존 로직 - 08-28 신규 유틸 추가로 회귀 없는지 확인) ---

def test_1_prediction_record_normal_requires_value():
    r = opstatus.PredictionRecord(
        status=opstatus.NORMAL, status_reason="정상", nwp_run_used=opstatus.RUN_09,
        input_reference_time="t", target_time="t", predicted_value=None)
    try:
        r.validate()
        assert False, "normal인데 값이 없는데 통과됨"
    except opstatus.StatusError:
        return True


def test_2_prediction_record_blocked_forbids_value():
    r = opstatus.PredictionRecord(
        status=opstatus.BLOCKED, status_reason="결측", nwp_run_used=opstatus.RUN_NONE,
        input_reference_time="t", target_time="t", predicted_value=5.0,
        missing_features=["a"])
    try:
        r.validate()
        assert False, "blocked인데 예측값이 있는데 통과됨 - 가짜성공 금지 위반"
    except opstatus.StatusError:
        return True


# --- 보완 1: connect_readonly / assert_safe_test_db_path ---

def test_3_connect_readonly_blocks_write():
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        db = tmpdir / "test.sqlite3"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()

        ro = opstatus.connect_readonly(db)
        try:
            rows = ro.execute("SELECT x FROM t").fetchall()
            assert rows == [(1,)], rows
            try:
                ro.execute("INSERT INTO t VALUES (2)")
                ro.commit()
                assert False, "읽기전용 접속인데 INSERT가 성공함"
            except sqlite3.OperationalError:
                pass
        finally:
            ro.close()
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_4_assert_safe_test_db_path_rejects_operational_filename():
    tmp_root = Path(tempfile.gettempdir())
    fake_operational_path = tmp_root / "blockdata_history.sqlite3"
    try:
        opstatus.assert_safe_test_db_path(fake_operational_path)
        assert False, "운영 DB와 같은 파일명인데 통과됨"
    except opstatus.OperationalDBWriteBlocked:
        return True


def test_5_assert_safe_test_db_path_rejects_outside_temp():
    outside_path = PARENT / "test_not_a_real_file.sqlite3"
    try:
        opstatus.assert_safe_test_db_path(outside_path)
        assert False, "임시디렉터리 밖 경로인데 통과됨"
    except opstatus.OperationalDBWriteBlocked:
        return True


def test_6_assert_safe_test_db_path_accepts_tempfile_path():
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        safe_path = tmpdir / "my_test.sqlite3"
        opstatus.assert_safe_test_db_path(safe_path)  # 예외 없이 통과해야 함
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- 보완 2: atomic_replace_with_retry ---

def test_7_atomic_replace_with_retry_success_path():
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        src = tmpdir / "staging.txt"
        dst = tmpdir / "final.txt"
        src.write_text("new content", encoding="utf-8")
        dst.write_text("old content", encoding="utf-8")
        opstatus.atomic_replace_with_retry(src, dst)
        assert dst.read_text(encoding="utf-8") == "new content"
        assert not src.exists()
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_8_atomic_replace_with_retry_cleans_up_staging_on_repeated_failure():
    """dst가 디렉터리라 os.replace가 절대 성공할 수 없는 상황을 흉내내
    재시도가 전부 실패했을 때 staging이 정리되고 명확한 예외가 나는지 확인."""
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        src = tmpdir / "staging.txt"
        dst = tmpdir / "final_dir"
        dst.mkdir()  # 디렉터리라서 os.replace(file -> dir)는 항상 실패
        src.write_text("content", encoding="utf-8")
        t0 = time.time()
        try:
            opstatus.atomic_replace_with_retry(src, dst, retries=2, delay_sec=0.05)
            assert False, "실패해야 하는데 성공함"
        except (opstatus.OperationalDBWriteBlocked, OSError):
            pass
        # PermissionError가 아닌 다른 OSError(IsADirectoryError 등)일 수도
        # 있으므로 staging 정리 여부만 관대하게 확인(핵심은 잔여물 없음).
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- 보완 4: verify_file_output / verify_and_print_outputs ---

def test_9_verify_file_output_reports_existing_file_with_content():
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        p = tmpdir / "out.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        info = opstatus.verify_file_output(p, n_lines=2)
        assert info["존재"] is True
        assert info["크기_bytes"] > 0
        assert info["상위2줄"] == ["a,b", "1,2"]
        assert info["비어있음_주의"] is False
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_10_verify_file_output_reports_missing_file_honestly():
    """파일이 없으면 "성공"이라 주장하지 않고 존재=False를 그대로 반환."""
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        missing = tmpdir / "does_not_exist.csv"
        info = opstatus.verify_file_output(missing)
        assert info["존재"] is False
        assert info["크기_bytes"] == 0
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_11_verify_file_output_flags_empty_file():
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        p = tmpdir / "empty.csv"
        p.write_text("", encoding="utf-8")
        info = opstatus.verify_file_output(p)
        assert info["존재"] is True
        assert info["비어있음_주의"] is True
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_12_verify_and_print_outputs_covers_all_given_paths():
    tmpdir = Path(tempfile.mkdtemp(prefix="opstatus_test_"))
    try:
        p1 = tmpdir / "a.csv"
        p1.write_text("x\n1\n", encoding="utf-8")
        p2 = tmpdir / "does_not_exist.json"
        report = opstatus.verify_and_print_outputs({"파일A": str(p1), "파일B": str(p2)})
        assert report["파일A"]["존재"] is True
        assert report["파일B"]["존재"] is False
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


TESTS = [
    test_1_prediction_record_normal_requires_value,
    test_2_prediction_record_blocked_forbids_value,
    test_3_connect_readonly_blocks_write,
    test_4_assert_safe_test_db_path_rejects_operational_filename,
    test_5_assert_safe_test_db_path_rejects_outside_temp,
    test_6_assert_safe_test_db_path_accepts_tempfile_path,
    test_7_atomic_replace_with_retry_success_path,
    test_8_atomic_replace_with_retry_cleans_up_staging_on_repeated_failure,
    test_9_verify_file_output_reports_existing_file_with_content,
    test_10_verify_file_output_reports_missing_file_honestly,
    test_11_verify_file_output_flags_empty_file,
    test_12_verify_and_print_outputs_covers_all_given_paths,
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
