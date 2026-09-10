# -*- coding: utf-8 -*-
"""verify_night_gate_wiring_v1_2026-08-28.py의 검사함수 자체를 합성
행으로 검증한다(운영 DB 미접근) - 08-31 실전 검증 전에 검사 로직 자체가
정상 동작하는지 확인."""
from __future__ import annotations

import importlib.util as ilu
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("vng", PARENT / "verify_night_gate_wiring_v1_2026-08-28.py")
vng = ilu.module_from_spec(spec)
sys.modules["vng"] = vng
spec.loader.exec_module(vng)


def test_1_check_1_passes_for_valid_lag_names():
    rows = [{"id": 1, "reason": "야간0채움: 발전출력_1시간전_kW,발전출력_6시간이동평균_kW"}]
    r = vng.check_1_only_lag_features(rows)
    assert r["통과"] is True, r
    return True


def test_2_check_1_catches_contaminated_feature():
    """야간0채움 목록에 발전출력 lag가 아닌 특성이 섞이면 위반으로 잡혀야 한다."""
    rows = [{"id": 2, "reason": "야간0채움: 발전출력_1시간전_kW,DSWRF"}]
    r = vng.check_1_only_lag_features(rows)
    assert r["통과"] is False, r
    assert any(b["특성"] == "DSWRF" for b in r["위반건"]), r
    return True


def test_3_check_4_passes_for_well_formed_reason():
    rows = [{"id": 3, "reason": "야간0채움: 발전출력_1시간전_kW"}]
    r = vng.check_4_reason_present_and_specific(rows)
    assert r["통과"] is True, r
    return True


def test_4_check_4_catches_empty_feature_list():
    rows = [{"id": 4, "reason": "야간0채움: "}]
    r = vng.check_4_reason_present_and_specific(rows)
    assert r["통과"] is False, r
    return True


def test_5_run_handles_empty_db_gracefully():
    """실행 결과 자체가 크래시 없이 항상 dict를 반환하는지(행이 0건이어도)."""
    result = vng.run()
    assert isinstance(result, dict)
    assert "야간0채움_성공행_건수" in result
    return True


TESTS = [
    test_1_check_1_passes_for_valid_lag_names,
    test_2_check_1_catches_contaminated_feature,
    test_3_check_4_passes_for_well_formed_reason,
    test_4_check_4_catches_empty_feature_list,
    test_5_run_handles_empty_db_gracefully,
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
