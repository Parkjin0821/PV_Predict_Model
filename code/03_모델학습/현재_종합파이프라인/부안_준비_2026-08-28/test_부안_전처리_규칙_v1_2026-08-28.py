# -*- coding: utf-8 -*-
"""부안_전처리_규칙_v1_2026-08-28.json 사전동결 규칙 자체 검증(오프라인,
설정 파일 읽기만 - 원본 Excel·운영 DB 접근 없음)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "config" / "부안_전처리_규칙_v1_2026-08-28.json"


def load():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def test_1_json_valid_and_required_keys():
    cfg = load()
    required = ["native_grid_minutes", "timestamp_alignment", "interpolation",
               "comm_error_policy", "status_column_policy",
               "plant_level_reconciliation", "excel_column_mapping",
               "promote_to_official"]
    missing = [k for k in required if k not in cfg]
    assert not missing, missing
    return True


def test_2_interpolation_limit_consistent():
    cfg = load()
    interp = cfg["interpolation"]
    assert interp["max_steps"] * interp["step_minutes"] == interp["max_gap_minutes"], interp
    assert interp["max_gap_minutes"] == 10, interp
    return True


def test_3_capacity_matches_8_times_125():
    cfg = load()
    rec = cfg["plant_level_reconciliation"]
    assert rec["expected_inverter_count"] == 8, rec
    total = sum(rec["inverter_capacity_kw"].values())
    assert abs(total - rec["capacity_sum_ac_kw"]) < 1e-9, total
    assert abs(total - 1000.0) < 1e-9, total  # 125kW x 8 = 1000kW(사용자 확인값)
    return True


def test_4_not_promoted_to_official():
    cfg = load()
    assert cfg["promote_to_official"] is False, cfg["promote_to_official"]
    return True


def test_5_waiting_status_not_prematurely_decided():
    """'waiting=야간'을 아직 확정하지 않았다는 걸 코드로도 강제 - 이후 누가
    실수로 이 필드를 'night_flag와 동일'로 바꾸면 이 테스트가 먼저 깨진다."""
    cfg = load()
    policy = cfg["status_column_policy"]["policy_waiting"]
    assert "미확정" in policy, policy
    return True


def test_6_duplicate_policy_never_silently_averages():
    cfg = load()
    policy = cfg["timestamp_alignment"]["duplicate_policy"]
    assert "average" not in policy.lower() and "mean" not in policy.lower(), policy
    return True


def test_7_defect_period_frozen_and_excludes_38_days():
    """08-28 사용자 확정: 04-15~05-25 결함구간 제외 지시 - 정밀 재확인한
    경계(04-15~05-22, end_exclusive=05-23)로 동결됐는지, half-open
    구간이 정확히 38일을 가리키는지 확인."""
    import datetime
    cfg = load()
    d = cfg["defect_period"]
    assert d["policy"] == "exclude_from_training_and_test", d
    start = datetime.date.fromisoformat(d["start_inclusive"])
    end = datetime.date.fromisoformat(d["end_exclusive"])
    assert (end - start).days == d["excluded_days_count"] == 38, (start, end, d)
    return True


def test_8_defect_period_excludes_isolated_good_days_too():
    """구간 내 개별적으로는 valid였던 날짜(04-17 등)도 결함구간 전체제외
    원칙에 따라 함께 제외 목록에 있어야 한다 - 예외적으로 봐주지 않음."""
    cfg = load()
    isolated = cfg["defect_period"]["isolated_valid_days_within_window_also_excluded"]
    assert "2026-04-17" in isolated and "2026-05-15" in isolated, isolated
    return True


TESTS = [
    test_1_json_valid_and_required_keys,
    test_2_interpolation_limit_consistent,
    test_3_capacity_matches_8_times_125,
    test_4_not_promoted_to_official,
    test_5_waiting_status_not_prematurely_decided,
    test_6_duplicate_policy_never_silently_averages,
    test_7_defect_period_frozen_and_excludes_38_days,
    test_8_defect_period_excludes_isolated_good_days_too,
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
