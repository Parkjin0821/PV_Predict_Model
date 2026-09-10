# -*- coding: utf-8 -*-
"""inverter_disaggregation_v1_2026-08-28.py 오프라인 테스트.

합성 total_kw/weights만 사용한다 - 실제 예측값 조회 없음.
"""
from __future__ import annotations

import importlib.util as ilu
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("idis", PARENT / "inverter_disaggregation_v1_2026-08-28.py")
idis = ilu.module_from_spec(spec)
sys.modules["idis"] = idis
spec.loader.exec_module(idis)

CFG = idis.load_config()


def test_1_config_not_promoted_to_official():
    assert CFG.get("promote_to_official") is False
    return True


def test_2_method_A_selected_for_초단기_3h():
    m = idis.select_method("초단기", "3h", CFG)
    assert m == "A_정격용량비례", m
    return True


def test_3_method_C_selected_for_초단기_1h():
    m = idis.select_method("초단기", "1h", CFG)
    assert m == "C_LightGBM비중", m
    return True


def test_4_일간_D_plus_1_now_adopted_as_A():
    """08-28 정식 CV(inverter_disaggregation_daily_cv_v1_2026-08-28.py) 결과
    A_정격용량비례 채택 - 더 이상 excluded가 아니다(과거엔 미검증으로
    제외됐었음, 이제는 select_method가 정상적으로 A를 반환해야 한다)."""
    m = idis.select_method("일간", "D+1", CFG)
    assert m == "A_정격용량비례", m
    assert "일간_D+1" not in CFG.get("excluded", {}), CFG.get("excluded")
    r = idis.disaggregate(219.0, "일간", "D+1", CFG)
    assert abs(r["합계_kW"] - 219.0) < 1e-6, r
    return True


def test_5_disaggregate_A_sum_preserved():
    r = idis.disaggregate(100.0, "초단기", "3h", CFG)
    assert r["방법"] == "A_정격용량비례"
    assert abs(r["합계_kW"] - 100.0) < 1e-6, r
    # 정격비례 배분 확인: 인버터1(50kW)은 219kW중 50/219 비율
    assert abs(r["인버터별_kW"]["1"] - 100.0 * 50.0 / 219.0) < 1e-6
    return True


def test_6_disaggregate_C_sum_preserved_with_explicit_weights():
    weights = {"1": 0.25, "2": 0.25, "3": 0.15, "4": 0.15, "5": 0.20}
    r = idis.disaggregate(80.0, "단기", "1h", CFG, weights_C=weights)
    assert r["방법"] == "C_LightGBM비중"
    assert abs(r["합계_kW"] - 80.0) < 1e-6, r
    assert abs(r["인버터별_kW"]["1"] - 20.0) < 1e-6
    return True


def test_7_disaggregate_C_without_weights_raises():
    """가중치 없이 C방법을 호출하면 임의생성하지 않고 예외를 내야 한다
    (사용자 지시: 공식 가중치 번들 없이 만들어내지 않는다)."""
    try:
        idis.disaggregate(80.0, "단기", "1h", CFG)
        assert False, "weights_C 없이도 통과됨 - 임의생성 금지 원칙 위반"
    except ValueError:
        return True


def test_8_disaggregate_C_bad_weight_sum_raises():
    weights = {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.0, "5": 0.0}  # 합계 1.5
    try:
        idis.disaggregate_C_weighted(80.0, weights)
        assert False, "가중치 합계가 1.0이 아닌데도 통과됨"
    except idis.DisaggregationError:
        return True


def test_9_assert_sum_preserved_raises_when_mismatched():
    per_inv = {"1": 10.0, "2": 10.0, "3": 10.0, "4": 10.0, "5": 10.0}  # 합계 50
    try:
        idis.assert_sum_preserved(60.0, per_inv, tolerance_kw=1e-6)
        assert False, "불일치인데 예외가 안 남"
    except idis.SumMismatchError:
        return True


def test_10_all_selected_horizons_produce_valid_disaggregation():
    """config selection에 등록된 모든 수평이 실제로 동작하는지 일괄 확인."""
    for key, method in CFG["selection"].items():
        tier, horizon = key.rsplit("_", 1)
        if method.startswith("C_"):
            weights = {inv: 1.0 / len(CFG["inverter_capacity_kw"])
                      for inv in CFG["inverter_capacity_kw"]}
            r = idis.disaggregate(219.0, tier, horizon, CFG, weights_C=weights)
        else:
            r = idis.disaggregate(219.0, tier, horizon, CFG)
        assert abs(r["합계_kW"] - 219.0) < 1e-6, (key, r)
    return True


TESTS = [
    test_1_config_not_promoted_to_official,
    test_2_method_A_selected_for_초단기_3h,
    test_3_method_C_selected_for_초단기_1h,
    test_4_일간_D_plus_1_now_adopted_as_A,
    test_5_disaggregate_A_sum_preserved,
    test_6_disaggregate_C_sum_preserved_with_explicit_weights,
    test_7_disaggregate_C_without_weights_raises,
    test_8_disaggregate_C_bad_weight_sum_raises,
    test_9_assert_sum_preserved_raises_when_mismatched,
    test_10_all_selected_horizons_produce_valid_disaggregation,
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
