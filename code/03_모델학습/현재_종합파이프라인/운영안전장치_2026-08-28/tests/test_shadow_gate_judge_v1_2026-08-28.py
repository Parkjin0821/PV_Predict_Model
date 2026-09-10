# -*- coding: utf-8 -*-
"""shadow_gate_judge_v1_2026-08-28.py 오프라인 테스트.

전부 합성 metrics 딕셔너리만 사용한다 - 실제 shadow 데이터는 아직
존재하지 않으므로(공식 shadow 평가 미시작) 접근하지 않는다.
"""
from __future__ import annotations

import importlib.util as ilu
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("sgj", PARENT / "shadow_gate_judge_v1_2026-08-28.py")
sgj = ilu.module_from_spec(spec)
sys.modules["sgj"] = sgj
spec.loader.exec_module(sgj)

CFG = sgj.load_config()


def test_1_config_is_now_user_frozen():
    """08-28 사용자가 채팅에서 제시된 수치 그대로 최종동결(user_frozen)
    했다 - shadow 실측 데이터가 하나도 없는 시점에 동결(결과를 본 뒤
    기준을 바꾸지 않는다는 원칙 그대로 지킨 것)."""
    assert CFG.get("_상태") == sgj.FROZEN_STATUS_VALUE, CFG.get("_상태")
    assert sgj.is_frozen(CFG) is True
    return True


def test_2_operational_7day_all_pass():
    metrics = {
        "scheduled_run_success_rate": 0.99,
        "prediction_generation_rate": 0.97,
        "unexplained_selected_feature_nan_count": 0,
        "future_leakage_count": 0,
        "save_reload_error_count": 0,
        "inverter_sum_mismatch_count": 0,
        "collection_gap_rate": 0.02,
        "fallback_usage_rate": 0.05,
    }
    r = sgj.judge_operational_7day(metrics, CFG)
    assert r["종합판정"] == "pass", r
    assert r["참고지표_판정미반영"]["collection_gap_rate"] == 0.02
    return True


def test_3_operational_7day_fail_on_low_success_rate():
    metrics = {
        "scheduled_run_success_rate": 0.80,  # 기준 0.95 미달
        "prediction_generation_rate": 0.97,
        "unexplained_selected_feature_nan_count": 0,
        "future_leakage_count": 0,
        "save_reload_error_count": 0,
        "inverter_sum_mismatch_count": 0,
    }
    r = sgj.judge_operational_7day(metrics, CFG)
    assert r["종합판정"] == "fail", r
    assert r["세부판정"]["scheduled_run_success_rate"]["판정"] == "fail"
    return True


def test_4_operational_7day_fail_on_leakage_even_if_rates_ok():
    """future_leakage_max=0이므로 단 1건이라도 있으면 다른 지표가 전부
    좋아도 fail이어야 한다(사용자 지시: 미래 누설은 절대 금지)."""
    metrics = {
        "scheduled_run_success_rate": 0.999,
        "prediction_generation_rate": 0.999,
        "unexplained_selected_feature_nan_count": 0,
        "future_leakage_count": 1,
        "save_reload_error_count": 0,
        "inverter_sum_mismatch_count": 0,
    }
    r = sgj.judge_operational_7day(metrics, CFG)
    assert r["종합판정"] == "fail", r
    return True


def test_5_operational_7day_missing_metrics_is_판정불가_not_pass():
    """측정값이 없는데 임의로 pass 처리하면 안 된다 - 가짜 성공 금지 원칙."""
    r = sgj.judge_operational_7day({}, CFG)
    assert r["종합판정"] == "판정불가(측정값 부족)", r
    for v in r["세부판정"].values():
        assert v["판정"] == "판정불가"
    return True


def test_6_accuracy_30day_all_pass():
    metrics = {
        "skill_score": 0.05,
        "nmae_delta_pp": 0.3,
        "rmse_delta_ratio": 0.05,
        "mae_winner": "shadow", "rmse_winner": "shadow",
        "bias_flags": {"hour_of_day": False, "season": False,
                       "weather_transition": False, "per_inverter": False},
        "kpx_reference_metric": 12.3,
    }
    r = sgj.judge_accuracy_30day(metrics, CFG)
    assert r["종합판정"] == "pass", r
    assert r["세부판정"]["bias_checks"]["hour_of_day"] == "이상없음"
    return True


def test_7_accuracy_30day_hold_on_nmae_worsening():
    metrics = {
        "skill_score": 0.02, "nmae_delta_pp": 1.5,  # 기준 1.0%p 초과
        "rmse_delta_ratio": 0.02,
        "mae_winner": "shadow", "rmse_winner": "shadow",
        "bias_flags": {}, "kpx_reference_metric": None,
    }
    r = sgj.judge_accuracy_30day(metrics, CFG)
    assert r["종합판정"] == "hold", r
    assert r["세부판정"]["nmae_worsening"]["판정"] == "hold"
    return True


def test_8_accuracy_30day_hold_on_mae_rmse_split():
    metrics = {
        "skill_score": 0.02, "nmae_delta_pp": 0.1, "rmse_delta_ratio": 0.01,
        "mae_winner": "shadow", "rmse_winner": "official",  # 승자 갈림
        "bias_flags": {}, "kpx_reference_metric": None,
    }
    r = sgj.judge_accuracy_30day(metrics, CFG)
    assert r["종합판정"] == "hold", r
    return True


def test_9_accuracy_30day_fail_on_skill_score_not_positive():
    metrics = {
        "skill_score": -0.01, "nmae_delta_pp": 0.1, "rmse_delta_ratio": 0.01,
        "mae_winner": "shadow", "rmse_winner": "shadow",
        "bias_flags": {}, "kpx_reference_metric": None,
    }
    r = sgj.judge_accuracy_30day(metrics, CFG)
    assert r["종합판정"] == "fail", r
    return True


def test_10_judge_top_level_no_warning_once_frozen():
    """08-28 동결 이후엔 더 이상 draft 경고가 나오면 안 된다 - 판정 결과를
    실제 shadow 결정에 써도 되는 상태라는 뜻."""
    out = sgj.judge({"scheduled_run_success_rate": 0.99, "prediction_generation_rate": 0.99,
                     "unexplained_selected_feature_nan_count": 0, "future_leakage_count": 0,
                     "save_reload_error_count": 0, "inverter_sum_mismatch_count": 0},
                    None, cfg=CFG)
    assert out["동결_경고"] is None, out["동결_경고"]
    return True


def test_11_judge_no_metrics_at_all_still_returns_status_only():
    out = sgj.judge(None, None, cfg=CFG)
    assert "operational_7day" not in out
    assert "accuracy_30day" not in out
    assert out["기준_상태"] == sgj.FROZEN_STATUS_VALUE
    return True


TESTS = [
    test_1_config_is_now_user_frozen,
    test_2_operational_7day_all_pass,
    test_3_operational_7day_fail_on_low_success_rate,
    test_4_operational_7day_fail_on_leakage_even_if_rates_ok,
    test_5_operational_7day_missing_metrics_is_판정불가_not_pass,
    test_6_accuracy_30day_all_pass,
    test_7_accuracy_30day_hold_on_nmae_worsening,
    test_8_accuracy_30day_hold_on_mae_rmse_split,
    test_9_accuracy_30day_fail_on_skill_score_not_positive,
    test_10_judge_top_level_no_warning_once_frozen,
    test_11_judge_no_metrics_at_all_still_returns_status_only,
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
