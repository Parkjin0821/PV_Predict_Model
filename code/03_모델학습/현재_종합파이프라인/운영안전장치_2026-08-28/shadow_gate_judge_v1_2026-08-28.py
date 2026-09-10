# -*- coding: utf-8 -*-
"""Shadow 사전등록 판정기 (사용자 지시 5번).

config/shadow_gate.json에 사전등록된 기준(7일 운영안정성 게이트 + 30일
정확도 게이트)을 읽어 "측정값 딕셔너리"에 대해 pass/hold/fail을 계산한다.

## 중요: 이 파일은 실제 shadow 데이터를 조회하지 않는다
judge_* 함수는 순수 함수다 - 측정값을 호출부(또는 CLI의 --metrics-json)가
전달해야 계산이 된다. shadow 평가가 아직 시작되지 않았으므로(사용자 지시:
"공식 shadow 평가는 아직 시작하지 말 것") 이 스크립트 자체가 운영 DB를
읽어 측정값을 만들어내는 로직은 이번 작업 범위에 넣지 않았다 - 그건 실제
shadow 데이터가 쌓인 뒤 별도로 만들 집계 함수의 몫이다.

## 동결 안내
config/shadow_gate.json의 "_상태"가 "draft_pending_user_freeze"인 동안은
이 판정기가 "초안 기준으로 계산한 결과"라는 경고를 항상 함께 출력한다.
사용자가 최종 검토 후 그 값을 "user_frozen"으로 바꾸기 전까지는 이 결과를
실제 shadow 통과/보류/실패 결정에 쓰지 않는다(사용자 지시: "결과를 본
후에는 기준을 바꾸지 않는다" - 값을 보기 전에 반드시 동결).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config" / "shadow_gate.json"

DRAFT_STATUS_VALUE = "draft_pending_user_freeze"
FROZEN_STATUS_VALUE = "user_frozen"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_frozen(cfg: dict) -> bool:
    return cfg.get("_상태") == FROZEN_STATUS_VALUE


def judge_operational_7day(metrics: dict[str, Any], cfg: dict) -> dict:
    """7일 운영안정성 게이트. metrics 키는 아래 6개(없으면 그 항목만 판정불가):
    scheduled_run_success_rate, prediction_generation_rate,
    unexplained_selected_feature_nan_count, future_leakage_count,
    save_reload_error_count, inverter_sum_mismatch_count
    info_only_metrics(collection_gap_rate, fallback_usage_rate)는 판정에
    쓰지 않고 그대로 노출만 한다(사용자 지시: 숨기지 않는다).
    """
    crit = cfg["operational_7day"]
    results: dict[str, dict] = {}

    def check_min(metric_key: str, cfg_key: str) -> None:
        val = metrics.get(metric_key)
        if val is None:
            results[metric_key] = {"판정": "판정불가", "사유": "측정값 없음"}
            return
        min_v = crit[cfg_key]
        results[metric_key] = {
            "측정값": val, "기준": ">= %.4f" % min_v,
            "판정": "pass" if val >= min_v else "fail",
        }

    def check_max(metric_key: str, cfg_key: str) -> None:
        val = metrics.get(metric_key)
        if val is None:
            results[metric_key] = {"판정": "판정불가", "사유": "측정값 없음"}
            return
        max_v = crit[cfg_key]
        results[metric_key] = {
            "측정값": val, "기준": "<= %s" % max_v,
            "판정": "pass" if val <= max_v else "fail",
        }

    check_min("scheduled_run_success_rate", "scheduled_run_success_rate_min")
    check_min("prediction_generation_rate", "prediction_generation_rate_min")
    check_max("unexplained_selected_feature_nan_count", "unexplained_selected_feature_nan_max")
    check_max("future_leakage_count", "future_leakage_max")
    check_max("save_reload_error_count", "save_reload_error_max")
    check_max("inverter_sum_mismatch_count", "inverter_sum_mismatch_max")

    info: dict[str, Any] = {}
    for m in crit.get("info_only_metrics", []):
        info[m] = metrics.get(m)

    verdicts = [v["판정"] for v in results.values()]
    if any(v == "fail" for v in verdicts):
        overall = "fail"
    elif any(v == "판정불가" for v in verdicts):
        overall = "판정불가(측정값 부족)"
    else:
        overall = "pass"

    return {
        "세부판정": results,
        "참고지표_판정미반영": info,
        "종합판정": overall,
        "기준_상태": cfg.get("_상태"),
    }


def judge_accuracy_30day(metrics: dict[str, Any], cfg: dict) -> dict:
    """30일 정확도 게이트(잠정 - 기존 공식 5계절 백테스트를 대체하지 않음).

    metrics 키:
    skill_score, nmae_delta_pp(shadow-공식, %p), rmse_delta_ratio(상대비율),
    mae_winner/rmse_winner("official" 또는 "shadow"),
    bias_flags: {hour_of_day/season/weather_transition/per_inverter: bool},
    kpx_reference_metric(참고용, 판정 미반영)
    """
    crit = cfg["accuracy_30day"]
    results: dict[str, dict] = {}

    skill = metrics.get("skill_score")
    if skill is None:
        results["skill_score"] = {"판정": "판정불가", "사유": "측정값 없음"}
    else:
        min_v = crit["skill_score_min_exclusive"]
        results["skill_score"] = {
            "측정값": skill, "기준": "> %.4f" % min_v,
            "판정": "pass" if skill > min_v else "fail",
        }

    nmae_delta = metrics.get("nmae_delta_pp")
    if nmae_delta is None:
        results["nmae_worsening"] = {"판정": "판정불가", "사유": "측정값 없음"}
    else:
        thr = crit["nmae_worsening_hold_above_pp"]
        results["nmae_worsening"] = {
            "측정값_%p": nmae_delta, "기준": "hold if > %.2f%%p" % thr,
            "판정": "hold" if nmae_delta > thr else "pass",
        }

    rmse_delta = metrics.get("rmse_delta_ratio")
    if rmse_delta is None:
        results["rmse_worsening"] = {"판정": "판정불가", "사유": "측정값 없음"}
    else:
        thr = crit["rmse_worsening_hold_above_ratio"]
        results["rmse_worsening"] = {
            "측정값_비율": rmse_delta, "기준": "hold if > %.2f" % thr,
            "판정": "hold" if rmse_delta > thr else "pass",
        }

    mae_w, rmse_w = metrics.get("mae_winner"), metrics.get("rmse_winner")
    if mae_w is not None and rmse_w is not None:
        split = mae_w != rmse_w
        results["mae_rmse_split"] = {
            "mae_승자": mae_w, "rmse_승자": rmse_w,
            "판정": "hold" if split else "pass",
        }
    else:
        results["mae_rmse_split"] = {"판정": "판정불가", "사유": "측정값 없음"}

    bias_flags = metrics.get("bias_flags") or {}
    bias_out: dict[str, str] = {}
    for b in crit.get("bias_checks", []):
        flag = bias_flags.get(b)
        bias_out[b] = "판정불가" if flag is None else (
            "요주의_재검토필요" if flag else "이상없음")
    results["bias_checks"] = bias_out

    results["kpx_참고"] = {
        "값": metrics.get("kpx_reference_metric"),
        "주의": "단독 219kW 발전소의 직접 참여성능처럼 표현하지 않는다(참고전용, 판정 미반영)",
    }

    scored_keys = ("skill_score", "nmae_worsening", "rmse_worsening", "mae_rmse_split")
    verdicts = [results[k]["판정"] for k in scored_keys]
    if any(v == "fail" for v in verdicts):
        overall = "fail"
    elif any(v == "hold" for v in verdicts):
        overall = "hold"
    elif any(v == "판정불가" for v in verdicts):
        overall = "판정불가(측정값 부족)"
    else:
        overall = "pass"

    return {
        "세부판정": results,
        "종합판정": overall,
        "기준_상태": cfg.get("_상태"),
    }


def judge(metrics_7day: dict[str, Any] | None, metrics_30day: dict[str, Any] | None,
         cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    out: dict[str, Any] = {
        "기준_상태": cfg.get("_상태"),
        "동결_경고": None if is_frozen(cfg) else (
            "★기준이 아직 사용자 최종동결(user_frozen) 전 초안(%s)이다 - "
            "이 판정 결과를 실제 shadow 통과/보류/실패 결정에 쓰지 말 것★"
            % cfg.get("_상태")),
    }
    if metrics_7day is not None:
        out["operational_7day"] = judge_operational_7day(metrics_7day, cfg)
    if metrics_30day is not None:
        out["accuracy_30day"] = judge_accuracy_30day(metrics_30day, cfg)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Shadow 사전등록 판정기 - metrics JSON 파일(오프라인, 이미 계산된 값)을 "
                   "입력받아 pass/hold/fail만 계산한다. 운영 DB를 직접 읽지 않는다.")
    ap.add_argument("--metrics-7day-json", type=Path, default=None,
                    help="operational_7day judge_* 함수 입력용 metrics JSON 파일 경로")
    ap.add_argument("--metrics-30day-json", type=Path, default=None,
                    help="accuracy_30day judge_* 함수 입력용 metrics JSON 파일 경로")
    args = ap.parse_args()

    m7 = json.loads(args.metrics_7day_json.read_text(encoding="utf-8")) \
        if args.metrics_7day_json else None
    m30 = json.loads(args.metrics_30day_json.read_text(encoding="utf-8")) \
        if args.metrics_30day_json else None

    if m7 is None and m30 is None:
        print("사용법: --metrics-7day-json <path> 그리고/또는 --metrics-30day-json <path>")
        print("(측정값 JSON이 없으면 이 스크립트는 아무 판정도 하지 않는다 - "
             "운영 DB를 대신 조회하지 않는다)")
        return 2

    result = judge(m7, m30)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
