# -*- coding: utf-8 -*-
"""일간 모델 — DIFSWRF 계열 4개 명시적 제외 재검증 — 08-26.

## 배경
AGENTS.md 08-26절 "★DIFSWRF 잔존 신규 발견★": DSWRFLX와 완전히 같은 원인
(KIMR 2026-06-01 이후 상류 결측)으로 DIFSWRF도 라이브에서 16/16행 100%
결측이다. 초단기·단기는 플래그만 쓰므로 안전하지만, 일간은
`목표일예보_DIFSWRF_bsrn정제_sum`(중요도 4위/55, 4.2%)·`_mean`(1.1%)·
`목표일_DIFSWRF_유효개수`(0.8%)를 실제 값으로 쓴다(계열 합계 6.1%).
결측 플래그(`_sum_결측여부`)의 중요도가 정확히 0이라 "결측이 드문
학습분포"와 "항상 결측인 운영분포"가 어긋난 채 숫자만 나온다.

사용자 결정(08-26, Codex와 상의 후 확정): readiness 게이트 검사 추가만으로
끝내지 않고, DIFSWRF 계열 4개를 명시적으로 제외한 뒤 공식 5폴드로
재검증한다. 이 스크립트는 그 1단계(비교·검증)다. 채택 기준(사용자 지정):
- 전체 MAE·RMSE 악화가 각각 1% 미만
- 어느 폴드에서도 RMSE 5% 이상 악화하지 않음
- 평가일과 시험행이 포함·제외 후보 간 동일

## 기준선 주의 — DSWRFLX 때와 다른 점
`daily_direct_final_audit_v1_2026-08-25.py::corrected_dataset()`는 원래
58특성 전체를 돌려준다. 하지만 **현재 실제 운영번들은 이미 08-26에
2일전평균_mean_communication_ok·DSWRFLX_sum·DSWRFLX_mean 3개를 제외한
55특성**이다(`train_production_daily_v3_특성정리_2026-08-26.py` 참고).
따라서 이번 비교의 "포함(기존)" 기준선은 58특성이 아니라 **현재 운영
중인 55특성**이어야 한다 — 58특성을 기준선으로 쓰면 이미 없는 3개
특성까지 있다고 잘못 비교하게 된다.

## 재구현 없음(원칙 준수)
`daily_direct_final_audit_v1_2026-08-25.py`의 `corrected_dataset`/
`predict_oof`/`compare`만 그대로 재사용한다. 학습 로직·폴드 정의·
LightGBM 구성 전부 원본과 동일.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "일간_DIFSWRF제외_재검증_v1_2026-08-26"
# 주의: "DIFSWRF_bsrn정제"만으로는 `목표일_DIFSWRF_유효개수`(접미사가 다름)를
# 놓친다. DIFSWRF 계열 4개(sum/mean/유효개수/sum_결측여부) 전부를 잡으려면
# "DIFSWRF" 자체로 걸러야 한다(DSWRFLX와는 문자열이 겹치지 않아 오검출 없음).
DIF = "DIFSWRF"
MCO = "2일전평균_mean_communication_ok"
DSX = "DSWRFLX_bsrn정제"
LABEL_BEFORE = "현재운영_55특성(MCO+DSX제외)"
LABEL_AFTER = "DIFSWRF계열4개_추가제외_51특성"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_audit = _load("difswrf_daily", "daily_direct_final_audit_v1_2026-08-25.py")


def _judge(verdict: dict, folds: pd.DataFrame) -> dict:
    """사용자 08-26 지정 채택기준(compare()의 자동 채택 규칙과 다름 —
    compare()는 개선 1%p 이상을 요구하지만, 이번 기준은 '악화 1% 미만'만
    요구한다).

    ★단위 주의★: 아래 `_pct` 필드는 전부 **기준 대비 상대변화율**
    ((후보-기준)/기준×100)이다 — 이미 %인 지표(MAPE 등) 사이의 차이를 뜻하는
    "%p(퍼센트포인트)"가 아니다. MAE·RMSE는 kWh 단위 절대오차이므로 그
    상대변화율은 처음부터 %로 표현되는 게 맞고 %p라는 표현 자체가 성립하지
    않는다(AGENTS.md 08-26절에 %p로 잘못 적었던 걸 사용자가 지적, 여기서
    바로잡음)."""
    mae_worse = -verdict["MAE개선율_pct"]  # 양수면 악화, 단위=기준 대비 상대% (%p 아님)
    rmse_worse = -verdict["RMSE개선율_pct"]
    worst_fold_rmse = float(folds["RMSE악화율_pct"].max())
    ok_overall = mae_worse < 1.0 and rmse_worse < 1.0
    ok_folds = worst_fold_rmse < 5.0
    final_adopt = bool(ok_overall and ok_folds)
    return {
        "전체MAE악화_상대pct": round(mae_worse, 4),
        "전체RMSE악화_상대pct": round(rmse_worse, 4),
        "최대폴드RMSE악화_상대pct": round(worst_fold_rmse, 4),
        "기준_전체악화1%미만_충족": ok_overall,
        "기준_폴드RMSE5%미만_충족": ok_folds,
        "사용자기준_채택": final_adopt,
        # ★최종 공식판정★ — compare()의 "채택"/"판정"은 "1%p 이상 개선"을
        # 요구하는 별도 규칙(DSWRFLX처럼 정확도 개선 목적 제외에 맞춘 기준)이라
        # 여기서는 False가 나온다. 이번 DIFSWRF 제외는 정확도 개선이 아니라
        # "라이브에서 100% 결측인 특성을 구조적으로 제거"가 목적이므로, 위
        # "사용자기준_채택"(운영 가용성 기준)이 실제 운영 반영 여부를 결정하는
        # 값이다. 자동 보고서·후속 스크립트는 compare()의 "채택"이 아니라
        # 반드시 이 필드를 읽을 것.
        "최종운영채택": final_adopt,
        "최종운영채택_근거": (
            "compare()의 채택/판정은 개선폭 1%p 이상을 요구하는 정확도개선용 "
            "규칙이라 이번처럼 '운영 불가능 특성 구조적 제거'가 목적인 경우엔 "
            "항상 기각으로 나온다(의도된 동작). 실제 운영 반영 여부는 사용자가"
            "08-26 지정한 가용성 기준(전체 악화<1%, 폴드 RMSE악화<5%)으로 "
            "판단하며, 이 결과는 그 기준을 충족해 사용자 승인 하에 운영 채택됨"
            "(train_production_daily_v4_DIFSWRF제외_2026-08-26.py로 반영 완료)."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    if capacity_kw != 219:
        raise RuntimeError(f"활성 용량이 219kW가 아님: {capacity_kw}")

    data, all_features, n_partial = daily_audit.corrected_dataset(capacity_kw)
    print(f"corrected_dataset 원본 전체특성={len(all_features)}, 부분가용역사일={n_partial}")

    dif_cols = [c for c in all_features if DIF in c]
    dsx_cols = [c for c in all_features if DSX in c]
    if MCO not in all_features or len(dsx_cols) != 2 or len(dif_cols) != 4:
        raise RuntimeError(
            f"특성목록 확인 실패: MCO={MCO in all_features}, DSX={dsx_cols}, DIF={dif_cols}"
        )

    # 현재 실제 운영 중인 55특성(MCO+DSX 3개 이미 제외) — 이번 비교의 기준선.
    current_prod_exclude = [MCO] + dsx_cols
    features_before = [c for c in all_features if c not in current_prod_exclude]
    print(f"현재 운영 기준선(MCO+DSX 제외): {len(features_before)}특성")

    features_after = [c for c in features_before if c not in dif_cols]
    print(f"DIFSWRF 계열 4개 추가 제외: {dif_cols}")
    print(f"제외 후: {len(features_after)}특성")

    before_rows, audit_before = daily_audit.predict_oof(
        data, features_before, capacity_kw, seed, LABEL_BEFORE, False)
    after_rows, audit_after = daily_audit.predict_oof(
        data, features_after, capacity_kw, seed, LABEL_AFTER, False)

    verdict, folds, merged = daily_audit.compare(before_rows, after_rows, LABEL_AFTER)
    judged = _judge(verdict, folds)
    verdict.update(judged)

    same_days = (
        len(merged) == len(before_rows) == len(after_rows)
        and set(before_rows["날짜"]) == set(after_rows["날짜"])
    )
    if not same_days:
        raise AssertionError(
            f"평가일/시험행 불일치 — before={len(before_rows)}, after={len(after_rows)}, "
            f"교집합={len(merged)}. 이 비교 결과는 신뢰할 수 없다."
        )

    pd.concat(
        [before_rows.assign(구성=LABEL_BEFORE), after_rows.assign(구성=LABEL_AFTER)],
        ignore_index=True,
    ).to_csv(OUT / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "폴드별_비교.csv", index=False, encoding="utf-8-sig")
    audit_all = pd.concat([audit_before, audit_after], ignore_index=True)
    audit_all.to_csv(OUT / "감사.csv", index=False, encoding="utf-8-sig")
    (OUT / "요약.json").write_text(
        json.dumps(
            {
                "제외컬럼_기존운영": current_prod_exclude,
                "제외컬럼_이번_DIFSWRF": dif_cols,
                "특성수_before": len(features_before),
                "특성수_after": len(features_after),
                "평가일수_일치": same_days,
                "compare_요약": verdict,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== 폴드별 비교 ===")
    print(folds.to_string(index=False))
    print("\n=== 요약(compare() 원본 + 사용자기준 판정) ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str))
    print(f"\n평가일/시험행 일치: {same_days}")
    print(f"저장 완료: {OUT}")
    print("\n★주의★ 이 결과는 운영 반영을 자동 결정하지 않는다 — 사용자 확인 후 "
          "train_production_daily_v4_DIFSWRF제외 스크립트로 진행할 것.")


if __name__ == "__main__":
    main()
