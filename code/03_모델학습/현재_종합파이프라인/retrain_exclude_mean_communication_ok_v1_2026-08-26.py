# -*- coding: utf-8 -*-
"""`2일전평균_mean_communication_ok` 후보특성 제외 재검증 — 08-26.

## 배경
`shadow_readiness_일간_v1_2026-08-26.py`가 일간 shadow를 `blocked`로
판정한 핵심 원인 — 공식 일간 58특성 번들이 `2일전평균_
mean_communication_ok`(장비 통신상태, Blockdata 인버터 텔레메트리에서
2일 지연 평균한 값)를 실제 선택특성으로 쓰는데, 라이브 Blockdata API
스키마에는 이 필드 자체가 원천적으로 없다(AGENTS.md 08-26,
`live_feature_assembler_단기_v1_2026-08-26.py`의 `_equipment_5min()`
주석에도 "Blockdata 실시간 스키마에는 인버터 온도·통신상태 필드가 없다"
로 이미 기록돼 있다). 임의값으로 채우면 안 되므로, 이 특성을 후보에서
빼고 기존 공식 5폴드로 성능이 유지되는지 재검증한다.

DSWRFLX 건과는 **원인이 완전히 다르다** — DSWRFLX는 상류(기상청 KIMR)
데이터 자체가 사라진 것이고, 이건 **애초에 Blockdata 실시간 API가 그
필드를 제공하지 않는 구조적 제약**이다(재개해도 안 생김). 그래서 별도
스크립트로 분리했다.

## 범위(중요 — 단기·초단기는 대상 아님)
AGENTS.md 08-26 정정 메모: "8개 운영모델 중 `mean_communication_ok`를
실제로 쓰는 건 일간뿐이다(단기 3개는 이 필드를 애초에 후보로 안 씀)."
그래서 이 스크립트는 **일간만** 다룬다 — `retrain_exclude_dswrflx_v1_
2026-08-26.py`처럼 초단기·단기용 몽키패치가 필요 없다.

## 재구현 없음(원칙 준수)
`daily_direct_final_audit_v1_2026-08-25.py`의 `corrected_dataset()`
(현재 공식 특성공학)로 데이터를 만들고, 이 컬럼 하나만 특성 목록에서
뺀 뒤 같은 파일의 `predict_oof()`·`compare()`를 그대로 재사용한다
(`retrain_exclude_dswrflx_v1_2026-08-26.py`와 동일 패턴 — 이번엔
일간만 다루므로 그 스크립트의 초단기·단기 부분은 그대로 두고 이
파일을 별도로 둔다).

## 이 스크립트가 하지 않는 것
- **운영모델(운영모델_v2)을 덮어쓰지 않는다.** 성능 비교표만 만든다.
- `predict_oof(save_models=False)` 기본값이라 joblib을 저장하지 않는다.

## Codex 실행 시 참고
- 출력: `outputs/mean_communication_ok_제외_재검증_v1_2026-08-26/`
  - `일간_동일행_예측정답.csv`
  - `일간_성능비교.csv` (MAE/RMSE/WAPE, 포함 vs 제외)
  - `일간_폴드별_비교.csv` (`daily_audit.compare()` 원본 출력 — 이미
    `validate="one_to_one"` 병합이라 동일행 비교가 기본 보장됨)
  - `감사로그.json`
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
OUT = ROOT / "outputs" / "mean_communication_ok_제외_재검증_v1_2026-08-26"
TARGET_COL = "2일전평균_mean_communication_ok"
LABEL_BEFORE = "MCO_포함(기존)"
LABEL_AFTER = f"{TARGET_COL}_제외"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_audit = _load("mco_excl_daily", "daily_direct_final_audit_v1_2026-08-25.py")


def run_daily_both(capacity_kw: float, seed: int):
    """corrected_dataset()으로 동일 특성공학을 만들고, TARGET_COL 하나만
    빼서 predict_oof()·compare()를 그대로 재사용한다(재구현 아님)."""
    data, features, n_partial = daily_audit.corrected_dataset(capacity_kw)
    if TARGET_COL not in features:
        raise RuntimeError(
            f"일간 특성목록에서 {TARGET_COL}을 못 찾았다 — 컬럼명이 바뀌었거나 "
            "가정이 틀렸을 수 있다(예: build_daily_dataset_v5의 EQUIPMENT_COLS "
            "구성이 바뀜). 중단."
        )
    print(f"[일간] 부분가용일 {n_partial}일, 전체특성 {len(features)}개, "
          f"제외 대상: {TARGET_COL}")
    # 실제 라이브 결측률도 참고용으로 남긴다(임의로 안 채웠는지 재확인용).
    missing_rate = float(data[TARGET_COL].isna().mean())
    print(f"[일간] {TARGET_COL} 학습자료 내 결측률: {missing_rate:.1%} "
          f"(라이브에서는 100% 결측 — 이 값은 과거 학습자료 기준 참고치일 뿐)")
    features_after = [c for c in features if c != TARGET_COL]

    before_rows, before_audit = daily_audit.predict_oof(
        data, features, capacity_kw, seed, LABEL_BEFORE)
    after_rows, after_audit = daily_audit.predict_oof(
        data, features_after, capacity_kw, seed, LABEL_AFTER)
    # daily_audit.compare()는 (요약dict, 폴드별DataFrame, 병합원본DataFrame)
    # 3-튜플을 반환한다(단일 DataFrame 아님 — 08-26 실제 실행에서 AttributeError로
    # 발견된 가정 오류, docstring도 이미 그렇게 돼 있었는데 여기서만 놓쳤었다).
    compare_verdict, compare_folds, compare_merged = daily_audit.compare(
        before_rows, after_rows, LABEL_AFTER)
    return (before_rows, after_rows, compare_verdict, compare_folds,
            before_audit, after_audit, missing_rate)


def daily_performance(before_rows: pd.DataFrame, after_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in (("기존", before_rows), ("제외", after_rows)):
        e = g["실제_kWh"] - g["예측_kWh"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        wape = float(e.abs().sum() / g["실제_kWh"].abs().sum() * 100)
        rows.append({"티어": "일간", "수평_h": "D+1", "n": len(g), "구성": label,
                    "MAE_kWh": round(mae, 2), "RMSE_kWh": round(rmse, 2), "WAPE_pct": round(wape, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) / "
          f"제외대상: {TARGET_COL} 단독 / 폴드: 공식 5폴드\n")

    (before_daily, after_daily, compare_verdict, compare_folds, daily_audit_before,
     daily_audit_after, missing_rate) = run_daily_both(capacity_kw, seed)

    daily_rows = pd.concat([
        before_daily.assign(구성=LABEL_BEFORE), after_daily.assign(구성=LABEL_AFTER),
    ], ignore_index=True)
    daily_rows.to_csv(OUT / "일간_동일행_예측정답.csv", index=False, encoding="utf-8-sig")
    compare_folds.to_csv(OUT / "일간_폴드별_비교.csv", index=False, encoding="utf-8-sig")
    daily_perf = daily_performance(before_daily, after_daily)
    daily_perf.to_csv(OUT / "일간_성능비교.csv", index=False, encoding="utf-8-sig")

    (OUT / "감사로그.json").write_text(
        json.dumps({
            "제외컬럼": TARGET_COL,
            "학습자료_결측률_참고치": missing_rate,
            "compare_요약": compare_verdict,
            "감사_기존": daily_audit_before.to_dict("records"),
            "감사_제외": daily_audit_after.to_dict("records"),
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n=== 일간 성능비교(포함 vs 제외) ===")
    print(daily_perf.to_string(index=False))
    print("\n=== 일간 폴드별 비교(daily_audit.compare 원본, 동일행 기준) ===")
    print(compare_folds.to_string(index=False))
    print("\n=== compare() 요약 판정(주의: 판단 기준이 우리 목적과 다름) ===")
    print(json.dumps(compare_verdict, ensure_ascii=False, indent=2, default=str))
    print(
        "\n★compare()의 '채택'/'판정' 필드 해석 주의★ 이 함수는 원래 "
        "'후보가 기준보다 1%p 이상 더 좋아야 채택'이라는 개선 목적 LOGO "
        "실험용이다. 우리는 성능을 개선하려는 게 아니라 **라이브에서 아예 "
        "못 구하는 특성을 어쩔 수 없이 빼는** 상황이므로, '채택=false'가 "
        "나와도 '빼면 안 된다'는 뜻이 아니다 — 중요한 건 '후보_MAE/RMSE가 "
        "기준 대비 크게 나빠지지 않았는지'(MAE개선율_pct·RMSE개선율_pct가 "
        "큰 음수가 아닌지, 최대폴드악화율_pct가 과도하지 않은지)다. "
        "이 판단은 자동화하지 않았다 — 아래 수치를 그대로 Claude에게 "
        "전달할 것."
    )
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
