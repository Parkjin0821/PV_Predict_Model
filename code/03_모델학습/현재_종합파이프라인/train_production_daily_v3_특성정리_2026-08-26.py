# -*- coding: utf-8 -*-
"""운영모델 v2의 일간 D+1 번들만 재생성 — 결측특성 3개 제외(08-26).

## 배경
`retrain_exclude_mean_communication_ok_v1_2026-08-26.py`,
`retrain_exclude_dswrflx_v1_2026-08-26.py`,
`retrain_exclude_daily_combined_v1_2026-08-26.py` 3개 스크립트로
공식 5폴드 검증을 마쳤다(전부 Claude가 로컬 Python으로 직접 실행,
결과는 AGENTS.md 08-26절 참고). 결론: `2일전평균_mean_communication_ok`·
`목표일예보_DSWRFLX_bsrn정제_sum`·`목표일예보_DSWRFLX_bsrn정제_mean`
3개를 동시에 빼도 나쁜 상호작용 없음(전체 MAE 108.67→106.38kWh,
−2.11%), 여름철만 MAE +8.1% 정도 저하되는 트레이드오프를 받아들이고
진행하기로 사용자 승인 받음(08-26). 이 3개 특성은 **선택이 아니라
필수 제외**다 — 라이브 Blockdata·KMA API 양쪽 다 구조적으로 값을
못 준다.

## 재구현 없음(원칙 준수)
`train_production_models_v2_2026-08-25.py::train_daily()`와 완전히
동일한 학습 절차(같은 `daily_direct_final_audit_v1_2026-08-25.py::
corrected_dataset()`, 같은 `hyperparameter_tuning_v1_2026-08-21.py::
tune_fold()`, 같은 LGBMRegressor 구성)를 그대로 따라가되, **특성
목록에서 3개만 제외**한다. 초단기 4종·단기 3종은 이번 변경과 무관해
**전혀 건드리지 않는다**(재학습·재생성 없음 — 08-24부터 이어온
"패치는 해당 티어만" 원칙).

## 안전장치
- 실행 전 기존 `운영모델_일간_D+1.joblib`·`운영모델_목록.json`을
  타임스탬프 붙여 백업했다(이 스크립트가 그 백업을 다시 만들지는
  않음 — 이미 수동으로 해뒀음, 아래 참고).
- 저장 직후 재적재해 예측값이 학습 시점과 일치하는지 확인한다
  (v1 방식과 동일한 reload-diff 체크).
- `운영모델_목록.json`은 일간 항목만 갱신하고 나머지 7개 모델 항목은
  그대로 보존한다(백업본에서 읽어와 재사용 — 다른 모델 재학습 안 함).

## 실행 후 반드시 할 것
`golden_replay_전체8종_v1_2026-08-25.py` 또는
`production_readiness_check_v1_2026-08-25.py`로 일간 최소 재검증할 것
(이 스크립트 자체는 reload-diff만 확인하지, golden replay 수준의
end-to-end 조립 검증은 안 함).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "운영모델_v2_2026-08-25"  # v2와 같은 폴더 — 일간만 교체
BUNDLE_PATH = OUT / "운영모델_일간_D+1.joblib"
REGISTRY_PATH = OUT / "운영모델_목록.json"
BACKUP_REGISTRY = OUT / "운영모델_목록_이전_백업_2026-08-26.json"

EXCLUDE_COLS = [
    "2일전평균_mean_communication_ok",
    "목표일예보_DSWRFLX_bsrn정제_sum",
    "목표일예보_DSWRFLX_bsrn정제_mean",
]
BUNDLE_VERSION = "v2_2026-08-25+일간특성정리_2026-08-26"
MISSING_RULE_TEXT = (
    "LightGBM native missing 위임 — 중앙값/평균 등 사후대체 없음. "
    "2일전평균_mean_communication_ok·목표일예보_DSWRFLX_bsrn정제_sum/_mean "
    "3개는 라이브에서 구조적으로 못 구해(Blockdata 스키마 부재·KMA KIMR "
    "2026-06-01 이후 상류 결측) 08-26에 후보에서 아예 제외함(선택이 아니라 "
    "필수 제외) — 공식 5폴드 검증 통과(AGENTS.md 08-26 참고)."
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_mod = _load("daily_v3_audit", "daily_direct_final_audit_v1_2026-08-25.py")
tuning = _load("daily_v3_tuning", "hyperparameter_tuning_v1_2026-08-21.py")


def main() -> None:
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(
            f"기존 일간 번들이 없다: {BUNDLE_PATH}. 백업이 이미 됐는지 먼저 확인할 것."
        )

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) seed={seed}\n")

    data, features, n_partial = daily_mod.corrected_dataset(capacity_kw)
    missing = [c for c in EXCLUDE_COLS if c not in features]
    if missing:
        raise RuntimeError(f"제외 대상인데 특성목록에 없음(가정 오류): {missing}")
    features_new = [c for c in features if c not in EXCLUDE_COLS]
    print(f"전체특성 {len(features)}개 → 제외 3개 → 최종 {len(features_new)}개")
    print(f"제외 대상: {EXCLUDE_COLS}")

    cap_day = capacity_kw * 24
    ACTUAL = "실제_일간발전량_kWh"
    train = data.dropna(subset=[ACTUAL])

    params, trace = tuning.tune_fold("LightGBM", train, features_new, ACTUAL, cap_day, seed)
    params = daily_mod.cast_params(params)
    model = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **params)
    model.fit(train[features_new], train[ACTUAL])

    bundle = {
        "tier": "일간", "horizon": "D+1", "target_transform": "daily_kWh", "타깃유형": "raw",
        "features": features_new, "model": model, "params": params,
        "missing_strategy": "LightGBM_native_missing+flags", "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "clip_상한": cap_day,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
        "번들버전": BUNDLE_VERSION, "결측처리규칙": MISSING_RULE_TEXT,
        "카파_역변환": None, "클러스터": None,
        "제외특성_08-26": EXCLUDE_COLS,
        "제외근거": ("retrain_exclude_mean_communication_ok_v1_2026-08-26.py, "
                  "retrain_exclude_dswrflx_v1_2026-08-26.py, "
                  "retrain_exclude_daily_combined_v1_2026-08-26.py 공식 5폴드 검증 "
                  "— AGENTS.md 08-26 절 참고"),
        "이력요구사항": "7일전·2일전 lag + 2일전기준 30일 이동통계 사용 — 라이브 운영 전 "
                    "최소 30~32일치 발전량 이력 축적(또는 과거자료 백필) 필요.",
    }
    joblib.dump(bundle, BUNDLE_PATH)
    loaded = joblib.load(BUNDLE_PATH)
    before = np.asarray(model.predict(train[features_new].tail(30)), float)
    after = np.asarray(loaded["model"].predict(train[features_new].tail(30)), float)
    reload_diff = float(np.max(np.abs(before - after)))
    print(f"\n[일간 D+1] 재학습 완료 학습행수={len(train):,} 특성수={len(features_new)} "
          f"학습기간={train.index.min().date()}~{train.index.max().date()} "
          f"재적재차이={reload_diff:.2e}")
    if reload_diff > 1e-9:
        raise RuntimeError(f"재적재 검증 실패(차이 {reload_diff}) — 번들 저장에 문제가 있을 수 있다.")

    # 운영모델_목록.json은 일간 항목만 갱신, 나머지 7개는 백업본에서 그대로 보존.
    if not BACKUP_REGISTRY.is_file():
        raise RuntimeError(f"이전 registry 백업이 없다: {BACKUP_REGISTRY} — 먼저 수동 백업할 것.")
    old_registry = json.loads(BACKUP_REGISTRY.read_text(encoding="utf-8"))
    old_registry["모델"]["일간_D+1"] = {
        "파일": str(BUNDLE_PATH), "학습행수": len(train), "특성수": len(features_new),
        "구조": "전체특성(3개제외)+폴드내부튜닝", "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
        "재적재차이": reload_diff, "제외특성": EXCLUDE_COLS,
    }
    old_registry["생성시각_일간패치"] = pd.Timestamp.now().isoformat()
    old_registry["번들버전_일간"] = BUNDLE_VERSION
    REGISTRY_PATH.write_text(
        json.dumps(old_registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n운영모델_목록.json 갱신 완료(일간만) → {REGISTRY_PATH}")
    print(f"저장 완료: {BUNDLE_PATH}")
    print("\n★다음 단계★ golden_replay_전체8종_v1_2026-08-25.py 또는 "
          "production_readiness_check_v1_2026-08-25.py로 일간 재검증할 것.")


if __name__ == "__main__":
    main()
