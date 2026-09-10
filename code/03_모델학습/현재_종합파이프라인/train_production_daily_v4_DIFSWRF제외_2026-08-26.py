# -*- coding: utf-8 -*-
"""운영모델 v2의 일간 D+1 번들만 재생성 — DIFSWRF 계열 4개 추가 제외(08-26 2차).

## 배경
`retrain_exclude_difswrf_daily_v1_2026-08-26.py`로 공식 5폴드 검증 완료
(현재 운영 55특성 vs DIFSWRF 계열 4개 추가 제외 51특성). 결과:
전체 MAE 악화 0.96%, RMSE 악화 0.55%, 폴드 최대 RMSE 악화 4.19%
— 사용자 지정 채택기준(전체 악화 각 1% 미만, 폴드 RMSE 5% 미만) 전부 충족.
산출물: `outputs/일간_DIFSWRF제외_재검증_v1_2026-08-26/요약.json`.

DIFSWRF도 DSWRFLX와 같은 원인(KIMR 2026-06-01 이후 상류 결측)으로 라이브
100% 결측이며, 결측 플래그 중요도가 0이라 학습분포-운영분포가 어긋난 채
숫자만 나오는 상태였다. 이 4개는 **선택이 아니라 필수 제외**다.

## 재구현 없음(원칙 준수)
`train_production_daily_v3_특성정리_2026-08-26.py`와 완전히 동일한 절차
(`daily_direct_final_audit_v1_2026-08-25.py::corrected_dataset()`,
`hyperparameter_tuning_v1_2026-08-21.py::tune_fold()`, 동일 LGBMRegressor
구성)를 그대로 따라가되, **제외 특성 목록만 3개→7개(기존 3개 + DIFSWRF
4개)로 확장**한다. 초단기 4종·단기 3종은 무관 — 전혀 건드리지 않는다.

## 안전장치
- 실행 전 기존 `운영모델_일간_D+1.joblib`·`운영모델_목록.json`을
  `retrain_exclude_difswrf_daily_v1_2026-08-26.py` 실행 직후, 이 스크립트
  실행 직전에 별도로 백업했다
  (`운영모델_일간_D+1_백업_2026-08-26_1545_DIFSWRF패치전.joblib`,
  `운영모델_목록_백업_2026-08-26_1545_DIFSWRF패치전.json`).
- 저장 직전 `bundle["features"]`와 재적재한 파일의 `features` 양쪽에서
  DIFSWRF·DSWRFLX·MCO 포함 특성 0건을 assert한다.
- 저장 직후 재적재해 예측값이 학습 시점과 일치하는지 확인한다(reload-diff).
- `운영모델_목록.json`은 일간 항목만 갱신하고 나머지 7개 모델 항목은
  백업본에서 그대로 보존한다(다른 모델 재학습 안 함).

## 실행 후 반드시 할 것
`golden_replay_전체8종_v1_2026-08-25.py`에 일간 DIFSWRF 부재 검사를 추가한
버전으로 전체 8종 재검증할 것.
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
OUT = ROOT / "outputs" / "운영모델_v2_2026-08-25"  # 기존과 같은 폴더 — 일간만 교체
BUNDLE_PATH = OUT / "운영모델_일간_D+1.joblib"
REGISTRY_PATH = OUT / "운영모델_목록.json"
BACKUP_REGISTRY = OUT / "운영모델_목록_백업_2026-08-26_1545_DIFSWRF패치전.json"
BACKUP_BUNDLE = OUT / "운영모델_일간_D+1_백업_2026-08-26_1545_DIFSWRF패치전.joblib"

EXCLUDE_COLS_PREV = [
    "2일전평균_mean_communication_ok",
    "목표일예보_DSWRFLX_bsrn정제_sum",
    "목표일예보_DSWRFLX_bsrn정제_mean",
]
EXCLUDE_COLS_DIFSWRF = [
    "목표일예보_DIFSWRF_bsrn정제_sum",
    "목표일예보_DIFSWRF_bsrn정제_mean",
    "목표일_DIFSWRF_유효개수",
    "목표일예보_DIFSWRF_bsrn정제_sum_결측여부",
]
EXCLUDE_COLS = EXCLUDE_COLS_PREV + EXCLUDE_COLS_DIFSWRF
BUNDLE_VERSION = "v2_2026-08-25+일간특성정리_2026-08-26+DIFSWRF제외_2026-08-26"
MISSING_RULE_TEXT = (
    "LightGBM native missing 위임 — 중앙값/평균 등 사후대체 없음. "
    "2일전평균_mean_communication_ok·목표일예보_DSWRFLX_bsrn정제_sum/_mean "
    "3개는 08-26 1차에 제외(Blockdata 스키마 부재·KMA KIMR 2026-06-01 이후 "
    "상류 결측). 목표일예보_DIFSWRF_bsrn정제_sum/_mean·목표일_DIFSWRF_유효개수·"
    "목표일예보_DIFSWRF_bsrn정제_sum_결측여부 4개는 08-26 2차에 제외 — "
    "DIFSWRF도 DSWRFLX와 같은 KIMR 원인으로 라이브 100% 결측인데 학습분포는 "
    "결측이 드물어(BSRN 정제 후 14.7%) 결측 플래그 중요도가 0이었다(학습-운영 "
    "분포 불일치). 총 7개 모두 선택이 아니라 필수 제외 — 공식 5폴드 검증 통과."
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_mod = _load("daily_v4_audit", "daily_direct_final_audit_v1_2026-08-25.py")
tuning = _load("daily_v4_tuning", "hyperparameter_tuning_v1_2026-08-21.py")


def _assert_clean(features: list[str], where: str) -> None:
    leaked = [c for c in features if c in EXCLUDE_COLS]
    if leaked:
        raise AssertionError(f"[{where}] 제외 대상 특성이 아직 남아있음: {leaked}")


def main() -> None:
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(f"기존 일간 번들이 없다: {BUNDLE_PATH}.")
    if not BACKUP_BUNDLE.is_file() or not BACKUP_REGISTRY.is_file():
        raise RuntimeError(
            f"이번 패치 전용 백업이 없다: {BACKUP_BUNDLE} / {BACKUP_REGISTRY} — 먼저 백업할 것."
        )

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) seed={seed}\n")

    data, features, n_partial = daily_mod.corrected_dataset(capacity_kw)
    missing = [c for c in EXCLUDE_COLS if c not in features]
    if missing:
        raise RuntimeError(f"제외 대상인데 원본 특성목록에 없음(가정 오류): {missing}")
    features_new = [c for c in features if c not in EXCLUDE_COLS]
    print(f"전체특성 {len(features)}개 → 누적제외 {len(EXCLUDE_COLS)}개(기존3+DIFSWRF4) → "
          f"최종 {len(features_new)}개")
    print(f"제외 대상: {EXCLUDE_COLS}")
    _assert_clean(features_new, "특성목록 확정 직후")

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
        "제외특성_08-26_1차": EXCLUDE_COLS_PREV,
        "제외특성_08-26_2차_DIFSWRF": EXCLUDE_COLS_DIFSWRF,
        "제외근거": ("retrain_exclude_mean_communication_ok_v1_2026-08-26.py, "
                  "retrain_exclude_dswrflx_v1_2026-08-26.py, "
                  "retrain_exclude_daily_combined_v1_2026-08-26.py(1차), "
                  "retrain_exclude_difswrf_daily_v1_2026-08-26.py(2차) 공식 5폴드 검증 "
                  "— AGENTS.md 08-26 절 참고"),
        "이력요구사항": "7일전·2일전 lag + 2일전기준 30일 이동통계 사용 — 라이브 운영 전 "
                    "최소 30~32일치 발전량 이력 축적(또는 과거자료 백필) 필요.",
    }
    _assert_clean(bundle["features"], "저장 직전")
    joblib.dump(bundle, BUNDLE_PATH)
    loaded = joblib.load(BUNDLE_PATH)
    _assert_clean(loaded["features"], "재적재 직후")
    before = np.asarray(model.predict(train[features_new].tail(30)), float)
    after = np.asarray(loaded["model"].predict(train[features_new].tail(30)), float)
    reload_diff = float(np.max(np.abs(before - after)))
    print(f"\n[일간 D+1] 재학습 완료 학습행수={len(train):,} 특성수={len(features_new)} "
          f"학습기간={train.index.min().date()}~{train.index.max().date()} "
          f"재적재차이={reload_diff:.2e}")
    if reload_diff > 1e-9:
        raise RuntimeError(f"재적재 검증 실패(차이 {reload_diff}) — 번들 저장에 문제가 있을 수 있다.")

    old_registry = json.loads(BACKUP_REGISTRY.read_text(encoding="utf-8"))
    old_registry["모델"]["일간_D+1"] = {
        "파일": str(BUNDLE_PATH), "학습행수": len(train), "특성수": len(features_new),
        "구조": "전체특성(7개제외:MCO+DSX3+DIFSWRF4)+폴드내부튜닝",
        "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
        "재적재차이": reload_diff, "제외특성": EXCLUDE_COLS,
    }
    old_registry["생성시각_일간패치_DIFSWRF"] = pd.Timestamp.now().isoformat()
    old_registry["번들버전_일간"] = BUNDLE_VERSION
    REGISTRY_PATH.write_text(
        json.dumps(old_registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n운영모델_목록.json 갱신 완료(일간만) → {REGISTRY_PATH}")
    print(f"저장 완료: {BUNDLE_PATH}")
    print("\n★다음 단계★ golden_replay_전체8종_v1_2026-08-25.py에 일간 DIFSWRF 부재 검사를 "
          "추가한 버전으로 전체 8종 재검증할 것.")


if __name__ == "__main__":
    main()
