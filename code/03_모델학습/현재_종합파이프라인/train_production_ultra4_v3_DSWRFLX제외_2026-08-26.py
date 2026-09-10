# -*- coding: utf-8 -*-
"""운영모델 v2의 초단기 +4h 번들만 재생성 — DSWRFLX 제외(08-26).

## 배경
`retrain_exclude_dswrflx_ultra4_v1_2026-08-26.py`(최신 공식
`청천지수+날씨군집화` 기준, Codex 지적 반영)로 공식 5폴드 검증을 이미
마쳤다 — nMAE +0.118%p, nRMSE +0.166%p로 손실은 작고, DSWRFLX를
군집 입력에 남겨뒀을 때 최근 폴드일수록 결측으로 버려지던 시험행
(0/0/17/825/5,828건)이 제외 후 전부 0건이 되는 부가효과까지 확인해
Claude가 채택을 권장했다(AGENTS.md 08-26절). DSWRFLX는 선택이 아니라
필수 제외다 — 라이브에서 2026-06-01 이후 KIMR 상류가 구조적으로
결측이라 재개해도 안 생긴다.

## ★몽키패치 시 주의(다른 스크립트와 다른 함정)★
`train_production_models_v2_2026-08-25.py`는 모듈 맨 위에서
`CLUSTER_SOURCE = improvement.CLUSTER_SOURCE`로 **값을 그 시점에
복사해 자기 모듈의 전역이름으로 캡처**해둔다. 그래서 `improvement.
CLUSTER_SOURCE = [...]`처럼 원본 쪽만 재할당해서는 이 모듈의
`fit_weather_clusters()`가 여전히 옛 목록을 본다(그 함수 안의
`CLUSTER_SOURCE`는 이 모듈 자신의 전역이름을 가리키지, `improvement.
CLUSTER_SOURCE`를 다시 찾아가지 않는다). 그래서 이 스크립트는
**`prod.CLUSTER_SOURCE`(로드한 모듈 자신의 전역이름)를 직접 패치**한다
— `retrain_exclude_dswrflx_ultra4_v1_2026-08-26.py`가 `improvement.
CLUSTER_SOURCE`를 패치한 것과는 대상이 다르다(그쪽은 `run_ultra_patch()`
가 `improvement.CLUSTER_SOURCE`를 매번 새로 참조하는 별도 모듈이라
그 방식이 맞았다 — 여기는 캡처된 복사본이라 다르게 패치해야 함).

## 재구현 없음
`train_production_models_v2_2026-08-25.py::train_ultra()`(h=4일 때
`fit_weather_clusters()`로 군집을 만들고 bundle에 스케일러·kmeans·
중앙값까지 저장하는, 실제 운영 번들 생성 함수)를 그대로 호출한다.
후보특성·군집원천 목록만 호출 전 몽키패치로 걸러냈다가 복원한다.

## 안전장치
- 실행 전 기존 `운영모델_초단기_h4.joblib`·`운영모델_목록.json`을
  타임스탬프 붙여 백업한다(이 스크립트가 직접 함 — 일간 패치 때처럼
  수동 백업 필요 없음).
- `train_ultra()` 내부의 `verify_full_chain()`이 재적재 후 예측 일치를
  이미 확인한다(카파 역변환이 조용히 스킵되는지도 자체 검증).
- `운영모델_목록.json`은 초단기_h4 항목만 갱신하고 나머지 7개(일간 포함
  — 이미 08-26에 재생성된 상태)는 **현재 파일에서 그대로 보존**한다.

## 실행 후 반드시 할 것
`golden_replay_전체8종_v1_2026-08-25.py`로 8종 재검증할 것.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "운영모델_v2_2026-08-25"
BUNDLE_PATH = OUT / "운영모델_초단기_h4.joblib"
REGISTRY_PATH = OUT / "운영모델_목록.json"
BACKUP_SUFFIX = "2026-08-26_h4패치전"
DSX = "DSWRFLX_bsrn정제"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prod = _load("ultra4_prod_v3", "train_production_models_v2_2026-08-25.py")
harness = prod.harness


def _remove_dsx() -> dict:
    before = {
        "전체후보": list(harness.FEATURE_SETS["전체후보"]),
        "FORECAST_COLUMNS": list(harness.sel.FORECAST_COLUMNS),
        "OBSERVED_COLUMNS": list(harness.sel.OBSERVED_COLUMNS),
        "CLUSTER_SOURCE": list(prod.CLUSTER_SOURCE),  # ★모듈 자신의 캡처본★
    }
    found = []
    if DSX in harness.FEATURE_SETS["전체후보"]:
        harness.FEATURE_SETS["전체후보"] = [c for c in harness.FEATURE_SETS["전체후보"] if c != DSX]
        found.append("FEATURE_SETS[전체후보]")
    if DSX in harness.sel.FORECAST_COLUMNS:
        harness.sel.FORECAST_COLUMNS = [c for c in harness.sel.FORECAST_COLUMNS if c != DSX]
        found.append("sel.FORECAST_COLUMNS")
    if DSX in harness.sel.OBSERVED_COLUMNS:
        harness.sel.OBSERVED_COLUMNS = [c for c in harness.sel.OBSERVED_COLUMNS if c != DSX]
        found.append("sel.OBSERVED_COLUMNS")
    if DSX in prod.CLUSTER_SOURCE:
        prod.CLUSTER_SOURCE = [c for c in prod.CLUSTER_SOURCE if c != DSX]
        found.append("prod.CLUSTER_SOURCE(모듈 자신의 캡처본)")
    if not found:
        raise RuntimeError(f"{DSX}를 아무 목록에서도 못 찾음 — 가정이 틀렸을 수 있다. 중단.")
    print(f"[몽키패치] {DSX} 제거됨: {found}")
    return before


def _restore(before: dict) -> None:
    harness.FEATURE_SETS["전체후보"] = before["전체후보"]
    harness.sel.FORECAST_COLUMNS = before["FORECAST_COLUMNS"]
    harness.sel.OBSERVED_COLUMNS = before["OBSERVED_COLUMNS"]
    prod.CLUSTER_SOURCE = before["CLUSTER_SOURCE"]


def main() -> None:
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(f"기존 +4h 번들이 없다: {BUNDLE_PATH}")
    if not REGISTRY_PATH.is_file():
        raise RuntimeError(f"운영모델_목록.json이 없다: {REGISTRY_PATH}")

    backup_bundle = OUT / f"운영모델_초단기_h4_{BACKUP_SUFFIX}.joblib"
    backup_registry = OUT / f"운영모델_목록_{BACKUP_SUFFIX}.json"
    shutil.copy2(BUNDLE_PATH, backup_bundle)
    shutil.copy2(REGISTRY_PATH, backup_registry)
    print(f"백업 완료: {backup_bundle.name}, {backup_registry.name}")

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) seed={seed}\n")

    saved = _remove_dsx()
    try:
        result = prod.train_ultra(4, capacity_kw, seed)
    finally:
        _restore(saved)

    print(f"\n[초단기 +4h] 재학습 결과: {json.dumps(result, ensure_ascii=False, default=str)}")
    if result.get("전체체인재검증차이", 1.0) > 1e-9:
        raise RuntimeError(f"재적재/역변환 검증 실패(차이 {result.get('전체체인재검증차이')}) — 중단.")

    # 운영모델_목록.json은 초단기_h4만 갱신, 나머지는 현재 파일(이미 일간
    # 패치가 반영된 상태) 그대로 보존.
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["모델"]["초단기_h4"] = result
    registry["초단기h4_DSWRFLX제외_08-26"] = True
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n운영모델_목록.json 갱신 완료(초단기_h4만) → {REGISTRY_PATH}")
    print(f"저장 완료: {BUNDLE_PATH}")
    print("\n★다음 단계★ golden_replay_전체8종_v1_2026-08-25.py로 8종 재검증할 것.")


if __name__ == "__main__":
    main()
