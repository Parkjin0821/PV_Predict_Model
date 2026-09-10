# -*- coding: utf-8 -*-
"""운영모델 v2의 초단기 4종·단기 3종 재생성 — DSWRFLX 명시적 완전제외.

## 배경
`retrain_exclude_dswrflx_v2_명시적제거_2026-08-26.py`(몽키패치 미사용,
7개 수평 자체검증, MODEL_DIR 격리)로 공식 5폴드 검증을 통과했고
**시간모델 7종 전부 DSWRFLX 제외를 채택**하기로 확정됐다(사용자
08-26). 이 스크립트는 그 검증된 방식 그대로 실제 운영 joblib 7개를
다시 만든다. 일간(58→55특성)은 이미 별도로 재생성·검증 완료됐으므로
이 스크립트가 다시 안 건드린다.

## 특성·군집 제외 방식(반복 정리)
`train_production_models_v2_2026-08-25.py::train_ultra()`/
`train_short()`(운영 번들을 실제로 만드는 함수)를 그대로 복제하되,
`features = base_cols + chosen` **직후** DSWRFLX를 명시적으로 제거하는
한 줄만 추가한다(몽키패치 안 씀 — `dpc`/`ultra`가 `harness`를 독립적으로
새로 로드해서 몽키패치가 실제 프레임 조립 경로에 안 먹혔던 이전 실패와
무관). +4h는 청천지수·날씨군집화·카파 역변환 구조를 그대로 유지하고,
이미 걸러진 `features`를 `fit_weather_clusters()`에 넘기므로 군집원천도
별도 패치 없이 자동으로 깨끗해진다.

## ★08-26 재수정: staging→전체검증→원자적 승격(Codex 지적 반영)★
이전 버전은 7개 번들을 하나씩 바로 최종 경로에 덮어썼다 — 4번째쯤에서
학습이 실패하면 **1~3번은 새 버전, 4~7번은 옛 버전으로 섞인 상태**가
되고, 그 상태로 golden replay를 돌리면 원인 파악이 어려워진다. 그래서
이번엔 3단계로 나눴다:

1. **백업**: 이번 실행에만 쓰는 고유 타임스탬프 폴더
   (`백업_YYYYMMDDTHHMMSS`)를 새로 만들어 기존 7개 번들+
   `운영모델_목록.json`을 복사한다. **폴더가 이미 있으면 그 자체를
   오류로 처리하고 중단한다** — 기존 백업을 절대 덮어쓰지 않는다.
2. **staging**: 7개 전부 `staging_YYYYMMDDTHHMMSS/`라는 별도 폴더에
   먼저 저장한다(최종 경로 미접촉). 각각 저장 직후 그 staging 파일을
   다시 열어 재적재·DSWRFLX 부재(특성+군집원천)·`verify_full_chain()`
   (재적재 예측 일치·카파 역변환 확인, 기존 함수 재사용)을 **전부
   통과해야만** 다음 단계로 넘어간다. **7개 중 하나라도 이 단계에서
   실패하면 즉시 중단** — 이 시점까지는 최종 경로(`운영모델_v2_
   2026-08-25/`)를 전혀 안 건드렸으므로 롤백할 것도 없다(기존 7개
   번들이 그대로 살아있다).
3. **원자적 승격 + 레지스트리 교체(하나의 try로 묶음, 08-26 재수정)**:
   7개 전부 staging 검증을 통과한 뒤에만 `os.replace(staging_경로,
   최종_경로)`로 하나씩 옮기고, 이어서 `운영모델_목록.json`도 임시
   파일에 먼저 쓰고 `os.replace`로 교체한다. **이 둘을 하나의 try
   블록으로 묶는다** — 번들 7개는 승격에 성공했는데 레지스트리 교체만
   실패하는 경우(신번들+구 레지스트리 혼합)까지 실패로 취급해야 하기
   때문이다(이전 버전은 승격 루프 안에서만 부분 롤백해서 이 경우를
   못 잡았다). **어느 단계에서 실패하든 `rollback_all()`이 백업의
   번들 7개+레지스트리(총 8개) 전부를 최종 경로로 복원**하고, 복원
   완료 여부를 8개 전부 바이트 단위로 검증한 뒤 원래 예외를 그대로
   다시 올린다. **복원은 `os.replace(backup, final)`이 아니라
   `shutil.copy2(backup, final)`을 쓴다** — `os.replace`로 백업을
   복원에 쓰면 그 파일이 백업 폴더에서 최종 경로로 "옮겨져" 없어지므로
   백업이 훼손된다. `copy2`로 복사만 하므로 백업은 몇 번을 복원해도
   그대로 남는다. 레지스트리 복원도 백업을 직접 `os.replace`하지 않고
   임시파일에 복사한 뒤 그 임시파일만 교체한다(백업 원본은 읽기 전용).

## 실행 안 함(사용자 지시)
이 파일은 **작성만 하고 실행하지 않았다.**

## 실행 후 반드시 할 것
`golden_replay_전체8종_v1_2026-08-25.py`(DSWRFLX 부재 검사 포함,
08-26 수정본)로 8종 재검증할 것.
"""
from __future__ import annotations

import filecmp
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "운영모델_v2_2026-08-25"
REGISTRY_PATH = OUT / "운영모델_목록.json"
DSX = "DSWRFLX_bsrn정제"

ULTRA_HORIZONS = (1, 2, 3, 4)
SHORT_HORIZONS = (1, 24, 48)
BUNDLE_NAMES = {("초단기", h): f"운영모델_초단기_h{h}.joblib" for h in ULTRA_HORIZONS}
BUNDLE_NAMES.update({("단기", h): f"운영모델_단기_h{h}.joblib" for h in SHORT_HORIZONS})


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prod = _load("prod7_v1", "train_production_models_v2_2026-08-25.py")
dpc, harness, ultra, clearsky, improvement, infer = (
    prod.dpc, prod.harness, prod.ultra, prod.clearsky, prod.improvement, prod.infer)
N_INVERTERS = prod.N_INVERTERS
DIF = prod.DIF
ULTRA_OFFICIAL = prod.ULTRA_OFFICIAL
ULTRA_STRUCTURE_TYPE = prod.ULTRA_STRUCTURE_TYPE
MISSING_RULE_TEXT = prod.MISSING_RULE_TEXT
BUNDLE_VERSION = f"{prod.BUNDLE_VERSION}+DSWRFLX제외_2026-08-26"
add_difswrf_flag = prod.e2e.add_difswrf_flag


def _strip_dsx(features: list[str]) -> list[str]:
    return [c for c in features if DSX not in c]


def _assert_clean(cols: list[str], where: str) -> None:
    leaked = [c for c in cols if DSX in c]
    if leaked:
        raise AssertionError(f"[{where}] DSWRFLX가 아직 남아있음: {leaked}")


def _verify_staged_bundle(bundle: dict, train: pd.DataFrame, path: Path) -> float:
    """staging에 저장한 직후 호출 — DSWRFLX 부재(특성+군집원천)와
    verify_full_chain(재적재 예측일치·카파역변환)을 전부 확인한다."""
    _assert_clean(bundle["features"], f"저장 직전(features) {path.name}")
    if bundle.get("클러스터"):
        _assert_clean(bundle["클러스터"]["입력컬럼"], f"저장 직전(군집원천) {path.name}")

    reloaded = joblib.load(path)
    _assert_clean(reloaded["features"], f"재적재 직후(features) {path.name}")
    if reloaded.get("클러스터"):
        _assert_clean(reloaded["클러스터"]["입력컬럼"], f"재적재 직후(군집원천) {path.name}")

    return prod.verify_full_chain(bundle, train, path)


def train_ultra_v4(h: int, capacity_kw: float, seed: int, save_dir: Path) -> dict:
    """prod.train_ultra()를 그대로 복제 + features 명시적 필터 한 줄만
    추가 + save_dir로 저장 위치를 바꿀 수 있게 함(staging 지원)."""
    frame = add_difswrf_flag(dpc.load_ultra_frame(h))
    candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    use_kappa = ULTRA_OFFICIAL[h] == "청천지수"
    frame["_청천_kW"] = np.clip(
        capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
    min_elev = clearsky.MIN_ELEVATION_DEG if use_kappa else 0.0
    daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]
    full = daylight[daylight["_목표_가용인버터수"] >= N_INVERTERS].copy()

    sel_input = full.dropna(subset=["목표_발전출력_kW"])
    chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
    features = base_cols + chosen
    features = _strip_dsx(features)  # ★유일한 실질 변경점★
    native_ok = harness.NATIVE_MISSING_OK | {DIF}
    required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW", "_청천_kW", "_카파"]
    train = full.dropna(subset=required)
    train["_카파_원본"] = train["_카파"]

    target = "_카파" if use_kappa else "목표_발전출력_kW"
    cluster_bundle = None
    if h == 4:
        train, features, cluster_bundle = prod.fit_weather_clusters(train, features, seed)
        model = ultra.make_model("LightGBM", seed)
        structure_name = "청천지수+날씨군집화"
    elif ULTRA_STRUCTURE_TYPE[h] == "구조선택":
        params, structure_name = improvement.choose_structure_kfold(
            "초단기", train, features, target, "kappa" if use_kappa else "raw", capacity_kw, seed)
        model = improvement.make_model("초단기", seed, params)
    else:
        model = ultra.make_model("LightGBM", seed)
        structure_name = "기본"
    model.fit(train[features], train[target])

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"운영모델_초단기_h{h}.joblib"
    bundle = {
        "tier": "초단기", "horizon_h": h, "target_transform": target,
        "타깃유형": ("kappa" if use_kappa else "raw"), "structure_name": structure_name,
        "features": features, "model": model, "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "clip_상한": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
        "번들버전": BUNDLE_VERSION, "결측처리규칙": MISSING_RULE_TEXT + " DSWRFLX_bsrn정제는 "
        "라이브 KIMR 2026-06-01 이후 상류 결측으로 08-26에 후보에서 명시적 완전제외(공식 5폴드 검증 통과).",
        "카파_역변환": ({"최소태양고도_deg": min_elev, "청천일사공식": "Haurwitz(1945)",
                     "clip범위_kW": [0, capacity_kw]} if use_kappa else None),
        "클러스터": cluster_bundle,
        "DSWRFLX_제외_08-26": True,
    }
    joblib.dump(bundle, path)
    chain_diff = _verify_staged_bundle(bundle, train, path)
    print(f"[초단기 +{h}h] staging 저장·검증 완료 학습행수={len(train):,} 특성수={len(features)} "
          f"구조={structure_name} 전체체인재검증차이={chain_diff:.2e} DSWRFLX포함=False(assert통과) "
          f"→ {path}")
    return {"티어": "초단기", "수평_h": h, "staging_경로": str(path),
            "학습행수": len(train), "특성수": len(features), "구조": structure_name,
            "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
            "전체체인재검증차이": chain_diff, "DSWRFLX_제외_확인": True}


def train_short_v4(h: int, capacity_kw: float, seed: int, save_dir: Path) -> dict:
    """prod.train_short()를 그대로 복제 + features 명시적 필터 한 줄 +
    save_dir 파라미터화(staging 지원)."""
    frame = add_difswrf_flag(dpc.load_short_frame(h))
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]
    full = daylight[daylight["_목표_가용인버터수"] >= N_INVERTERS].copy()

    chosen = harness.select_features_in_fold(
        full.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
    features = base_cols + chosen
    features = _strip_dsx(features)  # ★유일한 실질 변경점★
    native_ok = harness.NATIVE_MISSING_OK | {DIF}
    required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
    train = full.dropna(subset=required)

    params, structure_name = improvement.choose_structure_kfold(
        "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
    model = improvement.make_model("단기", seed, params)
    model.fit(train[features], train["목표_발전출력_kW"])

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"운영모델_단기_h{h}.joblib"
    bundle = {
        "tier": "단기", "horizon_h": h, "target_transform": "raw_kW",
        "타깃유형": "raw", "structure_name": structure_name,
        "features": features, "model": model, "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "clip_상한": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
        "번들버전": BUNDLE_VERSION, "결측처리규칙": MISSING_RULE_TEXT + " DSWRFLX_bsrn정제는 "
        "라이브 KIMR 2026-06-01 이후 상류 결측으로 08-26에 후보에서 명시적 완전제외(공식 5폴드 검증 통과).",
        "카파_역변환": None, "클러스터": None,
        "DSWRFLX_제외_08-26": True,
    }
    joblib.dump(bundle, path)
    chain_diff = _verify_staged_bundle(bundle, train, path)
    print(f"[단기 +{h}h] staging 저장·검증 완료 학습행수={len(train):,} 특성수={len(features)} "
          f"전체체인재검증차이={chain_diff:.2e} DSWRFLX포함=False(assert통과) → {path}")
    return {"티어": "단기", "수평_h": h, "staging_경로": str(path),
            "학습행수": len(train), "특성수": len(features), "구조": structure_name,
            "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
            "전체체인재검증차이": chain_diff, "DSWRFLX_제외_확인": True}


def make_backup(run_id: str) -> Path:
    backup_dir = OUT / f"백업_{run_id}"
    if backup_dir.exists():
        raise RuntimeError(
            f"백업 폴더가 이미 있다: {backup_dir} — 기존 백업을 덮어쓰지 않는다는 "
            "원칙 위반 위험이라 자동 진행하지 않는다. 폴더명이 겹친 원인을 먼저 "
            "확인할 것(같은 초 안에 재실행됐을 가능성)."
        )
    backup_dir.mkdir(parents=True)
    for (tier, h), name in BUNDLE_NAMES.items():
        src = OUT / name
        if not src.is_file():
            raise RuntimeError(f"백업 대상이 없다: {src}")
        shutil.copy2(src, backup_dir / name)
    if not REGISTRY_PATH.is_file():
        raise RuntimeError(f"운영모델_목록.json이 없다: {REGISTRY_PATH}")
    shutil.copy2(REGISTRY_PATH, backup_dir / REGISTRY_PATH.name)
    print(f"백업 완료(7개 번들+레지스트리) → {backup_dir}")
    return backup_dir


def promote_bundles(staged_results: dict[tuple[str, int], dict]) -> None:
    """staging 검증을 전부 통과한 뒤에만 호출된다. 7개를 하나씩
    os.replace로 최종 경로에 승격한다(같은 파일시스템 내 원자적 교체).
    실패하면 그대로 예외를 올린다 — 롤백은 호출부(main)가
    `rollback_all()`로 승격·레지스트리 교체를 한 덩어리로 묶어 처리한다
    (이 함수 자체는 부분 롤백을 하지 않음 — 08-26 재수정, 아래 참고)."""
    for key, result in staged_results.items():
        name = BUNDLE_NAMES[key]
        staged_path = Path(result["staging_경로"])
        os.replace(staged_path, OUT / name)
        print(f"승격 완료: {name}")


def promote_registry(staged_results: dict[tuple[str, int], dict]) -> None:
    """임시 파일에 먼저 쓰고 os.replace로 원자적 교체한다(쓰다 만
    registry가 최종 경로에 남지 않음)."""
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for (tier, h), result in staged_results.items():
        registry["모델"][f"{tier}_h{h}"] = {k: v for k, v in result.items() if k != "staging_경로"}
        registry["모델"][f"{tier}_h{h}"]["파일"] = str(OUT / BUNDLE_NAMES[(tier, h)])
    registry["DSWRFLX_제외_08-26"] = "초단기4종+단기3종(일간은 별도 08-26 작업에서 이미 반영됨)"

    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, REGISTRY_PATH)
    print(f"레지스트리 원자적 교체 완료 → {REGISTRY_PATH}")


def _restore_one(backup_path: Path, final_path: Path) -> bool:
    """백업 1개 항목을 최종 경로로 복원한다. ★`os.replace(backup, final)`을
    쓰지 않는다★ — 그건 백업 파일을 최종 경로로 "옮기는" 것이라 백업
    폴더에서 그 파일이 사라져버린다(08-26 Codex 지적). 대신
    `shutil.copy2`로 **복사**해서 백업은 그대로 보존한다. 복원 직후
    바이트 단위로 백업과 동일한지 `filecmp`로 확인하고 결과를 돌려준다."""
    shutil.copy2(backup_path, final_path)
    return filecmp.cmp(backup_path, final_path, shallow=False)


def rollback_all(backup_dir: Path, staged_results: dict[tuple[str, int], dict]) -> None:
    """★08-26 재수정(Codex 지적 반영)★: 승격(7개 번들)과 레지스트리
    교체 중 **어느 단계에서 실패하든** 이 함수 하나로 8개(번들 7 +
    레지스트리 1) 전부를 백업에서 복원한다. 이전 버전은 승격 루프
    안에서만 부분 롤백을 했는데, 그 경우 "번들 7개는 승격 성공했지만
    레지스트리 교체가 실패"하는 시나리오를 못 잡았다(신번들+구
    레지스트리가 남는 혼합 상태) — 이제 승격과 레지스트리 교체를
    main()에서 하나의 try 블록으로 묶고, 어느 쪽이 실패하든 이 함수가
    둘 다 복원한다.

    - 번들 7개: `_restore_one`(shutil.copy2, 백업 보존)로 복원.
    - 레지스트리: 마찬가지로 백업을 **직접 os.replace하지 않는다** —
      임시파일에 백업 내용을 복사한 뒤 그 임시파일을 `os.replace`로
      최종 경로에 교체한다(백업 원본은 읽기만 함, 그대로 보존됨).
    - 8개 전부 복원 후 백업과 바이트 단위로 동일한지 검증하고, 하나라도
      다르면 그 사실을 명시적으로 알린다(조용히 넘어가지 않음 — 이
      함수를 호출한 쪽에서 원래 예외를 다시 올리기 전에 반드시 이
      결과를 확인할 것).
    """
    print(f"\n★복원 시작 — 백업({backup_dir})에서 번들 7개+레지스트리 전부 복원(백업은 보존)★")
    failed: list[str] = []
    for key in staged_results:
        name = BUNDLE_NAMES[key]
        ok = _restore_one(backup_dir / name, OUT / name)
        print(f"  {'복원 확인됨' if ok else '★복원 검증 실패★'}: {name}")
        if not ok:
            failed.append(name)

    backup_registry = backup_dir / REGISTRY_PATH.name
    tmp_path = REGISTRY_PATH.with_suffix(".json.rollback_tmp")
    shutil.copy2(backup_registry, tmp_path)
    os.replace(tmp_path, REGISTRY_PATH)
    registry_ok = filecmp.cmp(backup_registry, REGISTRY_PATH, shallow=False)
    print(f"  {'복원 확인됨' if registry_ok else '★복원 검증 실패★'}: {REGISTRY_PATH.name}")
    if not registry_ok:
        failed.append(REGISTRY_PATH.name)

    if failed:
        raise RuntimeError(
            f"롤백을 시도했지만 다음 {len(failed)}개 항목이 백업과 여전히 다르다"
            f"(수동 확인 필요, 백업은 그대로 있음 — {backup_dir}): {failed}"
        )
    print(f"  복원 검증 완료 — 번들 7개+레지스트리(총 8개) 전부 백업과 바이트 단위로 동일함. "
          f"백업 폴더는 손대지 않고 그대로 남아있음: {backup_dir}")


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) seed={seed}")
    print("대상: 초단기 4종(+4h 청천지수+날씨군집화 구조 유지)·단기 3종 — DSWRFLX 명시적 완전제외\n")

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    staging_dir = OUT / f"staging_{run_id}"

    backup_dir = make_backup(run_id)
    print()

    # ── 1단계: staging에 7개 전부 저장·검증(최종 경로 미접촉) ──
    print(f"=== staging 저장·검증 시작 → {staging_dir} ===")
    staged_results: dict[tuple[str, int], dict] = {}
    for h in ULTRA_HORIZONS:
        staged_results[("초단기", h)] = train_ultra_v4(h, capacity_kw, seed, staging_dir)
    for h in SHORT_HORIZONS:
        staged_results[("단기", h)] = train_short_v4(h, capacity_kw, seed, staging_dir)
    print(f"\nstaging 7/7 전부 저장·검증 통과(DSWRFLX 0건 확인됨). "
          f"최종 경로는 아직 안 건드림 — 여기까지 실패했다면 롤백할 것도 없었다.\n")

    # ── 2단계: 원자적 승격 + 레지스트리 교체(전부 통과했을 때만 도달) ──
    # ★08-26 재수정★: 번들 승격(7개)과 레지스트리 교체를 **하나의 try**로
    # 묶는다 — 번들은 전부 승격됐는데 레지스트리 교체만 실패하는 경우도
    # "실패"로 취급해 8개(번들7+레지스트리) 전부 롤백해야 하기 때문이다
    # (이전 버전은 승격 루프 안에서만 부분 롤백해서 이 경우를 못 잡았다).
    print("=== 최종 경로로 원자적 승격 + 레지스트리 교체 시작 ===")
    try:
        promote_bundles(staged_results)
        promote_registry(staged_results)
    except Exception as exc:
        print(f"\n★승격 또는 레지스트리 교체 중 실패: {type(exc).__name__}: {exc}★")
        rollback_all(backup_dir, staged_results)
        raise

    print(f"\n승격 8/8(번들7+레지스트리) 완료. 백업: {backup_dir}")
    print("\n★다음 단계★ golden_replay_전체8종_v1_2026-08-25.py(DSWRFLX 부재 검사 포함)로 "
          "8종 재검증할 것.")


if __name__ == "__main__":
    main()
