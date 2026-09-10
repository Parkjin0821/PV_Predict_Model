# -*- coding: utf-8 -*-
"""초단기 4종·단기 3종 — DSWRFLX 명시적 완전제외 재검증(동일행 5폴드).

## 08-26 개정 이력(둘 다 실행 전 코드 리뷰로 잡힌 결함 — 실행해서 드러난 게 아님)

### 1차 결함(v1 몽키패치) — 앞선 스크립트들
`harness.FEATURE_SETS`/`sel.OBSERVED_COLUMNS`/`FORECAST_COLUMNS`를
몽키패치했지만, 실제 저장된 운영 joblib 7종을 직접 열어보니 전부
`features`에 `DSWRFLX_bsrn정제`가 남아있었다. 원인: `dpc.load_ultra_
frame()`/`load_short_frame()`·`ultra.build_ultra_short_frame()`가
**각자 자기 파일 안에서 `harness`를 독립적으로 새로 로드**해서, 패치가
실제 프레임 조립 경로에 안 먹힌다(AGENTS.md 08-26절에 상세 기록).
**해결**: 몽키패치를 버리고, `features = base_cols + chosen` **직후**
DSWRFLX를 명시적으로 제거하는 한 줄을 추가(leak 경로와 무관하게 최종
목록만 확정적으로 걸러냄).

### 2차 결함(이 파일의 이전 버전) — Codex 재지적, 반영 완료
초단기 전체를 `e2e_retrain_v5_공식B_v1_2026-08-24.py::run_ultra()`
하나로만 처리했는데, **+4h의 현재 공식은 v1이 아니라 v2 패치**
(`e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py::run_ultra_patch()`,
청천지수+날씨군집화+군집 스케일러/KMeans)다 — 이건 이전에 `retrain_
exclude_dswrflx_v1_2026-08-26.py`에서 이미 한 번 저지르고 고쳤던 바로
그 실수를 이 v2 스크립트에서 또 반복한 것이었다. **해결**: +4h 전용
`run_ultra4_v3()`를 `run_ultra_patch()`의 군집화·카파 역변환 로직
그대로 포크해 추가했다(구조선택 없이 청천지수 카파 타깃 + 날씨군집화,
`improvement.add_weather_clusters()` 그대로 재사용).

## 자체검증 확장(Codex 요청 반영)
"포크가 원본과 동일하게 동작하는가"를 **7개 수평 전부**에 대해
확인한다(이전엔 초단기 +3h 하나만 확인했었음):
- 초단기 +1h·+2h·+3h → `e2e.run_ultra()`(v1, 그 수평들의 현재 공식)와 비교
- 초단기 +4h → `run_ultra_patch()`(v2, +4h의 현재 공식)와 비교
- 단기 +1h·+24h·+48h → `e2e.run_short()`와 비교
전부 **행수·예측값까지 완전 일치**해야 통과다(포함 기준선=원본과
같아야 그 위에 얹는 "제외" 비교가 의미 있다).

## 저장 직전·재적재 직후 assert
제외(`exclude_dsx=True`) 대상 조합은 저장 직전 `bundle["features"]`와
재적재한 파일의 `features` 양쪽에서 DSWRFLX 포함 특성 0건을 assert한다.

## 실행 안 함(사용자 지시)
이 파일은 **작성만 하고 실행하지 않았다.** 운영모델도 전혀 안 건드린다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "DSWRFLX_명시적완전제외_재검증_v2_2026-08-26"
MODEL_DIR = OUT / "fold_models"
DSX = "DSWRFLX_bsrn정제"
LABEL_BEFORE = "DSX_포함(공식재현)"
LABEL_AFTER = f"{DSX}_명시적완전제외"
KEYS = ["티어", "수평_h", "폴드", "발행시각"]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("v2fix_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
v2patch = _load("v2fix_v2patch", "e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py")
dpc, harness, ultra, clearsky, improvement = e2e.dpc, e2e.harness, e2e.ultra, e2e.clearsky, e2e.improvement
OFFICIAL_WINDOWS = e2e.OFFICIAL_WINDOWS
N_INVERTERS = e2e.N_INVERTERS
DIF = e2e.DIF
PLANT_ID, PLANT_NAME = e2e.PLANT_ID, e2e.PLANT_NAME
ULTRA_OFFICIAL = e2e.ULTRA_OFFICIAL
ULTRA_STRUCTURE = e2e.ULTRA_STRUCTURE
add_difswrf_flag = e2e.add_difswrf_flag
_safe = e2e._safe


def _strip_dsx(features: list[str]) -> list[str]:
    return [c for c in features if DSX not in c]


def _assert_clean(features: list[str], where: str) -> None:
    leaked = [c for c in features if DSX in c]
    if leaked:
        raise AssertionError(f"[{where}] DSWRFLX가 아직 남아있음: {leaked}")


def _save_and_verify(model, bundle: dict, x: pd.DataFrame, path: Path, exclude_dsx: bool):
    if exclude_dsx:
        _assert_clean(bundle["features"], "저장 직전")
    before = np.asarray(model.predict(x), float)
    full_bundle = {**bundle, "model": model}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_bundle, path)
    loaded = joblib.load(path)
    if exclude_dsx:
        _assert_clean(loaded["features"], "재적재 직후")
    after = np.asarray(loaded["model"].predict(x), float)
    diff = float(np.max(np.abs(before - after))) if len(before) else 0.0
    return after, diff


# ─────────────────────── 초단기 +1h·+2h·+3h(v1 패턴) ───────────────────────

def run_ultra_v3(capacity_kw: float, seed: int, exclude_dsx: bool,
                 horizons: tuple = (1, 2, 3)) -> tuple[pd.DataFrame, list[dict]]:
    """e2e.run_ultra()의 정책B 경로를 그대로 복제 + features 명시적 필터
    한 줄만 추가. +4h는 다루지 않는다(아래 run_ultra4_v3 참고 — 현재
    공식이 v2 패턴이라 이 함수(v1 패턴)로 다루면 또 stale 모델이 된다)."""
    label = LABEL_AFTER if exclude_dsx else LABEL_BEFORE
    rows, audits = [], []
    for horizon in horizons:
        official = ULTRA_OFFICIAL[horizon]
        frame = add_difswrf_flag(dpc.load_ultra_frame(horizon))
        candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        use_kappa = "청천지수" in official
        frame["_청천_kW"] = np.clip(
            capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
        frame["_정책타깃_kW"] = frame["목표_발전출력_kW"]
        frame["_카파"] = frame["_정책타깃_kW"] / frame["_청천_kW"]
        min_elev = clearsky.MIN_ELEVATION_DEG if use_kappa else 0.0
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]

        for fold, s, e_ in OFFICIAL_WINDOWS:
            start, end = pd.Timestamp(s), pd.Timestamp(e_) + pd.Timedelta(days=1)
            full_mask = daylight["_목표_가용인버터수"] >= N_INVERTERS
            train_pool = daylight[daylight.index < start]
            train_b = train_pool[full_mask.loc[train_pool.index]]
            test_b = daylight[(daylight.index >= start) & (daylight.index < end) & full_mask]
            if len(train_b) < 300 or len(test_b) < 30:
                print(f"  [초단기 +{horizon}h {fold}] 표본부족 — 건너뜀")
                continue
            sel_input = train_b.drop(columns=["목표_발전출력_kW"]).assign(
                목표_발전출력_kW=train_b["_정책타깃_kW"]).dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            if exclude_dsx:
                features = _strip_dsx(features)  # ★유일한 실질 변경점★
            native_ok = harness.NATIVE_MISSING_OK | {DIF}
            required = [c for c in features if c not in native_ok] + ["_정책타깃_kW", "_청천_kW", "_카파"]
            train = train_b.dropna(subset=required)
            test = test_b.dropna(subset=[c for c in features if c not in native_ok]
                                 + ["목표_발전출력_kW", "_청천_kW"])
            if len(train) < 300 or len(test) < 30:
                continue

            target = "_카파" if use_kappa else "_정책타깃_kW"
            structure_name, params = "기본", None
            if ULTRA_STRUCTURE[horizon]:
                params, structure_name = improvement.choose_structure_kfold(
                    "초단기", train, features, target, "kappa" if use_kappa else "raw", capacity_kw, seed)
                model = improvement.make_model("초단기", seed, params)
            else:
                model = ultra.make_model("LightGBM", seed)
            model.fit(train[features], train[target])

            variant = official + ("+구조선택" if ULTRA_STRUCTURE[horizon] else "") + f"[{label}]"
            artifact = MODEL_DIR / _safe(f"초단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = _save_and_verify(model, {
                "tier": "초단기", "horizon_h": horizon, "variant": variant, "features": features,
                "params": params, "target_transform": target, "structure_name": structure_name,
                "정책": "B_구간제외", "용량프로필": "inverter_registered_sum_219",
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, test[features], artifact, exclude_dsx)
            pred = raw * test["_청천_kW"].to_numpy() if use_kappa else raw
            pred = np.clip(pred, 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon, unit="h")
            for issued, tt, y, p in zip(test.index, target_at, actual, pred):
                rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "초단기",
                            "수평_h": horizon, "폴드": fold, "발행시각": issued, "대상시각": tt,
                            "실제_kW": y, "예측_kW": p, "구성": label, "모델파일": str(artifact)})
            audits.append({"티어": "초단기", "수평_h": horizon, "폴드": fold, "구성": label,
                           "재적재차이": diff, "특성수": len(features),
                           "DSWRFLX_포함여부": any(DSX in c for c in features),
                           "학습행수": len(train), "시험행수": len(test)})
            print(f"[초단기 +{horizon}h {fold}] {label} 완료 (학습{len(train):,}/시험{len(test):,}, "
                  f"특성수={len(features)}, DSWRFLX포함={any(DSX in c for c in features)})")
    return pd.DataFrame(rows), audits


# ───────────────── 초단기 +4h(v2 패턴 — 청천지수+날씨군집화) ─────────────────

def run_ultra4_v3(capacity_kw: float, seed: int, exclude_dsx: bool) -> tuple[pd.DataFrame, list[dict]]:
    """v2patch.run_ultra_patch()(현재 공식 +4h: 청천지수 카파 타깃 +
    날씨군집화, 구조선택 없음)를 그대로 복제 + features 명시적 필터 한 줄만
    추가. DSX를 `add_weather_clusters()` 호출 **전에** 걸러내므로,
    `improvement.CLUSTER_SOURCE`를 따로 건드리지 않아도 군집 입력에도
    자동으로 안 들어간다(`cols = [c for c in CLUSTER_SOURCE if c in
    features ...]`가 걸러진 features를 보므로)."""
    label = LABEL_AFTER if exclude_dsx else LABEL_BEFORE
    horizon = 4
    rows, audits = [], []
    official = v2patch.ULTRA_OFFICIAL_V2[horizon]  # "청천지수"
    frame = add_difswrf_flag(dpc.load_ultra_frame(horizon))
    candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    use_kappa = "청천지수" in official
    frame["_청천_kW"] = np.clip(
        capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_정책타깃_kW"] = frame["목표_발전출력_kW"]  # 정책B, C 미대상
    frame["_카파"] = frame["_정책타깃_kW"] / frame["_청천_kW"]
    min_elev = clearsky.MIN_ELEVATION_DEG if use_kappa else 0.0
    daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]

    for fold, s, e_ in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e_) + pd.Timedelta(days=1)
        full_mask = daylight["_목표_가용인버터수"] >= N_INVERTERS
        train_pool = daylight[daylight.index < start]
        train_b = train_pool[full_mask.loc[train_pool.index]]
        test_b = daylight[(daylight.index >= start) & (daylight.index < end) & full_mask]
        if len(train_b) < 300 or len(test_b) < 30:
            print(f"  [초단기 +4h {fold}] 표본부족 — 건너뜀")
            continue
        sel_input = train_b.drop(columns=["목표_발전출력_kW"]).assign(
            목표_발전출력_kW=train_b["_정책타깃_kW"]).dropna(subset=["목표_발전출력_kW"])
        chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        if exclude_dsx:
            features = _strip_dsx(features)  # ★유일한 실질 변경점 — 군집화 전에 걸러짐★
        native_ok = harness.NATIVE_MISSING_OK | {DIF}
        required = [c for c in features if c not in native_ok] + ["_정책타깃_kW", "_청천_kW", "_카파"]
        train = train_b.dropna(subset=required)
        test = test_b.dropna(subset=[c for c in features if c not in native_ok]
                             + ["목표_발전출력_kW", "_청천_kW"])
        if len(train) < 300 or len(test) < 30:
            continue

        # ★v2 핵심★: 구조선택 대신 날씨군집화(run_ultra_patch()와 완전히 동일 호출).
        tr, te, feat2, cluster_cols = improvement.add_weather_clusters(train, test, features, seed)
        cluster_names = [f"날씨군집_{k}" for k in range(4)] if cluster_cols else []
        if exclude_dsx and any(DSX in c for c in cluster_cols):
            raise AssertionError(f"[{fold}] 군집원천에 DSWRFLX가 여전히 있음: {cluster_cols}")

        결측행_수 = 0
        군집배정_결측0건 = True
        if cluster_cols:
            miss_mask = te[cluster_cols].isna().any(axis=1)
            결측행_수 = int(miss_mask.sum())
            assigned = te[cluster_names].sum(axis=1)
            군집배정_결측0건 = bool((assigned == 1).all())
            if not 군집배정_결측0건:
                raise RuntimeError(f"[초단기 +4h {fold}] 군집 미배정 행 발생")

        target = "_카파" if use_kappa else "_정책타깃_kW"
        model = ultra.make_model("LightGBM", seed)
        model.fit(tr[feat2], tr[target])

        variant = official + "+날씨군집화" + f"[{label}]"
        artifact = MODEL_DIR / _safe(f"초단기_h4_{fold}_{variant}.joblib")
        raw, diff = _save_and_verify(model, {
            "tier": "초단기", "horizon_h": horizon, "variant": variant, "features": feat2,
            "params": None, "target_transform": target, "structure_name": "기본(군집화로대체)",
            "군집원천특성": cluster_cols, "정책": "B_구간제외",
            "용량프로필": "inverter_registered_sum_219",
            "train_end": str(train.index.max()), "test_start": str(test.index.min()),
        }, te[feat2], artifact, exclude_dsx)
        pred = raw * te["_청천_kW"].to_numpy() if use_kappa else raw
        pred = np.clip(pred, 0, capacity_kw)
        actual = te["목표_발전출력_kW"].to_numpy()
        target_at = te.index + pd.to_timedelta(horizon, unit="h")
        for issued, tt, y, p in zip(te.index, target_at, actual, pred):
            rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "초단기",
                        "수평_h": horizon, "폴드": fold, "발행시각": issued, "대상시각": tt,
                        "실제_kW": y, "예측_kW": p, "구성": label, "모델파일": str(artifact)})
        audits.append({"티어": "초단기", "수평_h": horizon, "폴드": fold, "구성": label,
                       "재적재차이": diff, "특성수": len(feat2),
                       "DSWRFLX_포함여부": any(DSX in c for c in feat2),
                       "군집원천특성수": len(cluster_cols), "시험_결측행수": 결측행_수,
                       "군집배정_전부정상": 군집배정_결측0건,
                       "학습행수": len(tr), "시험행수": len(te)})
        print(f"[초단기 +4h {fold}] {label} 완료 (학습{len(tr):,}/시험{len(te):,}, "
              f"특성수={len(feat2)}, 군집원천={len(cluster_cols)}개, "
              f"DSWRFLX포함={any(DSX in c for c in feat2)})")
    return pd.DataFrame(rows), audits


# ─────────────────────────── 단기 +1h·+24h·+48h ───────────────────────────

def run_short_v3(capacity_kw: float, seed: int, exclude_dsx: bool) -> tuple[pd.DataFrame, list[dict]]:
    """e2e.run_short()를 그대로 복제 + features 명시적 필터 한 줄만 추가."""
    label = LABEL_AFTER if exclude_dsx else LABEL_BEFORE
    rows, audits = [], []
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    for horizon in (1, 24, 48):
        frame = add_difswrf_flag(dpc.load_short_frame(horizon))
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        daylight = frame[frame["목표_낮시간"] > 0]

        for fold, s, e_ in OFFICIAL_WINDOWS:
            start, end = pd.Timestamp(s), pd.Timestamp(e_) + pd.Timedelta(days=1)
            train_b = daylight[(daylight.index < start) & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
            test_b = daylight[(daylight.index >= start) & (daylight.index < end)
                              & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
            if len(train_b) < 200 or len(test_b) < 30:
                print(f"  [단기 +{horizon}h {fold}] 표본부족 — 건너뜀")
                continue
            chosen = harness.select_features_in_fold(
                train_b.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            if exclude_dsx:
                features = _strip_dsx(features)  # ★유일한 실질 변경점★
            native_ok = harness.NATIVE_MISSING_OK | {DIF}
            required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
            train = train_b.dropna(subset=required)
            test = test_b.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue

            params, structure_name = improvement.choose_structure_kfold(
                "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
            model = improvement.make_model("단기", seed, params)
            model.fit(train[features], train["목표_발전출력_kW"])

            variant = f"기본+구조선택[{label}]"
            artifact = MODEL_DIR / _safe(f"단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = _save_and_verify(model, {
                "tier": "단기", "horizon_h": horizon, "variant": variant, "features": features,
                "params": params, "target_transform": "raw_kW", "structure_name": structure_name,
                "정책": "B_구간제외", "용량프로필": "inverter_registered_sum_219",
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, test[features], artifact, exclude_dsx)
            pred = np.clip(raw, 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon - 1, unit="h")
            for issued, tt, y, p in zip(test.index, target_at, actual, pred):
                rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "단기",
                            "수평_h": horizon, "폴드": fold, "발행시각": issued, "대상시각": tt,
                            "실제_kW": y, "예측_kW": p, "구성": label, "모델파일": str(artifact)})
            audits.append({"티어": "단기", "수평_h": horizon, "폴드": fold, "구성": label,
                           "재적재차이": diff, "특성수": len(features),
                           "DSWRFLX_포함여부": any(DSX in c for c in features),
                           "학습행수": len(train), "시험행수": len(test)})
            print(f"[단기 +{horizon}h {fold}] {label} 완료 (학습{len(train):,}/시험{len(test):,}, "
                  f"특성수={len(features)}, DSWRFLX포함={any(DSX in c for c in features)})")
    return pd.DataFrame(rows), audits


# ────────────────────────────── 자체검증(확장) ──────────────────────────────

def _compare_exact(label: str, official_df: pd.DataFrame, fork_df: pd.DataFrame) -> None:
    """official_df(진짜 공식 생성함수 결과) vs fork_df(이 파일의 exclude_dsx=False
    결과)가 행수·예측값까지 완전히 같은지 확인한다. 하나라도 다르면 즉시 예외 —
    포함 기준선이 원본과 다르면 그 위에 얹는 "제외" 비교 전체가 무의미해진다."""
    merged = official_df[KEYS + ["예측_kW"]].merge(
        fork_df[KEYS + ["예측_kW"]], on=KEYS, suffixes=("_공식", "_포크"), validate="one_to_one")
    if len(merged) != len(official_df) or len(merged) != len(fork_df):
        raise AssertionError(
            f"[{label}] 행수 불일치 — 공식={len(official_df)}, 포크={len(fork_df)}, "
            f"교집합={len(merged)}. 포크가 원본과 다른 표본을 만들고 있다."
        )
    max_diff = float((merged["예측_kW_공식"] - merged["예측_kW_포크"]).abs().max())
    print(f"  [{label}] 동일행 {len(merged)}개, 최대 예측차이: {max_diff:.2e}")
    if max_diff > 1e-9:
        raise AssertionError(
            f"[{label}] 자체검증 실패 — 포크(exclude_dsx=False)가 공식 생성함수와 "
            f"다른 예측을 낸다(최대차이 {max_diff}). 본 비교 결과를 신뢰하지 말 것."
        )


class _redirect_model_dir:
    """모듈 자신의 MODEL_DIR 전역을 임시로 안전한 경로로 바꿨다가
    with 블록을 벗어나면(예외가 나도) 반드시 원복한다.

    ★이전 DSWRFLX leak과는 다른 상황★: 그때는 `dpc`/`ultra`가 각자
    독립적으로 새로 로드한 *다른* 모듈 인스턴스의 `harness.sel`을 건드려서
    안 먹혔다. 여기서는 `run_ultra_patch()`/`run_short()`가 **자기 자신이
    속한 모듈의 전역이름** `MODEL_DIR`을 그대로 참조하므로, 그 모듈 객체
    (`v2patch`, `e2e`)를 직접 패치하면 확실히 반영된다 — 다른 모듈이 별도
    사본을 갖고 있지 않다.
    """

    def __init__(self, module, new_dir: Path):
        self.module = module
        self.new_dir = new_dir

    def __enter__(self):
        self.old_dir = self.module.MODEL_DIR
        self.new_dir.mkdir(parents=True, exist_ok=True)
        self.module.MODEL_DIR = self.new_dir
        return self.new_dir

    def __exit__(self, exc_type, exc, tb):
        self.module.MODEL_DIR = self.old_dir
        return False  # 예외는 그대로 전파(조용히 삼키지 않음)


def self_check_against_official(capacity_kw: float, seed: int) -> None:
    """7개 수평 전부: 이 파일의 exclude_dsx=False 결과가 '진짜 공식' 생성함수
    (초단기 +1~3h→e2e.run_ultra, +4h→v2patch.run_ultra_patch, 단기→
    e2e.run_short)와 행수·예측값까지 완전히 같은지 확인한다.

    ★안전장치(Codex 지적 반영)★: `run_ultra_patch()`/`run_short()`는
    `e2e.run_ultra()`와 달리 `model_dir` 매개변수가 없어 호출하면 자기
    모듈의 `MODEL_DIR`(진짜 공식 산출물이 있는 `outputs/E2E_v5_공식B_
    v2_⑥반영_2026-08-24/fold_models/`, `outputs/E2E_v5_공식B_v1_
    2026-08-24/fold_models/`)에 그대로 덮어쓴다 — 자체검증일 뿐인데
    실제 공식 산출물을 훼손할 뻔했다. `_redirect_model_dir`로 호출
    직전에만 `OUT/자체검증_공식/...`로 돌려놓고 `finally`로 반드시
    원복한다(중간에 예외가 나도 원복됨 — with문 보장).
    """
    print("=== 자체검증: exclude_dsx=False가 공식 생성함수와 완전히 같은가(7개 수평) ===")

    ultra123_model_dir = OUT / "자체검증_공식" / "초단기_123"
    ultra123_model_dir.mkdir(parents=True, exist_ok=True)
    official_ultra123, _ = e2e.run_ultra(capacity_kw, seed, horizons=(1, 2, 3),
                                         model_dir=ultra123_model_dir)
    fork_ultra123, _ = run_ultra_v3(capacity_kw, seed, exclude_dsx=False, horizons=(1, 2, 3))
    for h in (1, 2, 3):
        _compare_exact(f"초단기 +{h}h", official_ultra123[official_ultra123["수평_h"] == h],
                       fork_ultra123[fork_ultra123["수평_h"] == h])

    with _redirect_model_dir(v2patch, OUT / "자체검증_공식" / "초단기_h4"):
        official_ultra4, _ = v2patch.run_ultra_patch(capacity_kw, seed)
    fork_ultra4, _ = run_ultra4_v3(capacity_kw, seed, exclude_dsx=False)
    _compare_exact("초단기 +4h", official_ultra4, fork_ultra4)

    with _redirect_model_dir(e2e, OUT / "자체검증_공식" / "단기"):
        official_short, _ = e2e.run_short(capacity_kw, seed)
    fork_short, _ = run_short_v3(capacity_kw, seed, exclude_dsx=False)
    for h in (1, 24, 48):
        _compare_exact(f"단기 +{h}h", official_short[official_short["수평_h"] == h],
                       fork_short[fork_short["수평_h"] == h])

    print("자체검증 전부 통과 — 포크의 포함(exclude_dsx=False) 기준선이 "
          "현재 공식 백테스트와 행수·예측값까지 완전히 동일함을 확인했다.\n"
          f"(자체검증용 fold_models는 전부 {OUT / '자체검증_공식'}로 격리 저장됨 — "
          "진짜 공식 outputs/E2E_v5_공식B_*/fold_models/는 안 건드림.)")


# ─────────────────────────────── 비교·집계 ───────────────────────────────

def align_same_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = df[df["구성"] == LABEL_BEFORE]
    after = df[df["구성"] == LABEL_AFTER]
    merged = before[KEYS + ["실제_kW", "예측_kW"]].merge(
        after[KEYS + ["실제_kW", "예측_kW"]], on=KEYS, suffixes=("_기존", "_제외"), validate="one_to_one")
    mismatch = (merged["실제_kW_기존"] - merged["실제_kW_제외"]).abs() > 1e-9
    if bool(mismatch.any()):
        raise RuntimeError(f"동일 키인데 실측값이 다른 행 {int(mismatch.sum())}건 — 표본 정렬 이상. 중단.")
    coverage = pd.DataFrame([
        {"구성": LABEL_BEFORE, "자체_행수": len(before), "교집합_행수": len(merged), "교집합밖_행수": len(before) - len(merged)},
        {"구성": LABEL_AFTER, "자체_행수": len(after), "교집합_행수": len(merged), "교집합밖_행수": len(after) - len(merged)},
    ])
    return merged, coverage


def performance_delta(merged: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    rows = []
    for (tier, h), g in merged.groupby(["티어", "수평_h"], sort=False):
        actual = g["실제_kW_기존"].to_numpy()
        rec = {"티어": tier, "수평_h": h, "동일행_n": len(g)}
        for label, col in ((LABEL_BEFORE, "예측_kW_기존"), (LABEL_AFTER, "예측_kW_제외")):
            e = actual - g[col].to_numpy()
            mae, rmse = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
            rec[f"nMAE_pct_{label}"] = round(mae / capacity_kw * 100, 3)
            rec[f"nRMSE_pct_{label}"] = round(rmse / capacity_kw * 100, 3)
        rec["nMAE_pct_변화"] = round(rec[f"nMAE_pct_{LABEL_AFTER}"] - rec[f"nMAE_pct_{LABEL_BEFORE}"], 3)
        rec["nRMSE_pct_변화"] = round(rec[f"nRMSE_pct_{LABEL_AFTER}"] - rec[f"nRMSE_pct_{LABEL_BEFORE}"], 3)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) / "
          f"대상: 초단기 4종(+4h는 v2 청천지수+날씨군집화)+단기 3종 / 폴드: 공식 5폴드\n")

    self_check_against_official(capacity_kw, seed)

    print(f"=== [1/2] {LABEL_BEFORE} ===")
    ultra123_before, aud_u123_before = run_ultra_v3(capacity_kw, seed, exclude_dsx=False)
    ultra4_before, aud_u4_before = run_ultra4_v3(capacity_kw, seed, exclude_dsx=False)
    short_before, aud_s_before = run_short_v3(capacity_kw, seed, exclude_dsx=False)

    print(f"\n=== [2/2] {LABEL_AFTER} ===")
    ultra123_after, aud_u123_after = run_ultra_v3(capacity_kw, seed, exclude_dsx=True)
    ultra4_after, aud_u4_after = run_ultra4_v3(capacity_kw, seed, exclude_dsx=True)
    short_after, aud_s_after = run_short_v3(capacity_kw, seed, exclude_dsx=True)

    all_rows = pd.concat([ultra123_before, ultra4_before, short_before,
                          ultra123_after, ultra4_after, short_after], ignore_index=True)
    all_rows.to_csv(OUT / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")

    merged, coverage = align_same_rows(all_rows)
    coverage.to_csv(OUT / "표본커버리지.csv", index=False, encoding="utf-8-sig")
    perf = performance_delta(merged, capacity_kw)
    perf.to_csv(OUT / "성능비교.csv", index=False, encoding="utf-8-sig")

    all_audits = (aud_u123_before + aud_u4_before + aud_s_before
                  + aud_u123_after + aud_u4_after + aud_s_after)
    audit_df = pd.DataFrame(all_audits)
    audit_df.to_csv(OUT / "감사로그.csv", index=False, encoding="utf-8-sig")
    residual_leak = audit_df[(audit_df["구성"] == LABEL_AFTER) & (audit_df["DSWRFLX_포함여부"])]
    if len(residual_leak):
        raise AssertionError(f"제외 대상인데 DSWRFLX가 남은 조합이 있다:\n{residual_leak}")

    print("\n=== 표본 커버리지 ===")
    print(coverage.to_string(index=False))
    print("\n=== 성능비교(동일행 기준, 포함 vs 명시적 완전제외) ===")
    print(perf.to_string(index=False))
    print(f"\n감사로그: 제외 대상 {len(audit_df[audit_df['구성']==LABEL_AFTER])}개 조합 전부 "
          f"DSWRFLX_포함여부=False 확인됨(위반 0건).")
    print(f"\n저장 완료: {OUT}")
    print("\n★주의★ 이 결과는 채택 여부를 자동 결정하지 않는다 — Claude에게 전달해 판단받을 것.")


if __name__ == "__main__":
    main()
