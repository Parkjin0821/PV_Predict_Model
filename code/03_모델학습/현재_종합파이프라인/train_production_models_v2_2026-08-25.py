# -*- coding: utf-8 -*-
"""★운영용 모델 학습 v2 — Codex 리뷰(08-25) 반영, 번들 완전화★

v1(`train_production_models_v1_2026-08-25.py`)이 만든 8개 joblib은
**"이미 만들어진 특성으로 예측"만 재적재검증**했다. Codex가 지적한 대로
이건 실전 운영검사가 아니다 — 실제로는 원자료부터 특성을 만드는
과정까지 필요한데, 다음 두 가지가 번들에서 빠져 있었다.

1. **초단기 +4h(청천지수+날씨군집화)**: 학습 때 쓴 KMeans·스케일러·
   결측대체 중앙값이 저장 안 됨 → 새 데이터의 날씨군집을 재현할 방법이
   없었음.
2. **초단기 +3h·+4h(청천지수 κ 모델)**: κ→kW 역변환 레시피(태양고도·
   Haurwitz 청천일사·설비용량·clip)가 "target_transform=kappa"라는
   표시만 있고 실행 가능한 형태로 안 묶여 있었음.

## v1과 다른 점(이 두 가지만)
- h=4: `model_improvement_round2.add_weather_clusters`를 그대로 호출하는
  대신, 같은 계산을 인라인으로 재현해 **KMeans·스케일러·중앙값 객체
  자체를 번들에 저장**한다(재구현이 아니라 "부산물까지 붙잡는" 확장).
- h=3·h=4: 번들에 `카파_역변환` 딕셔너리(최소태양고도·capacity_kw·clip)를
  추가한다.
- 모든 번들에 `번들버전="v2_2026-08-25"`, `결측처리규칙` 텍스트 추가.
- 저장 직후 `production_inference_utils.predict_kw()`로 **군집배정+κ
  역변환을 포함한 전체 추론경로**가 학습 시점 값과 일치하는지 검증한다
  (v1은 raw model.predict()만 재검증해 이 부분을 놓쳤었다).

나머지(초단기 +1h·+2h, 단기 3개, 일간)는 v1과 완전히 동일한 로직 —
새로 손댈 이유가 없어서 그대로 재사용한다.

## 출력 (`outputs/운영모델_v2_2026-08-25/`)
joblib 8개(v1과 같은 파일명, 새 폴더에 저장) + `운영모델_목록.json`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "운영모델_v2_2026-08-25"
N_INVERTERS = 5
DIF = "DIFSWRF_bsrn정제"
BUNDLE_VERSION = "v2_2026-08-25"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("prod2_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc, harness, ultra, improvement, clearsky = e2e.dpc, e2e.harness, e2e.ultra, e2e.improvement, e2e.clearsky
add_difswrf_flag = e2e.add_difswrf_flag
daily_mod = _load("prod2_daily", "daily_direct_final_audit_v1_2026-08-25.py")
tuning = _load("prod2_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
infer = _load("prod2_infer", "production_inference_utils_v1_2026-08-25.py")

CLUSTER_SOURCE = improvement.CLUSTER_SOURCE  # 재사용 — 여기서 새로 정의하지 않음
ULTRA_OFFICIAL = {1: "raw", 2: "raw", 3: "청천지수", 4: "청천지수"}
ULTRA_STRUCTURE_TYPE = {1: "구조선택", 2: "구조선택", 3: "기본", 4: "날씨군집화(⑥)"}
MISSING_RULE_TEXT = ("LightGBM native missing 위임 — 중앙값/평균 등 사후대체 없음. "
                      "DIFSWRF_bsrn정제만 C전략(원값 NaN 유지+결측여부 플래그)으로 명시 처리.")


def fit_weather_clusters(train: pd.DataFrame, features: list[str], seed: int):
    """model_improvement_round2.add_weather_clusters와 정확히 같은 계산을
    수행하되, 부산물(scaler·km·med)까지 반환한다(재구현이 아니라 확장)."""
    cols = [c for c in CLUSTER_SOURCE if c in features and c in train]
    if len(cols) < 2 or len(train) < 200:
        raise RuntimeError(f"군집화 표본/입력컬럼 부족: cols={cols}, n={len(train)}")
    med = train[cols].median(numeric_only=True)
    scaler = StandardScaler().fit(train[cols].fillna(med))
    z = scaler.transform(train[cols].fillna(med))
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(z)
    labels = km.predict(z)
    out = train.copy()
    added = []
    for k in range(4):
        name = f"날씨군집_{k}"
        out[name] = (labels == k).astype(int)
        added.append(name)
    cluster_bundle = {"입력컬럼": cols, "결측대체_중앙값": med, "스케일러": scaler,
                      "kmeans": km, "군집수": 4}
    return out, features + added, cluster_bundle


def verify_full_chain(bundle: dict, train: pd.DataFrame, path: Path) -> float:
    """저장→재적재→predict_kw()까지 전체 추론경로로 재검증(raw model.predict만
    보는 게 아니라 군집배정·κ역변환까지 포함) — Codex 지적사항 반영."""
    loaded = joblib.load(path)
    sample = train.tail(50).copy()
    # predict_kw()가 "군집열 없는 원본 입력"에서 군집배정+κ역변환을 전부 다시
    # 해서, model.predict()를 직접 불러 같은 후처리를 한 것과 일치하는지 비교한다.
    base_cols = [c for c in sample.columns if not c.startswith("날씨군집_")]
    pred_via_util = infer.predict_kw(loaded, sample[base_cols])
    direct = np.asarray(bundle["model"].predict(sample[bundle["features"]]), float)
    clip_상한 = bundle.get("clip_상한", bundle["capacity_kw"])
    if bundle["타깃유형"] == "kappa":
        cs_kw = np.clip(bundle["capacity_kw"] * infer.clear_sky_ghi(sample["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
        direct_kw = np.clip(direct * cs_kw, 0, clip_상한)
    else:
        direct_kw = np.clip(direct, 0, clip_상한)
    diff = float(np.max(np.abs(pred_via_util - direct_kw)))
    # ★검증 자체의 검증★: 카파모델인데 raw 예측치(보통 0~1대)와
    # direct_kw(0~capacity_kw대) 스케일이 같으면 역변환이 조용히 스킵된
    # 것이므로(이번에 실제로 한 번 겪었음) 명시적으로 실패시킨다.
    if bundle["타깃유형"] == "kappa":
        raw_scale = float(np.max(np.abs(direct)))
        kw_scale = float(np.max(np.abs(direct_kw)))
        if kw_scale > 1.0 and raw_scale > 0 and abs(raw_scale - kw_scale) < 1e-9:
            raise RuntimeError(f"카파 역변환이 적용 안 된 것으로 보임(raw_scale={raw_scale}, kw_scale={kw_scale})")
    return diff


def train_ultra(h: int, capacity_kw: float, seed: int) -> dict:
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
    native_ok = harness.NATIVE_MISSING_OK | {DIF}
    required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW", "_청천_kW", "_카파"]
    train = full.dropna(subset=required)
    train["_카파_원본"] = train["_카파"]  # 전체체인 검증용으로만 남김(특성 아님)

    target = "_카파" if use_kappa else "목표_발전출력_kW"
    cluster_bundle = None
    if h == 4:
        train, features, cluster_bundle = fit_weather_clusters(train, features, seed)
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

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"운영모델_초단기_h{h}.joblib"
    bundle = {
        "tier": "초단기", "horizon_h": h, "target_transform": target,
        "타깃유형": ("kappa" if use_kappa else "raw"), "structure_name": structure_name,
        "features": features, "model": model, "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "clip_상한": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
        "번들버전": BUNDLE_VERSION, "결측처리규칙": MISSING_RULE_TEXT,
        "카파_역변환": ({"최소태양고도_deg": min_elev, "청천일사공식": "Haurwitz(1945)",
                     "clip범위_kW": [0, capacity_kw]} if use_kappa else None),
        "클러스터": cluster_bundle,
    }
    joblib.dump(bundle, path)
    chain_diff = verify_full_chain(bundle, train, path)
    print(f"[초단기 +{h}h] 학습완료 학습행수={len(train):,} 특성수={len(features)} "
          f"구조={structure_name} 학습기간={train.index.min().date()}~{train.index.max().date()} "
          f"전체체인재검증차이={chain_diff:.2e}")
    return {"파일": str(path), "학습행수": len(train), "특성수": len(features),
            "구조": structure_name, "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
            "전체체인재검증차이": chain_diff}


def train_short(h: int, capacity_kw: float, seed: int) -> dict:
    frame = add_difswrf_flag(dpc.load_short_frame(h))
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]
    full = daylight[daylight["_목표_가용인버터수"] >= N_INVERTERS].copy()

    chosen = harness.select_features_in_fold(
        full.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
    features = base_cols + chosen
    native_ok = harness.NATIVE_MISSING_OK | {DIF}
    required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
    train = full.dropna(subset=required)

    params, structure_name = improvement.choose_structure_kfold(
        "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
    model = improvement.make_model("단기", seed, params)
    model.fit(train[features], train["목표_발전출력_kW"])

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"운영모델_단기_h{h}.joblib"
    bundle = {
        "tier": "단기", "horizon_h": h, "target_transform": "raw_kW",
        "타깃유형": "raw", "structure_name": structure_name,
        "features": features, "model": model, "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "clip_상한": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
        "번들버전": BUNDLE_VERSION, "결측처리규칙": MISSING_RULE_TEXT,
        "카파_역변환": None, "클러스터": None,
    }
    joblib.dump(bundle, path)
    chain_diff = verify_full_chain(bundle, train, path)
    print(f"[단기 +{h}h] 학습완료 학습행수={len(train):,} 특성수={len(features)} "
          f"학습기간={train.index.min().date()}~{train.index.max().date()} 전체체인재검증차이={chain_diff:.2e}")
    return {"파일": str(path), "학습행수": len(train), "특성수": len(features),
            "구조": structure_name, "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
            "전체체인재검증차이": chain_diff}


def train_daily(capacity_kw: float, seed: int) -> dict:
    data, features, n_partial = daily_mod.corrected_dataset(capacity_kw)
    cap_day = capacity_kw * 24
    train = data.dropna(subset=["실제_일간발전량_kWh"]) if "실제_일간발전량_kWh" in data.columns else data.dropna(
        subset=[c for c in data.columns if "발전량" in c and "kWh" in c][0:1])
    target_col = "실제_일간발전량_kWh" if "실제_일간발전량_kWh" in train.columns else train.columns[
        [c for c in train.columns if "발전량" in c and "kWh" in c][0]]
    from lightgbm import LGBMRegressor
    params, trace = tuning.tune_fold("LightGBM", train, features, target_col, cap_day, seed)
    params = daily_mod.cast_params(params)
    model = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **params)
    model.fit(train[features], train[target_col])

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "운영모델_일간_D+1.joblib"
    bundle = {
        "tier": "일간", "horizon": "D+1", "target_transform": "daily_kWh", "타깃유형": "raw",
        "features": features, "model": model, "params": params,
        "missing_strategy": "LightGBM_native_missing+flags", "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "clip_상한": cap_day,  # 일간은 kWh 단위라 capacity_kw(kW)가 아니라 capacity_kw*24
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
        "번들버전": BUNDLE_VERSION, "결측처리규칙": MISSING_RULE_TEXT,
        "카파_역변환": None, "클러스터": None,
        "이력요구사항": "7일전·2일전 lag + 2일전기준 30일 이동통계 사용 — 라이브 운영 전 "
                    "최소 30~32일치 발전량 이력 축적(또는 과거자료 백필) 필요(Codex 08-25 지적).",
    }
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    before = np.asarray(model.predict(train[features].tail(30)), float)
    after = np.asarray(loaded["model"].predict(train[features].tail(30)), float)
    reload_diff = float(np.max(np.abs(before - after)))
    print(f"[일간 D+1] 학습완료 학습행수={len(train):,} 특성수={len(features)} "
          f"학습기간={train.index.min().date()}~{train.index.max().date()} 재적재차이={reload_diff:.2e}")
    return {"파일": str(path), "학습행수": len(train), "특성수": len(features),
            "학습기간": f"{train.index.min().date()}~{train.index.max().date()}", "재적재차이": reload_diff}


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(cfg["site"]["capacity_kw"])
    seed = int(cfg["random_seed"])
    print(f"운영모델 v2 학습 시작(번들 완전화) — 용량={capacity_kw}kW seed={seed}\n")

    registry = {}
    for h in (1, 2, 3, 4):
        registry[f"초단기_h{h}"] = train_ultra(h, capacity_kw, seed)
    for h in (1, 24, 48):
        registry[f"단기_h{h}"] = train_short(h, capacity_kw, seed)
    registry["일간_D+1"] = train_daily(capacity_kw, seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "운영모델_목록.json").write_text(
        json.dumps({"생성시각": pd.Timestamp.now().isoformat(), "capacity_kw": capacity_kw,
                    "seed": seed, "번들버전": BUNDLE_VERSION, "모델": registry},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n=== 전체 {len(registry)}개 운영모델(v2) 저장 완료: {OUT} ===")


if __name__ == "__main__":
    main()
