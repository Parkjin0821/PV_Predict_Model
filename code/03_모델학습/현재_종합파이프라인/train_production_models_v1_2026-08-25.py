# -*- coding: utf-8 -*-
"""★운영용(배포용) 모델 학습 — 가용한 전체 이력으로 학습, 시험구간 없음★

## 왜 필요한가
지금까지 저장된 `fold_models/*.joblib`은 전부 **5계절 백테스트용**이다.
각 폴드모델은 "그 계절 시작 전까지"만 학습했다(미래누출 검증을 위해
일부러 최근 데이터를 뺌) — 예를 들어 "5_초여름" 폴드모델도 2026-04-15
이전 데이터로만 학습돼 있어, 그 뒤 4개월치(04-15~08-04)가 빠져 있다.
**"지금 실전에 쓸 모델"로 이 백테스트 모델들을 쓰면 안 된다.**

이 스크립트는 **가용한 데이터 전체**(초단기·단기=v5 전체 기간,
일간=321개 목표일 전체)로 딱 하나씩 학습해, "이게 지금 시점 진짜
운영모델이다"라고 보관할 수 있는 산출물을 만든다.

## 구성(전부 공식 배포판과 동일 — 재구현 아님, 시험구간만 뺌)
- 초단기 +1h·+2h: raw+구조선택 / +3h: 청천지수 기본 /
  +4h: 청천지수+날씨군집화(⑥ 채택안)
- 단기 전체: 기본+구조선택
- 일간: 전체58특성+튜닝(08-25 최종감사와 동일 특성·전처리)
- DIFSWRF: C전략(원값 NaN 유지+결측여부, native missing) — 08-25에
  단기까지 전 티어 통일 확인된 방식.

## 재사용(재구현 금지 원칙)
`e2e_retrain_v5_공식B_v1_2026-08-24.py`(초단기·단기 로직, DIF/add_difswrf_flag),
`e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py`(+4h 날씨군집화),
`daily_direct_final_audit_v1_2026-08-25.py`(일간 corrected_dataset·
classify), `hyperparameter_tuning_v1_2026-08-21.tune_fold`,
`model_improvement_round2_v1_2026-08-21.choose_structure_kfold/make_model`.

## 출력 (`outputs/운영모델_v1_2026-08-25/`)
티어·수평별 joblib 8개 + `운영모델_목록.json`(각 모델의 학습기간·
특성수·재적재검증 결과).

## ★주의: 아직 "완전한 운영모델"은 아님★
이 스크립트는 **학습**만 한다. 매 15분/1시간/D-1 10시마다 자동으로
이 모델을 불러와 그 순간의 최신 입력으로 예측하는 "실행/스케줄러"
부분은 별도 구축이 필요하다(AGENTS.md 08-25 "운영 배포 설계안" 절
참고) — 이 파일은 그 설계의 1번 항목(운영용 모델)만 해결한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "운영모델_v1_2026-08-25"
N_INVERTERS = 5
DIF = "DIFSWRF_bsrn정제"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("prod_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc, harness, ultra, improvement, clearsky = e2e.dpc, e2e.harness, e2e.ultra, e2e.improvement, e2e.clearsky
add_difswrf_flag = e2e.add_difswrf_flag
v2patch = _load("prod_v2", "e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py")
daily_mod = _load("prod_daily", "daily_direct_final_audit_v1_2026-08-25.py")
tuning = _load("prod_tuning", "hyperparameter_tuning_v1_2026-08-21.py")

ULTRA_OFFICIAL = {1: "raw", 2: "raw", 3: "청천지수", 4: "청천지수"}
ULTRA_STRUCTURE_TYPE = {1: "구조선택", 2: "구조선택", 3: "기본", 4: "날씨군집화(⑥)"}


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
    full = daylight[daylight["_목표_가용인버터수"] >= N_INVERTERS].copy()  # 정책B, 시험구간 없이 전부 학습

    sel_input = full.dropna(subset=["목표_발전출력_kW"])
    chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
    features = base_cols + chosen
    native_ok = harness.NATIVE_MISSING_OK | {DIF}
    required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW", "_청천_kW", "_카파"]
    train = full.dropna(subset=required)

    target = "_카파" if use_kappa else "목표_발전출력_kW"
    if h == 4:
        # ⑥ 채택안: 청천지수+날씨군집화(model_improvement_round2.add_weather_clusters
        # 그대로 재사용 — test 인자가 필요한 시그니처라 train을 그대로 넣고 tr만 씀,
        # 군집 fit 자체는 train만으로 이뤄지므로 결과에 영향 없음).
        train, _te_unused, features, _cluster_cols = improvement.add_weather_clusters(
            train, train, features, seed)
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
        "tier": "초단기", "horizon_h": h, "target_transform": target, "structure_name": structure_name,
        "features": features, "model": model, "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
    }
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    before = np.asarray(model.predict(train[features].tail(50)), float)
    after = np.asarray(loaded["model"].predict(train[features].tail(50)), float)
    reload_diff = float(np.max(np.abs(before - after)))
    print(f"[초단기 +{h}h] 학습완료 학습행수={len(train):,} 특성수={len(features)} "
          f"구조={structure_name} 학습기간={train.index.min().date()}~{train.index.max().date()} "
          f"재적재차이={reload_diff:.2e}")
    return {"파일": str(path), "학습행수": len(train), "특성수": len(features),
            "구조": structure_name, "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
            "재적재차이": reload_diff}


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
    native_ok = harness.NATIVE_MISSING_OK | {DIF}  # 08-25 확정 C전략
    required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
    train = full.dropna(subset=required)

    params, structure_name = improvement.choose_structure_kfold(
        "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
    model = improvement.make_model("단기", seed, params)
    model.fit(train[features], train["목표_발전출력_kW"])

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"운영모델_단기_h{h}.joblib"
    bundle = {
        "tier": "단기", "horizon_h": h, "target_transform": "raw_kW", "structure_name": structure_name,
        "features": features, "model": model, "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
    }
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    before = np.asarray(model.predict(train[features].tail(50)), float)
    after = np.asarray(loaded["model"].predict(train[features].tail(50)), float)
    reload_diff = float(np.max(np.abs(before - after)))
    print(f"[단기 +{h}h] 학습완료 학습행수={len(train):,} 특성수={len(features)} "
          f"학습기간={train.index.min().date()}~{train.index.max().date()} 재적재차이={reload_diff:.2e}")
    return {"파일": str(path), "학습행수": len(train), "특성수": len(features),
            "구조": structure_name, "학습기간": f"{train.index.min().date()}~{train.index.max().date()}",
            "재적재차이": reload_diff}


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
        "tier": "일간", "horizon": "D+1", "target_transform": "daily_kWh",
        "features": features, "model": model, "params": params,
        "missing_strategy": "LightGBM_native_missing+flags", "정책": "B_구간제외",
        "용량프로필": "inverter_registered_sum_219", "capacity_kw": capacity_kw,
        "학습기간_시작": str(train.index.min()), "학습기간_끝": str(train.index.max()),
        "학습행수": len(train), "학습완료시각": pd.Timestamp.now().isoformat(),
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
    print(f"운영모델 학습 시작 — 용량={capacity_kw}kW seed={seed}\n")

    registry = {}
    for h in (1, 2, 3, 4):
        registry[f"초단기_h{h}"] = train_ultra(h, capacity_kw, seed)
    for h in (1, 24, 48):
        registry[f"단기_h{h}"] = train_short(h, capacity_kw, seed)
    registry["일간_D+1"] = train_daily(capacity_kw, seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "운영모델_목록.json").write_text(
        json.dumps({"생성시각": pd.Timestamp.now().isoformat(), "capacity_kw": capacity_kw,
                    "seed": seed, "모델": registry}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n=== 전체 {len(registry)}개 운영모델 저장 완료: {OUT} ===")


if __name__ == "__main__":
    main()
