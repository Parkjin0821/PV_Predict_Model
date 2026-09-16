# -*- coding: utf-8 -*-
"""부안·김제·영광 초단기(+1~4h)·중장기(일간) 발행시각 안전 전용 추론번들 생성.

## 배경
09-08 Shadow 발판 구축 점검에서 phase2 "공식모델"이 실제로는 직렬화된
추론 모델(.joblib) 없이 요약 json만 있다는 게 드러났다(라이브 연결을
안전보류한 핵심 이유 중 하나). 같은 날 D+1
(`rebuild_regional_dayahead_issue_safe_v1_2026-09-08.py`)과 +24h
(`build_regional_plus24h_issue_safe_v1_2026-09-08.py`)는 정식 배포용
번들(model.joblib+manifest.json, 원자저장+왕복검증+SHA-256)을 만들어뒀지만
초단기(+1~4h)·중장기(일간)는 아직 없었다 - `revalidate_all_horizons_
issue_safe_v1_2026-09-08.py`는 발행시각 누출 여부만 재확인하는 "재검증"
스크립트이고 모델을 저장하지 않는다.

이 스크립트는 그 재검증 스크립트의 프레임 구축·폴드 로직을 **그대로
재사용(재구현 안 함)**하면서, 전체 안전자료로 최종학습한 배포용 번들
저장 단계만 추가한다. NWP/실시간 API 호출은 전혀 없음(전부 저장된
과거자료).

## 대상(9개 번들)
- 초단기 +1h~+4h × 3지역(부안·김제·영광) = 12개... 가 아니라 지역당
  4개 수평이므로 3×4=12개 모델 파일이지만 "지역-수평" 단위 번들로는
  12개. 중장기는 김제·영광만(부안은 09-08 확정대로 가을 표본이 없어
  미구축 - 임의생성 금지, `not_built_no_existing_model` 그대로 유지).

## ★영광 초단기 특이사항★
09-07 사용자 확정("어쩔 수 없이 단일모델로 지정을 해놓은 뒤...")대로
영광 초단기의 잠정 공식모델은 **v1 단일모델(regime 미적용)**이다.
09-08 재검증 스크립트는 v3 파일에서 재노출된 `V1`(=v1과 동일 모듈)을
가져오되 `regime=True`로 실행해 참고용으로 regime 변형까지 확인했다
(결과가 "기존과 유사"함을 그때 이미 확인). 이 스크립트는 **배포용
번들은 반드시 regime=False(v1 단일모델)로 만든다** - 아래 각 매니페스트에
이 사실을 명시한다.

## 저장 규약(기존 D+1/+24h와 동일)
tmp 경로에 joblib.dump → 다시 load해 왕복예측 허용오차(1e-10) 확인 →
os.replace로 원자적 교체. manifest.json에 walk-forward 지표(참고용),
학습행수, 특성목록, SHA-256, 발행시각 누출감사, 생성시각을 기록한다.
★왕복검증은 클리핑 전 원시 예측으로 한다★ - 클리핑된 값과 재적재 후
원시 예측을 비교하면 capacity/0 경계를 넘는 행에서 항상 오탐된다(최초
실행에서 부안 +2h가 이 이유로 실패, 원인 규명 후 전 지역에 동일 수정).

## ★부수 발견: revalidate 스크립트의 김제 폴백 하이퍼파라미터 버그★
`revalidate_all_horizons_issue_safe_v1_2026-09-08.py`의 `run_ultra_model()`은
구조선택(`tune_chrono`)엔 v2 실제 하이퍼파라미터 dict를 폴백으로 넘기지만
바로 다음 줄 `params=(...).get 대상` 조회는 `{'raw': {}}`(빈 dict)로 별도
폴백한다 - 즉 STRUCTURES 속성이 없는 김제는 구조는 "선택"했지만 실제
학습은 LightGBM 기본 하이퍼파라미터로 됐다. 이 스크립트는 `GIMJE_
FALLBACK_STRUCTURES` 하나로 두 폴백을 통일해 v2 의도값으로 학습하므로,
이 번들의 김제 초단기 walk-forward MAE(예: +1h 80.28kW)는 AGENTS.md
09-08 재검증표 수치(81.59kW)보다 소폭 낮다(더 좋다) - 회귀가 아니라
버그 수정 결과, 각 manifest.json의 `found_bug_in_reference_script`에도
기록해뒀다. revalidate 원본 스크립트 자체는 이미 실행·기록된 재검증
용도라 건드리지 않았다(수정은 이 신규 스크립트에만 적용).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
RV_SCRIPT = ROOT / "revalidate_all_horizons_issue_safe_v1_2026-09-08.py"
SEED = 42

BUAN_DIR = ROOT / "부안_준비_2026-08-28"
GIMJE_DIR = ROOT / "김제_준비_2026-09-01"
YEONGGWANG_DIR = ROOT / "영광_준비_2026-09-03"
GWANGJU_DIR = ROOT / "광주_준비_2026-09-08"

CAPACITY_KW = {"부안": 998.715, "김제": 999.005, "영광": 639.94, "광주": 240.0}  # ★09-16 통일★ 발전소 API 정격(AC 계통연계)으로 4지역 기준 통일. 인버터 등록용량 합계(부안1000/김제1100/영광634)는 DC·명판측 값이라 clip 상한·nMAE 분모로 부적합 - Blockdata 구성감시용으로만 남긴다.

BUAN_LGBM_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, num_leaves=15, max_depth=4,
    min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
)
GIMJE_FALLBACK_STRUCTURES = {
    "raw": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=1.0)
}
MEDIUM_LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save(out_dir: Path, payload: dict, replay_X, replay_pred_expected) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.joblib"
    tmp = out_dir / "model.joblib.tmp"
    joblib.dump(payload, tmp)
    loaded = joblib.load(tmp)
    got = np.asarray(loaded["model"].predict(replay_X), float)
    if not np.allclose(np.asarray(replay_pred_expected, float), got, rtol=0, atol=1e-10):
        raise RuntimeError(f"{out_dir}: 번들 왕복 불일치")
    os.replace(tmp, final)
    return final, digest(final)


def write_manifest(out_dir: Path, result: dict) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------
# 부안 초단기 (+1h~+4h) - LightGBM vs 선형회귀, 지속성은 평가 전용
# ------------------------------------------------------------------
def build_buan_ultra(rv, bundle_version: str = "공식_초단기_issue_safe_v1_2026-09-08") -> dict:
    # ★09-14 추가★: 주기적 재학습(드리프트 감지+월1회 백스톱) 착수를 위해
    # 출력 폴더명을 인자화. 기본값은 기존 09-08 배포 경로 그대로라
    # bundle_version 없이 호출하면 이전과 완전히 동일하게 동작(왕복검증으로
    # 확인 필요). 학습 로직·피처·하이퍼파라미터는 전혀 안 건드림 - 출력
    # 경로 한 줄만 바뀜. AGENTS.md 09-14 항목 참고.
    m = load_module("buan_ultra_base", BUAN_DIR / "ultra_short_historical_backtest_v1_2026-08-31.py")
    s = m.load_plant_5min_series()
    base = m.build_features(s)
    features = ["value", "lag_15m", "lag_30m", "lag_1h", "roll_1h_mean", "lag_1day_same_time"]
    capacity = CAPACITY_KW["부안"]
    results = {}

    for mins in m.HORIZONS_MIN:
        d = base.copy()
        d["issue_time"] = d.index
        d["target_time"] = d.index + pd.Timedelta(minutes=mins)
        d["issue_day"] = d.issue_time.dt.normalize()
        d["y"] = s.shift(-(mins // 5))
        req = features + ["y"]
        days = pd.DatetimeIndex(np.sort(d.dropna(subset=req).issue_day.unique()))

        pooled: dict[str, dict] = {}
        for name in ("LightGBM", "선형회귀"):
            ys, ps = [], []
            for trd, ted in rv.folds(days, m.INITIAL_TRAIN_DAYS, m.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, "y")
                if len(tr) < 500 or te.empty:
                    continue
                model = LGBMRegressor(**BUAN_LGBM_PARAMS) if name == "LightGBM" else LinearRegression()
                model.fit(tr[features], tr.y)
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te.y.to_numpy())
            if not ys:
                raise RuntimeError(f"부안 초단기 +{mins//60}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**BUAN_LGBM_PARAMS) if selected == "LightGBM" else LinearRegression()
        final_model.fit(clean[features], clean.y)
        replay = clean.tail(min(128, len(clean)))
        # 왕복검증은 클리핑 전 원시 예측으로 한다(클리핑은 서빙 시 별도 적용되는
        # 결정론적 후처리일 뿐 - 클리핑된 값과 재적재 후 원시 예측을 비교하면
        # 값이 capacity/0 경계를 넘는 행에서 항상 "불일치"로 오탐된다).
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"부안 초단기 +{mins//60}h: 비유한 replay")

        horizon = f"+{mins // 60}h"
        out_dir = BUAN_DIR / "outputs" / bundle_version / horizon
        payload = {"model": final_model, "features": features, "region": "부안", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": selected}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "ready_for_live_assembler_after_nwp_gate", "region": "부안", "horizon": horizon,
            "selected_model": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "leakage_audit": {"max_target_relative_lag_used": 0,
                              "all_power_features_anchored_at_or_before_issue": True,
                              "note": "strict_split()으로 시험폴드 시작시각 이전 target만 학습에 사용(경계행 제외)"},
            "model_sha256": sha, "created_at_kst": datetime.now().astimezone().isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


# ------------------------------------------------------------------
# 김제 초단기 (+1h~+4h) - v2 모듈 재사용(구조는 기본 1종, revalidate와 동일 관례)
# ------------------------------------------------------------------
def build_gimje_ultra(rv) -> dict:
    v2 = load_module("gimje_ultra_v2", GIMJE_DIR / "ultra_short_term_v2_gimje_multihorizon_2026-09-01.py")
    capacity = CAPACITY_KW["김제"]
    results = {}

    for h in v2.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(v2, h)
        req = features + [target]
        # revalidate.run_ultra_model()과 동일하게 날짜축은 dropna 이전 전체
        # issue_day로 잡는다(폴드 경계는 위치 인덱스라 dropna로 날짜축 길이가
        # 바뀌면 09-08 재검증 수치와 어긋난다).
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))
        structures = v2.STRUCTURES if hasattr(v2, "STRUCTURES") else GIMJE_FALLBACK_STRUCTURES

        pooled: dict[str, dict] = {}
        for name, params in structures.items():
            ys, ps = [], []
            for trd, ted in rv.folds(days, v2.INITIAL_TRAIN_DAYS, v2.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, target)
                # revalidate.run_ultra_model()과 동일하게 시험행도 스마트지속성
                # 기준(kt_now·clearsky_power_target_kw) 계산 가능한 행만 남긴다.
                te = te.dropna(subset=["power_lag_0min", "kt_now", "clearsky_power_target_kw"])
                if len(tr) < v2.MIN_ROWS_PER_FOLD or te.empty:
                    continue
                model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
                model.fit(tr[features], tr[target])
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te[target].to_numpy())
            if not ys:
                raise RuntimeError(f"김제 초단기 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**structures[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        # 왕복검증은 클리핑 전 원시 예측으로 한다(사유는 부안 절 주석 참고).
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"김제 초단기 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = GIMJE_DIR / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08" / horizon
        payload = {"model": final_model, "features": features, "region": "김제", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": f"LightGBM({selected})"}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "ready_for_live_assembler_after_nwp_gate", "region": "김제", "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "leakage_audit": {"max_target_relative_lag_used": 0,
                              "all_power_features_anchored_at_or_before_issue": True,
                              "structure_source": "v2.STRUCTURES 없으면 GIMJE_FALLBACK_STRUCTURES(v2 원본과 동일 설정)로 폴백"},
            "found_bug_in_reference_script": (
                "revalidate_all_horizons_issue_safe_v1_2026-09-08.py의 run_ultra_model()은 "
                "구조선택(tune_chrono)엔 실제 v2 하이퍼파라미터 dict를 폴백값으로 넘기지만, "
                "바로 다음 줄의 params 조회는 {'raw': {}}(빈 dict)로 별도 폴백해 김제처럼 "
                "STRUCTURES 속성이 없는 지역은 실제로 LightGBM 기본 하이퍼파라미터로 학습됐다. "
                "이 번들은 두 폴백을 GIMJE_FALLBACK_STRUCTURES 하나로 통일해 v2 원래 설정으로 "
                "학습했으므로, AGENTS.md 09-08 재검증표 수치(예: +1h 81.59kW)보다 이 번들의 "
                "walk_forward_kW가 소폭 낮게(더 좋게) 나온다 - 회귀가 아니라 버그 수정 결과."
            ),
            "model_sha256": sha, "created_at_kst": datetime.now().astimezone().isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


# ------------------------------------------------------------------
# 영광 초단기 (+1h~+4h) - ★v1 단일모델(regime 미적용), 09-07 확정판★
# ------------------------------------------------------------------
def build_yeonggwang_ultra(rv) -> dict:
    v1 = load_module("yeonggwang_ultra_v1", YEONGGWANG_DIR / "ultra_short_term_v1_yeonggwang_2026-09-07.py")
    capacity = CAPACITY_KW["영광"]
    results = {}

    for h in v1.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(v1, h)
        req = features + [target]
        # revalidate.run_ultra_model()과 동일하게 날짜축은 dropna 이전 전체
        # issue_day로 잡는다(사유는 김제 절 주석 참고).
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))
        structures = v1.STRUCTURES

        pooled: dict[str, dict] = {}
        for name, params in structures.items():
            ys, ps = [], []
            for trd, ted in rv.folds(days, v1.INITIAL_TRAIN_DAYS, v1.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, target)
                # revalidate.run_ultra_model()과 동일 필터(사유는 김제 절 주석 참고).
                te = te.dropna(subset=["power_lag_0min", "kt_now", "clearsky_power_target_kw"])
                if len(tr) < v1.MIN_ROWS_PER_FOLD or te.empty:
                    continue
                model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
                model.fit(tr[features], tr[target])
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te[target].to_numpy())
            if not ys:
                raise RuntimeError(f"영광 초단기 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**structures[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        # 왕복검증은 클리핑 전 원시 예측으로 한다(사유는 부안 절 주석 참고).
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"영광 초단기 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = YEONGGWANG_DIR / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08" / horizon
        payload = {"model": final_model, "features": features, "region": "영광", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": f"LightGBM({selected})", "regime_applied": False}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "ready_for_live_assembler_after_nwp_gate", "region": "영광", "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "regime_applied": False,
            "regime_note": "09-07 사용자 확정대로 v1 단일모델(regime 미적용)로 번들화. "
                           "09-08 재검증 표의 수치는 참고용 regime=True 버전이며 '기존과 유사'함이 "
                           "이미 확인됐으나 이 번들의 walk_forward_kW와는 소폭 다를 수 있음.",
            "leakage_audit": {"max_target_relative_lag_used": 0,
                              "all_power_features_anchored_at_or_before_issue": True},
            "model_sha256": sha, "created_at_kst": datetime.now().astimezone().isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


# ------------------------------------------------------------------
# 광주 초단기 (+1h~+4h) - 09-08 신규(영광 v1과 완전 동일 방법론, 데이터만
# 광주 자체 - 기존 08-26 NWP의존 초단기와는 별개의 새 배포용 번들)
# ------------------------------------------------------------------
def build_gwangju_ultra(rv) -> dict:
    v1 = load_module("gwangju_ultra_v1", ROOT / "ultra_short_term_v1_gwangju_2026-09-08.py")
    capacity = CAPACITY_KW["광주"]
    results = {}

    for h in v1.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(v1, h)
        req = features + [target]
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))
        structures = v1.STRUCTURES

        pooled: dict[str, dict] = {}
        for name, params in structures.items():
            ys, ps = [], []
            for trd, ted in rv.folds(days, v1.INITIAL_TRAIN_DAYS, v1.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, target)
                te = te.dropna(subset=["power_lag_0min", "kt_now", "clearsky_power_target_kw"])
                if len(tr) < v1.MIN_ROWS_PER_FOLD or te.empty:
                    continue
                model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
                model.fit(tr[features], tr[target])
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te[target].to_numpy())
            if not ys:
                raise RuntimeError(f"광주 초단기 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**structures[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"광주 초단기 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = GWANGJU_DIR / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08" / horizon
        payload = {"model": final_model, "features": features, "region": "광주", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": f"LightGBM({selected})"}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "ready_for_live_assembler_no_nwp_gate_needed", "region": "광주", "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "note": "기존 08-26 광주 초단기(live_feature_assembler_초단기_v1_2026-08-26.py)는 "
                   "DSWRF/TCDC/LCDC 등 NWP파생 특성을 써서 현재 NWP 결측에 막혀있음(09-08 실측 확인) - "
                   "이 번들은 그것과 별개의 신규 NWP 무관 모델(부안/김제/영광과 동일 설계).",
            "leakage_audit": {"max_target_relative_lag_used": 0,
                              "all_power_features_anchored_at_or_before_issue": True},
            "model_sha256": sha, "created_at_kst": datetime.now().astimezone().isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


# ------------------------------------------------------------------
# 부안 초단기 v2(날씨피처 추가판, 09-08 저녁) - "부안만 오차 크다" 감사
# 결과 라이브 ASOS 수집(station 243)은 이미 있는데 조립기가 안 쓰고
# 있었을 뿐임을 확인, 김제와 동일 레시피로 날씨 피처 포함 재학습.
# 기존 power_only 번들(build_buan_ultra)은 그대로 두고 별도 출력경로에
# 신규 배포용 번들을 만든다(비교·롤백 가능하게).
# ------------------------------------------------------------------
def build_buan_ultra_v2_weather(rv) -> dict:
    v1 = load_module("buan_ultra_v1_weather", BUAN_DIR / "ultra_short_term_v1_buan_2026-09-08.py")
    capacity = CAPACITY_KW["부안"]
    results = {}

    for h in v1.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(v1, h)
        req = features + [target]
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))
        structures = v1.STRUCTURES

        pooled: dict[str, dict] = {}
        for name, params in structures.items():
            ys, ps = [], []
            for trd, ted in rv.folds(days, v1.INITIAL_TRAIN_DAYS, v1.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, target)
                te = te.dropna(subset=["power_lag_0min", "kt_now", "clearsky_power_target_kw"])
                if len(tr) < v1.MIN_ROWS_PER_FOLD or te.empty:
                    continue
                model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
                model.fit(tr[features], tr[target])
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te[target].to_numpy())
            if not ys:
                raise RuntimeError(f"부안 초단기v2 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**structures[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"부안 초단기v2 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = BUAN_DIR / "outputs" / "공식_초단기_v2_날씨피처_2026-09-08" / horizon
        payload = {"model": final_model, "features": features, "region": "부안", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": f"LightGBM({selected})"}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "ready_for_live_assembler_no_nwp_gate_needed", "region": "부안", "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "note": "09-08 저녁 사용자 지적('부안만 오차 크다')으로 신규 제작 - 기존 08-28 번들"
                   "(공식_초단기_issue_safe_v1_2026-09-08)은 날씨 관측치 없이 발전량 lag만 썼는데, "
                   "부안용 라이브 ASOS 수집(station 243, kma_live_inputs_v1_2026-08-28)은 이미 "
                   "정상 동작 중이었음을 실측 확인(09-08 20:21 최신행 존재) - 조립기 연결만 빠져 "
                   "있었으므로 새로 수집을 시작할 필요 없이 김제와 동일 레시피(station 243 공유, "
                   "일사계 없어 obs_ghi_wm2 제외)로 재학습. 기존 번들은 비교용으로 그대로 둔다.",
            "leakage_audit": {"max_target_relative_lag_used": 0,
                              "all_power_features_anchored_at_or_before_issue": True},
            "model_sha256": sha, "created_at_kst": datetime.now().astimezone().isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


# ------------------------------------------------------------------
# 중장기(일간) - 김제·영광만 (부안은 09-08 확정대로 미구축, 임의생성 금지)
# ------------------------------------------------------------------
def build_medium(region: str, region_dir: Path, module_path: Path, rv) -> dict:
    mod = load_module(f"{region}_medium", module_path)
    d, _meta = mod.build_daily_dataset()
    d = d.copy()
    # revalidate와 동일 교정: 발행 당일 미완결값이 아니라 그 이전 완결일값만 사용.
    lookup = d.set_index("target_day")[mod.TARGET]
    d["daily_energy_lag1_kwh"] = (d["issue_day"] - pd.Timedelta(days=1)).map(lookup)

    features = mod.CANDIDATE_FEATURES
    req = list(features) + [mod.TARGET]
    days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))

    ys, pm, pp = [], [], []
    for trd, ted in rv.folds(days, mod.INITIAL_TRAIN_DAYS, mod.TEST_BLOCK_DAYS):
        tr = d[d.issue_day.isin(trd)].dropna(subset=req)
        te = d[d.issue_day.isin(ted)].dropna(subset=req)
        if len(tr) < mod.MIN_ROWS_PER_FOLD or te.empty:
            continue
        model = LGBMRegressor(**MEDIUM_LGBM_PARAMS)
        model.fit(tr[features], tr[mod.TARGET])
        pm.append(np.clip(model.predict(te[features]), 0, None))
        ys.append(te[mod.TARGET].to_numpy())
        pp.append(te.daily_energy_lag1_kwh.to_numpy())
    if not ys:
        raise RuntimeError(f"{region} 중장기: 유효 폴드 없음")
    y = np.concatenate(ys)
    model_score = rv.score(y, np.concatenate(pm))
    persistence_score = rv.score(y, np.concatenate(pp))

    clean = d.dropna(subset=req).sort_values("issue_day")
    final_model = LGBMRegressor(**MEDIUM_LGBM_PARAMS)
    final_model.fit(clean[features], clean[mod.TARGET])
    replay = clean.tail(min(64, len(clean)))
    # 왕복검증은 클리핑 전 원시 예측으로 한다(사유는 부안 초단기 절 주석 참고).
    pred = np.asarray(final_model.predict(replay[features]), float)
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{region} 중장기: 비유한 replay")

    out_dir = region_dir / "outputs" / "공식_중장기_issue_safe_v1_2026-09-08"
    payload = {"model": final_model, "features": list(features), "region": region, "horizon": "일간(D+1 kWh)",
              "target": mod.TARGET, "model_name": "LightGBM"}
    final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
    result = {
        "status": "ready_for_live_assembler_after_nwp_gate", "region": region, "horizon": "일간",
        "features": list(features), "target": mod.TARGET, "training_rows": len(clean),
        "walk_forward_kWh": {"model": model_score, "persistence_last_complete_day": persistence_score},
        "mae_improvement_pct_vs_persistence": round(
            (1 - model_score["MAE_kW"] / persistence_score["MAE_kW"]) * 100, 2
        ),
        "leakage_audit": {"daily_energy_lag1_기준": "발행시각 이전 마지막 완결일(issue_day-1)만 사용, "
                                                    "발행 당일 미완결 롤업 참조 0건"},
        "model_sha256": sha, "created_at_kst": datetime.now().astimezone().isoformat(),
    }
    write_manifest(out_dir, result)
    return result


def main() -> None:
    rv = load_module("revalidate_all_horizons", RV_SCRIPT)

    summary = {
        "부안": {"초단기": build_buan_ultra(rv), "초단기_v2_날씨피처": build_buan_ultra_v2_weather(rv),
               "중장기": {"status": "not_built_no_existing_model",
                        "reason": "가을 표본 없음(09-08 확정, 재문의 금지) - 임의생성 안 함"}},
        "김제": {"초단기": build_gimje_ultra(rv), "중장기": build_medium("김제", GIMJE_DIR,
                  GIMJE_DIR / "medium_term_daily_v1_gimje_2026-09-01.py", rv)},
        "영광": {"초단기": build_yeonggwang_ultra(rv), "중장기": build_medium("영광", YEONGGWANG_DIR,
                  YEONGGWANG_DIR / "medium_term_daily_v1_yeonggwang_2026-09-07.py", rv)},
        "광주": {"초단기": build_gwangju_ultra(rv), "중장기": {"status": "out_of_scope_09-08",
                                                     "reason": "오늘 요청 범위는 초단기뿐(기존 08-26 일간 파이프라인 별도 존재)"}},
    }

    out = ROOT / "outputs" / "3지역_초단기_중장기_배포용번들_v1_2026-09-08"
    out.mkdir(parents=True, exist_ok=True)
    (out / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
