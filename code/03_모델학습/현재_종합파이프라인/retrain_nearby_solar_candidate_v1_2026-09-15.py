# -*- coding: utf-8 -*-
"""부안·김제 초단기 - 인근일사량(전주146) 정식 피처 추가 후보 재학습 + MAE
비교(09-15, 사용자 지시 "정식 피처 추가하고 재학습해서 MAE 비교해줘").

`build_regional_ultrashort_mediumterm_issue_safe_v1_2026-09-08.py`의
build_buan_ultra_v2_weather()/build_gimje_ultra()와 **완전히 동일한
walk-forward·구조선택·저장 로직을 그대로 재사용**(재구현 아님) - 유일한
차이는 모듈을 신규 nearbysolar 버전으로 바꾸고, 저장 경로를 별도 후보
폴더로 분리한 것뿐(프로덕션 번들은 전혀 건드리지 않음).

비교 기준(베이스라인): 기존 프로덕션 manifest.json의 walk_forward_kW를
그대로 읽어온다(재학습 안 함 - 이미 기록된 실측치, 동일 폴드/동일
방법론으로 산출된 값이라 재현 불필요).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
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

CAPACITY_KW = {"부안": 1000.0, "김제": 1100.0}

BUAN_STRUCTURES = {
    "raw": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=1.0),
    "shallow_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=7, max_depth=3,
                        min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                        reg_alpha=0.3, reg_lambda=2.0),
    "leaf_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=31, max_depth=-1,
                     min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
                     reg_alpha=0.5, reg_lambda=3.0),
}
GIMJE_STRUCTURES = {
    "raw": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=1.0),
}

BUAN_BASELINE_MANIFEST = BUAN_DIR / "outputs" / "공식_초단기_v2_날씨피처_2026-09-08"
GIMJE_BASELINE_MANIFEST = GIMJE_DIR / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(path: Path) -> str:
    import hashlib
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


def load_baseline_mae(baseline_dir: Path) -> dict:
    out = {}
    for horizon_dir in sorted(baseline_dir.glob("+*h")):
        mf = horizon_dir / "manifest.json"
        if not mf.is_file():
            continue
        d = json.loads(mf.read_text(encoding="utf-8"))
        wf = d.get("walk_forward_kW", {})
        sel = d.get("selected_model") or d.get("selected_structure")
        mae = wf.get(sel, {}).get("MAE_kW") if sel else None
        out[horizon_dir.name] = {"selected": sel, "MAE_kW": mae, "features": d.get("features")}
    return out


def build_buan_candidate(rv) -> dict:
    m = load_module("buan_nearbysolar", BUAN_DIR / "ultra_short_term_v2_buan_2026-09-15_nearbysolar.py")
    capacity = CAPACITY_KW["부안"]
    results = {}
    for h in m.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(m, h)
        req = features + [target]
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))

        pooled: dict[str, dict] = {}
        for name, params in BUAN_STRUCTURES.items():
            ys, ps = [], []
            for trd, ted in rv.folds(days, m.INITIAL_TRAIN_DAYS, m.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, target)
                te = te.dropna(subset=["power_lag_0min", "kt_now", "clearsky_power_target_kw"])
                if len(tr) < m.MIN_ROWS_PER_FOLD or te.empty:
                    continue
                model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
                model.fit(tr[features], tr[target])
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te[target].to_numpy())
            if not ys:
                raise RuntimeError(f"부안 인근일사량후보 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**BUAN_STRUCTURES[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"부안 인근일사량후보 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = BUAN_DIR / "outputs" / "부안_초단기_v2_인근일사량_후보_2026-09-15" / horizon
        payload = {"model": final_model, "features": features, "region": "부안", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": selected}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "candidate_not_deployed", "region": "부안", "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "leakage_audit": {"nearby_ghi_join": "merge_asof(direction=backward, tolerance=3h) on grid_time_kst - "
                                                  "기존 obs_temp_c 등과 동일 안전조건, 미래 관측 참조 없음",
                              "all_power_features_anchored_at_or_before_issue": True},
            "note": "09-15 사용자 지시로 신규 생성 - 전주146 인근 실측 일사량(nearby_ghi_wm2) 정식 피처 추가 후보. "
                   "라이브 미배포, live_feature_assembler_ultrashort 미연결.",
            "model_sha256": sha, "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


def build_gimje_candidate(rv) -> dict:
    m = load_module("gimje_nearbysolar", GIMJE_DIR / "ultra_short_term_v3_gimje_2026-09-15_nearbysolar.py")
    capacity = CAPACITY_KW["김제"]
    results = {}
    for h in m.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(m, h)
        req = features + [target]
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))

        pooled: dict[str, dict] = {}
        for name, params in GIMJE_STRUCTURES.items():
            ys, ps = [], []
            for trd, ted in rv.folds(days, m.INITIAL_TRAIN_DAYS, m.TEST_BLOCK_DAYS):
                tr, te, _removed = rv.strict_split(d, trd, ted, features, target)
                te = te.dropna(subset=["power_lag_0min", "kt_now", "clearsky_power_target_kw"])
                if len(tr) < m.MIN_ROWS_PER_FOLD or te.empty:
                    continue
                model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
                model.fit(tr[features], tr[target])
                ps.append(np.clip(model.predict(te[features]), 0, capacity))
                ys.append(te[target].to_numpy())
            if not ys:
                raise RuntimeError(f"김제 인근일사량후보 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**GIMJE_STRUCTURES[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"김제 인근일사량후보 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = GIMJE_DIR / "outputs" / "김제_초단기_v3_인근일사량_후보_2026-09-15" / horizon
        payload = {"model": final_model, "features": features, "region": "김제", "horizon": horizon,
                   "capacity_kw": capacity, "model_name": f"LightGBM({selected})"}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "candidate_not_deployed", "region": "김제", "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "leakage_audit": {"nearby_ghi_join": "merge_asof(direction=backward, tolerance=3h) on grid_time_kst - "
                                                  "기존 obs_temp_c 등과 동일 안전조건, 미래 관측 참조 없음",
                              "all_power_features_anchored_at_or_before_issue": True},
            "note": "09-15 사용자 지시로 신규 생성 - 전주146 인근 실측 일사량(nearby_ghi_wm2) 정식 피처 추가 후보. "
                   "라이브 미배포, live_feature_assembler_ultrashort 미연결.",
            "model_sha256": sha, "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


def main() -> None:
    rv = load_module("revalidate_all_horizons", RV_SCRIPT)

    buan_candidate = build_buan_candidate(rv)
    gimje_candidate = build_gimje_candidate(rv)
    buan_baseline = load_baseline_mae(BUAN_BASELINE_MANIFEST)
    gimje_baseline = load_baseline_mae(GIMJE_BASELINE_MANIFEST)

    comparison = {"부안": {}, "김제": {}}
    for horizon, cand in buan_candidate.items():
        base_mae = buan_baseline.get(horizon, {}).get("MAE_kW")
        cand_mae = cand["walk_forward_kW"][cand["selected_structure"]]["MAE_kW"]
        improve = round((1 - cand_mae / base_mae) * 100, 2) if base_mae else None
        comparison["부안"][horizon] = {
            "기존(피처無GHI)_MAE_kW": base_mae, "기존_selected": buan_baseline.get(horizon, {}).get("selected"),
            "신규(인근GHI포함)_MAE_kW": cand_mae, "신규_selected": cand["selected_structure"],
            "개선율_pct": improve,
        }
    for horizon, cand in gimje_candidate.items():
        base_mae = gimje_baseline.get(horizon, {}).get("MAE_kW")
        cand_mae = cand["walk_forward_kW"][cand["selected_structure"]]["MAE_kW"]
        improve = round((1 - cand_mae / base_mae) * 100, 2) if base_mae else None
        comparison["김제"][horizon] = {
            "기존(피처無GHI)_MAE_kW": base_mae, "기존_selected": gimje_baseline.get(horizon, {}).get("selected"),
            "신규(인근GHI포함)_MAE_kW": cand_mae, "신규_selected": cand["selected_structure"],
            "개선율_pct": improve,
        }

    out = ROOT / "outputs" / "인근일사량_후보_MAE비교_v1_2026-09-15"
    out.mkdir(parents=True, exist_ok=True)
    (out / "비교결과.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
