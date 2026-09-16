# -*- coding: utf-8 -*-
"""광주·영광 초단기 - GHI(obs_ghi_wm2) 제거 ablation 실험(09-15, 사용자
지시 "GHI의 MAE를 구해보고 필요 없으면 삭제하는게 좋지 않을까").

부안·김제(09-15(7))에 GHI를 "추가"해본 것의 대칭 실험 - 이번엔 이미
GHI를 쓰는 광주·영광에서 GHI를 "제거"했을 때 MAE가 어떻게 되는지
확인한다. 기존 라이브 배포 모듈(`ultra_short_term_v1_gwangju_2026-09-08.py`,
`ultra_short_term_v1_yeonggwang_2026-09-07.py`)을 그대로 재사용(재구현
안 함) - BASE_FEATURES에서 obs_ghi_wm2 한 줄만 몽키패치로 제거, 나머지
로직·하이퍼파라미터·lag·시간피처는 전혀 안 건드림. 프로덕션 번들은
전혀 안 건드리고 별도 후보 폴더에만 저장.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
RV_SCRIPT = ROOT / "revalidate_all_horizons_issue_safe_v1_2026-09-08.py"
SEED = 42

GWANGJU_DIR = ROOT / "광주_준비_2026-09-08"
YEONGGWANG_DIR = ROOT / "영광_준비_2026-09-03"

CAPACITY_KW = {"광주": 240.58, "영광": 634.0}

GWANGJU_BASELINE_MANIFEST = GWANGJU_DIR / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08"
YEONGGWANG_BASELINE_MANIFEST = YEONGGWANG_DIR / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08"


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


def load_baseline_mae(baseline_dir: Path) -> dict:
    out = {}
    for horizon_dir in sorted(baseline_dir.glob("+*h")):
        mf = horizon_dir / "manifest.json"
        if not mf.is_file():
            continue
        d = json.loads(mf.read_text(encoding="utf-8"))
        wf = d.get("walk_forward_kW", {})
        sel = d.get("selected_structure") or d.get("selected_model")
        mae = wf.get(sel, {}).get("MAE_kW") if sel else None
        out[horizon_dir.name] = {"selected": sel, "MAE_kW": mae}
    return out


def build_no_ghi_candidate(region: str, region_dir: Path, module_name: str, module_path: Path,
                           out_subdir: str, rv) -> dict:
    m = load_module(module_name, module_path)
    # ★핵심★ obs_ghi_wm2만 제거, 나머지 피처 순서·내용 그대로.
    m.BASE_FEATURES = [f for f in m.BASE_FEATURES if f != "obs_ghi_wm2"]
    capacity = CAPACITY_KW[region]
    structures = m.STRUCTURES
    results = {}
    for h in m.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(m, h)
        req = features + [target]
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))

        pooled: dict[str, dict] = {}
        for name, params in structures.items():
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
                raise RuntimeError(f"{region} GHI제거후보 +{h}h/{name}: 유효 폴드 없음")
            pooled[name] = rv.score(np.concatenate(ys), np.concatenate(ps))

        selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
        clean = d.dropna(subset=req).sort_values("target_time")
        final_model = LGBMRegressor(**structures[selected], random_state=SEED, n_jobs=-1, verbosity=-1)
        final_model.fit(clean[features], clean[target])
        replay = clean.tail(min(128, len(clean)))
        pred = np.asarray(final_model.predict(replay[features]), float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"{region} GHI제거후보 +{h}h: 비유한 replay")

        horizon = f"+{h}h"
        out_dir = region_dir / "outputs" / out_subdir / horizon
        payload = {"model": final_model, "features": features, "region": region, "horizon": horizon,
                   "capacity_kw": capacity, "model_name": f"LightGBM({selected})"}
        final_path, sha = atomic_save(out_dir, payload, replay[features], pred)
        result = {
            "status": "candidate_not_deployed", "region": region, "horizon": horizon,
            "selected_structure": selected, "features": features, "capacity_kw": capacity,
            "training_rows": len(clean), "walk_forward_kW": pooled,
            "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / capacity * 100, 4),
            "note": "09-15 사용자 지시 - obs_ghi_wm2 제거 ablation(GHI가 실제로 MAE에 필요한지 검증). "
                   "라이브 미배포.",
            "model_sha256": sha, "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        }
        write_manifest(out_dir, result)
        results[horizon] = result
    return results


def main() -> None:
    rv = load_module("revalidate_all_horizons", RV_SCRIPT)

    gwangju_candidate = build_no_ghi_candidate(
        "광주", GWANGJU_DIR, "gwangju_no_ghi", ROOT / "ultra_short_term_v1_gwangju_2026-09-08.py",
        "광주_초단기_v2_GHI제거_후보_2026-09-15", rv,
    )
    yeonggwang_candidate = build_no_ghi_candidate(
        "영광", YEONGGWANG_DIR, "yeonggwang_no_ghi",
        YEONGGWANG_DIR / "ultra_short_term_v1_yeonggwang_2026-09-07.py",
        "영광_초단기_v2_GHI제거_후보_2026-09-15", rv,
    )
    gwangju_baseline = load_baseline_mae(GWANGJU_BASELINE_MANIFEST)
    yeonggwang_baseline = load_baseline_mae(YEONGGWANG_BASELINE_MANIFEST)

    comparison = {"광주": {}, "영광": {}}
    for horizon, cand in gwangju_candidate.items():
        base_mae = gwangju_baseline.get(horizon, {}).get("MAE_kW")
        cand_mae = cand["walk_forward_kW"][cand["selected_structure"]]["MAE_kW"]
        improve = round((1 - cand_mae / base_mae) * 100, 2) if base_mae else None
        comparison["광주"][horizon] = {
            "기존(GHI포함)_MAE_kW": base_mae, "GHI제거_MAE_kW": cand_mae, "변화율_pct": improve,
        }
    for horizon, cand in yeonggwang_candidate.items():
        base_mae = yeonggwang_baseline.get(horizon, {}).get("MAE_kW")
        cand_mae = cand["walk_forward_kW"][cand["selected_structure"]]["MAE_kW"]
        improve = round((1 - cand_mae / base_mae) * 100, 2) if base_mae else None
        comparison["영광"][horizon] = {
            "기존(GHI포함)_MAE_kW": base_mae, "GHI제거_MAE_kW": cand_mae, "변화율_pct": improve,
        }

    out = ROOT / "outputs" / "GHI제거_ablation_MAE비교_v1_2026-09-15"
    out.mkdir(parents=True, exist_ok=True)
    (out / "비교결과.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
