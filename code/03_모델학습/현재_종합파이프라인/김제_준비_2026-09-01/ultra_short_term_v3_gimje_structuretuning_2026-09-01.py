# -*- coding: utf-8 -*-
"""김제 초단기(+1h~+4h) 구조튜닝(09-01, v3).

v2(ultra_short_term_v2_gimje_multihorizon_2026-09-01.py) 대비 광주 08-21
방식(각 외부 폴드의 "학습구간 내부" 80/20 홀드아웃에서만 구조 선택, 시험
폴드는 구조선택에 절대 안 씀 - leakage 방지) 재사용. v2는 고정 LightGBM
설정 하나만 썼는데, 광주 대비 nMAE가 1.5~1.6배 나빴던 것(09-01 정규화
비교) 중 구조튜닝 미적용분을 얼마나 메우는지 확인.

구조 후보 3개(광주 "얕은 규제형/리프규제형" 아이디어 재사용):
- raw: v2와 동일(대조군)
- shallow_reg: 더 얕고 강하게 규제(과적합 억제 우선)
- leaf_reg: 리프 수는 늘리되 min_child_samples로 강하게 규제

내부홀드아웃은 날짜 단위 80/20(행 단위 아님 - 같은 날 다른 슬롯이
train/holdout에 걸치면 시간적으로 거의 leakage와 같아서 날짜로 나눔).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "gimje_ultra_v2", HERE / "ultra_short_term_v2_gimje_multihorizon_2026-09-01.py"
)
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

OUT_DIR = HERE / "outputs" / "김제_초단기_구조튜닝_v3_2026-09-01"
SEED = 42
HOLDOUT_FRAC = 0.2

STRUCTURES = {
    "raw(v2와 동일)": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          reg_alpha=0.1, reg_lambda=1.0),
    "shallow_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=7, max_depth=3,
                        min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                        reg_alpha=0.3, reg_lambda=2.0),
    "leaf_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=31, max_depth=-1,
                     min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
                     reg_alpha=0.5, reg_lambda=3.0),
}


def fit_predict(params: dict, train: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
    model.fit(train[features], train[target])
    return np.clip(model.predict(eval_df[features]), 0, None)


def select_structure_internal_holdout(train_days: pd.DatetimeIndex, d: pd.DataFrame,
                                      target_col: str, features: list[str], rng: np.random.RandomState) -> str:
    """학습구간 내부에서만 80/20 나눠 구조 후보를 고른다 - 시험폴드는 안 씀."""
    days = np.array(sorted(train_days))
    n_holdout = max(1, int(len(days) * HOLDOUT_FRAC))
    holdout_days = set(rng.choice(days, size=n_holdout, replace=False))
    inner_train_days = [dd for dd in days if dd not in holdout_days]

    inner_train = d[d["day"].isin(inner_train_days)].dropna(subset=features + [target_col])
    inner_holdout = d[d["day"].isin(holdout_days)].dropna(subset=features + [target_col])
    if len(inner_train) < 30 or len(inner_holdout) < 10:
        return "raw(v2와 동일)"

    best_name, best_mae = None, np.inf
    for name, params in STRUCTURES.items():
        pred = fit_predict(params, inner_train, inner_holdout, features, target_col)
        mae = float(np.mean(np.abs(inner_holdout[target_col].to_numpy() - pred)))
        if mae < best_mae:
            best_mae, best_name = mae, name
    return best_name


def run_walkforward_tuned(d: pd.DataFrame, target_col: str, features: list[str], folds: list[tuple]) -> dict:
    rng = np.random.RandomState(SEED)
    all_true, all_model = [], []
    fold_rows = []
    structure_votes = {}
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출"
        train = d[d["day"].isin(train_days)].dropna(subset=features + [target_col])
        test = d[d["day"].isin(test_days)].dropna(subset=features + [target_col])
        if len(train) < v2.MIN_ROWS_PER_FOLD or len(test) < 1:
            continue

        chosen = select_structure_internal_holdout(train_days, d, target_col, features, rng)
        structure_votes[chosen] = structure_votes.get(chosen, 0) + 1
        pred_model = fit_predict(STRUCTURES[chosen], train, test, features, target_col)

        y_true = test[target_col].to_numpy()
        all_true.append(y_true); all_model.append(pred_model)
        fold_rows.append({"폴드": i, "시험일수": len(test_days), "시험행수": len(test),
                          "선택된구조": chosen,
                          "모델_MAE": v2.pooled_score(y_true, pred_model)["MAE_kW"]})

    if not all_true:
        return {"폴드수": 0, "폴드별": [], "오류": "유효 폴드 없음"}
    y_true_all = np.concatenate(all_true)
    return {
        "폴드수": len(fold_rows), "폴드별": fold_rows, "구조선택_투표분포": structure_votes,
        "pooled_모델(구조튜닝)": v2.pooled_score(y_true_all, np.concatenate(all_model)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, meta = v2.load_base()

    results = {}
    for h in v2.LEAD_HOURS:
        d, target_col, features = v2.build_horizon_frame(base, h)
        days = pd.DatetimeIndex(np.sort(d["day"].unique()))
        folds = v2.expanding_folds_full_coverage(days, v2.INITIAL_TRAIN_DAYS, v2.TEST_BLOCK_DAYS)
        tuned = run_walkforward_tuned(d, target_col, features, folds)
        results[f"+{h}h"] = tuned
        mae = tuned.get("pooled_모델(구조튜닝)", {}).get("MAE_kW")
        print(f"[+{h}h] 구조튜닝 완료 - pooled MAE={mae}, 구조분포={tuned.get('구조선택_투표분포')}")

    # v2(raw 고정) 결과와 비교하기 위해 v2 결과 JSON을 재사용(있으면).
    v2_result_path = HERE / "outputs" / "김제_초단기멀티호라이즌_v2_2026-09-01" / "김제_초단기_1to4h_결과.json"
    v2_perf = {}
    if v2_result_path.exists():
        v2_data = json.loads(v2_result_path.read_text(encoding="utf-8"))
        for h in v2.LEAD_HOURS:
            v2_perf[f"+{h}h"] = v2_data["리드타임별_결과"][f"+{h}h"]["성능"]["pooled_모델"]["MAE_kW"]

    comparison = {}
    for h in v2.LEAD_HOURS:
        key = f"+{h}h"
        tuned_mae = results[key].get("pooled_모델(구조튜닝)", {}).get("MAE_kW")
        raw_mae = v2_perf.get(key)
        comparison[key] = {
            "v2_raw_MAE": raw_mae, "v3_구조튜닝_MAE": tuned_mae,
            "개선율_pct": round((1 - tuned_mae / raw_mae) * 100, 1) if raw_mae and tuned_mae else None,
        }

    summary = {**meta, "리드타임별_구조튜닝_결과": results, "v2_대비_비교": comparison,
              "_판정": "잠정치 - promote_to_official 대상 아님."}
    (OUT_DIR / "김제_초단기_구조튜닝_결과.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
