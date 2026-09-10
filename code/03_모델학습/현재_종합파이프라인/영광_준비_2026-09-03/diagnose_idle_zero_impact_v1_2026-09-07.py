# -*- coding: utf-8 -*-
"""영광 초단기(+1h) 오차가 idle_zero(저조도 무발전) 구간에 쏠려있는지 실측 확인(09-07).

`ultra_short_term_v1_yeonggwang_2026-09-07.py`와 완전히 동일한
walk-forward/구조튜닝 로직을 그대로 재사용(import)하되, 시험행에
quality_status를 붙여서 카테고리별 오차를 분해한다. 학습/모델 자체는
건드리지 않음 - 순수 진단용.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "ultra_short_yg", HERE / "ultra_short_term_v1_yeonggwang_2026-09-07.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)  # type: ignore

OUT_DIR = HERE / "outputs" / "영광_초단기_v1_2026-09-07"


def run_h1_with_quality() -> dict:
    plant = pd.read_parquet(M.PLANT_5MIN)
    plant["grid_time_kst"] = pd.to_datetime(plant["grid_time_kst"])
    plant = plant.sort_values("grid_time_kst").reset_index(drop=True)
    quality_lookup = plant.set_index("grid_time_kst")["quality_status"]

    base, _ = M.load_base()
    d, target_col, features = M.build_horizon_frame(base, 1)
    d["target_quality_status"] = (
        d["grid_time_kst"] + pd.Timedelta(hours=1)).map(quality_lookup)

    days = pd.DatetimeIndex(np.sort(d["day"].unique()))
    folds = M.expanding_folds_full_coverage(days, M.INITIAL_TRAIN_DAYS, M.TEST_BLOCK_DAYS)

    rng = np.random.RandomState(M.SEED)
    rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        need = features + [target_col, "power_lag_0min", "kt_now", "clearsky_power_target_kw"]
        train = d[d["day"].isin(train_days)].dropna(subset=features + [target_col])
        test = d[d["day"].isin(test_days)].dropna(subset=need)
        if len(train) < M.MIN_ROWS_PER_FOLD or len(test) < 1:
            continue
        chosen = M.select_structure_internal_holdout(train_days, d, target_col, features, rng)
        pred = M.fit_predict(M.STRUCTURES[chosen], train, test, features, target_col)
        err = np.abs(test[target_col].to_numpy() - pred)
        rows.append(pd.DataFrame({
            "abs_err_kw": err,
            "target_quality_status": test["target_quality_status"].to_numpy(),
        }))

    all_rows = pd.concat(rows, ignore_index=True)
    overall_mae = float(all_rows["abs_err_kw"].mean())
    overall_n = len(all_rows)

    by_status = all_rows.groupby("target_quality_status", dropna=False).agg(
        n=("abs_err_kw", "size"), mae_kw=("abs_err_kw", "mean")
    ).reset_index()
    by_status["행_비중_pct"] = round(by_status["n"] / overall_n * 100, 2)
    by_status["nMAE_pct"] = round(by_status["mae_kw"] / M.CAPACITY_KW * 100, 3)
    by_status = by_status.sort_values("n", ascending=False)

    idle_mask = all_rows["target_quality_status"] == "complete_with_idle_zero"
    non_idle_mask = ~idle_mask
    idle_n = int(idle_mask.sum())
    idle_mae = float(all_rows.loc[idle_mask, "abs_err_kw"].mean()) if idle_n else None
    non_idle_mae = float(all_rows.loc[non_idle_mask, "abs_err_kw"].mean())

    # idle_zero 행을 아예 평가에서 뺐다면 전체 nMAE가 어떻게 바뀌는지
    counterfactual_mae = non_idle_mae
    counterfactual_nmae = round(counterfactual_mae / M.CAPACITY_KW * 100, 3)

    return {
        "전체_n": overall_n,
        "전체_MAE_kW": round(overall_mae, 3),
        "전체_nMAE_pct": round(overall_mae / M.CAPACITY_KW * 100, 3),
        "카테고리별": by_status.to_dict(orient="records"),
        "idle_zero_n": idle_n,
        "idle_zero_행비중_pct": round(idle_n / overall_n * 100, 2),
        "idle_zero_MAE_kW": round(idle_mae, 3) if idle_mae is not None else None,
        "idle_zero_제외시_MAE_kW": round(counterfactual_mae, 3),
        "idle_zero_제외시_nMAE_pct": counterfactual_nmae,
        "idle_zero_제외효과_pct포인트": round(
            (overall_mae / M.CAPACITY_KW * 100) - counterfactual_nmae, 3),
    }


def main() -> None:
    result = run_h1_with_quality()
    (OUT_DIR / "idle_zero_영향분해_+1h.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
