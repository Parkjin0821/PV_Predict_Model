# -*- coding: utf-8 -*-
"""부안 총출력모델 "모델 선정" 마무리(08-31) - v4 특성셋·데이터(저상관
특성제거+결함구간제외)에서 LightGBM이 왜 선택됐는지 단순 베이스라인
(선형회귀)과 정면비교해 근거를 남긴다. 지금까지는 LightGBM만 써왔고
비교대상이 없었다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v4mod", HERE / "total_output_weather_model_v4_결함구간제외_2026-08-31.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

OUT_DIR = HERE / "outputs" / "총출력모델_선정_2026-08-31"
SEED = 42


def run() -> dict:
    cfg = v4.load_cfg()
    df = v4.build_dataset(cfg)
    feats = v4.feature_columns(cfg)
    target = cfg["target"]

    issue_days = pd.DatetimeIndex(np.sort(df["issue_day"].unique()))
    folds = v4.expanding_window_folds(issue_days, cfg["initial_train_issue_days"],
                                      cfg["test_block_issue_days"])

    rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        train = df[df["issue_day"].isin(train_days)].dropna(subset=feats + [target])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=feats + [target])
        if len(train) < 50 or len(test) < 5:
            continue
        y_true = test[target].to_numpy()
        pers = test["lag_1day_same_slot_kw"].to_numpy()

        lgbm = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                             max_depth=4, min_child_samples=15, subsample=0.9,
                             colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbosity=-1)
        lgbm.fit(train[feats], train[target])
        pred_lgbm = np.clip(lgbm.predict(test[feats]), 0, None)

        lin = LinearRegression()
        lin.fit(train[feats], train[target])
        pred_lin = np.clip(lin.predict(test[feats]), 0, None)

        rows.append({
            "폴드": i,
            "LightGBM_MAE": round(mean_absolute_error(y_true, pred_lgbm), 2),
            "LightGBM_RMSE": round(float(np.sqrt(mean_squared_error(y_true, pred_lgbm))), 2),
            "선형회귀_MAE": round(mean_absolute_error(y_true, pred_lin), 2),
            "선형회귀_RMSE": round(float(np.sqrt(mean_squared_error(y_true, pred_lin))), 2),
            "지속성_MAE": round(mean_absolute_error(y_true, pers), 2),
        })

    lgbm_mae = float(np.mean([r["LightGBM_MAE"] for r in rows]))
    lin_mae = float(np.mean([r["선형회귀_MAE"] for r in rows]))
    lgbm_rmse = float(np.mean([r["LightGBM_RMSE"] for r in rows]))
    lin_rmse = float(np.mean([r["선형회귀_RMSE"] for r in rows]))
    pers_mae = float(np.mean([r["지속성_MAE"] for r in rows]))

    return {
        "폴드별": rows,
        "전체평균": {
            "LightGBM_MAE_kW": round(lgbm_mae, 2), "LightGBM_RMSE_kW": round(lgbm_rmse, 2),
            "선형회귀_MAE_kW": round(lin_mae, 2), "선형회귀_RMSE_kW": round(lin_rmse, 2),
            "지속성_MAE_kW": round(pers_mae, 2),
        },
        "LightGBM_vs_선형회귀_MAE_개선_pct": round((1 - lgbm_mae / lin_mae) * 100, 1),
        "LightGBM_vs_선형회귀_RMSE_개선_pct": round((1 - lgbm_rmse / lin_rmse) * 100, 1),
        "_선정근거": "비선형 특성(구름량·강수확률 등이 발전량과 비선형 관계일 가능성, "
                  "일사량-발전량 clipping 등)과 특성 간 상호작용을 선형모델은 못 잡는다는 "
                  "가설을 실측으로 검증 - 개선폭이 크면 LightGBM 선택 근거, 작거나 역전되면 "
                  "선형모델의 단순함이 더 낫다는 뜻이라 숨기지 않고 그대로 판단할 것.",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "부안_총출력모델_LightGBM_vs_선형회귀.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
