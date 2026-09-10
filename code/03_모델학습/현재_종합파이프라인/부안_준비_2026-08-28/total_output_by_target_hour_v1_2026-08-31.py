# -*- coding: utf-8 -*-
"""부안 총출력 D+1급 모델 - 8개 대상시각(target hour)별 개별 성능 +
낮시간전용 nMAE/nRMSE(08-31, Codex 지적: 앞서 "짧은리드/긴리드" 2분할이
리드타임과 시간대(태양고도)를 완전히 혼입시켰음 - 단일 10시 발행자료
만으로는 리드타임 효과와 시간대 효과를 분리할 수 없다는 걸 그대로
인정하고, 최소한 8개 대상시각 개별 성능과 낮시간전용 지표는 따로 낸다).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v5mod", HERE / "total_output_weather_model_v5_방법론정정_2026-08-31.py")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

OUT_DIR = HERE / "outputs" / "총출력_대상시각별_2026-08-31"
TARGET_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def wf_for_subset(sub: pd.DataFrame, features: list[str], model_kind: str) -> dict | None:
    issue_days = pd.DatetimeIndex(np.sort(sub["issue_day"].unique()))
    folds = v5.expanding_folds_full_coverage(issue_days, v5.INITIAL_TRAIN_DAYS, v5.TEST_BLOCK_DAYS)
    all_true, all_pred, all_pers = [], [], []
    for train_days, test_days in folds:
        train = sub[sub["issue_day"].isin(train_days)].dropna(subset=features + [v5.TARGET])
        test = sub[sub["issue_day"].isin(test_days)].dropna(subset=features + [v5.TARGET])
        if len(train) < 30 or len(test) < 1:
            continue
        if model_kind == "lightgbm":
            m = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                              max_depth=4, min_child_samples=10, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=v5.SEED, n_jobs=-1, verbosity=-1)
        else:
            m = LinearRegression()
        m.fit(train[features], train[v5.TARGET])
        pred = np.clip(m.predict(test[features]), 0, None)
        all_true.append(test[v5.TARGET].to_numpy())
        all_pred.append(pred)
        all_pers.append(test["lag_1day_same_slot_kw"].to_numpy())
    if not all_true:
        return None
    y = np.concatenate(all_true); p = np.concatenate(all_pred); pe = np.concatenate(all_pers)
    model_score = v5.pooled_score(y, p)
    pers_score = v5.pooled_score(y, pe)
    mean_actual = float(np.mean(y)) if len(y) else np.nan
    return {
        "n": model_score["n"], "평균실측_kW": round(mean_actual, 2),
        "모델_MAE_kW": model_score["MAE_kW"], "모델_RMSE_kW": model_score["RMSE_kW"],
        "지속성_MAE_kW": pers_score["MAE_kW"], "지속성_RMSE_kW": pers_score["RMSE_kW"],
        "MAE_개선율_pct": round((1 - model_score["MAE_kW"] / pers_score["MAE_kW"]) * 100, 1) if pers_score["MAE_kW"] else None,
        "RMSE_개선율_pct": round((1 - model_score["RMSE_kW"] / pers_score["RMSE_kW"]) * 100, 1) if pers_score["RMSE_kW"] else None,
        "nMAE_pct(모델MAE/평균실측)": round(model_score["MAE_kW"] / mean_actual * 100, 2) if mean_actual else None,
        "nRMSE_pct(모델RMSE/평균실측)": round(model_score["RMSE_kW"] / mean_actual * 100, 2) if mean_actual else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, removed = v5.build_dataset()

    per_hour = {}
    for h in TARGET_HOURS:
        sub = df[df["target_time_kst"].dt.hour == h]
        lgbm = wf_for_subset(sub, v5.FEATURES_15, "lightgbm")
        lin = wf_for_subset(sub, v5.FEATURES_15, "linear")
        per_hour[f"{h:02d}시"] = {"LightGBM": lgbm, "선형회귀": lin}

    # 낮시간전용(physical_daylight==1) - 시간대 효과를 걷어낸 nMAE/nRMSE
    daylight_sub = df[df["physical_daylight"] == 1]
    daylight_lgbm = wf_for_subset(daylight_sub, v5.FEATURES_15, "lightgbm")
    daylight_lin = wf_for_subset(daylight_sub, v5.FEATURES_15, "linear")

    result = {
        "대상시각별(8종_개별)": per_hour,
        "낮시간전용(physical_daylight==1)": {"LightGBM": daylight_lgbm, "선형회귀": daylight_lin},
        "_주의": "리드타임 효과와 시간대(태양고도) 효과가 이 데이터셋(발행10시→익일8슬롯) "
              "구조상 완전히 얽혀있어 분리 불가 - 8개 대상시각 개별값과 낮시간전용 지표로 "
              "최대한 투명하게 보여줄 뿐, '리드타임 단독효과'라는 결론은 내지 않는다.",
    }
    (OUT_DIR / "부안_대상시각별_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
