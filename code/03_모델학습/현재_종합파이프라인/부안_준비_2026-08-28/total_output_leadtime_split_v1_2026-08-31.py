# -*- coding: utf-8 -*-
"""부안 총출력 D+1급 모델을 리드타임 구간별로 쪼개서 재비교(08-31).

## 배경
지금까지(v1~v5)는 발행10시→익일 8슬롯(0/3/6/9/12/15/18/21시, 리드타임
+14h~+35h)을 전부 하나로 뭉쳐서 LightGBM vs 선형회귀를 비교했다. 이건
광주 기준으로는 "일간~단기48h" 범위 하나뿐이고, 리드타임이 짧은 쪽/긴
쪽에서 같은 모델이 이기는지는 검증한 적이 없다. 새 데이터 없이 기존
v5 파이프라인(정정 완료된 방법론: target_time_kst기준 결함구간 제외,
폴드내부 특성선택 이미 검증됨, pooled 지표, 60일 이후 전체구간 커버)을
재사용해 리드타임 짧은/긴 구간으로만 나눠 다시 돌린다.

## 구간 정의
- 짧은리드(target 0/3/6/9시, 리드타임 +14~+23h) - 발행일 다음날 새벽~오전
- 긴리드(target 12/15/18/21시, 리드타임 +26~+35h) - 발행일 다음날 오후~밤
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

OUT_DIR = HERE / "outputs" / "총출력_리드타임구간별_2026-08-31"
SHORT_HOURS = {0, 3, 6, 9}
LONG_HOURS = {12, 15, 18, 21}


def run_for_hours(df: pd.DataFrame, hours: set[int], label: str) -> dict:
    sub = df[df["target_time_kst"].dt.hour.isin(hours)].copy()
    issue_days = pd.DatetimeIndex(np.sort(sub["issue_day"].unique()))
    folds = v5.expanding_folds_full_coverage(issue_days, v5.INITIAL_TRAIN_DAYS, v5.TEST_BLOCK_DAYS)

    def wf(features, model_kind):
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
        return {
            "n": model_score["n"],
            "모델_MAE_kW": model_score["MAE_kW"], "모델_RMSE_kW": model_score["RMSE_kW"],
            "지속성_MAE_kW": pers_score["MAE_kW"], "지속성_RMSE_kW": pers_score["RMSE_kW"],
            "MAE_개선율_pct": round((1 - model_score["MAE_kW"] / pers_score["MAE_kW"]) * 100, 1) if pers_score["MAE_kW"] else None,
            "RMSE_개선율_pct": round((1 - model_score["RMSE_kW"] / pers_score["RMSE_kW"]) * 100, 1) if pers_score["RMSE_kW"] else None,
        }

    lgbm15 = wf(v5.FEATURES_15, "lightgbm")
    lin15 = wf(v5.FEATURES_15, "linear")

    return {
        "구간": label, "target_hours": sorted(hours),
        "issue_days": int(len(issue_days)), "폴드수": len(folds),
        "LightGBM_15특성": lgbm15,
        "선형회귀_15특성": lin15,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, removed = v5.build_dataset()
    short = run_for_hours(df, SHORT_HOURS, "짧은리드(target 0/3/6/9시, +14~23h)")
    long_ = run_for_hours(df, LONG_HOURS, "긴리드(target 12/15/18/21시, +26~35h)")

    result = {
        "짧은리드": short, "긴리드": long_,
        "_판정": "각 구간에서 LightGBM이 여전히 선형회귀를 이기는지가 핵심 확인사항. "
               "잠정치 - promote_to_official 대상 아님.",
    }
    (OUT_DIR / "부안_리드타임구간별_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
