# -*- coding: utf-8 -*-
"""요인 재검증 2단계(공통 모듈) - 다중공선성 정리 + 모델 선정(LightGBM vs
XGBoost vs 선형회귀). 부안·김제·영광 3개 지역 러너 스크립트가 이 모듈을
그대로 import해서 쓴다(지역별 중복 재구현 안 함).

## 배경
사용자 09-03 지시("자정부터 할 작업에 다중공선성·모델선정·백테스트도
추가해줄 수 있어?"). 오늘 낮에 한 v6 permutation importance(특성
"선정")와 달리, 이 2단계는 **임계값이 이미 확립된 기계적 작업**이라
(VIF≥10·쌍상관≥0.8 - `backtest_harness_v1_2026-08-20밤.py`의 08-20
확정 O'Brien 2007 기준, LightGBM/XGBoost 하이퍼파라미터도 그 파일의
`make_model()` 그대로) 예약 실행으로 돌려도 안전하다고 판단 - 오늘
1단계(방법론 자체를 새로 정하는 판단)와는 성격이 다르다.

## 재사용한 것(재구현 안 함)
`backtest_harness_v1_2026-08-20밤.py`의 `compute_vif`·
`prune_multicollinearity`(VIF_SEVERE=10.0·PAIR_CORR_HIGH=0.8 그대로)·
`make_model("LightGBM"/"XGBoost", seed)` 전부 그대로 import.
선형회귀만 sklearn 표준 LinearRegression 추가.

## 다중공선성 처리 범위
baseline(과거발전량 lag/rolling)은 가지치기 대상에서 제외한다(v6도
같은 원칙 - 자기회귀 근간이라 건드리지 않음). 오늘 permutation
importance로 채택된 **후보(candidate)만** VIF+쌍상관 검토 대상.

## 결측 처리(선형회귀 추가로 인한 유일한 변경점)
강수량_mm·적설_cm은 결측률이 95% 안팎이고(비가/눈이 온 시간만 값이
있는 구조 - 결측=관측 자체가 없었다는 뜻이 물리적으로 "무강수"에
가까움) LightGBM·XGBoost는 native missing으로 처리하지만 선형회귀는
NaN을 못 받는다 - 이 두 컬럼만 0으로 채워 세 모델 동일 조건으로
비교한다(임의 대체 아니라 물리적으로 근거 있는 값).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
HARNESS_SCRIPT = HERE / "backtest_harness_v1_2026-08-20밤.py"
ZERO_FILL_COLS = {"issue_asos_강수량_mm", "issue_asos_적설_cm"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_harness():
    return _load_module("phase2_harness_20260903", HARNESS_SCRIPT)


def prune_candidates(frame: pd.DataFrame, target_col: str, candidate_cols: list[str],
                      harness) -> tuple[list[str], list[str]]:
    """v6이 채택한 후보 안에서 VIF+쌍상관 가지치기(harness 그대로 재사용).
    전체 프레임(오늘 importance 계산에 쓴 것과 동일한 daylight 풀) 기준
    - 오늘 v6 선정 자체가 이미 85/15 분할 기준이었으므로 동일 기준 유지."""
    x = frame[candidate_cols + [target_col]].dropna()
    target_corr = {}
    for c in candidate_cols:
        r = np.corrcoef(x[c], x[target_col])[0, 1]
        target_corr[c] = float(r) if np.isfinite(r) else 0.0
    kept = harness.prune_multicollinearity(x, candidate_cols, target_corr)
    dropped = [c for c in candidate_cols if c not in kept]
    return kept, dropped


def run_model_selection(frame: pd.DataFrame, folds: list[tuple], target_col: str,
                        features: list[str], native_missing_ok: set[str],
                        pooled_score_fn: Callable, seed: int, harness,
                        persistence_col: str = "lag_1day_same_slot_kw") -> dict:
    """LightGBM·XGBoost·선형회귀를 동일 폴드·동일 특성으로 walk-forward
    비교한다. required는 harness의 native-missing 관례 그대로."""
    required = [c for c in features if c not in native_missing_ok]
    linear_frame = frame.copy()
    for c in ZERO_FILL_COLS:
        if c in linear_frame.columns:
            linear_frame[c] = linear_frame[c].fillna(0.0)

    results: dict[str, dict] = {}
    for model_kind in ["LightGBM", "XGBoost", "선형회귀"]:
        all_true, all_pred, all_pers = [], [], []
        fold_rows = []
        src = linear_frame if model_kind == "선형회귀" else frame
        for i, (train_days, test_days) in enumerate(folds, start=1):
            assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤"
            train = src[src["issue_day"].isin(train_days)].dropna(subset=required + [target_col])
            test = src[src["issue_day"].isin(test_days)].dropna(subset=required + [target_col])
            if len(train) < 100 or len(test) < 5:
                continue
            model = (LinearRegression() if model_kind == "선형회귀"
                     else harness.make_model(model_kind, seed))
            model.fit(train[features], train[target_col])
            pred = np.clip(model.predict(test[features]), 0, None)
            y_true = test[target_col].to_numpy()
            pers = test[persistence_col].to_numpy()
            all_true.append(y_true); all_pred.append(pred); all_pers.append(pers)
            fold_rows.append({"폴드": i, "시험행수": len(test),
                              **{f"모델_{k}": v for k, v in pooled_score_fn(y_true, pred).items()
                                 if k != "n"}})
        if not all_true:
            results[model_kind] = {"폴드수": 0, "폴드별": [], "pooled_모델": None, "pooled_지속성": None}
            continue
        y_true_all = np.concatenate(all_true)
        pred_all = np.concatenate(all_pred)
        pers_all = np.concatenate(all_pers)
        results[model_kind] = {
            "폴드수": len(fold_rows), "폴드별": fold_rows,
            "pooled_모델": pooled_score_fn(y_true_all, pred_all),
            "pooled_지속성": pooled_score_fn(y_true_all, pers_all),
        }
    return results


def run_phase2(frame: pd.DataFrame, folds: list[tuple], target_col: str,
               baseline_cols: list[str], candidate_cols: list[str],
               native_missing_ok: set[str], pooled_score_fn: Callable,
               seed: int, out_dir: Path, region_label: str,
               persistence_col: str = "lag_1day_same_slot_kw") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    harness = _load_harness()

    daylight_pool = frame[frame.get("physical_daylight", 1) == 1].dropna(subset=[target_col])
    kept_candidates, dropped_candidates = prune_candidates(daylight_pool, target_col, candidate_cols, harness)
    final_features = list(dict.fromkeys(baseline_cols + kept_candidates))
    print(f"[{region_label}] 다중공선성 가지치기 - 후보 {len(candidate_cols)}개 중 "
          f"{len(dropped_candidates)}개 제거: {dropped_candidates}")
    print(f"[{region_label}] 최종 특성 {len(final_features)}개(baseline {len(baseline_cols)} + "
          f"후보 {len(kept_candidates)})")

    model_results = run_model_selection(
        frame, folds, target_col, final_features, native_missing_ok,
        pooled_score_fn, seed, harness, persistence_col)

    summary_rows = []
    for name, r in model_results.items():
        if r["pooled_모델"] is None:
            summary_rows.append({"모델": name, "MAE_kW": None, "RMSE_kW": None})
            continue
        summary_rows.append({"모델": name, "MAE_kW": r["pooled_모델"]["MAE_kW"],
                             "RMSE_kW": r["pooled_모델"]["RMSE_kW"],
                             "지속성_MAE_kW": r["pooled_지속성"]["MAE_kW"],
                             "지속성_RMSE_kW": r["pooled_지속성"]["RMSE_kW"]})
    summary_df = pd.DataFrame(summary_rows).sort_values("MAE_kW", na_position="last")
    summary_df.to_csv(out_dir / "모델선정_결과.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{region_label}] === 모델 비교(pooled) ===")
    print(summary_df.to_string(index=False))

    winner = summary_df.iloc[0]["모델"] if len(summary_df) and summary_df.iloc[0]["MAE_kW"] is not None else None

    payload = {
        "지역": region_label, "최종특성": final_features,
        "다중공선성_제거": dropped_candidates, "모델비교": summary_rows,
        "선정모델(MAE기준)": winner,
    }
    (out_dir / "phase2_요약.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{region_label}] 선정 모델(MAE 기준): {winner}")
    print(f"[{region_label}] 저장 완료: {out_dir}")
    return payload
