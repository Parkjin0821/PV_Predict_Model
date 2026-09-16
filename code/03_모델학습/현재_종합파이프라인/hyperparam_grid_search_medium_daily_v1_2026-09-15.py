# -*- coding: utf-8 -*-
"""D+1 일간 모델 규제 하이퍼파라미터 1차원 스윕(그리드) - 09-15
정확도개선 파일럿 #3(Claude, 사용자 요청).

## 방법 (재구현 없음)
부안·김제 각 지역의 `medium_term_daily_v1_*.py`에서 `build_daily_dataset()`·
`expanding_folds_full_coverage()`·`CANDIDATE_FEATURES`·`TARGET`을 그대로
importlib로 재사용(피처·데이터·폴드 구성 전부 기존 검증본 그대로). 바뀌는
건 LGBMRegressor 하이퍼파라미터뿐 - walk-forward 루프 자체는 각 원본의
`run_walkforward()`와 동일 로직을 파라미터화만 해서 재현(완전한 재구현이
아니라 이미 있는 4~5줄짜리 루프를 하이퍼파라미터 인자로 바꾼 것).

전체 조합 그리드(수천 개)를 다 돌리지 않고, 현재 채택된 기준값
(max_depth=4, num_leaves=15, min_child_samples=15, subsample=colsample=0.9,
reg_alpha=0.1, reg_lambda=1.0) 대비 **한 번에 한 축만** 바꾸는 1차원
스윕으로 방향성(더 조이면 좋아지는지 나빠지는지)부터 확인한다 - 표본이
적은 부안·표본이 많은 김제 둘 다 같은 방향인지 다른 방향인지가 핵심
질문(09-14 대화에서 "이미 상당히 보수적이라 더 조이면 부안은 과소적합
위험" 가설을 실측으로 검증).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")

REGIONS = {
    "부안": ROOT / "부안_준비_2026-08-28" / "medium_term_daily_v1_buan_2026-09-14.py",
    "김제": ROOT / "김제_준비_2026-09-01" / "medium_term_daily_v1_gimje_2026-09-01.py",
    "영광": ROOT / "영광_준비_2026-09-03" / "medium_term_daily_v1_yeonggwang_2026-09-07.py",
}

BASE_PARAMS = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, max_depth=4,
                    min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1)

# 각 항목: (파라미터명 또는 튜플, 스윕값 목록). subsample_colsample은 두
# 파라미터를 항상 같이 바꿈(원본도 항상 같은 값 0.9로 묶여있었음).
SWEEPS = [
    ("max_depth", [3, 4, 5, 6]),
    ("num_leaves", [7, 15, 31, 63]),
    ("reg_alpha", [0.0, 0.1, 0.5, 2.0]),
    ("reg_lambda", [0.5, 1.0, 3.0, 10.0]),
    ("min_child_samples", [5, 15, 30]),
    ("subsample_colsample", [0.7, 0.9, 1.0]),
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pooled_mae(y_true: np.ndarray, pred: np.ndarray) -> float:
    return round(float(np.mean(np.abs(y_true - pred))), 1)


def run_walkforward_with_params(m, df: pd.DataFrame, folds: list[tuple], params: dict) -> dict:
    """m.run_walkforward()와 동일 구조(폴드·leak assert·clip 전부 재사용)
    - 하이퍼파라미터만 인자로 받도록 파라미터화."""
    CF, T = m.CANDIDATE_FEATURES, m.TARGET
    cap = getattr(m, "DAILY_CAPACITY_KWH", None)
    all_true, all_model = [], []
    for train_days, test_days in folds:
        assert train_days.max() < test_days.min(), "시간누출"
        train = df[df["issue_day"].isin(train_days)].dropna(subset=CF + [T])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=CF + [T])
        if len(train) < m.MIN_ROWS_PER_FOLD or len(test) < 1:
            continue
        model = LGBMRegressor(**params)
        model.fit(train[CF], train[T])
        pred = model.predict(test[CF])
        pred = np.clip(pred, 0, cap) if cap else np.clip(pred, 0, None)
        all_true.append(test[T].to_numpy())
        all_model.append(pred)
    if not all_true:
        return {"n": 0, "MAE_kWh": None}
    y_true_all = np.concatenate(all_true)
    return {"n": int(len(y_true_all)), "MAE_kWh": pooled_mae(y_true_all, np.concatenate(all_model))}


def main() -> None:
    loaded = {}
    for region, path in REGIONS.items():
        m = load_module(path)
        df, _meta = m.build_daily_dataset()
        days = pd.DatetimeIndex(np.sort(df["issue_day"].unique()))
        folds = m.expanding_folds_full_coverage(days, m.INITIAL_TRAIN_DAYS, m.TEST_BLOCK_DAYS)
        loaded[region] = (m, df, folds)

    baseline = {}
    print("=== 기준값(현재 채택 하이퍼파라미터) ===")
    for region, (m, df, folds) in loaded.items():
        r = run_walkforward_with_params(m, df, folds, BASE_PARAMS)
        baseline[region] = r["MAE_kWh"]
        print(f"{region}: n={r['n']} MAE={r['MAE_kWh']}kWh")

    print()
    for param, values in SWEEPS:
        print(f"=== {param} 스윕 ===")
        for v in values:
            params = dict(BASE_PARAMS)
            if param == "subsample_colsample":
                params["subsample"] = v
                params["colsample_bytree"] = v
            else:
                params[param] = v
            row = []
            for region, (m, df, folds) in loaded.items():
                r = run_walkforward_with_params(m, df, folds, params)
                delta = None
                if r["MAE_kWh"] is not None and baseline[region]:
                    delta = round((r["MAE_kWh"] / baseline[region] - 1) * 100, 1)
                row.append(f"{region}={r['MAE_kWh']}kWh({'+' if (delta or 0) >= 0 else ''}{delta}%)")
            marker = " ← 기준값" if v == BASE_PARAMS.get(
                "subsample" if param == "subsample_colsample" else param) else ""
            print(f"  {param}={v}: " + " | ".join(row) + marker)
        print()

    print("=== 결합 테스트(1D스윕에서 3지역 중 2곳 이상 동시개선된 방향들을 결합) ===")
    combos = {
        "max_depth=3": dict(max_depth=3),
        "reg_lambda=10.0": dict(reg_lambda=10.0),
        "max_depth=3+reg_lambda=10.0": dict(max_depth=3, reg_lambda=10.0),
        "max_depth=3+num_leaves=7": dict(max_depth=3, num_leaves=7),
        "max_depth=3+num_leaves=7+reg_lambda=10.0": dict(max_depth=3, num_leaves=7, reg_lambda=10.0),
    }
    for name, override in combos.items():
        params = dict(BASE_PARAMS); params.update(override)
        row = []
        for region, (m, df, folds) in loaded.items():
            r = run_walkforward_with_params(m, df, folds, params)
            delta = round((r["MAE_kWh"] / baseline[region] - 1) * 100, 1) if r["MAE_kWh"] else None
            row.append(f"{region}={r['MAE_kWh']}kWh({'+' if (delta or 0) >= 0 else ''}{delta}%)")
        print(f"  {name}: " + " | ".join(row))


if __name__ == "__main__":
    main()
