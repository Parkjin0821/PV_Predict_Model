"""모델 공통 평가지표, 시간순 검증, 앙상블 가중치 함수."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return {"표본수": 0}
    error = p - y
    absolute = np.abs(error)
    denominator = float(np.abs(y).sum())
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - p) ** 2))
    wape = float(absolute.sum() / denominator * 100) if denominator > 0 else None
    return {
        "표본수": int(len(y)),
        "평균오차_kW": float(error.mean()),
        "평균절대오차_kW": float(absolute.mean()),
        "평균제곱근오차_kW": float(np.sqrt(np.mean(error**2))),
        "가중절대비율오차_pct": wape,
        "정확도_pct": float(100 - wape) if wape is not None else None,
        "결정계수": float(1 - residual / total) if total > 0 else None,
    }


def expanding_folds(target_times: pd.Series, config: dict) -> list[tuple[np.ndarray, np.ndarray, str]]:
    times = pd.to_datetime(target_times)
    folds = []
    for number, window in enumerate(config["cross_validation_windows"], start=1):
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        train = (times < start).to_numpy()
        valid = ((times >= start) & (times <= end)).to_numpy()
        if train.any() and valid.any():
            folds.append((train, valid, f"검증{number}"))
    return folds


def optimize_nonnegative_weights(actual: np.ndarray, predictions: np.ndarray) -> dict:
    """절대오차 합을 최소화하는 0 이상·합계 1 가중치를 구한다."""
    y = np.asarray(actual, dtype=float)
    matrix = np.asarray(predictions, dtype=float)
    valid = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
    y, matrix = y[valid], matrix[valid]
    count = matrix.shape[1]
    if len(y) == 0 or count == 0:
        raise ValueError("가중치 계산에 사용할 유효 예측값이 없습니다.")

    # 절대값의 꺾이는 점 때문에 SLSQP가 초기값에서 멈추는 것을 피하도록
    # 작은 평활항을 둔 절대오차 근사식을 사용한다.
    smooth = 1e-2

    def objective(weights: np.ndarray) -> float:
        error = matrix @ weights - y
        return float(np.sqrt(error**2 + smooth**2).sum())

    def gradient(weights: np.ndarray) -> np.ndarray:
        error = matrix @ weights - y
        factor = error / np.sqrt(error**2 + smooth**2)
        return matrix.T @ factor

    initial = np.repeat(1 / count, count)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1)}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    # 최적화 실패 또는 수치정체가 있어도 최선 단일모델보다 나쁜 조합은 채택하지 않는다.
    single_objectives = [objective(np.eye(count)[number]) for number in range(count)]
    best_single = int(np.argmin(single_objectives))
    if objective(weights) > single_objectives[best_single] + 1e-6:
        weights = np.eye(count)[best_single]
    return {
        "가중치": weights.tolist(),
        "최적화성공": bool(result.success),
        "사용표본수": int(len(y)),
        "목적함수_절대오차합": objective(weights),
        "최선단일모델번호": best_single,
        "최선단일모델_목적함수": float(single_objectives[best_single]),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
