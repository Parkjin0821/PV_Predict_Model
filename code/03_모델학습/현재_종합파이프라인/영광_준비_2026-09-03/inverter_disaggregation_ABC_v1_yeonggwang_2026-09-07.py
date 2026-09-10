# -*- coding: utf-8 -*-
"""영광 인버터별 분해 A·B·C(09-07) - 부안 v2+v3+v4(08-31) 로직 재사용,
13대(12x50kW+1x34kW) 구성에 맞춰 일반화.

## 재사용 범위(★코드 틀만 - 데이터·결과는 영광 자체★)
부안 `inverter_disaggregation_ABC_v2_필터수정_walkforward_2026-08-31.py`가
확정한 3가지 배분방법(A/B/C)과 평가체계(_score, walk-forward)를 그대로
가져오되, 부안은 8대가 **전부 동일용량(125kW)**이라 방법A를 "균등
1/N"으로 짰다. 영광은 13대 중 12대 50kW+1대 34kW로 **동일하지 않음**
(사용자 지시: "지역의 인버터 개수에 맞춰서") - 그래서 방법A만 **정격
용량 비례배분**으로 일반화하고, B(계절중앙값+fallback)·C(인버터별
LightGBM 비중)는 N=13에도 그대로 적용 가능해 로직 변경 없음.

영광은 부안과 달리 알려진 결함구간이 없어(완전가용률 99.947%, 09-01
확인) v3의 결함구간 제외 단계는 생략(부안 v3와 다른 점, v2 수준에서
바로 v4 스타일 "마지막 폴드 전체구간 커버"만 적용).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
SRC = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
    r"\영광_인버터별_5분_야간0포함.parquet"
)
OUT_DIR = HERE / "outputs" / "영광_인버터분해_ABC_v1_2026-09-07"
SEED = 42
MIN_SLOT_COUNT = 260  # 부안과 동일 기준(하루 288슬롯의 약 90%) - 완전가용일 판정
SUM_TOL_KWH = 1e-6
INITIAL_TRAIN_DAYS = 90  # 영광의 다른 walk-forward 스크립트(초단기·중장기)와 동일하게 맞춤
TEST_BLOCK_DAYS = 30
OK_STATUSES = ["observed", "physical_zero_night", "physical_zero_idle_supported"]
# 부안의 ok 집합(observed·physical_zero_night류)과 동일 원칙 - 보간/결측/통신장애는 제외.


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "겨울"
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    return "가을"


def load_daily_per_inverter() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(SRC, columns=[
        "grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night",
        "inverter_capacity_kw"])
    df["ok"] = df["quality_status_after_night"].isin(OK_STATUSES)
    df["date"] = df["grid_time_kst"].dt.date
    df["kwh_5min"] = df["ac_power_kw"] * (5 / 60)

    capacity_by_inv = (
        df.groupby("inverter_number")["inverter_capacity_kw"].median().round(1).to_dict()
    )

    daily = df[df["ok"]].groupby(["date", "inverter_number"])["kwh_5min"].sum().unstack("inverter_number")
    count = df[df["ok"]].groupby(["date", "inverter_number"]).size().unstack("inverter_number")
    n_inv = df["inverter_number"].nunique()
    full_all = count.notna().all(axis=1) & (count.min(axis=1) >= MIN_SLOT_COUNT)
    daily_full = daily[full_all].dropna()
    meta = {
        "인버터_수": int(n_inv),
        "인버터별_정격용량_kw": {int(k): float(v) for k, v in capacity_by_inv.items()},
        "정격용량_합계_kw": round(float(sum(capacity_by_inv.values())), 1),
        "완전가용일_판정기준": f"{n_inv}대 전부 관측 + 슬롯수>={MIN_SLOT_COUNT}(하루 288슬롯의 약 90%)",
        "완전가용일수": int(len(daily_full)),
        "전체_고유일수": int(daily.shape[0]),
    }
    return daily_full, meta


def _score(daily_full: pd.DataFrame, pred: np.ndarray, actual: np.ndarray, method_name: str) -> dict:
    inv_cols = list(daily_full.columns)
    total = actual.sum(axis=1)
    pred_total = pred.sum(axis=1)
    max_sum_err = float(np.max(np.abs(pred_total - total))) if len(total) else 0.0
    if len(total) and max_sum_err > SUM_TOL_KWH:
        raise AssertionError(f"{method_name}: 합계보존 위반(최대오차 {max_sum_err} kWh)")

    err = actual - pred
    mae_per_inv = np.abs(err).mean(axis=0)
    mean_actual = actual.mean(axis=0)
    nmae_pct = np.divide(mae_per_inv, mean_actual, out=np.zeros_like(mae_per_inv), where=mean_actual != 0) * 100

    per_inverter = [{
        "인버터": int(inv), "MAE_kWh": round(float(mae_per_inv[i]), 2),
        "nMAE_pct": round(float(nmae_pct[i]), 2),
    } for i, inv in enumerate(inv_cols)]
    return {
        "방법": method_name,
        "시험표본_행수": int(len(total)),
        "합계보존_최대오차_kWh": round(max_sum_err, 8),
        "인버터별": per_inverter,
        "평균_nMAE_pct": round(float(nmae_pct.mean()), 2),
        "최대_nMAE_pct": round(float(nmae_pct.max()), 2),
    }


def expanding_folds_full_coverage(n_days: int, initial: int, block: int) -> list[tuple]:
    folds, end = [], initial
    while end < n_days:
        test_end = min(end + block, n_days)
        folds.append((end, test_end))
        end = test_end
    return folds


def run_method_a_capacity_proportional(daily_full: pd.DataFrame, capacity_by_inv: dict) -> dict:
    inv_cols = list(daily_full.columns)
    cap = np.array([capacity_by_inv[inv] for inv in inv_cols], dtype=float)
    weight = cap / cap.sum()  # ★부안(균등1/N)과 다른 부분 - 정격용량 비례★
    total = daily_full.sum(axis=1).to_numpy()
    pred = total[:, None] * weight[None, :]
    return _score(daily_full, pred, daily_full.to_numpy(),
                 "A_정격용량비례(12x50kW+1x34kW 비례배분)")


def run_method_b_wf(daily_full: pd.DataFrame, folds: list[tuple]) -> dict:
    seasons = pd.Series([_season(d.month) for d in daily_full.index], index=daily_full.index)
    shares_all = daily_full.div(daily_full.sum(axis=1), axis=0)
    total = daily_full.sum(axis=1)

    fold_results, all_pred, all_actual = [], [], []
    for i, (train_end, test_end) in enumerate(folds, start=1):
        train_idx = daily_full.index[:train_end]
        test_idx = daily_full.index[train_end:test_end]
        train_shares = shares_all.loc[train_idx]
        train_seasons = seasons.loc[train_idx]

        pred_rows = []
        fallback_days = 0
        for day in test_idx:
            s = seasons.loc[day]
            pool = train_shares[train_seasons == s]
            if len(pool) < 3:
                pool = train_shares
                fallback_days += 1
            median_share = pool.median()
            norm = median_share / median_share.sum()
            pred_rows.append(total.loc[day] * norm.to_numpy())
        pred = np.array(pred_rows)
        actual = daily_full.loc[test_idx].to_numpy()
        fold_score = _score(daily_full.loc[test_idx], pred, actual, f"B_폴드{i}")
        fold_results.append({"폴드": i, "학습일수": train_end, "시험일수": len(test_idx),
                             "fallback_적용일수": fallback_days,
                             "평균_nMAE_pct": fold_score["평균_nMAE_pct"]})
        all_pred.append(pred); all_actual.append(actual)

    pred_cat = np.concatenate(all_pred, axis=0)
    actual_cat = np.concatenate(all_actual, axis=0)
    overall = _score(daily_full, pred_cat, actual_cat, "B_계절중앙값(walk-forward)")
    overall["폴드별"] = fold_results
    return overall


def run_method_c_wf(daily_full: pd.DataFrame, folds: list[tuple]) -> dict:
    inv_cols = list(daily_full.columns)
    shares_all = daily_full.div(daily_full.sum(axis=1), axis=0)
    total = daily_full.sum(axis=1)
    doy = pd.Series([d.timetuple().tm_yday for d in daily_full.index], index=daily_full.index)
    feat = pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "log_total_kwh": np.log1p(total),
    }, index=daily_full.index)

    fold_results, all_pred, all_actual = [], [], []
    for i, (train_end, test_end) in enumerate(folds, start=1):
        train_idx = daily_full.index[:train_end]
        test_idx = daily_full.index[train_end:test_end]

        models = {}
        for inv in inv_cols:
            m = LGBMRegressor(n_estimators=80, learning_rate=0.08, num_leaves=7,
                              max_depth=3, min_child_samples=15, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=1.5,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
            m.fit(feat.loc[train_idx], shares_all.loc[train_idx, inv])
            models[inv] = m

        raw = np.column_stack([np.clip(models[inv].predict(feat.loc[test_idx]), 0, None)
                               for inv in inv_cols])
        row_sum = raw.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        norm = raw / row_sum
        pred = total.loc[test_idx].to_numpy()[:, None] * norm
        actual = daily_full.loc[test_idx].to_numpy()
        fold_score = _score(daily_full.loc[test_idx], pred, actual, f"C_폴드{i}")
        fold_results.append({"폴드": i, "학습일수": train_end, "시험일수": len(test_idx),
                             "평균_nMAE_pct": fold_score["평균_nMAE_pct"]})
        all_pred.append(pred); all_actual.append(actual)

    pred_cat = np.concatenate(all_pred, axis=0)
    actual_cat = np.concatenate(all_actual, axis=0)
    overall = _score(daily_full, pred_cat, actual_cat, "C_LightGBM비중(walk-forward)")
    overall["폴드별"] = fold_results
    return overall


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_full, meta = load_daily_per_inverter()
    capacity_by_inv = meta["인버터별_정격용량_kw"]

    folds = expanding_folds_full_coverage(len(daily_full), INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
    season_coverage = pd.Series([_season(d.month) for d in daily_full.index]).value_counts().to_dict()

    a_full = run_method_a_capacity_proportional(daily_full, capacity_by_inv)
    b = run_method_b_wf(daily_full, folds)
    c = run_method_c_wf(daily_full, folds)

    test_days_all = daily_full.index[INITIAL_TRAIN_DAYS:folds[-1][1]] if folds else daily_full.index[:0]
    a_on_test_only = (run_method_a_capacity_proportional(daily_full.loc[test_days_all], capacity_by_inv)
                      if len(test_days_all) else None)

    ranking = sorted([
        {"방법": "A_정격용량비례", "평균_nMAE_pct": (a_on_test_only or a_full)["평균_nMAE_pct"]},
        {"방법": "B_계절중앙값(walk-forward)", "평균_nMAE_pct": b["평균_nMAE_pct"]},
        {"방법": "C_LightGBM비중(walk-forward)", "평균_nMAE_pct": c["평균_nMAE_pct"]},
    ], key=lambda r: r["평균_nMAE_pct"])

    result = {
        **meta,
        "계절별_표본일수": season_coverage,
        "walk_forward_폴드수": len(folds),
        "폴드구성": [{"폴드": i + 1, "학습일수": s, "시험일수": e - s} for i, (s, e) in enumerate(folds)],
        "시험구간_전체커버여부": (folds[-1][1] == len(daily_full)) if folds else False,
        "A_전체기간": a_full,
        "A_시험구간만(B·C와_공정비교용)": a_on_test_only,
        "B": b,
        "C": c,
        "순위_시험구간_공정비교": ranking,
        "_방법론출처": "부안 inverter_disaggregation_ABC_v2/v3/v4(2026-08-31) 로직 재사용 - "
                     "A만 균등1/N에서 정격용량비례로 일반화(영광은 13대 중 12대 50kW+1대 34kW로 "
                     "부안의 8대 동일용량 전제가 안 맞음), B·C는 변경 없이 그대로 적용.",
        "_판정": "잠정치 - 부안처럼 조건부 공식채택(promote_to_official) 여부는 사용자 확인 후 결정.",
    }
    (OUT_DIR / "영광_인버터분해_ABC_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("B", "C")}, ensure_ascii=False, indent=2))
    print("순위:", json.dumps(ranking, ensure_ascii=False))


if __name__ == "__main__":
    main()
