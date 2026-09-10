# -*- coding: utf-8 -*-
"""부안 인버터분해 A·B·C v2(08-31 후속) - 필터 버그 수정 + 진짜 시간순
walk-forward로 재실행.

## 무엇을 고쳤나
inverter_disaggregation_check_v1_2026-08-28.py(및 이걸 그대로 복사한
inverter_disaggregation_ABC_v1_2026-08-31.py)의 "ok" 필터가
`isin(["observed", "night_zero"])`였는데, 실제 상태값은
`"night_zero_physical"`이라 철자가 달라 전혀 매칭되지 않았다 - 야간
정상0값이 전부 결측 취급되어 "8대 완전가용일"이 59일(여름51·봄8,
겨울·가을 0일)로 심하게 축소됐었다. `"night_zero_physical"`로 고치니
181일(겨울78·봄52·여름51)로 늘어난다 - Claude가 직접 재확인.

## 왜 이번엔 LOO 대신 walk-forward인가
Codex가 08-31 LOO 결과에 지적한 대로, LOO는 대상일 "이후" 날짜도
학습에 포함돼 실제 운영(과거만 알고 미래를 예측)을 재현하지 못한다.
181일이면 08-28 총출력모델과 비슷한 규모의 expanding-window
walk-forward(초기학습+시험블록)를 쓸 수 있어 이 한계를 구조적으로
없앨 수 있다 - 08-28에 59일이라 walk-forward를 못 썼던 이유(폴드당
시험표본 한자릿수) 자체가 표본buggy 때문이었으므로, 버그를 고친
지금은 원래 방법(walk-forward)으로 되돌아가는 게 맞다.

## 한계(여전히 남음)
- 가을 표본은 여전히 0일(부안 데이터 자체의 근본 한계, 이번 버그와 무관).
- Codex의 부안_인버터별_5분_야간0포함.parquet을 읽기전용으로만 사용.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
SRC = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_인버터별_5분_야간0포함.parquet"
)
OUT_DIR = HERE / "outputs" / "인버터분해_ABC_v2_필터수정_2026-08-31"
SEED = 42
MIN_SLOT_COUNT = 260
SUM_TOL_KWH = 1e-6
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "겨울"
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    return "가을"


def load_daily_per_inverter() -> pd.DataFrame:
    df = pd.read_parquet(SRC, columns=[
        "grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"])
    # 공통 품질정책은 physical_zero_night를 사용한다. 기존 저장자료의
    # night_zero_physical도 재현 가능하도록 읽기 호환만 유지한다.
    df["ok"] = df["quality_status_after_night"].isin(
        ["observed", "physical_zero_night", "night_zero_physical"]
    )
    df["date"] = df["grid_time_kst"].dt.date
    df["kwh_5min"] = df["ac_power_kw"] * (5 / 60)
    daily = df[df["ok"]].groupby(["date", "inverter_number"])["kwh_5min"].sum().unstack("inverter_number")
    count = df[df["ok"]].groupby(["date", "inverter_number"]).size().unstack("inverter_number")
    full8 = count.notna().all(axis=1) & (count.min(axis=1) >= MIN_SLOT_COUNT)
    return daily[full8].dropna()


def _score(daily8: pd.DataFrame, pred: np.ndarray, actual: np.ndarray, method_name: str) -> dict:
    inv_cols = list(daily8.columns)
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


def expanding_folds(n_days: int, initial: int, block: int) -> list[tuple]:
    folds = []
    end = initial
    while end + block <= n_days:
        folds.append((end, end + block))
        end += block
    return folds


def run_method_a(daily8: pd.DataFrame) -> dict:
    total = daily8.sum(axis=1).to_numpy()
    pred = np.repeat(total[:, None] * (1.0 / daily8.shape[1]), daily8.shape[1], axis=1)
    return _score(daily8, pred, daily8.to_numpy(), "A_정격용량비례(균등12.5%)")


def run_method_b_wf(daily8: pd.DataFrame, folds: list[tuple]) -> dict:
    inv_cols = list(daily8.columns)
    seasons = pd.Series([_season(d.month) for d in daily8.index], index=daily8.index)
    shares_all = daily8.div(daily8.sum(axis=1), axis=0)
    total = daily8.sum(axis=1)

    fold_results = []
    all_pred, all_actual = [], []
    for i, (train_end, test_end) in enumerate(folds, start=1):
        train_idx = daily8.index[:train_end]
        test_idx = daily8.index[train_end:test_end]
        train_shares = shares_all.loc[train_idx]
        train_seasons = seasons.loc[train_idx]

        pred_rows = []
        for day in test_idx:
            s = seasons.loc[day]
            pool = train_shares[train_seasons == s]
            if len(pool) < 3:
                pool = train_shares  # 계절표본 부족 시 학습구간 전체 중앙값으로 대체(기록됨)
            median_share = pool.median()
            norm = median_share / median_share.sum()
            pred_rows.append(total.loc[day] * norm.to_numpy())
        pred = np.array(pred_rows)
        actual = daily8.loc[test_idx].to_numpy()
        fold_score = _score(daily8.loc[test_idx], pred, actual, f"B_폴드{i}")
        fold_results.append({"폴드": i, "학습일수": train_end, "시험일수": len(test_idx),
                             "평균_nMAE_pct": fold_score["평균_nMAE_pct"]})
        all_pred.append(pred)
        all_actual.append(actual)

    pred_cat = np.concatenate(all_pred, axis=0)
    actual_cat = np.concatenate(all_actual, axis=0)
    overall = _score(daily8, pred_cat, actual_cat, "B_계절중앙값(walk-forward)")
    overall["폴드별"] = fold_results
    return overall


def run_method_c_wf(daily8: pd.DataFrame, folds: list[tuple]) -> dict:
    inv_cols = list(daily8.columns)
    shares_all = daily8.div(daily8.sum(axis=1), axis=0)
    total = daily8.sum(axis=1)
    doy = pd.Series([d.timetuple().tm_yday for d in daily8.index], index=daily8.index)
    feat = pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "log_total_kwh": np.log1p(total),
    }, index=daily8.index)

    fold_results = []
    all_pred, all_actual = [], []
    for i, (train_end, test_end) in enumerate(folds, start=1):
        train_idx = daily8.index[:train_end]
        test_idx = daily8.index[train_end:test_end]

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
        actual = daily8.loc[test_idx].to_numpy()
        fold_score = _score(daily8.loc[test_idx], pred, actual, f"C_폴드{i}")
        fold_results.append({"폴드": i, "학습일수": train_end, "시험일수": len(test_idx),
                             "평균_nMAE_pct": fold_score["평균_nMAE_pct"]})
        all_pred.append(pred)
        all_actual.append(actual)

    pred_cat = np.concatenate(all_pred, axis=0)
    actual_cat = np.concatenate(all_actual, axis=0)
    overall = _score(daily8, pred_cat, actual_cat, "C_LightGBM비중(walk-forward)")
    overall["폴드별"] = fold_results
    return overall


def run() -> dict:
    daily8 = load_daily_per_inverter()
    season_coverage = pd.Series([_season(d.month) for d in daily8.index]).value_counts().to_dict()
    folds = expanding_folds(len(daily8), INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    a = run_method_a(daily8)  # A는 전체 181일에 그대로(결정론적, CV 불필요)
    b = run_method_b_wf(daily8, folds)
    c = run_method_c_wf(daily8, folds)

    # A도 B·C와 같은 시험구간(폴드에 포함된 날짜)에서만 비교해야 공정 -
    # walk-forward 시험구간 전체(첫 60일 제외)에서 A를 별도 재계산.
    test_days_all = daily8.index[INITIAL_TRAIN_DAYS:folds[-1][1]] if folds else daily8.index[:0]
    a_on_test_only = run_method_a(daily8.loc[test_days_all]) if len(test_days_all) else None

    ranking = sorted([
        {"방법": "A_정격용량비례", "평균_nMAE_pct": (a_on_test_only or a)["평균_nMAE_pct"]},
        {"방법": "B_계절중앙값(walk-forward)", "평균_nMAE_pct": b["평균_nMAE_pct"]},
        {"방법": "C_LightGBM비중(walk-forward)", "평균_nMAE_pct": c["평균_nMAE_pct"]},
    ], key=lambda r: r["평균_nMAE_pct"])

    return {
        "필터수정_영향": "night_zero -> night_zero_physical 오타수정으로 완전가용일 59일->181일",
        "표본일수": int(len(daily8)),
        "계절별_표본일수": season_coverage,
        "walk_forward_폴드수": len(folds),
        "A_전체181일": a,
        "A_시험구간만(B·C와_공정비교용)": a_on_test_only,
        "B": b,
        "C": c,
        "순위_시험구간_공정비교": ranking,
        "_판정주의": "가을 표본 0일은 여전함(이번 수정과 무관한 별개 한계). walk-forward라 "
                    "이번엔 미래누출 없음(각 폴드는 그 이전 날짜로만 학습). promote_to_official "
                    "대상 아님 - 08-31 2차 중간점검.",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "부안_인버터분해_ABC_v2_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
