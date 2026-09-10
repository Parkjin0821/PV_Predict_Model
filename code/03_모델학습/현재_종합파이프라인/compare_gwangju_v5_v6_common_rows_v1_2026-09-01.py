# -*- coding: utf-8 -*-
"""Compare Gwangju v5/v6 on identical test rows and report coverage changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
V5 = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24"
V6 = ROOT / "outputs" / "E2E_v6_동료대조_공식B_v2_2026-09-01"


def metric(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    err = actual.astype(float) - pred.astype(float)
    return float(err.abs().mean()), float(np.sqrt(np.mean(np.square(err))))


def coverage_by_fold() -> pd.DataFrame:
    rows = []
    for filename, time_cols in [
        ("행단위_ac_power_예측정답.csv", ["발행시각", "대상시각"]),
        ("행단위_daily_예측정답.csv", ["발행시각", "대상일"]),
    ]:
        a = pd.read_csv(V5 / filename, parse_dates=time_cols)
        b = pd.read_csv(V6 / filename, parse_dates=time_cols)
        keys = ["티어", "수평_h", "폴드", *time_cols]
        merged = a[keys].merge(b[keys], on=keys, how="outer", indicator=True)
        for (tier, horizon, fold), g in merged.groupby(["티어", "수평_h", "폴드"], dropna=False):
            rows.append({"티어": tier, "수평_h": horizon, "폴드": fold,
                         "v5_only_제외시험행": int((g["_merge"] == "left_only").sum()),
                         "v6_only_신규시험행": int((g["_merge"] == "right_only").sum()),
                         "공통시험행": int((g["_merge"] == "both").sum())})
    return pd.DataFrame(rows)


def compare_ac() -> tuple[list[dict], list[dict], list[dict]]:
    a = pd.read_csv(V5 / "행단위_ac_power_예측정답.csv", parse_dates=["발행시각", "대상시각"])
    b = pd.read_csv(V6 / "행단위_ac_power_예측정답.csv", parse_dates=["발행시각", "대상시각"])
    keys = ["티어", "수평_h", "폴드", "발행시각", "대상시각"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise RuntimeError("duplicate AC evaluation keys")
    merged = a.merge(b, on=keys, how="outer", suffixes=("_v5", "_v6"), indicator=True)
    overall, folds, coverage = [], [], []
    for (tier, horizon), g in merged.groupby(["티어", "수평_h"], dropna=False):
        common = g[g["_merge"] == "both"].copy()
        if not np.allclose(common["실제_kW_v5"], common["실제_kW_v6"], equal_nan=True):
            raise RuntimeError(f"actual target mismatch: {tier}/{horizon}")
        m5, r5 = metric(common["실제_kW_v5"], common["예측_kW_v5"])
        m6, r6 = metric(common["실제_kW_v6"], common["예측_kW_v6"])
        overall.append({"티어": tier, "수평_h": horizon, "동일시험행_n": len(common),
                        "v5_MAE": m5, "v5_RMSE": r5, "v6_MAE": m6, "v6_RMSE": r6,
                        "MAE_개선율_pct": (m5 - m6) / m5 * 100,
                        "RMSE_개선율_pct": (r5 - r6) / r5 * 100})
        coverage.append({"티어": tier, "수평_h": horizon, "v5_only_제외시험행": int((g["_merge"] == "left_only").sum()),
                         "v6_only_신규시험행": int((g["_merge"] == "right_only").sum()), "공통시험행": len(common)})
        for fold, fg in common.groupby("폴드"):
            fm5, fr5 = metric(fg["실제_kW_v5"], fg["예측_kW_v5"])
            fm6, fr6 = metric(fg["실제_kW_v6"], fg["예측_kW_v6"])
            folds.append({"티어": tier, "수평_h": horizon, "폴드": fold, "동일시험행_n": len(fg),
                          "v5_MAE": fm5, "v5_RMSE": fr5, "v6_MAE": fm6, "v6_RMSE": fr6,
                          "MAE_악화율_pct": (fm6 - fm5) / fm5 * 100,
                          "RMSE_악화율_pct": (fr6 - fr5) / fr5 * 100})
    return overall, folds, coverage


def compare_daily() -> tuple[dict, list[dict], dict]:
    a = pd.read_csv(V5 / "행단위_daily_예측정답.csv", parse_dates=["발행시각", "대상일"])
    b = pd.read_csv(V6 / "행단위_daily_예측정답.csv", parse_dates=["발행시각", "대상일"])
    keys = ["티어", "수평_h", "폴드", "발행시각", "대상일"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise RuntimeError("duplicate daily evaluation keys")
    g = a.merge(b, on=keys, how="outer", suffixes=("_v5", "_v6"), indicator=True)
    common = g[g["_merge"] == "both"].copy()
    if not np.allclose(common["실제_kWh_v5"], common["실제_kWh_v6"], equal_nan=True):
        raise RuntimeError("daily actual target mismatch on common rows")
    m5, r5 = metric(common["실제_kWh_v5"], common["예측_kWh_v5"])
    m6, r6 = metric(common["실제_kWh_v6"], common["예측_kWh_v6"])
    overall = {"티어": "일간", "수평_h": "D+1", "동일시험행_n": len(common),
               "v5_MAE": m5, "v5_RMSE": r5, "v6_MAE": m6, "v6_RMSE": r6,
               "MAE_개선율_pct": (m5 - m6) / m5 * 100,
               "RMSE_개선율_pct": (r5 - r6) / r5 * 100}
    folds = []
    for fold, fg in common.groupby("폴드"):
        fm5, fr5 = metric(fg["실제_kWh_v5"], fg["예측_kWh_v5"])
        fm6, fr6 = metric(fg["실제_kWh_v6"], fg["예측_kWh_v6"])
        folds.append({"티어": "일간", "수평_h": "D+1", "폴드": fold, "동일시험행_n": len(fg),
                      "v5_MAE": fm5, "v5_RMSE": fr5, "v6_MAE": fm6, "v6_RMSE": fr6,
                      "MAE_악화율_pct": (fm6 - fm5) / fm5 * 100,
                      "RMSE_악화율_pct": (fr6 - fr5) / fr5 * 100})
    coverage = {"티어": "일간", "수평_h": "D+1", "v5_only_제외시험행": int((g["_merge"] == "left_only").sum()),
                "v6_only_신규시험행": int((g["_merge"] == "right_only").sum()), "공통시험행": len(common)}
    return overall, folds, coverage


def main() -> None:
    overall, folds, coverage = compare_ac()
    d_overall, d_folds, d_coverage = compare_daily()
    overall.append(d_overall); folds.extend(d_folds); coverage.append(d_coverage)
    perf = pd.DataFrame(overall)
    fold_perf = pd.DataFrame(folds)
    cov = pd.DataFrame(coverage)
    # Conservative decision: no negative pooled change and no >=5% fold degradation.
    worst = fold_perf.groupby(["티어", "수평_h"], dropna=False)[["MAE_악화율_pct", "RMSE_악화율_pct"]].max().reset_index()
    decision = perf.merge(worst, on=["티어", "수평_h"], how="left")
    decision["pooled_악화없음"] = (decision["MAE_개선율_pct"] >= 0) & (decision["RMSE_개선율_pct"] >= 0)
    decision["폴드5pct_위반없음"] = (decision["MAE_악화율_pct"] < 5) & (decision["RMSE_악화율_pct"] < 5)
    decision["판정"] = np.where(decision["pooled_악화없음"] & decision["폴드5pct_위반없음"], "승격검토가능", "v5유지")
    perf.to_csv(V6 / "v5대비_동일시험행_성능_전체.csv", index=False, encoding="utf-8-sig")
    fold_perf.to_csv(V6 / "v5대비_동일시험행_성능_폴드별.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(V6 / "v5대비_시험행_커버리지변화.csv", index=False, encoding="utf-8-sig")
    coverage_by_fold().to_csv(V6 / "v5대비_시험행_커버리지변화_폴드별.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(V6 / "v5대비_수평별_판정.csv", index=False, encoding="utf-8-sig")
    summary = {"comparison_basis": "v5/v6 identical test-row intersection, pooled metrics",
               "all_horizons_promotable": bool((decision["판정"] == "승격검토가능").all()),
               "production_modified": False, "rows": decision.to_dict("records")}
    (V6 / "v5대비_비교요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(decision.to_string(index=False))
    print("\ncoverage")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
