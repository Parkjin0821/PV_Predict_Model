# -*- coding: utf-8 -*-
"""초단기 DIFSWRF 결측 처리 4후보 공정 비교.

공식 B(부분가용/결함 타깃 제외), 정상구간 5폴드, 동일 시험행을 고정한다.
A 제외 / B raw NaN native / C raw NaN+mask / D fold-train NWP imputation+mask.
D의 복원기는 각 폴드 학습기간의 NWP·태양기하만 사용하며 발전량·ASOS
관측·시험기간 DIFSWRF를 학습에 사용하지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "DIFSWRF_결측전략비교_v1_2026-08-24"
DIF = "DIFSWRF_bsrn정제"
MASK = "DIFSWRF_결측여부"
IMPUTED = "DIFSWRF_폴드내추정"
N_INV = 5

WINDOWS = [
    ("1_여름", "2025-06-01", "2025-08-14"),
    ("2_가을_결함종료후", "2025-11-17", "2025-12-14"),
    ("3_겨울", "2025-12-15", "2026-02-14"),
    ("4_봄", "2026-02-15", "2026-04-14"),
    ("5_초여름", "2026-04-15", "2026-08-04"),
]
STRATEGIES = ["A_DIFSWRF제외", "B_NaN자체처리", "C_NaN+결측표시", "D_폴드내NWP추정+결측표시"]
IMPUTER_FEATURES = [
    "DSWRF", "DSWRFLX_bsrn정제", "TCDC", "LCDC", "MCDC", "HCDC",
    "TMP", "SKY", "REH", "WSD", "POP", "VEC", "목표_태양고도_deg",
    "일주기_sin", "일주기_cos", "연주기_sin", "연주기_cos",
]


def load_module(name: str, filename: str):
    p = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpc = load_module("difs_dpc", "defect_policy_comparison_v1_2026-08-21.py")
harness = dpc.harness
ultra = dpc.ultra


def model(seed: int):
    return LGBMRegressor(
        n_estimators=220, learning_rate=0.04, num_leaves=31,
        min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
    )


def metrics(y, p):
    e = np.asarray(y) - np.asarray(p)
    return float(np.abs(e).mean()), float(np.sqrt(np.mean(e ** 2)))


def dhi_upper(elev_deg: pd.Series) -> np.ndarray:
    """BSRN PPL의 보수적 상한. Sa는 최대측에 가까운 1400 W/m²로 둔다."""
    mu = np.clip(np.sin(np.deg2rad(elev_deg.to_numpy(float))), 0.0, None)
    return 1400.0 * 0.95 * np.power(mu, 1.2) + 50.0


def impute_fold(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    predictors = [c for c in IMPUTER_FEATURES if c in train.columns]
    observed = train[DIF].notna()
    if int(observed.sum()) < 500:
        raise RuntimeError(f"DIFSWRF 복원기 학습표본 부족: {int(observed.sum())}")
    imp = LGBMRegressor(
        n_estimators=180, learning_rate=0.04, num_leaves=24,
        min_child_samples=40, reg_lambda=0.5, random_state=seed,
        n_jobs=4, verbosity=-1,
    )
    # 오직 학습기간 중 DIFSWRF가 실제 존재하는 행으로만 복원기를 학습한다.
    imp.fit(train.loc[observed, predictors], train.loc[observed, DIF])

    tr = train[DIF].copy()
    te = test[DIF].copy()
    tr_missing, te_missing = tr.isna(), te.isna()
    if tr_missing.any():
        pred = imp.predict(train.loc[tr_missing, predictors])
        tr.loc[tr_missing] = np.clip(pred, 0, dhi_upper(train.loc[tr_missing, "목표_태양고도_deg"]))
    if te_missing.any():
        pred = imp.predict(test.loc[te_missing, predictors])
        te.loc[te_missing] = np.clip(pred, 0, dhi_upper(test.loc[te_missing, "목표_태양고도_deg"]))
    return tr, te, int(observed.sum()), predictors


def run_horizon(h: int, seed: int, capacity_kw: float):
    frame = dpc.load_ultra_frame(h)
    daylight = frame[frame["목표_낮시간"] > 0].copy()
    candidates = [c for c in ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
                  if c != DIF]
    base_cols = [c for c in frame.columns if c not in (ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS)
                 and not c.startswith("_") and c not in ("목표_발전출력_kW", "목표_낮시간")]
    fold_rows, pred_rows, hour_rows, imputer_rows = [], [], [], []

    for fold, s, e in WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index < end)]

        # 공식 B: 타깃시각 5대 완전가용만 학습/평가. DIFS 결측으로 행을 버리지 않는다.
        train_b = train_all[(train_all["_목표_가용인버터수"] >= N_INV)
                            & train_all["목표_발전출력_kW"].notna()].copy()
        test_b = test_all[(test_all["_목표_가용인버터수"] >= N_INV)
                          & test_all["목표_발전출력_kW"].notna()].copy()

        chosen = harness.select_features_in_fold(train_b, candidates, 0.3, True, True)
        common_features = base_cols + chosen
        # DIFS 이외의 기존 필수특성만으로 공통 행을 확정한다.
        required = [c for c in common_features if c not in harness.NATIVE_MISSING_OK]
        required += ["목표_발전출력_kW", "_지속성_직전출력_kW"]
        train = train_b.dropna(subset=required).copy()
        test = test_b.dropna(subset=required).copy()
        train[MASK] = train[DIF].isna().astype("int8")
        test[MASK] = test[DIF].isna().astype("int8")
        train[IMPUTED], test[IMPUTED], imp_n, imp_features = impute_fold(train, test, seed)

        definitions = {
            "A_DIFSWRF제외": common_features,
            "B_NaN자체처리": common_features + [DIF],
            "C_NaN+결측표시": common_features + [DIF, MASK],
            "D_폴드내NWP추정+결측표시": common_features + [IMPUTED, MASK],
        }
        y_train = train["목표_발전출력_kW"].to_numpy()
        y_test = test["목표_발전출력_kW"].to_numpy()
        target_time = test.index + pd.Timedelta(hours=h)

        for strategy, features in definitions.items():
            m = model(seed)
            m.fit(train[features], y_train)
            pred = np.clip(m.predict(test[features]), 0, capacity_kw)
            mae, rmse = metrics(y_test, pred)
            fold_rows.append({
                "수평_h": h, "폴드": fold, "전략": strategy,
                "학습행수": len(train), "시험행수": len(test),
                "DIFS_학습결측률_pct": round(100 * train[DIF].isna().mean(), 2),
                "DIFS_시험결측률_pct": round(100 * test[DIF].isna().mean(), 2),
                "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4),
            })
            tmp = pd.DataFrame({
                "수평_h": h, "폴드": fold, "전략": strategy,
                "발행시각": test.index, "대상시각": target_time,
                "실제_kW": y_test, "예측_kW": pred,
                "DIFS_원값결측": test[MASK].to_numpy(),
            })
            pred_rows.append(tmp)
            for hour, g in tmp.groupby(tmp["대상시각"].dt.hour):
                h_mae, h_rmse = metrics(g["실제_kW"], g["예측_kW"])
                hour_rows.append({"수평_h": h, "폴드": fold, "전략": strategy,
                                  "대상시_hour": int(hour), "n": len(g),
                                  "MAE_kW": round(h_mae, 4), "RMSE_kW": round(h_rmse, 4)})
        imputer_rows.append({"수평_h": h, "폴드": fold, "복원기_학습행수": imp_n,
                             "복원기_입력특성": "|".join(imp_features)})
        print(f"  +{h}h {fold}: 동일시험 {len(test):,}행, DIFS 결측 {test[DIF].isna().mean()*100:.1f}%")
    return fold_rows, pred_rows, hour_rows, imputer_rows


def main():
    dpc.require_v5_outputs()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed, capacity_kw = int(cfg["random_seed"]), float(cfg["site"]["capacity_kw"])
    all_fold, all_pred, all_hour, all_imp = [], [], [], []
    for h in (1, 2, 3, 4):
        a, b, c, d = run_horizon(h, seed, capacity_kw)
        all_fold += a; all_pred += b; all_hour += c; all_imp += d

    fold_df = pd.DataFrame(all_fold)
    pred_df = pd.concat(all_pred, ignore_index=True)
    hour_df = pd.DataFrame(all_hour)
    imp_df = pd.DataFrame(all_imp)

    summary = []
    for (h, strategy), g in fold_df.groupby(["수평_h", "전략"]):
        weights = g["시험행수"].to_numpy(float)
        summary.append({
            "수평_h": h, "전략": strategy, "폴드수": len(g),
            "가중MAE_kW": round(float(np.average(g["MAE_kW"], weights=weights)), 4),
            "가중RMSE_kW": round(float(np.average(g["RMSE_kW"], weights=weights)), 4),
            "최악폴드_RMSE_kW": round(float(g["RMSE_kW"].max()), 4),
            "시험행수합": int(g["시험행수"].sum()),
        })
    summary_df = pd.DataFrame(summary)
    for h, g in summary_df.groupby("수평_h"):
        base = float(g.loc[g["전략"] == "A_DIFSWRF제외", "가중RMSE_kW"].iloc[0])
        summary_df.loc[g.index, "A대비_RMSE개선_pct"] = (base - g["가중RMSE_kW"]) / base * 100

    fold_df.to_csv(OUT / "폴드별_비교.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "수평별_요약.csv", index=False, encoding="utf-8-sig")
    hour_df.to_csv(OUT / "시간대별_비교.csv", index=False, encoding="utf-8-sig")
    pred_df.to_parquet(OUT / "동일시험행_예측.parquet", index=False)
    imp_df.to_csv(OUT / "복원기_감사.csv", index=False, encoding="utf-8-sig")
    (OUT / "요약.txt").write_text(summary_df.to_string(index=False), encoding="utf-8")
    print("\n=== 수평별 요약 ===")
    print(summary_df.to_string(index=False))
    print(f"\n산출물: {OUT}")


if __name__ == "__main__":
    main()
