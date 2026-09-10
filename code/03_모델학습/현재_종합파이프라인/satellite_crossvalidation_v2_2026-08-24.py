# -*- coding: utf-8 -*-
"""GK2A 위성특성 v5 예비 교차검증.

기존 5개 시범창을 각 창 내부의 과거 학습/이후 시험으로 분리한다.
공식 B(목표시각 인버터 5대 완전가용), 동일 시험행, LightGBM 동일 설정으로
위성 없는 기준선 A와 VI006/IR105 후보 B~E를 비교한다.

주의: 44일 수집이 끝나기 전의 예비 게이트다. 30일 학습이 필요한
Heliosat 정규화(F)는 이 스크립트에서 평가하지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "위성특성_예비교차검증_v2_2026-08-24"
V5_DIR = ROOT / "outputs" / "v5_복구_2026-08-21"
SAT_TRIAL = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_trial_14d_v1_2026-08-21\광주_위성픽셀_14일시범_VI006_IR105.csv")
SAT_SEASON = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_reverify_5season_v1_2026-08-21\광주_위성픽셀_5계절재검증_VI006_IR105.csv")
N_INV = 5
DIF = "DIFSWRF_bsrn정제"
DIF_MASK = "DIFSWRF_결측여부"

WINDOWS = [
    ("1_여름", "2025-07-17", "2025-07-24", "2025-07-27"),
    ("2_가을", "2025-12-05", "2025-12-12", "2025-12-15"),
    ("3_겨울", "2025-12-16", "2025-12-23", "2025-12-26"),
    ("4_봄", "2026-04-01", "2026-04-08", "2026-04-11"),
    ("5_초여름_14일", "2026-04-27", "2026-05-07", "2026-05-11"),
]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpc = load_module("satcv_dpc", "defect_policy_comparison_v1_2026-08-21.py")
harness = dpc.harness
ultra = dpc.ultra


def model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=220, learning_rate=0.04, num_leaves=31,
        min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
    )


def load_satellite() -> pd.DataFrame:
    a = pd.read_csv(SAT_SEASON, parse_dates=["시각_kst"])
    b = pd.read_csv(SAT_TRIAL, parse_dates=["시각_kst"])
    cols = ["시각_kst", "VI006_픽셀값", "VI006_5x5평균", "IR105_픽셀값", "IR105_5x5평균"]
    sat = pd.concat([a[cols], b[cols]], ignore_index=True)
    sat = sat.drop_duplicates("시각_kst", keep="last").sort_values("시각_kst").set_index("시각_kst")

    # 몇 달 떨어진 창의 값을 변화량으로 연결하지 않는다.
    segment = sat.index.to_series().diff().gt(pd.Timedelta(hours=6)).cumsum()
    for ch in ("VI006", "IR105"):
        sat[f"{ch}_공간차"] = sat[f"{ch}_픽셀값"] - sat[f"{ch}_5x5평균"]
        sat[f"{ch}_시간차"] = sat.groupby(segment)[f"{ch}_5x5평균"].diff()
    sat["위성관측시각"] = sat.index
    return sat


def broadcast(index: pd.DatetimeIndex, sat: pd.DataFrame) -> pd.DataFrame:
    left = pd.DataFrame({"발행시각": pd.DatetimeIndex(index)}).sort_values("발행시각")
    right = sat.reset_index(drop=True).sort_values("위성관측시각")
    out = pd.merge_asof(
        left, right, left_on="발행시각", right_on="위성관측시각",
        direction="backward", tolerance=pd.Timedelta(minutes=165),
    ).set_index("발행시각")
    out["위성경과분"] = (out.index.to_series() - out["위성관측시각"]).dt.total_seconds() / 60.0
    out["VI006_결측"] = out["VI006_5x5평균"].isna().astype("int8")
    out["IR105_결측"] = out["IR105_5x5평균"].isna().astype("int8")
    return out


VI_STATIC = ["VI006_픽셀값", "VI006_5x5평균", "VI006_공간차", "위성경과분", "VI006_결측"]
IR_STATIC = ["IR105_픽셀값", "IR105_5x5평균", "IR105_공간차", "위성경과분", "IR105_결측"]
VARIANTS = {
    "A_위성없음": [],
    "B_VI006": VI_STATIC,
    "C_IR105": IR_STATIC,
    "D_두채널정적": list(dict.fromkeys(VI_STATIC + IR_STATIC)),
    "E_두채널+과거변화": list(dict.fromkeys(VI_STATIC + IR_STATIC + ["VI006_시간차", "IR105_시간차"])),
}


def metrics(y, p):
    err = np.asarray(y, float) - np.asarray(p, float)
    return float(np.abs(err).mean()), float(np.sqrt(np.mean(err ** 2)))


def run_horizon(h: int, sat_features: pd.DataFrame, seed: int, capacity_kw: float):
    frame = dpc.load_ultra_frame(h)
    for col in sat_features.columns:
        if col != "위성관측시각":
            frame[col] = sat_features[col].reindex(frame.index)
    daylight = frame[frame["목표_낮시간"] > 0].copy()

    all_candidates = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_candidates = [c for c in all_candidates if c != DIF]
    base_cols = [c for c in frame.columns if c not in all_candidates
                 and c not in set(sum(VARIANTS.values(), []))
                 and c != "위성관측시각" and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    fold_rows, pred_rows = [], []
    for fold, train_start, test_start, end in WINDOWS:
        tr0, ts, en = map(pd.Timestamp, (train_start, test_start, end))
        train_all = daylight[(daylight.index >= tr0) & (daylight.index < ts)]
        test_all = daylight[(daylight.index >= ts) & (daylight.index < en)]
        train_b = train_all[(train_all["_목표_가용인버터수"] >= N_INV)
                            & train_all["목표_발전출력_kW"].notna()].copy()
        test_b = test_all[(test_all["_목표_가용인버터수"] >= N_INV)
                          & test_all["목표_발전출력_kW"].notna()].copy()
        if len(train_b) < 100 or len(test_b) < 30:
            fold_rows.append({"수평_h": h, "폴드": fold, "후보": "평가불가",
                              "학습행수": len(train_b), "시험행수": len(test_b),
                              "사유": "공식B 표본부족"})
            continue

        chosen = harness.select_features_in_fold(train_b, base_candidates, 0.3, True, True)
        common = base_cols + chosen + [DIF, DIF_MASK]
        train_b[DIF_MASK] = train_b[DIF].isna().astype("int8")
        test_b[DIF_MASK] = test_b[DIF].isna().astype("int8")
        required = [c for c in common if c not in harness.NATIVE_MISSING_OK and c != DIF]
        required += ["목표_발전출력_kW", "_지속성_직전출력_kW"]
        train = train_b.dropna(subset=required).copy()
        test = test_b.dropna(subset=required).copy()
        if len(train) < 100 or len(test) < 30:
            fold_rows.append({"수평_h": h, "폴드": fold, "후보": "평가불가",
                              "학습행수": len(train), "시험행수": len(test),
                              "사유": "공통 입력표본부족"})
            continue

        ytr = train["목표_발전출력_kW"].to_numpy()
        yte = test["목표_발전출력_kW"].to_numpy()
        for variant, extra in VARIANTS.items():
            features = common + extra
            m = model(seed)
            m.fit(train[features], ytr)
            pred = np.clip(m.predict(test[features]), 0, capacity_kw)
            mae, rmse = metrics(yte, pred)
            fold_rows.append({
                "수평_h": h, "폴드": fold, "후보": variant,
                "학습행수": len(train), "시험행수": len(test),
                "시험_VI유효_pct": round(100 * test["VI006_5x5평균"].notna().mean(), 2),
                "시험_IR유효_pct": round(100 * test["IR105_5x5평균"].notna().mean(), 2),
                "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4), "사유": "",
            })
            pred_rows.append(pd.DataFrame({
                "수평_h": h, "폴드": fold, "후보": variant,
                "발행시각": test.index, "대상시각": test.index + pd.Timedelta(hours=h),
                "실제_kW": yte, "예측_kW": pred,
                "위성경과분": test["위성경과분"].to_numpy(),
                "VI006_결측": test["VI006_결측"].to_numpy(),
                "IR105_결측": test["IR105_결측"].to_numpy(),
            }))
        print(f"+{h}h {fold}: 동일시험 {len(test):,}행")
    return fold_rows, pred_rows


def main():
    dpc.require_v5_outputs()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(cfg["random_seed"])
    capacity_kw = float(cfg["site"]["capacity_kw"])
    capacity_profile = cfg["site"].get("capacity_profile", "legacy")

    sat = load_satellite()
    first_frame = dpc.load_ultra_frame(1)
    sat_features = broadcast(first_frame.index, sat)
    all_fold, all_pred = [], []
    for h in (1, 2, 3, 4):
        rows, preds = run_horizon(h, sat_features, seed, capacity_kw)
        all_fold += rows
        all_pred += preds

    fold_df = pd.DataFrame(all_fold)
    pred_df = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    valid = fold_df[fold_df["후보"].isin(VARIANTS)].copy()
    summary = []
    for (h, variant), g in valid.groupby(["수평_h", "후보"]):
        summary.append({
            "수평_h": h, "후보": variant, "평가폴드수": g["폴드"].nunique(),
            "폴드동일가중_MAE_kW": round(float(g["MAE_kW"].mean()), 4),
            "폴드동일가중_RMSE_kW": round(float(g["RMSE_kW"].mean()), 4),
            "시험행수합": int(g["시험행수"].sum()),
        })
    summary_df = pd.DataFrame(summary)
    for h, g in summary_df.groupby("수평_h"):
        base = g[g["후보"] == "A_위성없음"].iloc[0]
        idx = g.index
        summary_df.loc[idx, "A대비_MAE개선_pct"] = (
            (base["폴드동일가중_MAE_kW"] - g["폴드동일가중_MAE_kW"])
            / base["폴드동일가중_MAE_kW"] * 100
        )
        summary_df.loc[idx, "A대비_RMSE개선_pct"] = (
            (base["폴드동일가중_RMSE_kW"] - g["폴드동일가중_RMSE_kW"])
            / base["폴드동일가중_RMSE_kW"] * 100
        )
        for i in idx:
            variant = summary_df.loc[i, "후보"]
            merged = valid[(valid["수평_h"] == h) & (valid["후보"].isin(["A_위성없음", variant]))].pivot(
                index="폴드", columns="후보", values="RMSE_kW")
            if variant == "A_위성없음":
                worst = 0.0
            else:
                delta = (merged[variant] - merged["A_위성없음"]) / merged["A_위성없음"] * 100
                worst = float(delta.max())
            summary_df.loc[i, "최대폴드_RMSE악화_pct"] = round(worst, 3)

    summary_df["예비채택통과"] = (
        (summary_df["후보"] != "A_위성없음")
        & (summary_df["A대비_MAE개선_pct"] >= 1.0)
        & (summary_df["A대비_RMSE개선_pct"] >= 1.0)
        & (summary_df["최대폴드_RMSE악화_pct"] < 5.0)
        & (summary_df["평가폴드수"] == len(WINDOWS))
    )
    fold_df["용량프로필"] = capacity_profile
    summary_df["용량프로필"] = capacity_profile
    fold_df.to_csv(OUT / "폴드별_동일시험행_비교.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "수평별_예비판정.csv", index=False, encoding="utf-8-sig")
    if not pred_df.empty:
        pred_df.to_parquet(OUT / "행단위_예측.parquet", index=False)
    audit = {
        "status": "예비검증_공식채택금지",
        "capacity_kw": capacity_kw, "capacity_profile": capacity_profile,
        "satellite_rows": len(sat), "satellite_unique_times": int(sat.index.nunique()),
        "VI006_valid": int(sat["VI006_5x5평균"].notna().sum()),
        "IR105_valid": int(sat["IR105_5x5평균"].notna().sum()),
        "rules": ["v5", "공식B", "동일시험행", "위성시각<=발행시각", "165분 tolerance", "F는44일완료후"],
    }
    (OUT / "감사.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 수평별 예비판정 ===")
    print(summary_df.to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
