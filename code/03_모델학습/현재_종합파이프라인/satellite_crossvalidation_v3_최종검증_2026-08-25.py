# -*- coding: utf-8 -*-
"""GK2A 위성특성 v5 최종검증 — 연속44일 완결 + 후보F(Heliosat 정규화) 포함.

## 전제
`위성특성_교차검증_프로토콜_v1_2026-08-24.md`(04_평가검증)를 그대로 따른다.
`satellite_crossvalidation_v2_2026-08-24.py`(예비, A~E만)를 재사용하되
다음 두 가지만 바꾼다.
1. 위성 원본에 신규 완결된 연속44일 CSV(352행, 중복0)를 합친다.
2. **후보 F**(E + 학습폴드 내부에서만 만든 Heliosat식 청천/구름 정규화)를
   추가한다. 연속44일 중 앞 30일(03-28~04-26)을 각 고정 KST 시각대별
   청천기준(하위5%)·구름기준(상위95%) 학습이력으로 쓰고, 뒤 14일
   (04-27~05-10, 기존 14일 시범과 동일 시험구간)에서 정규화지수를 적용해
   같은 시험행으로 A~E와 재비교한다(AGENTS.md 2026-08-21 절의 설계를
   그대로 따르되, 프로토콜의 누출방지 규칙에 맞춰 **분위수 계산은
   03-28~04-26까지만** 쓴다 — 08-21 메모의 "44일 전체 이력" 표현은 이
   더 엄격한 04-24 프로토콜로 대체됨).
3. 1~4번 창(여름·가을·겨울·봄 단기 시범창)은 학습이력이 30일 미만이라
   F를 시도하지 않고 **평가불가**로 명시한다(프로토콜 "특정 폴드의
   학습표본이 부족하면 평가불가로 표시하며 다른 폴드에 합쳐 숨기지
   않는다"). 5번 창만 학습구간을 03-28로 넓혀(연속44일 덕분에 새로
   가능해짐) F를 실제로 평가한다.

## 재사용(재구현 금지 원칙)
`satellite_crossvalidation_v2_2026-08-24.py`의 `load_module`·`model`·
`broadcast`·`metrics`·VI_STATIC/IR_STATIC 정의를 그대로 가져다 쓴다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "위성특성_최종검증_v3_2026-08-25"
SAT_TRIAL = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_trial_14d_v1_2026-08-21\광주_위성픽셀_14일시범_VI006_IR105.csv")
SAT_SEASON = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_reverify_5season_v1_2026-08-21\광주_위성픽셀_5계절재검증_VI006_IR105.csv")
SAT_CONT44 = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_continuous44d_v1_2026-08-21\광주_위성픽셀_연속44일_VI006_IR105.csv")
N_INV = 5
DIF = "DIFSWRF_bsrn정제"
DIF_MASK = "DIFSWRF_결측여부"
HELIOSAT_TRAIN_START = pd.Timestamp("2026-03-28")  # 연속44일 앞단(청천/구름 이력용)
MIN_HOUR_SAMPLES = 15   # 시각대별 분위수를 신뢰하려면 최소 이만큼의 학습일 필요
MIN_FOLD_HISTORY_DAYS = 25  # 이보다 이력이 짧은 창은 F 자체를 시도하지 않음

# 5번 창만 04-27~05-10 시험(기존 14일 시범과 동일)을 그대로 두고, 학습을
# 03-28까지 넓힌다 — 나머지 1~4번은 v2 예비검증과 완전히 동일하게 유지해
# 재현성을 확인한다.
WINDOWS = [
    ("1_여름", "2025-07-17", "2025-07-24", "2025-07-27"),
    ("2_가을", "2025-12-05", "2025-12-12", "2025-12-15"),
    ("3_겨울", "2025-12-16", "2025-12-23", "2025-12-26"),
    ("4_봄", "2026-04-01", "2026-04-08", "2026-04-11"),
    ("5_초여름_연속44일", "2026-03-28", "2026-04-27", "2026-05-11"),
]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2 = load_module("satcv_v2", "satellite_crossvalidation_v2_2026-08-24.py")
dpc, harness, ultra = v2.dpc, v2.harness, v2.ultra
broadcast, metrics, model = v2.broadcast, v2.metrics, v2.model
VI_STATIC, IR_STATIC = v2.VI_STATIC, v2.IR_STATIC


def load_satellite_v3() -> pd.DataFrame:
    a = pd.read_csv(SAT_SEASON, parse_dates=["시각_kst"])
    b = pd.read_csv(SAT_TRIAL, parse_dates=["시각_kst"])
    c = pd.read_csv(SAT_CONT44, parse_dates=["시각_kst"])
    cols = ["시각_kst", "VI006_픽셀값", "VI006_5x5평균", "IR105_픽셀값", "IR105_5x5평균"]
    # 연속44일이 가장 완결된 원본이므로 중복 시각은 이쪽 값을 우선한다
    # (trial14의 뒤 14일은 continuous44가 같은 원본파일을 재사용했으므로
    # 이론상 완전히 같은 값이어야 한다 — 아래서 이 가정도 검증한다).
    sat = pd.concat([a[cols], b[cols], c[cols]], ignore_index=True)
    overlap_check = sat[sat.duplicated("시각_kst", keep=False)].sort_values("시각_kst")
    conflicts = 0
    if len(overlap_check):
        g = overlap_check.groupby("시각_kst")[["VI006_5x5평균", "IR105_5x5평균"]].nunique()
        conflicts = int(((g["VI006_5x5평균"] > 1) | (g["IR105_5x5평균"] > 1)).sum())
    sat = sat.drop_duplicates("시각_kst", keep="last").sort_values("시각_kst").set_index("시각_kst")

    segment = sat.index.to_series().diff().gt(pd.Timedelta(hours=6)).cumsum()
    for ch in ("VI006", "IR105"):
        sat[f"{ch}_공간차"] = sat[f"{ch}_픽셀값"] - sat[f"{ch}_5x5평균"]
        sat[f"{ch}_시간차"] = sat.groupby(segment)[f"{ch}_5x5평균"].diff()
    sat["위성관측시각"] = sat.index
    return sat, conflicts


def heliosat_stats(train_hist: pd.DataFrame) -> pd.DataFrame:
    """고정 KST 시각대별 청천(하위5%)·구름(상위95%) 기준을 학습이력만으로 계산."""
    g = train_hist.copy()
    g["시각대"] = g.index.hour
    out = g.groupby("시각대").agg(
        VI_clear=("VI006_5x5평균", lambda s: s.quantile(0.05)),
        VI_cloud=("VI006_5x5평균", lambda s: s.quantile(0.95)),
        IR_clear=("IR105_5x5평균", lambda s: s.quantile(0.95)),  # 온도 높음=청천
        IR_cloud=("IR105_5x5평균", lambda s: s.quantile(0.05)),  # 온도 낮음=구름
        n=("VI006_5x5평균", "count"),
    )
    return out


def apply_heliosat(frame: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    hour = frame["위성관측시각"].dt.hour
    s = stats.reindex(hour).reset_index(drop=True)
    s.index = frame.index
    vi = frame["VI006_5x5평균"]
    ir = frame["IR105_5x5평균"]
    denom_vi = (s["VI_cloud"] - s["VI_clear"]).replace(0, np.nan)
    idx_vi = ((vi - s["VI_clear"]) / denom_vi).clip(0, 1)
    denom_ir = (s["IR_clear"] - s["IR_cloud"]).replace(0, np.nan)
    idx_ir = ((s["IR_clear"] - ir) / denom_ir).clip(0, 1)
    enough = (s["n"] >= MIN_HOUR_SAMPLES).to_numpy()
    idx_vi = idx_vi.where(enough)
    idx_ir = idx_ir.where(enough)
    out = frame.copy()
    out["VI006_청천지수"] = idx_vi
    out["IR105_청천지수"] = idx_ir
    return out


F_EXTRA = ["VI006_청천지수", "IR105_청천지수"]
VARIANTS = dict(v2.VARIANTS)
VARIANTS["F_청천지수"] = list(dict.fromkeys(
    VI_STATIC + IR_STATIC + ["VI006_시간차", "IR105_시간차"] + F_EXTRA))


def run_horizon(h: int, sat: pd.DataFrame, seed: int, capacity_kw: float):
    frame = dpc.load_ultra_frame(h)
    first_broadcast = broadcast(frame.index, sat)
    daylight_full = frame[frame["목표_낮시간"] > 0].copy()

    all_candidates = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_candidates = [c for c in all_candidates if c != DIF]
    base_cols = [c for c in frame.columns if c not in all_candidates
                 and c not in set(sum(v2.VARIANTS.values(), []) + F_EXTRA + ["VI006_시간차", "IR105_시간차"])
                 and c != "위성관측시각" and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    fold_rows, pred_rows = [], []
    for fold, train_start, test_start, end in WINDOWS:
        tr0, ts, en = map(pd.Timestamp, (train_start, test_start, end))
        base = first_broadcast.reindex(daylight_full.index)
        for col in base.columns:
            if col != "위성관측시각":
                daylight_full[col] = base[col]
        daylight_full["위성관측시각"] = base["위성관측시각"]

        train_all = daylight_full[(daylight_full.index >= tr0) & (daylight_full.index < ts)]
        test_all = daylight_full[(daylight_full.index >= ts) & (daylight_full.index < en)]
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

        # --- 후보 F: 학습구간 이력이 충분한 창에서만 시도 ---
        sat_hist = sat[(sat.index >= tr0) & (sat.index < ts)]
        hist_days = int(sat_hist.index.normalize().nunique())
        f_ready = hist_days >= MIN_FOLD_HISTORY_DAYS
        if f_ready:
            stats = heliosat_stats(sat_hist)
            train_f = apply_heliosat(train, stats)
            test_f = apply_heliosat(test, stats)
        else:
            fold_rows.append({"수평_h": h, "폴드": fold, "후보": "F_청천지수",
                              "학습행수": len(train), "시험행수": len(test),
                              "사유": f"위성 학습이력 {hist_days}일 < {MIN_FOLD_HISTORY_DAYS}일(Heliosat 정규화 불가)"})

        ytr = train["목표_발전출력_kW"].to_numpy()
        yte = test["목표_발전출력_kW"].to_numpy()
        for variant, extra in VARIANTS.items():
            if variant == "F_청천지수":
                if not f_ready:
                    continue
                tr_use, te_use = train_f, test_f
            else:
                tr_use, te_use = train, test
            features = common + extra
            m = model(seed)
            m.fit(tr_use[features], ytr)
            pred = np.clip(m.predict(te_use[features]), 0, capacity_kw)
            mae, rmse = metrics(yte, pred)
            row = {
                "수평_h": h, "폴드": fold, "후보": variant,
                "학습행수": len(tr_use), "시험행수": len(te_use),
                "시험_VI유효_pct": round(100 * te_use["VI006_5x5평균"].notna().mean(), 2),
                "시험_IR유효_pct": round(100 * te_use["IR105_5x5평균"].notna().mean(), 2),
                "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4), "사유": "",
            }
            if variant == "F_청천지수":
                row["시험_VI청천지수유효_pct"] = round(100 * te_use["VI006_청천지수"].notna().mean(), 2)
                row["시험_IR청천지수유효_pct"] = round(100 * te_use["IR105_청천지수"].notna().mean(), 2)
                row["학습이력일수"] = hist_days
            fold_rows.append(row)
            pred_rows.append(pd.DataFrame({
                "수평_h": h, "폴드": fold, "후보": variant,
                "발행시각": te_use.index, "대상시각": te_use.index + pd.Timedelta(hours=h),
                "실제_kW": yte, "예측_kW": pred,
            }))
        print(f"+{h}h {fold}: 동일시험 {len(test):,}행 (F준비={f_ready}, 학습이력={hist_days}일)")
    return fold_rows, pred_rows


def summarize(fold_df: pd.DataFrame, windows: list, variants: dict) -> pd.DataFrame:
    valid = fold_df[fold_df["후보"].isin(variants)].copy()
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
        base = g[g["후보"] == "A_위성없음"]
        if base.empty:
            continue
        base = base.iloc[0]
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
            if variant == "A_위성없음" or "A_위성없음" not in merged.columns or variant not in merged.columns:
                worst = 0.0
            else:
                common_folds = merged.dropna()
                delta = (common_folds[variant] - common_folds["A_위성없음"]) / common_folds["A_위성없음"] * 100
                worst = float(delta.max()) if len(delta) else 0.0
            summary_df.loc[i, "최대폴드_RMSE악화_pct"] = round(worst, 3)

    summary_df["전체폴드평가통과"] = (
        (summary_df["후보"] != "A_위성없음")
        & (summary_df["A대비_MAE개선_pct"] >= 1.0)
        & (summary_df["A대비_RMSE개선_pct"] >= 1.0)
        & (summary_df["최대폴드_RMSE악화_pct"] < 5.0)
        & (summary_df["평가폴드수"] == len(windows))
    )
    return summary_df


def main():
    dpc.require_v5_outputs()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(cfg["random_seed"])
    capacity_kw = float(cfg["site"]["capacity_kw"])
    capacity_profile = cfg["site"].get("capacity_profile", "legacy")

    sat, conflicts = load_satellite_v3()
    print(f"위성 통합 원본: {len(sat):,}행, 시각중복 값불일치 {conflicts}건")

    all_fold, all_pred = [], []
    for h in (1, 2, 3, 4):
        rows, preds = run_horizon(h, sat, seed, capacity_kw)
        all_fold += rows
        all_pred += preds

    fold_df = pd.DataFrame(all_fold)
    pred_df = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    fold_df["용량프로필"] = capacity_profile

    ae_summary = summarize(fold_df, WINDOWS, v2.VARIANTS)
    f_only = fold_df[fold_df["후보"] == "F_청천지수"].copy()
    f_ready_rows = f_only[f_only["사유"] == ""]

    fold_df.to_csv(OUT / "폴드별_동일시험행_비교.csv", index=False, encoding="utf-8-sig")
    ae_summary.to_csv(OUT / "수평별_판정_A_E.csv", index=False, encoding="utf-8-sig")
    if not pred_df.empty:
        pred_df.to_parquet(OUT / "행단위_예측.parquet", index=False)

    # F는 5번 창(유일한 평가가능 창)에서만 A~E와 나란히 재비교한다.
    f_window_compare = fold_df[(fold_df["폴드"] == "5_초여름_연속44일")
                               & (fold_df["후보"].isin(list(VARIANTS)))].copy()
    f_window_compare.to_csv(OUT / "F_5번창_전후보_비교.csv", index=False, encoding="utf-8-sig")

    f_verdict = []
    for h, g in f_window_compare.groupby("수평_h"):
        a_row = g[g["후보"] == "A_위성없음"]
        f_row = g[g["후보"] == "F_청천지수"]
        if a_row.empty or f_row.empty:
            continue
        a_mae, a_rmse = a_row.iloc[0]["MAE_kW"], a_row.iloc[0]["RMSE_kW"]
        f_mae, f_rmse = f_row.iloc[0]["MAE_kW"], f_row.iloc[0]["RMSE_kW"]
        best_raw = g[g["후보"].isin(["B_VI006", "C_IR105", "D_두채널정적", "E_두채널+과거변화"])].sort_values("RMSE_kW").iloc[0]
        f_verdict.append({
            "수평_h": h,
            "A_MAE": a_mae, "A_RMSE": a_rmse,
            "F_MAE": f_mae, "F_RMSE": f_rmse,
            "F_A대비_MAE개선_pct": round((a_mae - f_mae) / a_mae * 100, 3),
            "F_A대비_RMSE개선_pct": round((a_rmse - f_rmse) / a_rmse * 100, 3),
            "최선원시값후보": best_raw["후보"], "최선원시값_RMSE": best_raw["RMSE_kW"],
            "F_원시최선대비_RMSE개선_pct": round((best_raw["RMSE_kW"] - f_rmse) / best_raw["RMSE_kW"] * 100, 3),
        })
    f_verdict_df = pd.DataFrame(f_verdict)
    f_verdict_df.to_csv(OUT / "F_최종판정.csv", index=False, encoding="utf-8-sig")

    audit = {
        "status": "최종검증_연속44일반영",
        "capacity_kw": capacity_kw, "capacity_profile": capacity_profile,
        "satellite_rows": len(sat), "satellite_unique_times": int(sat.index.nunique()),
        "satellite_시각중복_값불일치": conflicts,
        "VI006_valid": int(sat["VI006_5x5평균"].notna().sum()),
        "IR105_valid": int(sat["IR105_5x5평균"].notna().sum()),
        "F_평가가능_폴드": sorted(f_ready_rows["폴드"].unique().tolist()) if len(f_ready_rows) else [],
        "F_평가불가_사유": f_only[f_only["사유"] != ""][["수평_h", "폴드", "사유"]].drop_duplicates().to_dict("records"),
        "규칙": ["v5", "공식B", "동일시험행", "위성시각<=발행시각", "165분 tolerance",
                "F 분위수는 03-28~04-26 학습이력만 사용(프로토콜 04-24 누출방지 규칙)"],
    }
    (OUT / "감사.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== A~E 전체폴드 판정(1~5번 창, F 제외) ===")
    print(ae_summary.to_string(index=False))
    print("\n=== F 5번창 단독비교(유일 평가가능 창) ===")
    print(f_verdict_df.to_string(index=False) if len(f_verdict_df) else "F 평가가능 조합 없음")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
