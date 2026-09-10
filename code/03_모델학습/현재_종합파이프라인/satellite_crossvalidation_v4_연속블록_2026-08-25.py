# -*- coding: utf-8 -*-
"""GK2A 위성특성 최종검증 재시도 — 연속44일 블록 내부 expanding rolling-origin.

## 왜 한 번 더 하나 (v3의 설계 결함 교정)
`satellite_crossvalidation_v3_최종검증_2026-08-25.py`에서 후보 F는 5개 창
중 1개(초여름)에서만 평가됐다. 나머지 4개 창은 위성 학습이력이 7일뿐이라
Heliosat 30일 전제를 못 채웠기 때문이다. 그 결과 **F는 "전체폴드 통과"
기준을 원리적으로 만족할 수 없는 상태로 판정**됐다 — 이건 방법의 실패가
아니라 평가설계의 결함이다.

이번에는 **연속44일 블록(2026-03-28~05-10) 안에서만** expanding
rolling-origin 4폴드를 구성해 A~F가 전부 같은 4폴드·같은 시험행을 받게
한다. 이러면 F도 공정하게 "전체폴드" 기준으로 판정된다.

## v3에서 바꾸는 것 (딱 두 가지)
1. 폴드: 연속블록 내부 expanding rolling-origin 4개. 학습은 항상 03-28에서
   시작해 폴드마다 늘어나고, Heliosat 분위수도 **그 폴드 학습구간만으로**
   다시 계산한다(누출방지 프로토콜 규칙 유지).
2. 후보 `F_lean` 추가: 현재 F는 E(픽셀값·공간차·시간차 전부) 위에 지수를
   얹은 형태라 특성이 희석됐을 수 있다. "정규화 자체가 도움이 되는가"를
   분리해 보기 위해 **A + 청천지수 2개 + 결측표시 + 경과분**만 쓰는 얇은
   버전을 함께 평가한다.

나머지(모델·전처리·누출방지·공식B·219kW·동일시험행)는 v3/v2 그대로 쓴다.

## ★사전등록 채택기준(실행 전 고정, 사후 변경 금지)★
- A 대비 폴드동일가중 MAE·RMSE가 **둘 다 1% 이상 개선**
- **어느 폴드에서도 RMSE 5% 이상 악화 없음**
- **4개 폴드 전부에서 평가**될 것
- 추가: F/F_lean은 같은 폴드의 **최선 원시값 후보(B~E)도 이겨야** 채택
  (못 이기면 더 단순한 원시값 후보를 택한다)
통과 못 하면 위성은 연구후보로 보류하고 v5는 위성 없이 확정한다.
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
OUT = ROOT / "outputs" / "위성특성_최종검증_v4_연속블록_2026-08-25"
N_INV = 5
DIF = "DIFSWRF_bsrn정제"
DIF_MASK = "DIFSWRF_결측여부"
MIN_HOUR_SAMPLES = 15

# 연속44일 블록 내부 expanding rolling-origin.
# (학습시작, 시험시작, 시험종료) — 학습은 항상 03-28부터, 시험창은 순차 이동.
BLOCK_START = "2026-03-28"
WINDOWS = [
    ("R1_0427_0430", BLOCK_START, "2026-04-27", "2026-05-01"),
    ("R2_0501_0503", BLOCK_START, "2026-05-01", "2026-05-04"),
    ("R3_0504_0507", BLOCK_START, "2026-05-04", "2026-05-08"),
    ("R4_0508_0510", BLOCK_START, "2026-05-08", "2026-05-11"),
]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v3 = load_module("satcv_v3", "satellite_crossvalidation_v3_최종검증_2026-08-25.py")
v2 = v3.v2
dpc, harness, ultra = v3.dpc, v3.harness, v3.ultra
broadcast, metrics, model = v3.broadcast, v3.metrics, v3.model
VI_STATIC, IR_STATIC = v3.VI_STATIC, v3.IR_STATIC
heliosat_stats, apply_heliosat = v3.heliosat_stats, v3.apply_heliosat

F_EXTRA = ["VI006_청천지수", "IR105_청천지수"]
LEAN = ["VI006_청천지수", "IR105_청천지수", "위성경과분", "VI006_결측", "IR105_결측"]
VARIANTS = dict(v2.VARIANTS)
VARIANTS["F_청천지수"] = list(dict.fromkeys(
    VI_STATIC + IR_STATIC + ["VI006_시간차", "IR105_시간차"] + F_EXTRA))
VARIANTS["F_lean_청천지수만"] = list(LEAN)
HELIOSAT_VARIANTS = {"F_청천지수", "F_lean_청천지수만"}
RAW_SAT_VARIANTS = ["B_VI006", "C_IR105", "D_두채널정적", "E_두채널+과거변화"]

ALL_SAT_FEATURES = set(sum(VARIANTS.values(), []))


def run_horizon(h: int, sat: pd.DataFrame, seed: int, capacity_kw: float):
    frame = dpc.load_ultra_frame(h)
    sat_features = broadcast(frame.index, sat)
    daylight = frame[frame["목표_낮시간"] > 0].copy()
    base = sat_features.reindex(daylight.index)
    for col in base.columns:
        daylight[col] = base[col]

    all_candidates = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_candidates = [c for c in all_candidates if c != DIF]
    base_cols = [c for c in frame.columns if c not in all_candidates
                 and c not in ALL_SAT_FEATURES
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

        # Heliosat 분위수는 이 폴드의 학습구간 위성이력만으로 계산(누출방지)
        sat_hist = sat[(sat.index >= tr0) & (sat.index < ts)]
        hist_days = int(sat_hist.index.normalize().nunique())
        stats = heliosat_stats(sat_hist)
        train_f = apply_heliosat(train, stats)
        test_f = apply_heliosat(test, stats)

        ytr = train["목표_발전출력_kW"].to_numpy()
        yte = test["목표_발전출력_kW"].to_numpy()
        for variant, extra in VARIANTS.items():
            tr_use, te_use = (train_f, test_f) if variant in HELIOSAT_VARIANTS else (train, test)
            features = common + extra
            m = model(seed)
            m.fit(tr_use[features], ytr)
            pred = np.clip(m.predict(te_use[features]), 0, capacity_kw)
            mae, rmse = metrics(yte, pred)
            row = {
                "수평_h": h, "폴드": fold, "후보": variant,
                "학습행수": len(tr_use), "시험행수": len(te_use), "학습이력일수": hist_days,
                "시험_VI유효_pct": round(100 * te_use["VI006_5x5평균"].notna().mean(), 2),
                "시험_IR유효_pct": round(100 * te_use["IR105_5x5평균"].notna().mean(), 2),
                "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4), "사유": "",
            }
            if variant in HELIOSAT_VARIANTS:
                row["시험_VI청천지수유효_pct"] = round(100 * te_use["VI006_청천지수"].notna().mean(), 2)
                row["시험_IR청천지수유효_pct"] = round(100 * te_use["IR105_청천지수"].notna().mean(), 2)
            fold_rows.append(row)
            pred_rows.append(pd.DataFrame({
                "수평_h": h, "폴드": fold, "후보": variant,
                "발행시각": te_use.index, "대상시각": te_use.index + pd.Timedelta(hours=h),
                "실제_kW": yte, "예측_kW": pred,
            }))
        print(f"+{h}h {fold}: 학습{len(train):,}행({hist_days}일이력) / 동일시험{len(test):,}행")
    return fold_rows, pred_rows


def main():
    dpc.require_v5_outputs()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(cfg["random_seed"])
    capacity_kw = float(cfg["site"]["capacity_kw"])
    capacity_profile = cfg["site"].get("capacity_profile", "legacy")

    sat, conflicts = v3.load_satellite_v3()
    print(f"위성 통합 원본: {len(sat):,}행, 시각중복 값불일치 {conflicts}건")
    print(f"사전등록 기준: MAE·RMSE 둘다 ≥1% 개선 / 폴드 RMSE 악화 <5% / "
          f"{len(WINDOWS)}폴드 전부 평가 / F계열은 최선 원시값후보도 능가\n")

    all_fold, all_pred = [], []
    for h in (1, 2, 3, 4):
        rows, preds = run_horizon(h, sat, seed, capacity_kw)
        all_fold += rows
        all_pred += preds

    fold_df = pd.DataFrame(all_fold)
    fold_df["용량프로필"] = capacity_profile
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
        a = g[g["후보"] == "A_위성없음"]
        if a.empty:
            continue
        a = a.iloc[0]
        idx = g.index
        summary_df.loc[idx, "A대비_MAE개선_pct"] = (
            (a["폴드동일가중_MAE_kW"] - g["폴드동일가중_MAE_kW"]) / a["폴드동일가중_MAE_kW"] * 100)
        summary_df.loc[idx, "A대비_RMSE개선_pct"] = (
            (a["폴드동일가중_RMSE_kW"] - g["폴드동일가중_RMSE_kW"]) / a["폴드동일가중_RMSE_kW"] * 100)
        # 최선 원시값 후보(폴드동일가중 RMSE 최소)
        raws = g[g["후보"].isin(RAW_SAT_VARIANTS)]
        best_raw_rmse = float(raws["폴드동일가중_RMSE_kW"].min()) if len(raws) else np.nan
        best_raw_name = raws.sort_values("폴드동일가중_RMSE_kW").iloc[0]["후보"] if len(raws) else ""
        for i in idx:
            variant = summary_df.loc[i, "후보"]
            piv = valid[(valid["수평_h"] == h) & (valid["후보"].isin(["A_위성없음", variant]))].pivot(
                index="폴드", columns="후보", values="RMSE_kW")
            if variant == "A_위성없음" or variant not in piv.columns or "A_위성없음" not in piv.columns:
                worst = 0.0
            else:
                d = piv.dropna()
                delta = (d[variant] - d["A_위성없음"]) / d["A_위성없음"] * 100
                worst = float(delta.max()) if len(delta) else 0.0
            summary_df.loc[i, "최대폴드_RMSE악화_pct"] = round(worst, 3)
            if variant in HELIOSAT_VARIANTS and not np.isnan(best_raw_rmse):
                summary_df.loc[i, "최선원시값후보"] = best_raw_name
                summary_df.loc[i, "원시최선대비_RMSE개선_pct"] = round(
                    (best_raw_rmse - summary_df.loc[i, "폴드동일가중_RMSE_kW"]) / best_raw_rmse * 100, 3)

    base_pass = (
        (summary_df["후보"] != "A_위성없음")
        & (summary_df["A대비_MAE개선_pct"] >= 1.0)
        & (summary_df["A대비_RMSE개선_pct"] >= 1.0)
        & (summary_df["최대폴드_RMSE악화_pct"] < 5.0)
        & (summary_df["평가폴드수"] == len(WINDOWS))
    )
    heliosat_extra = summary_df["후보"].isin(HELIOSAT_VARIANTS)
    extra_ok = (~heliosat_extra) | (summary_df.get("원시최선대비_RMSE개선_pct", pd.Series(np.nan, index=summary_df.index)) > 0)
    summary_df["사전등록기준_통과"] = base_pass & extra_ok

    fold_df.to_csv(OUT / "폴드별_동일시험행_비교.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "수평별_최종판정.csv", index=False, encoding="utf-8-sig")
    if not pred_df.empty:
        pred_df.to_parquet(OUT / "행단위_예측.parquet", index=False)

    passed = summary_df[summary_df["사전등록기준_통과"]]
    audit = {
        "status": "최종검증_v4_연속블록_expanding_rolling_origin",
        "설계변경": ["연속44일 블록 내부 4폴드로 A~F 동일평가", "F_lean 추가"],
        "사전등록기준": ["MAE·RMSE 둘다 >=1% 개선", "폴드 RMSE 악화 <5%",
                    f"{len(WINDOWS)}폴드 전부 평가", "F계열은 최선 원시값후보도 능가"],
        "capacity_kw": capacity_kw, "capacity_profile": capacity_profile,
        "폴드정의": [{"폴드": w[0], "학습": f"{w[1]}~{w[2]}", "시험": f"{w[2]}~{w[3]}"} for w in WINDOWS],
        "통과조합수": int(len(passed)),
        "통과조합": passed[["수평_h", "후보"]].to_dict("records"),
    }
    (OUT / "감사.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    cols = ["수평_h", "후보", "평가폴드수", "폴드동일가중_MAE_kW", "폴드동일가중_RMSE_kW",
            "A대비_MAE개선_pct", "A대비_RMSE개선_pct", "최대폴드_RMSE악화_pct",
            "원시최선대비_RMSE개선_pct", "사전등록기준_통과"]
    cols = [c for c in cols if c in summary_df.columns]
    print("\n=== 수평별 최종판정(연속블록 4폴드, A~F 동일평가) ===")
    print(summary_df[cols].to_string(index=False))
    print(f"\n사전등록기준 통과 조합: {len(passed)}개")
    if len(passed):
        print(passed[["수평_h", "후보"]].to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
