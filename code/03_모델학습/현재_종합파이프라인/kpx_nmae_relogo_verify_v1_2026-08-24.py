# -*- coding: utf-8 -*-
"""KPX NMAE 재채점에서 뒤집힌 3건을 실제 배포판 기준(LOGO)으로 재검증.

## 배경
`kpx_nmae_rescore_candidates_v1_2026-08-24.py`가 KPX NMAE로 재채점하니
round2 4건이 뒤집혔는데, 그중 3건(초단기+1h·+2h의 OOF잔차보정, 단기+1h의
날씨군집화)은 round2의 "바닥 기준선"과 비교한 것이라 ⑥ round2(초단기
+2h)·일간 특성군 순방향 판정과 같은 함정에 걸릴 위험이 있었다
(AGENTS.md "round2류 바닥기준선 버그" 재발방지 규칙).

이 스크립트는 **바닥이 아니라 실제 배포판(3차 v3 공식구성)**을
기준으로 삼아 같은 후보를 다시 검증한다 — LOGO 재발방지 규칙 그대로
적용, 채점은 KPX NMAE.

## 실제 배포 기준(3차 v3 공식구성)
- 초단기 +1h·+2h: raw + 폴드내부 구조선택(choose_structure_kfold)
- 단기 전체: 기본 + 폴드내부 구조선택(choose_structure_kfold)

## 후보(문헌 근거 확인됨 — AGENTS.md 08-24 "Google Scholar 사전게이트" 절)
- 초단기 +1h·+2h: 배포판(구조선택) 위에 **OOF 잔차보정**을 추가
- 단기 +1h: 배포판(구조선택) 위에 **날씨군집화**를 추가

## 판정규칙(사전동결)
KPX NMAE 개선율 ≥1% AND 어느 정상 폴드도 5%이상 악화 없음.

## 산출물 (`outputs/KPX_NMAE_LOGO재검증_2026-08-24/`)
`판정표.csv`, `폴드별.csv`
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
OUT = ROOT / "outputs" / "KPX_NMAE_LOGO재검증_2026-08-24"
CAPACITY_KW = 219.0
UTIL_THRESHOLD = 0.10


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("relogo_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc, harness, ultra, clearsky, improvement = e2e.dpc, e2e.harness, e2e.ultra, e2e.clearsky, e2e.improvement
OFFICIAL_WINDOWS = e2e.OFFICIAL_WINDOWS
N_INVERTERS = e2e.N_INVERTERS
DIF = e2e.DIF


def nmae(y, p, capacity=CAPACITY_KW, threshold=UTIL_THRESHOLD):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y >= threshold * capacity
    if mask.sum() == 0:
        return float("nan"), 0
    e = np.abs(y[mask] - p[mask]) / capacity * 100
    return float(e.mean()), int(mask.sum())


def verify_ultra_oof(horizon: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """초단기 +1h·+2h: 배포판(raw+구조선택) vs 배포판+OOF잔차보정."""
    frame = e2e.add_difswrf_flag(dpc.load_ultra_frame(horizon))
    candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    frame["_청천_kW"] = np.clip(CAPACITY_KW * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
    daylight = frame[frame["목표_낮시간"] > 0]

    rows, folds = [], []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        full_mask = daylight["_목표_가용인버터수"] >= N_INVERTERS
        train_pool = daylight[daylight.index < start]
        train = train_pool[full_mask.loc[train_pool.index]]
        test = daylight[(daylight.index >= start) & (daylight.index < end) & full_mask]
        if len(train) < 300 or len(test) < 30:
            continue
        chosen = harness.select_features_in_fold(
            train.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        native_ok = harness.NATIVE_MISSING_OK | {DIF}
        required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW", "_청천_kW", "_카파"]
        train = train.dropna(subset=required)
        test = test.dropna(subset=[c for c in features if c not in native_ok] + ["목표_발전출력_kW", "_청천_kW"])
        if len(train) < 300 or len(test) < 30:
            continue

        # 배포판: raw + 구조선택
        params, sname = improvement.choose_structure_kfold(
            "초단기", train, features, "목표_발전출력_kW", "raw", CAPACITY_KW, seed)
        base_model = improvement.make_model("초단기", seed, params)
        base_model.fit(train[features], train["목표_발전출력_kW"])
        p_base = np.clip(base_model.predict(test[features]), 0, CAPACITY_KW)

        # 후보: 배포판 + OOF 잔차보정(그 위에 얹음)
        corr = improvement.fit_oof_residual(
            train, features, "목표_발전출력_kW", "raw", "초단기", CAPACITY_KW, seed, params)
        p_cand = improvement.apply_residual(p_base.copy(), test, corr, CAPACITY_KW)

        y = test["목표_발전출력_kW"].to_numpy()
        for yy, pb, pc in zip(y, p_base, p_cand):
            rows.append({"폴드": fold, "실제": yy, "배포판": pb, "후보": pc})
        nb, _ = nmae(y, p_base); nc, cnt = nmae(y, p_cand)
        folds.append({"폴드": fold, "NMAE_배포판": round(nb, 3) if np.isfinite(nb) else None,
                      "NMAE_후보": round(nc, 3) if np.isfinite(nc) else None,
                      "정산표본수": cnt})
        print(f"  [초단기+{horizon}h {fold}] 구조={sname} 학습{len(train):,}/시험{len(test):,}")
    return pd.DataFrame(rows), folds


def verify_short_cluster(horizon: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """단기 +1h: 배포판(기본+구조선택) vs 배포판+날씨군집화."""
    frame = e2e.add_difswrf_flag(dpc.load_short_frame(horizon))
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]

    rows, folds = [], []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train = daylight[(daylight.index < start) & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
        test = daylight[(daylight.index >= start) & (daylight.index < end)
                        & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
        if len(train) < 200 or len(test) < 30:
            continue
        chosen = harness.select_features_in_fold(
            train.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
        train = train.dropna(subset=required)
        test = test.dropna(subset=required)
        if len(train) < 200 or len(test) < 30:
            continue

        # 배포판: 기본 + 구조선택
        params, sname = improvement.choose_structure_kfold(
            "단기", train, features, "목표_발전출력_kW", "raw", CAPACITY_KW, seed)
        base_model = improvement.make_model("단기", seed, params)
        base_model.fit(train[features], train["목표_발전출력_kW"])
        p_base = np.clip(base_model.predict(test[features]), 0, CAPACITY_KW)

        # 후보: 배포판 구조 위에 날씨군집화 특성 추가 → 그 구조로 재학습
        tr, te, feat2, cluster_cols = improvement.add_weather_clusters(train, test, features, seed)
        cand_model = improvement.make_model("단기", seed, params)
        cand_model.fit(tr[feat2], tr["목표_발전출력_kW"])
        p_cand = np.clip(cand_model.predict(te[feat2]), 0, CAPACITY_KW)

        y = test["목표_발전출력_kW"].to_numpy()
        for yy, pb, pc in zip(y, p_base, p_cand):
            rows.append({"폴드": fold, "실제": yy, "배포판": pb, "후보": pc})
        nb, _ = nmae(y, p_base); nc, cnt = nmae(y, p_cand)
        folds.append({"폴드": fold, "NMAE_배포판": round(nb, 3) if np.isfinite(nb) else None,
                      "NMAE_후보": round(nc, 3) if np.isfinite(nc) else None,
                      "정산표본수": cnt})
        print(f"  [단기+{horizon}h {fold}] 구조={sname} 군집특성{len(cluster_cols)}개 학습{len(train):,}/시험{len(test):,}")
    return pd.DataFrame(rows), folds


def verdict_from_rows(rows: pd.DataFrame) -> dict:
    n_base, _ = nmae(rows["실제"], rows["배포판"])
    n_cand, _ = nmae(rows["실제"], rows["후보"])
    gain = (n_base - n_cand) / n_base * 100 if n_base else float("nan")
    worst, worst_fold = -np.inf, ""
    for fold, g in rows.groupby("폴드"):
        fb, _ = nmae(g["실제"], g["배포판"])
        fc, cnt = nmae(g["실제"], g["후보"])
        if cnt < 10 or not np.isfinite(fb) or fb == 0:
            continue
        w = (fc - fb) / fb * 100
        if w > worst:
            worst, worst_fold = w, fold
    if worst == -np.inf:
        worst, worst_fold = 0.0, "(표본부족)"
    adopt = bool(np.isfinite(gain) and gain >= 1 and worst < 5)
    reason = "채택" if adopt else ("개선율1%미만" if not np.isfinite(gain) or gain < 1 else f"계절악화({worst_fold} {worst:.1f}%)")
    return {"NMAE_배포판": round(n_base, 3), "NMAE_후보": round(n_cand, 3),
           "개선율_pct": round(gain, 2) if np.isfinite(gain) else None,
           "최대계절악화_pct": round(worst, 2), "채택": adopt, "판정사유": reason}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    print("실제 배포판 기준(LOGO) 재검증 — KPX NMAE 채점\n")

    results = {}
    all_folds = []

    print("=== 초단기+1h: 배포판(raw+구조선택) vs +OOF잔차보정 ===")
    r1, f1 = verify_ultra_oof(1, seed)
    v1 = verdict_from_rows(r1)
    results["초단기+1h_OOF잔차보정"] = v1
    for fr in f1: fr["조합"] = "초단기+1h_OOF잔차보정"; all_folds.append(fr)
    print(f"  -> {v1}\n")

    print("=== 초단기+2h: 배포판(raw+구조선택) vs +OOF잔차보정 ===")
    r2, f2 = verify_ultra_oof(2, seed)
    v2 = verdict_from_rows(r2)
    results["초단기+2h_OOF잔차보정"] = v2
    for fr in f2: fr["조합"] = "초단기+2h_OOF잔차보정"; all_folds.append(fr)
    print(f"  -> {v2}\n")

    print("=== 단기+1h: 배포판(기본+구조선택) vs +날씨군집화 ===")
    r3, f3 = verify_short_cluster(1, seed)
    v3 = verdict_from_rows(r3)
    results["단기+1h_날씨군집화"] = v3
    for fr in f3: fr["조합"] = "단기+1h_날씨군집화"; all_folds.append(fr)
    print(f"  -> {v3}\n")

    verdict_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "조합"})
    verdict_df.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_folds).to_csv(OUT / "폴드별.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("=== 최종 판정(실제 배포판 대비, KPX NMAE) ===")
    print(verdict_df.to_string(index=False))
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
