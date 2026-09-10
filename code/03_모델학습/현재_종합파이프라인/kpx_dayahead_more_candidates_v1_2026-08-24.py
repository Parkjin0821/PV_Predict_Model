# -*- coding: utf-8 -*-
"""KPX day-ahead 모델 추가 개선후보 2종 — 청천지수(kappa) 변환, 트리 스태킹.

## 근거(Google Scholar 사전확인, 실행 전 기록 — AGENTS.md 08-24 절)
1. **청천지수(clear-sky index/kappa) 변환**: 강한 근거. "Physics-informed
   day-ahead PV power forecasting with seasonal trend mitigation and
   clear-sky template integration"(ScienceDirect, 2025)와 다수 문헌이
   청천지수 정규화로 결정론적 태양 성분을 제거해 day-ahead 예측을 쉽게
   만든다고 보고한다. **우리 프로젝트 자체도 초단기 +3h·+4h에서 이미
   이 방식을 공식 채택 중**이라 이중으로 근거 있음(외부문헌 + 내부
   기 검증사례).
2. **LightGBM-XGBoost 스태킹 앙상블**: 강한 근거, 다수 논문(2025-2026).
   "A stacked Gradient Boosting-XGBoost ensemble with ridge meta-learner
   for accurate short-term solar PV power forecasting"(Scientific
   Reports 2026) 등. 배만수 부장님 제공 자료(에너지 자립 실현
   발표자료)에서도 LightGBM-XGBoost 스태킹이 최고 성능이었다(AGENTS.md
   "딥러닝 정식 도입" 절 참고) — 국내 유사 사례로 이중 근거.

## 비교기준
`kpx_dayahead_candidate_reverify_v1_2026-08-24.py`와 동일 — 실제
배포판(raw+구조선택, XGBoost)을 기준으로 삼는다(바닥 기준선 아님).
스태킹 가중치는 학습폴드 내부 80/20 홀드아웃에서만 적합(누출 없음).

## 산출물 (`outputs/KPX_dayahead_추가후보_2026-08-24/`)
`판정표.csv`, `폴드별.csv`
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
OUT = ROOT / "outputs" / "KPX_dayahead_추가후보_2026-08-24"
CAPACITY_KW = 219.0
UTIL_THRESHOLD = 0.10


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dr = _load("mc_dr", "kpx_dayahead_candidate_reverify_v1_2026-08-24.py")
kpx, improvement, harness, N_INVERTERS, WINDOWS = dr.kpx, dr.improvement, dr.harness, dr.N_INVERTERS, dr.WINDOWS
clearsky = _load("mc_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
model_common = _load("mc_common", "model_common.py")


def nmae(y, p, capacity=CAPACITY_KW, threshold=UTIL_THRESHOLD):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y >= threshold * capacity
    if mask.sum() == 0:
        return float("nan"), 0
    e = np.abs(y[mask] - p[mask]) / capacity * 100
    return float(e.mean()), int(mask.sum())


def run(seed: int):
    frame, candidate_cols = dr.build_frame_with_obs(CAPACITY_KW)
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간") and not c.startswith("__target_obs__")]
    frame["_청천_kW"] = np.clip(CAPACITY_KW * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
    daylight = frame[frame["목표_낮시간"] > 0]

    rows, fold_summaries = [], []
    for fold, s, e in WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        all_target = daylight[(daylight.index >= start) & (daylight.index < end)]
        full_mask_all = daylight["_목표_가용인버터수"] >= N_INVERTERS
        train_all = daylight[(daylight.index < start) & full_mask_all]
        test_all = all_target[full_mask_all.loc[all_target.index]] if len(all_target) else all_target
        if len(train_all) < 200 or len(test_all) < 30:
            print(f"[{fold}] 표본부족 — 건너뜀")
            continue
        chosen = harness.select_features_in_fold(
            train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW", "_청천_kW", "_카파"]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=[c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW", "_청천_kW"])
        if len(train) < 200 or len(test) < 30:
            print(f"[{fold}] 결측제거 후 표본부족 — 건너뜀")
            continue

        # ── 실제 배포판: raw + 구조선택(XGBoost) ──
        params, sname = improvement.choose_structure_kfold(
            "단기", train, features, "목표_발전출력_kW", "raw", CAPACITY_KW, seed)
        base_model = improvement.make_model("단기", seed, params)
        base_model.fit(train[features], train["목표_발전출력_kW"])
        p_base = np.clip(base_model.predict(test[features]), 0, CAPACITY_KW)
        y = test["목표_발전출력_kW"].to_numpy()

        preds = {}

        # ── 후보A: 청천지수(kappa) 변환 + 구조선택 ──
        params_k, sname_k = improvement.choose_structure_kfold(
            "단기", train, features, "_카파", "kappa", CAPACITY_KW, seed)
        model_k = improvement.make_model("단기", seed, params_k)
        model_k.fit(train[features], train["_카파"])
        raw_k = model_k.predict(test[features])
        preds["청천지수변환"] = np.clip(raw_k * test["_청천_kW"].to_numpy(), 0, CAPACITY_KW)

        # ── 후보B: LightGBM-XGBoost 스태킹(가중치는 폴드 내부 80/20에서만 적합) ──
        cut = int(len(train) * 0.8)
        inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]
        xgb_inner = improvement.make_model("단기", seed, params)
        xgb_inner.fit(inner_tr[features], inner_tr["목표_발전출력_kW"])
        lgb_inner = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4)
        lgb_inner.fit(inner_tr[features], inner_tr["목표_발전출력_kW"])
        va_pred = pd.DataFrame({
            "xgb": np.clip(xgb_inner.predict(inner_va[features]), 0, CAPACITY_KW),
            "lgb": np.clip(lgb_inner.predict(inner_va[features]), 0, CAPACITY_KW),
        })
        w_result = model_common.optimize_nonnegative_weights(inner_va["목표_발전출력_kW"], va_pred)
        weights = np.asarray(w_result["가중치"], dtype=float)

        lgb_full = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4)
        lgb_full.fit(train[features], train["목표_발전출력_kW"])
        p_lgb_test = np.clip(lgb_full.predict(test[features]), 0, CAPACITY_KW)
        preds["스태킹(XGB+LGBM)"] = np.clip(
            np.column_stack([p_base, p_lgb_test]) @ weights, 0, CAPACITY_KW)

        for cand_name, p_cand in preds.items():
            for yy, pb, pc in zip(y, p_base, p_cand):
                rows.append({"후보": cand_name, "폴드": fold, "실제": yy, "배포판": pb, "후보예측": pc})
        nb, _ = nmae(y, p_base)
        fold_row = {"폴드": fold, "NMAE_배포판": round(nb, 3), "구조_배포판": sname,
                   "구조_kappa": sname_k, "스태킹가중치_XGB_LGBM": weights.tolist(),
                   "학습행수": len(train), "시험행수": len(test)}
        for cand_name, p_cand in preds.items():
            nc, cnt = nmae(y, p_cand)
            fold_row[f"NMAE_{cand_name}"] = round(nc, 3) if np.isfinite(nc) else None
        fold_summaries.append(fold_row)
        print(f"[{fold}] 배포판구조={sname} kappa구조={sname_k} 가중치(XGB,LGBM)={weights.round(3).tolist()} "
              f"학습{len(train):,}/시험{len(test):,} NMAE_배포판={nb:.3f}%")

    return pd.DataFrame(rows), fold_summaries


def verdict_from_rows(rows: pd.DataFrame) -> dict:
    n_base, _ = nmae(rows["실제"], rows["배포판"])
    n_cand, _ = nmae(rows["실제"], rows["후보예측"])
    gain = (n_base - n_cand) / n_base * 100 if n_base else float("nan")
    worst, worst_fold = -np.inf, ""
    for fold, g in rows.groupby("폴드"):
        fb, _ = nmae(g["실제"], g["배포판"])
        fc, cnt = nmae(g["실제"], g["후보예측"])
        if cnt < 10 or not np.isfinite(fb) or fb == 0:
            continue
        w = (fc - fb) / fb * 100
        if w > worst:
            worst, worst_fold = w, fold
    if worst == -np.inf:
        worst, worst_fold = 0.0, "(표본부족)"
    adopt = bool(np.isfinite(gain) and gain >= 1 and worst < 5)
    reason = ("정산대상표본부족" if not np.isfinite(gain) else
             "채택" if adopt else
             ("개선율1%미만" if gain < 1 else f"계절악화({worst_fold} {worst:.1f}%)"))
    return {"NMAE_배포판": round(n_base, 3), "NMAE_후보": round(n_cand, 3),
           "개선율_pct": round(gain, 2) if np.isfinite(gain) else None,
           "최대계절악화_pct": round(worst, 2), "채택": adopt, "판정사유": reason}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    print("KPX day-ahead 모델 추가 개선후보(청천지수변환, 스태킹) 검증\n"
          "기준: 실제 배포판(raw+구조선택, XGBoost), 채점: KPX 공식 NMAE\n")

    rows, fold_summaries = run(seed)
    if not len(rows):
        raise SystemExit("시험행 없음")

    verdicts = []
    for cand_name, g in rows.groupby("후보"):
        v = verdict_from_rows(g)
        v["후보"] = cand_name
        verdicts.append(v)
    verdict_df = pd.DataFrame(verdicts)[["후보", "NMAE_배포판", "NMAE_후보", "개선율_pct",
                                        "최대계절악화_pct", "채택", "판정사유"]]
    verdict_df.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_summaries).to_csv(OUT / "폴드별.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 최종 판정(실제 배포판 대비, KPX NMAE) ===")
    print(verdict_df.to_string(index=False))
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
