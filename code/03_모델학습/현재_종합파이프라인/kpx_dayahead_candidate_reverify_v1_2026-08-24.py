# -*- coding: utf-8 -*-
"""KPX day-ahead 재현 모델(협약계획서 성과지표 7.87%를 실제로 만드는 모델)에
개선후보 3종 재검증 — 처음부터 실제 배포판 기준(LOGO 원칙 준수).

## 왜 이 모델인가
협약 연구개발계획서 "9.재생 발전 예측 정확도"의 4차년도 목표(7%,
현재 7.87%)는 **`kpx_mode_backtest_v5_공식B_v1_2026-08-24.py`의 D-1
day-ahead 모델에서만 나온다** — 초단기(+1~4h) 티어는 이 수치에 전혀
관여하지 않는다(AGENTS.md "초단기+3h만 하면 의미 없는가" 절 참고).
지금까지 이 day-ahead 모델엔 개선후보를 한 번도 시도한 적이 없었다.

## 오늘의 재발방지 규칙을 처음부터 지킨다
⑥ round2·일간 특성군에서 "바닥 기준선과 비교" 버그를 세 번 겪었다
(AGENTS.md "round2류 바닥기준선 버그" 재발방지 규칙). 이 스크립트는
**처음부터 바닥이 아니라 실제 배포판**(`kpx_mode_backtest_v5_공식B_
v1_2026-08-24.py`가 만드는 raw+구조선택 모델)을 기준으로 후보를
얹는다 — round2 방식(별도 파이프라인에서 처음부터 다시 판정)을
반복하지 않는다.

## 후보 3종과 학술근거(Google Scholar 사전확인 — 실행 전에 확인함)
1. **NWP 분위사상(quantile mapping)**: 강한 근거. "Adapting quantile
   mapping to bias correct solar radiation data"(Solar Energy, 2024,
   ScienceDirect) 등 다수 논문이 NWP 일사량 예보의 분위사상 보정을
   표준 기법으로 다룬다.
2. **날씨군집화**: 강한 근거(원문 확인). Amarasinghe et al.(2020),
   *AIMS Energy* 8(2), 252-271 — 운량 기반 클러스터링으로 RMSE
   7.78~10.49% 개선 보고(AGENTS.md 08-24 절 참고).
3. **OOF 잔차보정(구름전이·강수상태 조건부)**: 개념수준 근거(원문
   완독은 403으로 실패, 다수 논문이 같은 개념을 보고 — AGENTS.md
   08-24 절 참고).

## 채점·판정
KPX 공식 NMAE(이용률 10%이상 시각만, 용량정규화, 219kW) — 이 모델이
이미 만드는 `요약_정산시뮬레이션.csv`와 동일한 정의를 재사용한다.
판정규칙(사전동결): 개선율≥1% AND 어느 폴드도 5%이상 악화 없음.

## 산출물 (`outputs/KPX_dayahead_개선후보재검증_2026-08-24/`)
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
OUT = ROOT / "outputs" / "KPX_dayahead_개선후보재검증_2026-08-24"
CAPACITY_KW = 219.0
UTIL_THRESHOLD = 0.10


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kpx = _load("dr_kpx", "kpx_mode_backtest_v5_공식B_v1_2026-08-24.py")
improvement = _load("dr_round2", "model_improvement_round2_v1_2026-08-21.py")
harness, N_INVERTERS, WINDOWS = kpx.harness, kpx.N_INVERTERS, kpx.WINDOWS


def nmae(y, p, capacity=CAPACITY_KW, threshold=UTIL_THRESHOLD):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y >= threshold * capacity
    if mask.sum() == 0:
        return float("nan"), 0
    e = np.abs(y[mask] - p[mask]) / capacity * 100
    return float(e.mean()), int(mask.sum())


def build_frame_with_obs(capacity_kw: float):
    src = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                      parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(kpx.V5_DIR / "집계_1시간_자료_v5.parquet")
    common = src.index.intersection(v5_1h.index)
    src = src.loc[common].copy()
    src["plant_output_kw"] = v5_1h.loc[common, "발전출력_kW"]
    src["inverters_available"] = v5_1h.loc[common, "가용인버터수"]

    candidate_cols = harness.FEATURE_SETS["전체후보"]
    frame = kpx.build_kpx_frame(src, candidate_cols)
    # ★분위사상용 hidden 관측컬럼★: 대상시각의 실측(사후관측, 채점에만 씀
    # — round2와 동일 관례, 학습 특성으로는 안 들어감)
    for obs in set(improvement.NWP_OBS_PAIRS.values()):
        if obs in src.columns:
            frame[f"__target_obs__{obs}"] = src[obs].reindex(frame.index)
    return frame, candidate_cols


def run(seed: int) -> tuple[pd.DataFrame, list[dict]]:
    frame, candidate_cols = build_frame_with_obs(CAPACITY_KW)
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간") and not c.startswith("__target_obs__")]
    daylight = frame[frame["목표_낮시간"] > 0]

    rows, fold_summaries = [], []
    for fold, s, e in WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        all_target = daylight[(daylight.index >= start) & (daylight.index < end)]
        full_mask_all = daylight["_목표_가용인버터수"] >= N_INVERTERS
        train_all = daylight[(daylight.index < start) & full_mask_all]
        test_all = all_target[full_mask_all.loc[all_target.index]] if len(all_target) else all_target
        if len(train_all) < 200 or len(test_all) < 30:
            print(f"[{fold}] 표본부족(train={len(train_all)}, test={len(test_all)}) — 건너뜀(예: 이상구간 폴드는 정책B상 커버리지 0)")
            continue
        chosen = harness.select_features_in_fold(
            train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=required)
        if len(train) < 200 or len(test) < 30:
            print(f"[{fold}] 결측제거 후 표본부족 — 건너뜀")
            continue

        # ── 실제 배포판(raw+구조선택) ──
        params, sname = improvement.choose_structure_kfold(
            "단기", train, features, "목표_발전출력_kW", "raw", CAPACITY_KW, seed)
        base_model = improvement.make_model("단기", seed, params)
        base_model.fit(train[features], train["목표_발전출력_kW"])
        p_base = np.clip(base_model.predict(test[features]), 0, CAPACITY_KW)
        y = test["목표_발전출력_kW"].to_numpy()

        preds = {"배포판": p_base}

        # ── 후보1: NWP 분위사상(같은 구조 재사용, 특성만 QM 변환) ──
        tr_qm, te_qm, qm_log = improvement.apply_quantile_mapping(train, test, features)
        m_qm = improvement.make_model("단기", seed, params)
        m_qm.fit(tr_qm[features], tr_qm["목표_발전출력_kW"])
        preds["NWP분위사상"] = np.clip(m_qm.predict(te_qm[features]), 0, CAPACITY_KW)

        # ── 후보2: OOF 잔차보정(배포판 예측 위에 얹음) ──
        corr = improvement.fit_oof_residual(
            train, features, "목표_발전출력_kW", "raw", "단기", CAPACITY_KW, seed, params)
        preds["OOF잔차보정"] = improvement.apply_residual(p_base.copy(), test, corr, CAPACITY_KW)

        # ── 후보3: 날씨군집화(같은 구조 재사용, 특성에 군집 추가) ──
        tr_cl, te_cl, feat_cl, cluster_cols = improvement.add_weather_clusters(train, test, features, seed)
        m_cl = improvement.make_model("단기", seed, params)
        m_cl.fit(tr_cl[feat_cl], tr_cl["목표_발전출력_kW"])
        preds["날씨군집화"] = np.clip(m_cl.predict(te_cl[feat_cl]), 0, CAPACITY_KW)

        for cand_name, p_cand in preds.items():
            if cand_name == "배포판":
                continue
            for yy, pb, pc in zip(y, p_base, p_cand):
                rows.append({"후보": cand_name, "폴드": fold, "실제": yy, "배포판": pb, "후보예측": pc})
        nb, _ = nmae(y, p_base)
        fold_row = {"폴드": fold, "NMAE_배포판": round(nb, 3) if np.isfinite(nb) else None, "구조": sname,
                   "QM특성수": len(qm_log), "군집원천특성수": len(cluster_cols),
                   "학습행수": len(train), "시험행수": len(test)}
        for cand_name, p_cand in preds.items():
            if cand_name == "배포판":
                continue
            nc, cnt = nmae(y, p_cand)
            fold_row[f"NMAE_{cand_name}"] = round(nc, 3) if np.isfinite(nc) else None
            fold_row[f"정산표본_{cand_name}"] = cnt
        fold_summaries.append(fold_row)
        print(f"[{fold}] 구조={sname} QM특성{len(qm_log)}개 군집특성{len(cluster_cols)}개 "
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
    print("KPX day-ahead 모델(협약계획서 7.87% 산출 모델) 개선후보 3종 재검증\n"
          "기준: 실제 배포판(raw+구조선택), 채점: KPX 공식 NMAE\n")

    rows, fold_summaries = run(seed)
    if not len(rows):
        raise SystemExit("시험행 없음 — 폴드/커버리지 확인 필요")

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
