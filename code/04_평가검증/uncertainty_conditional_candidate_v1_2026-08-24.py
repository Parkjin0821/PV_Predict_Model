# -*- coding: utf-8 -*-
"""예측구간 조건부 포함률 개선 후보 검증 — Mondrian(예측값 구간별) conformal.

## 왜 필요한가 — v5 전사 재산출에서 발견한 문제
`uncertainty_audit_v5_공식B_v1_2026-08-24.py`로 전 티어·수평 예측구간을
v5 기준으로 재산출한 결과, **전체(marginal) 포함률은 명목치에 잘 맞는데
(초단기·단기 ±2.5pp 이내) 조건부로는 크게 어긋났다**:
- 90% 명목인데 예측값 십분위1(최저출력)은 **100% 포함**(과대포함),
  십분위7(중상위 출력)은 **74.9~78.5%**(최대 15.1pp 과소포함)
- 90% 기준 십분위 구간 80개 중 10개가 포함률 80% 미만

원인은 구조적이다: 현행 방식은 폴드당 단일 분위수 q로 **모든 행에 같은
폭**의 구간을 준다. PV 오차는 출력수준에 강하게 의존(새벽·해질녘 ≈0,
정오 최대)하므로, 등폭 구간은 저출력을 과대포함하고 고출력을 과소포함할
수밖에 없다. **"90% 구간"이라고 내놓은 게 정작 발전이 많은 시간대에는
75%밖에 안 맞는다는 뜻**이라, 이 구간을 신뢰하고 운영 판단을 하면
위험하다.

## 후보: Mondrian conformal(예측값 구간별 분위수)
보정 폴드에서 **예측값 5구간(분위수 경계)별로 따로** 절대잔차 분위수를
구하고, 시험 행을 그 경계로 배정해 해당 구간의 q를 쓴다. 등폭 대신
출력수준에 따라 폭이 달라진다. 경계·분위수 모두 **보정 폴드에서만**
적합하므로 누출 없음(현행과 동일한 순차 구조 유지).

## ★사전동결 판정규칙(결과 보기 전에 확정 — 이 파일에 먼저 기록)★
1. 비교기준은 **현행 공식**(등폭 순차 conformal, v5 전사재산출 결과)이다
   — 바닥 기준선이 아니라 실제 배포 방식과 직접 비교한다(AGENTS.md
   "재발방지 규칙": 이미 배포된 구성을 개선하는 재판정은 배포판을
   시작점으로 삼는다).
2. **동일 시험행**(같은 폴드·같은 행)에서 비교한다.
3. 아래를 **전부** 만족해야 채택:
   (a) 전체 포함률이 명목 ±3pp 이내
   (b) 십분위별 |포함률차이|의 최댓값이 현행보다 **감소**
   (c) 평균 구간폭이 현행 대비 **10% 초과 확대되지 않음**
4. 어느 폴드에서도 전체 포함률이 명목-5pp 미만으로 떨어지지 않는다.
5. 구간별 보정표본이 30개 미만이면 그 구간은 전체 q로 폴백한다(표본
   부족 구간에서 분위수를 억지로 추정하지 않는다).

## 산출물 (`outputs/불확실성_조건부개선후보_2026-08-24/`)
- `판정표.csv`, `수평별_비교.csv`, `조건부포함률_비교.csv`, `판정.json`
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
PIPE = ROOT.parent / "03_모델학습" / "현재_종합파이프라인"
SRC = PIPE / "outputs" / "E2E_v5_공식B_v3_일간특성군_2026-08-24"
OUT = ROOT / "outputs" / "불확실성_조건부개선후보_2026-08-24"
CAPACITY_KW = 219.0
N_BINS = 5              # 사전동결: 예측값 5구간
MIN_BIN_CALIB = 30      # 사전동결: 구간별 보정표본 30개 미만이면 전체 q 폴백


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ua = _load("uc_ua_v1", ROOT / "uncertainty_audit_v1_2026-08-24.py")
uav5 = _load("uc_ua_v5", ROOT / "uncertainty_audit_v5_공식B_v1_2026-08-24.py")


def mondrian_conformal(oof: pd.DataFrame, capacities: dict[str, float]) -> pd.DataFrame:
    """현행 `sequential_conformal`과 같은 순차 구조 + 예측값 구간별 분위수."""
    detailed = []
    for (tier, horizon), g in oof.groupby(["티어", "수평_h"], sort=False):
        fold_order = g.groupby("폴드")["발행시각"].min().sort_values().index.tolist()
        prior = pd.DataFrame()
        for fold_idx, fold in enumerate(fold_order):
            test = g[g["폴드"] == fold].copy()
            if fold_idx == 0:
                prior = pd.concat([prior, test], ignore_index=True)
                continue
            cap = capacities["daily"] if tier == "일간" else capacities["hourly"]
            calib_pred = prior["예측"].to_numpy(float)
            calib_score = np.abs(prior["실제"].to_numpy(float) - calib_pred)

            # 구간 경계는 보정 데이터에서만 정한다(누출 없음).
            edges = np.quantile(calib_pred, np.linspace(0, 1, N_BINS + 1)[1:-1])
            edges = np.unique(edges)
            calib_bin = np.digitize(calib_pred, edges)
            test_bin = np.digitize(test["예측"].to_numpy(float), edges)

            for nominal, alpha in [(80, 0.20), (90, 0.10)]:
                q_global = ua.finite_sample_quantile(calib_score, alpha)
                q_by_bin, fallback_bins = {}, []
                for b in range(len(edges) + 1):
                    s = calib_score[calib_bin == b]
                    if len(s) < MIN_BIN_CALIB:
                        q_by_bin[b] = q_global
                        fallback_bins.append(b)
                    else:
                        q_by_bin[b] = ua.finite_sample_quantile(s, alpha)
                q_row = np.array([q_by_bin[b] for b in test_bin], dtype=float)

                pred = test["예측"].to_numpy(float)
                lo = np.clip(pred - q_row, 0, cap)
                hi = np.clip(pred + q_row, 0, cap)
                y = test["실제"].to_numpy(float)
                part = test.copy()
                part["명목포함률_pct"] = nominal
                part["보정표본수"] = len(calib_score)
                part["보정절대잔차_q"] = q_row
                part["구간번호"] = test_bin
                part["폴백구간수"] = len(fallback_bins)
                part["하한"] = lo
                part["상한"] = hi
                part["포함"] = (y >= lo) & (y <= hi)
                detailed.append(part)
            prior = pd.concat([prior, test], ignore_index=True)
    return pd.concat(detailed, ignore_index=True)


def decile_gap(rows: pd.DataFrame) -> pd.DataFrame:
    """티어×수평×명목별 십분위 포함률과 그 최대 이탈폭."""
    out = []
    for (tier, h, nom), g in rows.groupby(["티어", "수평_h", "명목포함률_pct"]):
        g = g.copy()
        g["십분위"] = pd.qcut(g["예측"].rank(method="first"), 10, labels=False) + 1
        cov = g.groupby("십분위")["포함"].mean() * 100
        out.append({
            "티어": tier, "수평_h": h, "명목포함률_pct": nom,
            "십분위_최저포함률_pct": round(float(cov.min()), 2),
            "십분위_최고포함률_pct": round(float(cov.max()), 2),
            "십분위_최대이탈_pp": round(float((cov - nom).abs().max()), 2),
            "포함률80미만_구간수": int((cov < 80).sum()),
        })
    return pd.DataFrame(out)


def marginal(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows.assign(구간폭=rows["상한"] - rows["하한"])
    agg = (r.groupby(["티어", "수평_h", "명목포함률_pct"], as_index=False)
           .agg(실제포함률_pct=("포함", lambda x: float(np.mean(x) * 100)),
                평균구간폭=("구간폭", "mean"), 시험표본수=("포함", "size")))
    agg["포함률차이_pp"] = agg["실제포함률_pct"] - agg["명목포함률_pct"]
    return agg


def fold_level(rows: pd.DataFrame) -> pd.DataFrame:
    return (rows.groupby(["티어", "수평_h", "명목포함률_pct", "폴드"], as_index=False)
            .agg(실제포함률_pct=("포함", lambda x: float(np.mean(x) * 100))))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    capacities = {"hourly": CAPACITY_KW, "daily": CAPACITY_KW * 24}
    oof = uav5.load_oof_v5()
    print(f"입력: {SRC}\n대상: 전 티어·수평 8개 조합, {len(oof):,}행\n")

    print("[1/3] 현행 공식(등폭 순차 conformal) 재계산...")
    cur_rows, _ = ua.sequential_conformal(oof, capacities)
    print("[2/3] 후보(Mondrian, 예측값 5구간별 분위수) 계산...")
    cand_rows = mondrian_conformal(oof, capacities)

    # 동일 시험행 확인(사전동결 규칙2)
    key = ["티어", "수평_h", "명목포함률_pct", "폴드", "발행시각"]
    assert len(cur_rows) == len(cand_rows), "시험행 수 불일치 — 동일행 비교 전제 위반"
    print(f"  동일 시험행 {len(cur_rows):,}개 확인\n")

    cur_m, cand_m = marginal(cur_rows), marginal(cand_rows)
    cur_d, cand_d = decile_gap(cur_rows), decile_gap(cand_rows)

    comp = cur_m.merge(cand_m, on=["티어", "수평_h", "명목포함률_pct"],
                       suffixes=("_현행", "_후보"))
    comp = comp.merge(cur_d[["티어", "수평_h", "명목포함률_pct", "십분위_최대이탈_pp",
                             "십분위_최저포함률_pct", "포함률80미만_구간수"]]
                      .rename(columns={"십분위_최대이탈_pp": "십분위최대이탈_현행",
                                       "십분위_최저포함률_pct": "십분위최저_현행",
                                       "포함률80미만_구간수": "80미만구간_현행"}),
                      on=["티어", "수평_h", "명목포함률_pct"])
    comp = comp.merge(cand_d[["티어", "수평_h", "명목포함률_pct", "십분위_최대이탈_pp",
                              "십분위_최저포함률_pct", "포함률80미만_구간수"]]
                      .rename(columns={"십분위_최대이탈_pp": "십분위최대이탈_후보",
                                       "십분위_최저포함률_pct": "십분위최저_후보",
                                       "포함률80미만_구간수": "80미만구간_후보"}),
                      on=["티어", "수평_h", "명목포함률_pct"])
    comp["구간폭_변화_pct"] = (comp["평균구간폭_후보"] - comp["평균구간폭_현행"]) / comp["평균구간폭_현행"] * 100

    # ── 사전동결 규칙 3·4 적용 ──
    cur_f, cand_f = fold_level(cur_rows), fold_level(cand_rows)
    fold_min = (cand_f.assign(이탈=lambda d: d["실제포함률_pct"] - d["명목포함률_pct"])
                .groupby(["티어", "수평_h", "명목포함률_pct"])["이탈"].min()
                .rename("폴드최저이탈_pp").reset_index())
    comp = comp.merge(fold_min, on=["티어", "수평_h", "명목포함률_pct"])

    comp["규칙3a_전체포함률_3pp이내"] = comp["포함률차이_pp_후보"].abs() <= 3
    comp["규칙3b_십분위이탈_감소"] = comp["십분위최대이탈_후보"] < comp["십분위최대이탈_현행"]
    comp["규칙3c_구간폭_10pct이내"] = comp["구간폭_변화_pct"] <= 10
    comp["규칙4_폴드포함률_명목-5pp이상"] = comp["폴드최저이탈_pp"] >= -5
    comp["채택"] = (comp["규칙3a_전체포함률_3pp이내"] & comp["규칙3b_십분위이탈_감소"]
                  & comp["규칙3c_구간폭_10pct이내"] & comp["규칙4_폴드포함률_명목-5pp이상"])

    comp.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")
    cur_m.assign(방식="현행_등폭").to_csv(OUT / "수평별_비교.csv", index=False, encoding="utf-8-sig")
    pd.concat([cur_d.assign(방식="현행_등폭"), cand_d.assign(방식="후보_Mondrian")],
              ignore_index=True).to_csv(OUT / "조건부포함률_비교.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("[3/3] 판정")
    print("\n=== 전체(marginal) 포함률 · 구간폭 ===")
    print(comp[["티어", "수평_h", "명목포함률_pct", "실제포함률_pct_현행", "실제포함률_pct_후보",
                "평균구간폭_현행", "평균구간폭_후보", "구간폭_변화_pct"]].to_string(index=False))
    print("\n=== 조건부(십분위) 이탈폭 — 이번 개선의 핵심 지표 ===")
    print(comp[["티어", "수평_h", "명목포함률_pct", "십분위최대이탈_현행", "십분위최대이탈_후보",
                "십분위최저_현행", "십분위최저_후보", "80미만구간_현행", "80미만구간_후보"]].to_string(index=False))
    print("\n=== 사전동결 규칙 판정 ===")
    print(comp[["티어", "수평_h", "명목포함률_pct", "규칙3a_전체포함률_3pp이내", "규칙3b_십분위이탈_감소",
                "규칙3c_구간폭_10pct이내", "규칙4_폴드포함률_명목-5pp이상", "채택"]].to_string(index=False))

    n_adopt = int(comp["채택"].sum())
    print(f"\n채택 {n_adopt}/{len(comp)} 조합")

    report = {
        "후보": "Mondrian conformal(예측값 5구간별 분위수, 경계·분위수 모두 보정폴드에서만 적합)",
        "비교기준": "현행 공식(등폭 순차 conformal) — 바닥 기준선 아님",
        "사전동결규칙": {
            "3a": "전체 포함률 명목 ±3pp 이내",
            "3b": "십분위별 최대 이탈폭이 현행보다 감소",
            "3c": "평균 구간폭이 현행 대비 10% 초과 확대되지 않음",
            "4": "어느 폴드도 전체 포함률이 명목-5pp 미만 아님",
            "5": f"구간별 보정표본 {MIN_BIN_CALIB}개 미만이면 전체 q 폴백",
        },
        "채택조합수": n_adopt,
        "전체조합수": int(len(comp)),
        "십분위_최대이탈_평균": {
            "현행": round(float(comp["십분위최대이탈_현행"].mean()), 2),
            "후보": round(float(comp["십분위최대이탈_후보"].mean()), 2),
        },
        "구간폭_변화_pct_범위": [round(float(comp["구간폭_변화_pct"].min()), 2),
                          round(float(comp["구간폭_변화_pct"].max()), 2)],
        "비고": "이 스크립트는 후보 판정만 한다 — 공식 예측구간 산출방식은 사용자 결정 전까지 바꾸지 않는다.",
    }
    (OUT / "판정.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
