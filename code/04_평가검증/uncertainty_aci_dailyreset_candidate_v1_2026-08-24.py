# -*- coding: utf-8 -*-
"""예측구간 개선후보2 — 문헌 근거(daily-reset ACI) 정식 해법 검증.

## 출처와 이 스크립트가 정직하게 밝히는 한계
Gibbs & Candès(2021) "Adaptive Conformal Inference Under Distribution
Shift"의 표준 ACI 업데이트식과, 이를 태양광에 특화한 "daily miscoverage
reset" 변형(MDPI Energies 19(6):1495, "Model-Agnostic, Probabilistic,
Hour-Ahead Solar PV Forecasting Using Adaptive Conformal Inference";
Scientific Reports 2026, 유사 변형)을 재현한다.

★정직하게 밝힘★: MDPI 논문 원문은 이번에도 HTTP 403으로 못 읽었다
(2026-08-21에 이미 겪은 것과 같은 문제 — AGENTS.md에 "인용논문 원문
확보 못하면 비교표에서 제외" 원칙이 있다). Scientific Reports는
읽었으나 WebFetch의 자동요약(소형 모델)을 거친 것이라 **γ(학습률)
정확한 수치·"하루" 경계의 정확한 정의는 논문에서 확인 못 했다**.
표준 ACI 업데이트식(요약에서도 확인된, Gibbs & Candès 원 논문과 동일한
형태)만 논문 근거로 확실히 쓰고, γ는 Gibbs & Candès 원논문이 실험에서
쓰는 통상 범위(0.01~0.05)에서 γ=0.02를 기본값으로 잡아 **명시적으로
우리가 고른 값**이라고 밝힌다. "하루" 경계는 대상시각(예측 대상)의
날짜로 정의한다(발행시각이 아니라 — 리셋의 목적이 "야간 0오차 누적이
주간 보정을 왜곡하는 것 방지"이므로 대상시각 기준이 논리적으로 맞다).

## 표준 ACI 업데이트식(Gibbs & Candès 2021, 식 그대로)
α_{t+1} = α_t + γ·(α_target − 1{y_t ∉ C_t(x_t)})
- 구간을 벗어나면(미포함) α_t가 커져서 다음 구간이 넓어짐
- 포함되면 α_t가 작아져서 다음 구간이 좁아짐
- 구간: C_t(x_t) = [pred_t − q_{1−α_t}, pred_t + q_{1−α_t}]
  (q는 이전 폴드 전체의 절대잔차 분포에서 뽑음 — 현재 폴드 안에서는
  칼리브레이션 집합을 늘리지 않는다, 누출 없음. 현재 공식과 동일한
  "이전 폴드까지만" 원칙 유지)
- ★신규(daily reset)★: 대상시각의 날짜가 바뀌면 α_t를 α_target으로
  리셋

## 비교기준·사전동결 규칙
`uncertainty_conditional_candidate_v1_2026-08-24.py`와 **동일한
규칙**(3a/3b/3c/4/5)을 그대로 재사용해 Mondrian 후보와 나란히 비교
가능하게 한다(비교기준=현행 등폭, 바닥 아님).

## 산출물 (`outputs/불확실성_ACI일일리셋후보_2026-08-24/`)
`판정표.csv`, `조건부포함률_비교.csv`, `감마민감도.csv`, `판정.json`
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
OUT = ROOT / "outputs" / "불확실성_ACI일일리셋후보_2026-08-24"
CAPACITY_KW = 219.0
ALPHA_TARGET = {80: 0.20, 90: 0.10}
GAMMA_DEFAULT = 0.02
ALPHA_CLIP = (0.01, 0.5)  # 사전동결: 분위수 계산이 안 깨지게 범위 제한


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ua = _load("adr_ua_v1", ROOT / "uncertainty_audit_v1_2026-08-24.py")
uav5 = _load("adr_ua_v5", ROOT / "uncertainty_audit_v5_공식B_v1_2026-08-24.py")
cand = _load("adr_cand", ROOT / "uncertainty_conditional_candidate_v1_2026-08-24.py")


def aci_daily_reset(oof: pd.DataFrame, capacities: dict[str, float], gamma: float) -> pd.DataFrame:
    detailed = []
    for (tier, horizon), g in oof.groupby(["티어", "수평_h"], sort=False):
        fold_order = g.groupby("폴드")["발행시각"].min().sort_values().index.tolist()
        prior = pd.DataFrame()
        for fold_idx, fold in enumerate(fold_order):
            test = g[g["폴드"] == fold].sort_values("대상").copy()
            if fold_idx == 0:
                prior = pd.concat([prior, test], ignore_index=True)
                continue
            cap = capacities["daily"] if tier == "일간" else capacities["hourly"]
            calib_score = np.abs(prior["실제"].to_numpy(float) - prior["예측"].to_numpy(float))

            for nominal in (80, 90):
                a_target = ALPHA_TARGET[nominal]
                alpha_t = a_target
                cur_day = None
                lo_list, hi_list, q_list, alpha_list, reset_list = [], [], [], [], []
                for _, row in test.iterrows():
                    day = pd.Timestamp(row["대상"]).normalize()
                    reset = False
                    if cur_day is not None and day != cur_day:
                        alpha_t = a_target
                        reset = True
                    cur_day = day
                    alpha_t = float(np.clip(alpha_t, *ALPHA_CLIP))
                    q = ua.finite_sample_quantile(calib_score, alpha_t)
                    pred = float(row["예측"])
                    lo, hi = max(0.0, pred - q), min(cap, pred + q)
                    covered = (row["실제"] >= lo) and (row["실제"] <= hi)
                    alpha_list.append(alpha_t); q_list.append(q)
                    lo_list.append(lo); hi_list.append(hi); reset_list.append(reset)
                    alpha_t = alpha_t + gamma * (a_target - (0 if covered else 1))

                part = test.copy()
                part["명목포함률_pct"] = nominal
                part["하한"] = lo_list; part["상한"] = hi_list
                part["포함"] = [(y >= l) & (y <= h) for y, l, h in zip(test["실제"], lo_list, hi_list)]
                part["alpha_t"] = alpha_list
                part["보정절대잔차_q"] = q_list
                part["일일리셋"] = reset_list
                part["보정표본수"] = len(calib_score)
                detailed.append(part)
            prior = pd.concat([prior, test], ignore_index=True)
    return pd.concat(detailed, ignore_index=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    capacities = {"hourly": CAPACITY_KW, "daily": CAPACITY_KW * 24}
    oof = uav5.load_oof_v5()
    print(f"대상: 전 티어·수평 8개 조합, {len(oof):,}행, γ={GAMMA_DEFAULT}"
          f"(문헌 통상범위 0.01~0.05 중 기본값 — 원문 미확보로 직접 지정)\n")

    print("[1/3] 현행 공식(등폭 순차 conformal)...")
    cur_rows, _ = ua.sequential_conformal(oof, capacities)
    print("[2/3] 후보(ACI + daily reset, γ=%.2f)..." % GAMMA_DEFAULT)
    cand_rows = aci_daily_reset(oof, capacities, GAMMA_DEFAULT)
    assert len(cur_rows) == len(cand_rows), "시험행 수 불일치"
    print(f"  동일 시험행 {len(cur_rows):,}개 확인\n")

    cur_m, cand_m = cand.marginal(cur_rows), cand.marginal(cand_rows)
    cur_d, cand_d = cand.decile_gap(cur_rows), cand.decile_gap(cand_rows)

    comp = cur_m.merge(cand_m, on=["티어", "수평_h", "명목포함률_pct"], suffixes=("_현행", "_후보"))
    comp = comp.merge(cur_d[["티어", "수평_h", "명목포함률_pct", "십분위_최대이탈_pp", "포함률80미만_구간수"]]
                      .rename(columns={"십분위_최대이탈_pp": "십분위최대이탈_현행", "포함률80미만_구간수": "80미만구간_현행"}),
                      on=["티어", "수평_h", "명목포함률_pct"])
    comp = comp.merge(cand_d[["티어", "수평_h", "명목포함률_pct", "십분위_최대이탈_pp", "포함률80미만_구간수"]]
                      .rename(columns={"십분위_최대이탈_pp": "십분위최대이탈_후보", "포함률80미만_구간수": "80미만구간_후보"}),
                      on=["티어", "수평_h", "명목포함률_pct"])
    comp["구간폭_변화_pct"] = (comp["평균구간폭_후보"] - comp["평균구간폭_현행"]) / comp["평균구간폭_현행"] * 100

    cur_f, cand_f = cand.fold_level(cur_rows), cand.fold_level(cand_rows)
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
    pd.concat([cur_d.assign(방식="현행_등폭"), cand_d.assign(방식="후보_ACI일일리셋")],
              ignore_index=True).to_csv(OUT / "조건부포함률_비교.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("[3/3] 판정")
    print("\n=== 전체(marginal) 포함률 · 구간폭 ===")
    print(comp[["티어", "수평_h", "명목포함률_pct", "실제포함률_pct_현행", "실제포함률_pct_후보",
                "평균구간폭_현행", "평균구간폭_후보", "구간폭_변화_pct"]].to_string(index=False))
    print("\n=== 조건부(십분위) 이탈폭 ===")
    print(comp[["티어", "수평_h", "명목포함률_pct", "십분위최대이탈_현행", "십분위최대이탈_후보",
                "80미만구간_현행", "80미만구간_후보"]].to_string(index=False))
    print("\n=== 사전동결 규칙 판정 ===")
    print(comp[["티어", "수평_h", "명목포함률_pct", "규칙3a_전체포함률_3pp이내", "규칙3b_십분위이탈_감소",
                "규칙3c_구간폭_10pct이내", "규칙4_폴드포함률_명목-5pp이상", "채택"]].to_string(index=False))

    n_adopt = int(comp["채택"].sum())
    print(f"\n채택 {n_adopt}/{len(comp)} 조합")

    print("\n[참고] γ 민감도 체크(0.01, 0.05)...")
    sens_rows = []
    for gamma in (0.01, 0.05):
        r = aci_daily_reset(oof, capacities, gamma)
        m = cand.marginal(r)
        d = cand.decile_gap(r)
        j = m.merge(d[["티어", "수평_h", "명목포함률_pct", "십분위_최대이탈_pp"]], on=["티어", "수평_h", "명목포함률_pct"])
        j["gamma"] = gamma
        sens_rows.append(j)
    sens_df = pd.concat(sens_rows, ignore_index=True)
    sens_df.to_csv(OUT / "감마민감도.csv", index=False, encoding="utf-8-sig")
    for gamma, g in sens_df.groupby("gamma"):
        print(f"  γ={gamma}: 십분위최대이탈 평균 {g['십분위_최대이탈_pp'].mean():.2f}pp, "
              f"|포함률차이|평균 {g['포함률차이_pp'].abs().mean():.2f}pp")

    report = {
        "후보": "ACI(Gibbs & Candès 2021) + daily miscoverage reset(태양광 특화 변형)",
        "출처": {
            "표준ACI업데이트식": "Gibbs & Candès (2021), Adaptive Conformal Inference Under Distribution Shift — 원문 확인, 식 그대로 구현",
            "daily_reset_아이디어": "MDPI Energies 19(6):1495 — 원문 HTTP 403으로 미확보(2026-08-21과 동일 문제, 비교표 수치 인용 안 함); "
                                  "Scientific Reports 2026(s41598-026-40911-x) — WebFetch 자동요약으로 확인, 원문 직접 읽지 않음",
            "gamma_값": f"{GAMMA_DEFAULT}(논문 원문 미확보로 우리가 지정 — Gibbs & Candès 통상 실험범위 0.01~0.05 내)",
            "하루경계_정의": "대상시각(예측 대상) 날짜 기준(우리가 정함 — 논문에서 확인 못함)",
        },
        "비교기준": "현행 공식(등폭 순차 conformal), Mondrian 후보와 동일 사전동결 규칙",
        "채택조합수": n_adopt,
        "전체조합수": int(len(comp)),
        "비고": "이 스크립트도 후보 판정만 한다 — 공식 방식은 사용자 결정 전까지 바꾸지 않는다.",
    }
    (OUT / "판정.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
