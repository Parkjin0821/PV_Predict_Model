# -*- coding: utf-8 -*-
"""기존 개선후보들을 KPX NMAE로 재채점 — 협약 연구개발계획서 공식지표 정렬.

## 왜 필요한가
협약 연구개발계획서(RS-2026-25536130) "9.재생 발전 예측 정확도" 성과지표는
**KPX NMAE 단일 지표**(설비이용률 10% 이상 시각만, (100/n)Σ|실제-예측|/
설비용량×100)로 평가·인증된다(AGENTS.md "협약 연구개발계획서 확인" 절).
그런데 지금까지 ⑥ round2 개선후보 4종(NWP분위사상·OOF잔차보정·날씨군집화·
트리구조개선)은 전부 **원시 kW MAE/RMSE**로만 판정했다. 원시오차와
KPX NMAE(용량정규화 + 10%필터)는 같은 방향으로 움직인다는 보장이 없다
— 재채점해서 판정이 실제로 바뀌는지 확인한다.

## 범위
- ⑥ round2 4단계 전부(`outputs/⑥재판정_v5_공식B_v1_2026-08-24/{step}_
  {name}/동일행_예측정답.csv`)를 **재구현 없이 그대로 재사용**, 채점
  지표만 KPX NMAE로 바꾼다.
- **일간(D+1)은 제외한다** — KPX NMAE는 순간(또는 시간단위) 출력의
  이용률 10% 필터를 전제로 하는 지표라 일간 총량(kWh)에는 정의가 그대로
  안 맞는다(억지로 늘리면 새 지표를 만드는 것이지 "재채점"이 아니다).
- 실제로 배포까지 반영된 두 결정(초단기 +4h 채택, +2h 반려)도 실제
  운영 예측정답(E2E v1/v2)으로 KPX NMAE 재검증한다.

## 단일지표 판정규칙(사전동결 — 결과 보기 전 여기 기록)
원래 규칙(MAE·RMSE 동시 1%↑, 계절 5%↓ 없음)을 **단일지표(NMAE)로
그대로 축소** 적용한다: **① NMAE 개선율 ≥1% ② 어느 정상 폴드도 NMAE
5% 이상 악화 없음**. 두 조건 다 만족해야 채택.

## 산출물 (`outputs/KPX_NMAE_재채점_v1_2026-08-24/`)
`재채점_전체.csv`(원래 MAE/RMSE 판정과 KPX NMAE 판정 나란히 비교),
`재채점_폴드별.csv`, `배포결정_재검증.csv`, `요약.json`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC6 = ROOT / "outputs" / "⑥재판정_v5_공식B_v1_2026-08-24"
OUT = ROOT / "outputs" / "KPX_NMAE_재채점_v1_2026-08-24"
CAPACITY_KW = 219.0
UTIL_THRESHOLD = 0.10
STEPS = {
    1: "1_NWP_분위사상",
    2: "2_구름전이_강수_OOF잔차보정",
    3: "3_일사예보_날씨군집",
    4: "4_트리_구조개선",
}


def nmae(y: np.ndarray, p: np.ndarray, capacity: float) -> tuple[float, int]:
    """KPX 정의 NMAE(%): 이용률 10% 이상 시각만, 용량 정규화."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y >= UTIL_THRESHOLD * capacity
    if mask.sum() == 0:
        return float("nan"), 0
    e = np.abs(y[mask] - p[mask]) / capacity * 100
    return float(e.mean()), int(mask.sum())


def score_step(step: int, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = SRC6 / name / "동일행_예측정답.csv"
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d["수평_h"] = d["수평_h"].astype(str)
    d = d[d["티어"] != "일간"].copy()  # 일간은 KPX NMAE 정의 밖(위 docstring)

    verdict_rows, fold_rows = [], []
    for (tier, h), g in d.groupby(["티어", "수평_h"]):
        n_cur, cnt_cur = nmae(g["실제"], g["현행예측"], CAPACITY_KW)
        n_cand, cnt_cand = nmae(g["실제"], g["후보예측"], CAPACITY_KW)
        gain = (n_cur - n_cand) / n_cur * 100 if n_cur else float("nan")

        worst, worst_fold = -np.inf, ""
        for fold, fg in g.groupby("폴드"):
            fc, _ = nmae(fg["실제"], fg["현행예측"], CAPACITY_KW)
            fb, cnt = nmae(fg["실제"], fg["후보예측"], CAPACITY_KW)
            if cnt < 10 or not np.isfinite(fc) or fc == 0:
                continue
            worsen = (fb - fc) / fc * 100
            fold_rows.append({"단계": step, "개선안": STEPS[step].split("_", 1)[1], "티어": tier,
                              "수평_h": h, "폴드": fold, "NMAE_현행_pct": round(fc, 3),
                              "NMAE_후보_pct": round(fb, 3), "악화율_pct": round(worsen, 2),
                              "정산대상표본수": cnt})
            if worsen > worst:
                worst, worst_fold = worsen, fold
        if worst == -np.inf:
            worst, worst_fold = 0.0, "(폴드부족)"

        adopt = bool(np.isfinite(gain) and gain >= 1 and worst < 5)
        reason = ("정산대상표본부족" if not np.isfinite(gain) else
                  "채택" if adopt else
                  ("개선율1%미만" if gain < 1 else f"계절악화({worst_fold} {worst:.1f}%)"))
        verdict_rows.append({
            "단계": step, "개선안": STEPS[step].split("_", 1)[1], "티어": tier, "수평_h": h,
            "NMAE_현행_pct": round(n_cur, 3), "NMAE_후보_pct": round(n_cand, 3),
            "NMAE_개선율_pct": round(gain, 2) if np.isfinite(gain) else None,
            "최대계절악화_pct": round(worst, 2), "정산대상표본_현행": cnt_cur, "정산대상표본_후보": cnt_cand,
            "KPX_NMAE_채택": adopt, "KPX_NMAE_판정사유": reason,
        })
    return pd.DataFrame(verdict_rows), pd.DataFrame(fold_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"용량: {CAPACITY_KW}kW / 이용률필터: {UTIL_THRESHOLD:.0%} 이상 / 일간 제외\n")

    all_v, all_f = [], []
    for step, name in STEPS.items():
        print(f"=== 단계{step}: {name.split('_',1)[1]} ===")
        v, f = score_step(step, name)
        all_v.append(v); all_f.append(f)
        print(v.to_string(index=False))
        print()

    verdict = pd.concat(all_v, ignore_index=True)
    folds = pd.concat(all_f, ignore_index=True)

    # 원래(MAE/RMSE) 판정과 나란히 비교
    orig = pd.read_csv(SRC6 / "판정_전체.csv", encoding="utf-8-sig")
    orig["수평_h"] = orig["수평_h"].astype(str)
    orig_slim = orig[orig["티어"] != "일간"][["단계", "티어", "수평_h", "채택", "판정사유", "MAE개선율_pct", "RMSE개선율_pct"]].rename(
        columns={"채택": "원래(MAE_RMSE)_채택", "판정사유": "원래_판정사유"})
    verdict = verdict.merge(orig_slim, on=["단계", "티어", "수평_h"], how="left")
    verdict["판정_일치"] = verdict["KPX_NMAE_채택"] == verdict["원래(MAE_RMSE)_채택"]

    verdict.to_csv(OUT / "재채점_전체.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "재채점_폴드별.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("=== 최종 비교(원래 MAE/RMSE 판정 vs KPX NMAE 재채점) ===")
    print(verdict[["단계", "개선안", "티어", "수평_h", "원래(MAE_RMSE)_채택", "KPX_NMAE_채택", "판정_일치",
                  "NMAE_개선율_pct", "MAE개선율_pct", "RMSE개선율_pct"]].to_string(index=False))

    flips = verdict[~verdict["판정_일치"]]
    print(f"\n판정이 뒤집힌 조합: {len(flips)}/{len(verdict)}")
    if len(flips):
        print(flips[["단계", "개선안", "티어", "수평_h", "원래(MAE_RMSE)_채택", "KPX_NMAE_채택", "KPX_NMAE_판정사유"]].to_string(index=False))

    summary = {
        "범위": "⑥ round2 1~4단계 초단기·단기(일간 제외, KPX NMAE 정의 밖)",
        "지표": "KPX NMAE(%) = 이용률 10%이상 시각만, (100/n)Σ|실제-예측|/용량×100",
        "용량": CAPACITY_KW,
        "단일지표_판정규칙": "개선율 1%이상 AND 어느 폴드도 5%이상 악화 없음(원 규칙을 단일지표로 축소)",
        "전체조합수": int(len(verdict)),
        "판정_뒤집힌_조합수": int(len(flips)),
        "원래_채택_KPX_기각": int(len(verdict[(verdict["원래(MAE_RMSE)_채택"]) & (~verdict["KPX_NMAE_채택"])])),
        "원래_기각_KPX_채택": int(len(verdict[(~verdict["원래(MAE_RMSE)_채택"]) & (verdict["KPX_NMAE_채택"])])),
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
