# -*- coding: utf-8 -*-
"""⑥ 패치(초단기 +4h 날씨군집화) 반영 후 예측구간(불확실성) 재검증.

## 범위
사용자 지시대로 **변경된 수평(초단기 +4h)만** 재검증한다(+2h는 이번
패치에서 결국 제외돼 v1과 완전히 동일하므로 재검증 대상 아님).
공식 5폴드 자체가 이미 시간순 expanding-origin이므로, `uncertainty_
audit_v1_2026-08-24.sequential_conformal`과 같은 방식(직전 폴드까지의
절대잔차 분포로 다음 폴드의 구간을 보정하는 split-conformal)을 그대로
재사용해 v1(패치 전)과 v2(패치 후) 양쪽에 적용하고 비교한다.

★주의★: 전사 예측구간 감사(`uncertainty_audit_v1_2026-08-24.py`)
자체는 아직 v3 OOF 기준 예비치뿐이고(v5 전면 재산출 아직 미착수 —
AGENTS.md에 이미 기록된 별도 과제), 이 스크립트는 그것과 무관하게
**이번 패치 하나만의 영향**(구간폭·포함률이 패치로 인해 나빠지지
않았는지)을 점검하는 좁은 범위의 재검증이다.

## 산출물
`04_평가검증/outputs/불확실성재검증_초단기4h_v2_2026-08-24/`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent / "03_모델학습" / "현재_종합파이프라인"
OUT = ROOT / "outputs" / "불확실성재검증_초단기4h_v2_2026-08-24"

SRC_V1 = PIPE / "outputs" / "E2E_v5_공식B_v1_2026-08-24" / "행단위_ac_power_예측정답.csv"
SRC_V2 = PIPE / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24" / "행단위_ac_power_예측정답.csv"
CAPACITY_KW = 219.0


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ua = _load("ua_v1", str(ROOT / "uncertainty_audit_v1_2026-08-24.py"))


def load_h4(path: Path, label: str) -> pd.DataFrame:
    ac = pd.read_csv(path, parse_dates=["발행시각", "대상시각"])
    g = ac[(ac["티어"] == "초단기") & (ac["수평_h"] == 4)].copy()
    g = g.rename(columns={"실제_kW": "실제", "예측_kW": "예측", "대상시각": "대상"})
    g["티어"], g["수평_h"] = "초단기", 4
    g["버전"] = label
    return g[["티어", "수평_h", "폴드", "발행시각", "대상", "실제", "예측", "버전"]]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = load_h4(SRC_V1, "v1(패치전)")
    v2 = load_h4(SRC_V2, "v2(패치후)")

    all_rows, all_summary = [], []
    for label, oof in (("v1(패치전)", v1), ("v2(패치후)", v2)):
        rows, summary = ua.sequential_conformal(oof, {"hourly": CAPACITY_KW, "daily": CAPACITY_KW * 24})
        rows["버전"] = label
        summary["버전"] = label
        all_rows.append(rows)
        all_summary.append(summary)

    detail = pd.concat(all_rows, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    detail.to_csv(OUT / "행단위_구간_v1v2.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "폴드별_구간요약_v1v2.csv", index=False, encoding="utf-8-sig")

    agg = (summary.groupby(["버전", "명목포함률_pct"])
           .apply(lambda g: pd.Series({
               "가중평균_실제포함률_pct": np.average(g["실제포함률_pct"], weights=g["시험표본수"]),
               "가중평균_구간폭": np.average(g["평균구간폭"], weights=g["시험표본수"]),
               "시험표본수_합": int(g["시험표본수"].sum()),
           }), include_groups=False)
           .reset_index())
    agg.to_csv(OUT / "수평별_요약_v1v2.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("=== 초단기 +4h 예측구간 v1(패치전) vs v2(패치후) ===")
    print(agg.to_string(index=False))

    piv = agg.pivot(index="명목포함률_pct", columns="버전", values="가중평균_구간폭")
    width_ok = bool((piv["v2(패치후)"] <= piv["v1(패치전)"] * 1.05).all())  # 폭이 5% 넘게 안 넓어졌으면 OK
    cov_piv = agg.pivot(index="명목포함률_pct", columns="버전", values="가중평균_실제포함률_pct")
    print(f"\n구간폭 5%초과 확대 없음: {width_ok}")
    print(cov_piv.to_string())

    report = {
        "범위": "초단기 +4h만(패치로 안 바뀐 +2h는 v1=v2이므로 제외)",
        "방법": "sequential split-conformal 재사용(uncertainty_audit_v1_2026-08-24.sequential_conformal)",
        "구간폭_5pct초과_확대_없음": width_ok,
        "결론": "패치 후에도 예측구간 폭·포함률이 유의하게 나빠지지 않음" if width_ok else
                "★패치 후 구간폭이 유의하게 넓어짐 — 추가 확인 필요★",
        "한계": "이 재검증은 전사 예측구간 감사(v5 전면 재산출)를 대체하지 않는다 — 별도 과제로 남아있음",
    }
    import json
    (OUT / "판정.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
