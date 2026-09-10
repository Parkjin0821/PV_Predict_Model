# -*- coding: utf-8 -*-
"""예측구간(불확실성) 공식 채택 — 90%는 daily-reset ACI, 80%·일간은 현행(등폭).

## 사용자 확정 결정(2026-08-24)
`uncertainty_aci_dailyreset_candidate_v1_2026-08-24.py`의 사전동결 규칙
판정 결과를 그대로 공식화한다 — **재판정하지 않는다, 이미 정해진 결과를
공식 산출물로 조립만 한다**(재구현 금지 원칙).

### 채택 매트릭스(판정.json/판정표.csv 그대로)
| 티어·수평 | 80% | 90% |
|---|---|---|
| 초단기 +1h·+2h·+3h·+4h | 현행(등폭) | **daily-reset ACI** |
| 단기 +1h | 현행(등폭) | 현행(등폭) — ACI가 십분위이탈 기준(3b) 근소 미달 |
| 단기 +24h·+48h | 현행(등폭) | **daily-reset ACI** |
| 일간 D+1 | 현행(등폭) | 현행(등폭) — ACI가 수학적으로 현행과 동일(1일1행이라 매 스텝 리셋) |

6개 조합(단기+24h·+48h, 초단기 4개 수평의 90% 구간)만 daily-reset ACI로
교체하고 나머지 10개 조합(위 표의 나머지 칸)은 현행 그대로 둔다.

## 범위 — Blockdata는 변경 없음
Blockdata 출력 스키마(`blockdata_export_v1_2026-08-21.build_spec` 등)를
확인했다. **예측구간(하한·상한) 필드가 애초에 없다** — Blockdata는
점예측(`plant_ac_power_kw`)만 나른다. 그래서 이번 채택은 Blockdata
산출물에 영향이 없고, `04_평가검증` 아래의 분석·보고용 산출물에만
반영한다.

## 산출물 (`outputs/불확실성_공식채택_2026-08-24/`)
- `예측구간_공식_행단위.csv.gz` — 전 티어·수평·명목포함률, 방식 컬럼 포함
- `예측구간_공식_수평별_요약.csv`
- `채택_구성.json` — 어느 조합에 어느 방식을 썼는지 + 근거
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
OUT = ROOT / "outputs" / "불확실성_공식채택_2026-08-24"
CAPACITY_KW = 219.0

# 사용자 확정 채택 매트릭스: (티어, 수평_h, 명목포함률) -> "ACI" | "현행"
ACI_ADOPTED = {
    ("단기", 24, 90), ("단기", 48, 90),
    ("초단기", 1, 90), ("초단기", 2, 90), ("초단기", 3, 90), ("초단기", 4, 90),
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ua = _load("off_ua_v1", ROOT / "uncertainty_audit_v1_2026-08-24.py")
uav5 = _load("off_ua_v5", ROOT / "uncertainty_audit_v5_공식B_v1_2026-08-24.py")
cand = _load("off_cand", ROOT / "uncertainty_conditional_candidate_v1_2026-08-24.py")
adr = _load("off_adr", ROOT / "uncertainty_aci_dailyreset_candidate_v1_2026-08-24.py")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    capacities = {"hourly": CAPACITY_KW, "daily": CAPACITY_KW * 24}
    oof = uav5.load_oof_v5()
    print(f"입력: 전 티어·수평 8개 조합, {len(oof):,}행\n")

    print("[1/2] 현행(등폭) + ACI(daily reset, γ=%.2f) 계산..." % adr.GAMMA_DEFAULT)
    cur_rows, _ = ua.sequential_conformal(oof, capacities)
    aci_rows = adr.aci_daily_reset(oof, capacities, adr.GAMMA_DEFAULT)

    # 수평_h를 비교 가능하게 정규화(단기/초단기는 int, 일간은 "D+1" 문자열)
    def norm_h(v):
        return v if v == "D+1" else int(v)

    cur_rows = cur_rows.assign(방식="현행_등폭")
    aci_rows = aci_rows.assign(방식="ACI_daily_reset")

    print("[2/2] 채택 매트릭스대로 조립...")
    parts = []
    for (tier, h, nom), key_tag in (
        (row, "x") for row in
        [(t, h, n) for t in oof["티어"].unique() for h in oof[oof["티어"] == t]["수평_h"].unique() for n in (80, 90)]
    ):
        use_aci = (tier, norm_h(h), nom) in ACI_ADOPTED
        src = aci_rows if use_aci else cur_rows
        sub = src[(src["티어"] == tier) & (src["수평_h"] == h) & (src["명목포함률_pct"] == nom)]
        parts.append(sub)
    official = pd.concat(parts, ignore_index=True)

    official.to_csv(OUT / "예측구간_공식_행단위.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    agg = cand.marginal(official)
    dec = cand.decile_gap(official)
    agg_full = agg.merge(dec[["티어", "수평_h", "명목포함률_pct", "십분위_최대이탈_pp", "포함률80미만_구간수"]],
                         on=["티어", "수평_h", "명목포함률_pct"])
    method_col = official.drop_duplicates(["티어", "수평_h", "명목포함률_pct"])[
        ["티어", "수평_h", "명목포함률_pct", "방식"]]
    agg_full = agg_full.merge(method_col, on=["티어", "수평_h", "명목포함률_pct"])
    agg_full.to_csv(OUT / "예측구간_공식_수평별_요약.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 공식 예측구간(최종 조립) ===")
    print(agg_full[["티어", "수평_h", "명목포함률_pct", "방식", "실제포함률_pct", "평균구간폭",
                    "십분위_최대이탈_pp", "포함률80미만_구간수"]].to_string(index=False))

    n_aci = len(agg_full[agg_full["방식"] == "ACI_daily_reset"])
    print(f"\nACI_daily_reset 적용 조합: {n_aci}/{len(agg_full)}")

    config = {
        "결정일": "2026-08-24",
        "결정": "사용자 확정 — 90% 구간 6개 조합만 daily-reset ACI로 교체, 나머지는 현행(등폭) 유지",
        "채택_매트릭스": [
            {"티어": t, "수평_h": h, "명목포함률_pct": n, "방식": "ACI_daily_reset"}
            for (t, h, n) in sorted(ACI_ADOPTED)
        ],
        "근거": {
            "표준ACI": "Gibbs & Candès (2021), Adaptive Conformal Inference Under Distribution Shift",
            "daily_reset_아이디어": "MDPI Energies 19(6):1495(원문 미확보, 403) / "
                                  "Scientific Reports 2026 s41598-026-40911-x(자동요약 확인)",
            "gamma": adr.GAMMA_DEFAULT,
            "사전동결_판정규칙": "uncertainty_conditional_candidate_v1_2026-08-24.py 문서화(3a/3b/3c/4/5)",
        },
        "판정근거파일": str(ROOT / "outputs" / "불확실성_ACI일일리셋후보_2026-08-24" / "판정표.csv"),
        "Blockdata_영향": "없음 — Blockdata 스키마에 예측구간(하한·상한) 필드가 애초에 없어 점예측만 나른다",
        "재검증범위": "이 스크립트는 재판정하지 않는다 — 이미 확정된 판정 결과를 공식 산출물로 조립만 함",
    }
    (OUT / "채택_구성.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
