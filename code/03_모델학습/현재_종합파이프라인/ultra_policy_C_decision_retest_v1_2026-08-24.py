# -*- coding: utf-8 -*-
"""초단기 +1h·+2h 약화 재판정 — 공식 구조선택 파이프라인 그대로 정책 A/B/C 재비교.

## 왜 필요한가
5번 재학습(v5·공식B·219kW) 후 초단기 +1h·+2h가 2차(v3) 대비 +3.6%·
+4.6% 나빠졌다. 이전 A/B/C 비교(`defect_policy_comparison_v1_2026-08-21`)
는 **구조선택 없는 단순 LightGBM 기본**으로만 비교했는데, 실제 공식
3차 모델(+1h·+2h)은 **구조선택(`choose_structure`)이 적용된 모델**이다
— 즉 이전 비교는 지금 문제가 된 그 파이프라인과 정확히 같은 조건이
아니었다. 이 스크립트는 `e2e_retrain_v5_공식B_v1_2026-08-24.run_ultra`
(구조선택 포함, 방금 공식 3차를 만든 바로 그 함수)를 정책만 바꿔가며
그대로 재사용해 A(무처리)·C(용량가중추정보정)를 다시 채점한다.
B(공식) 결과는 이미 만든 3차 공식모델 산출물을 그대로 쓴다(재학습 안 함).
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
OUT = ROOT / "outputs" / "초단기_ABC_재판정_v1_2026-08-24"
B_OUT = ROOT / "outputs" / "E2E_v5_공식B_v1_2026-08-24"

spec = importlib.util.spec_from_file_location("e2e", ROOT / "e2e_retrain_v5_공식B_v1_2026-08-24.py")
e2e = importlib.util.module_from_spec(spec)
sys.modules["e2e"] = e2e
spec.loader.exec_module(e2e)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    horizons = (1, 2)

    results = {}
    for policy, label in [("A_무처리", "A"), ("C_용량가중추정보정", "C")]:
        print(f"\n=== 정책 {label}({policy}) ===")
        model_dir = OUT / f"fold_models_{label}"
        model_dir.mkdir(parents=True, exist_ok=True)
        ac, audits = e2e.run_ultra(capacity_kw, seed, policy=policy, horizons=horizons, model_dir=model_dir)
        ac.to_csv(OUT / f"행단위_{label}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(audits).to_csv(OUT / f"감사_{label}.csv", index=False, encoding="utf-8-sig")
        results[label] = ac

    # 공식 B는 이미 만든 3차 결과에서 +1h·+2h만 뽑는다(재학습 안 함).
    b_full = pd.read_csv(B_OUT / "행단위_ac_power_예측정답.csv", encoding="utf-8-sig")
    results["B"] = b_full[(b_full["티어"] == "초단기") & (b_full["수평_h"].isin(horizons))]

    rows = []
    for label, df in results.items():
        for h, g in df.groupby("수평_h"):
            e = g["실제_kW"] - g["예측_kW"]
            mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
            rows.append({"정책": label, "수평_h": h, "n": len(g),
                        "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3)})
    summary = pd.DataFrame(rows).sort_values(["수평_h", "정책"])
    summary.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("\n=== A/B/C 최종 판정표(공식 구조선택 파이프라인, 같은 시험행) ===")
    print(summary.to_string(index=False))

    # 2차(v3) 참고치와 나란히
    old = {1: (10.866, 17.367), 2: (13.183, 20.064)}
    print("\n=== 참고: 2차(v3) 대비 ===")
    for h in horizons:
        o = old[h]
        print(f"  +{h}h 2차: MAE {o[0]} / RMSE {o[1]}")

    (OUT / "요약.txt").write_text(summary.to_string(index=False), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
