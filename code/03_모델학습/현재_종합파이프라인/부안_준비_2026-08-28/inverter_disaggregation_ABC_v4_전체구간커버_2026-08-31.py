# -*- coding: utf-8 -*-
"""부안 인버터분해 A·B·C v4(08-31 3차 후속) - v3에 "마지막 구간 평가
누락" 정정까지 적용.

## 왜 필요한가
v3(179일, 5폴드)를 직접 대조해보니 마지막 19일(160~178)이 어느 폴드
시험구간에도 들어가지 않아 평가에서 완전히 빠져있었다 - 총출력모델
v5에서 Codex 지적으로 고친 것과 같은 종류의 문제를 인버터분해에서도
스스로 찾아 정정한다. 마지막 폴드가 남은 날을 전부 흡수하도록 바꿔
179일 전부가 평가에 포함되게 한다. 나머지(필터 수정·결함구간 제외·
walk-forward)는 v3와 완전히 동일.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE / "inverter_disaggregation_ABC_v3_결함구간제외_2026-08-31.py"
OUT_DIR = HERE / "outputs" / "인버터분해_ABC_v4_전체구간커버_2026-08-31"


def load_v3_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("abc_v3", V3)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expanding_folds_full_coverage(n_days: int, initial: int, block: int) -> list[tuple]:
    folds = []
    end = initial
    while end < n_days:
        test_end = min(end + block, n_days)
        folds.append((end, test_end))
        end = test_end
    return folds


def main() -> None:
    v3 = load_v3_module()
    m2 = v3.load_v2_module()
    daily8 = m2.load_daily_per_inverter()
    idx = pd.to_datetime(pd.Index(daily8.index))
    in_defect = (idx >= v3.DEFECT_START) & (idx < v3.DEFECT_END_EXCLUSIVE)
    removed = sorted(idx[in_defect].strftime("%Y-%m-%d").tolist())
    daily8 = daily8.loc[~in_defect]

    folds = expanding_folds_full_coverage(len(daily8), m2.INITIAL_TRAIN_DAYS, m2.TEST_BLOCK_DAYS)

    a = m2.run_method_a(daily8)
    b = m2.run_method_b_wf(daily8, folds)
    c = m2.run_method_c_wf(daily8, folds)

    test_days_all = daily8.index[m2.INITIAL_TRAIN_DAYS:folds[-1][1]] if folds else daily8.index[:0]
    a_on_test_only = m2.run_method_a(daily8.loc[test_days_all]) if len(test_days_all) else None

    season_coverage = pd.Series([m2._season(d.month) for d in daily8.index]).value_counts().to_dict()

    result = {
        "결함구간_제외_날수": len(removed),
        "표본일수": int(len(daily8)),
        "계절별_표본일수": season_coverage,
        "폴드구성": [{"폴드": i + 1, "학습일수": s, "시험일수": e - s} for i, (s, e) in enumerate(folds)],
        "시험구간_전체커버여부": (folds[-1][1] == len(daily8)) if folds else False,
        "A_시험구간만(공정비교용)": a_on_test_only,
        "B": b,
        "C": c,
        "순위_시험구간_공정비교": sorted([
            {"방법": "A_정격용량비례", "평균_nMAE_pct": (a_on_test_only or a)["평균_nMAE_pct"]},
            {"방법": "B_계절중앙값(walk-forward)", "평균_nMAE_pct": b["평균_nMAE_pct"]},
            {"방법": "C_LightGBM비중(walk-forward)", "평균_nMAE_pct": c["평균_nMAE_pct"]},
        ], key=lambda r: r["평균_nMAE_pct"]),
        "_비고": "v3(마지막19일 미평가) 대비 변화폭 확인용 - B 채택 결론이 유지되는지가 핵심.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "부안_인버터분해_ABC_v4_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
