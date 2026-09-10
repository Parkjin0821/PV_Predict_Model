# -*- coding: utf-8 -*-
"""부안 인버터분해 A·B·C v3(08-31 재후속) - v2에 부안 결함구간
(2026-04-15~05-22, half-open) 전체제외 원칙 적용.

## 왜 필요한가
v2(inverter_disaggregation_ABC_v2_필터수정_walkforward_2026-08-31.py)의
181일 표본을 직접 대조해보니 2026-04-29·05-15 2일이 부안 결함구간
(부안_전처리_규칙_v1_2026-08-28.json의 defect_period,
start_inclusive=2026-04-15/end_exclusive=2026-05-23)에 들어있었다.
08-28에 이미 "구간 내 개별적으로 정상으로 보이는 날도 예외 없이 전체
제외"라는 원칙을 정했는데(총출력 5·6단계 모델엔 적용됐음) 이 인버터분해
스크립트엔 빠져있었다 - 원칙대로 다시 뺀다. 나머지 로직(필터 수정,
walk-forward)은 v2와 완전히 동일.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V2 = HERE / "inverter_disaggregation_ABC_v2_필터수정_walkforward_2026-08-31.py"
OUT_DIR = HERE / "outputs" / "인버터분해_ABC_v3_결함구간제외_2026-08-31"

DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")


def load_v2_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("abc_v2", V2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    m = load_v2_module()
    daily8 = m.load_daily_per_inverter()

    idx = pd.to_datetime(pd.Index(daily8.index))
    in_defect = (idx >= DEFECT_START) & (idx < DEFECT_END_EXCLUSIVE)
    removed = sorted(idx[in_defect].strftime("%Y-%m-%d").tolist())
    daily8 = daily8.loc[~in_defect]

    folds = m.expanding_folds(len(daily8), m.INITIAL_TRAIN_DAYS, m.TEST_BLOCK_DAYS)

    a = m.run_method_a(daily8)
    b = m.run_method_b_wf(daily8, folds)
    c = m.run_method_c_wf(daily8, folds)

    test_days_all = daily8.index[m.INITIAL_TRAIN_DAYS:folds[-1][1]] if folds else daily8.index[:0]
    a_on_test_only = m.run_method_a(daily8.loc[test_days_all]) if len(test_days_all) else None

    season_coverage = pd.Series([m._season(d.month) for d in daily8.index]).value_counts().to_dict()

    result = {
        "결함구간_제외": {"제외기간": "2026-04-15~2026-05-22(half-open end_exclusive 05-23)",
                       "제외된_날": removed, "제외건수": len(removed)},
        "표본일수_v2대비": {"v2": 181, "v3": int(len(daily8))},
        "계절별_표본일수": season_coverage,
        "walk_forward_폴드수": len(folds),
        "A_시험구간만(B·C와_공정비교용)": a_on_test_only,
        "B": b,
        "C": c,
        "순위_시험구간_공정비교": sorted([
            {"방법": "A_정격용량비례", "평균_nMAE_pct": (a_on_test_only or a)["평균_nMAE_pct"]},
            {"방법": "B_계절중앙값(walk-forward)", "평균_nMAE_pct": b["평균_nMAE_pct"]},
            {"방법": "C_LightGBM비중(walk-forward)", "평균_nMAE_pct": c["평균_nMAE_pct"]},
        ], key=lambda r: r["평균_nMAE_pct"]),
        "_비고": "v2 대비 변화폭이 이 결과의 핵심 확인사항 - 결함구간 2일 제외가 "
               "실제로 결론(B 채택)을 바꾸는지 여부.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "부안_인버터분해_ABC_v3_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
