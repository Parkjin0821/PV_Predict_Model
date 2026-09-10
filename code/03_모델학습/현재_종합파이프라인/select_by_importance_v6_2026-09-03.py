# -*- coding: utf-8 -*-
"""6차(09-03) - Pearson 선형필터의 한계(사용자 지적 + Google Scholar 확인)를
반영해, 사전 선형 필터 없이 48개 후보 전부를 LightGBM에 넣고 permutation
importance(비선형 관계도 잡아내는 지표)로 재선정한다.

## 근거(09-03 확인, 실행 전 필수 게이트)
- Pearson은 선형 스크리닝용, 비선형은 SHAP·MI·트리기반 중요도로 별도
  확인하는 게 최근 PV예측 문헌의 표준 조합(Sciencedirect 2021 LSTM+Pearson,
  Nature Sci Rep 2026 correlation+ML, Sciencedirect 2024 tree+SHAP for
  solar radiation 등, 09-03 세션에서 WebSearch로 확인).
- 이번 라운드는 SHAP 라이브러리 미설치라 sklearn permutation_importance
  사용(트리기반 비선형 중요도 계열, 별도 설치 없이 재현 가능·재구현 아님
  - sklearn 표준 함수 그대로 호출).

## 방법
1. v5의 후보 조립 함수를 그대로 재사용(48개: 환경·설비 36 + 시간 12,
   인버터효율 포함하면 49) - 사전 Pearson 필터 없이 baseline(과거발전량)
   + 전체 후보를 한 LightGBM에 학습.
2. permutation_importance(반복 20회, MAE 감소량 기준)로 각 특성의
   실제 기여도를 측정 - 선형/비선형 관계 둘 다 잡힘(모델이 이미
   비선형을 학습했으므로, 그 모델을 셔플했을 때 성능이 얼마나
   나빠지는지로 중요도를 잰다 - 상관계수와 무관한 방식).
3. 중요도>0(셔플 시 성능이 실제로 나빠지는 것)인 특성만 채택, 나머지는
   가지치기.
4. 최종 선택셋으로 재학습해 MAE/RMSE를 09-03 v5 결과(46개, MAE 8.352/
   RMSE 11.974)와 비교.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v5", ROOT / "select_and_check_all_factors_v5_2026-09-03.py")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\중요도기반_재검증_v6_2026-09-03")
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(v5.sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity_kw = float(df["reported_capacity_kw"].dropna().iloc[0])

    env_equip_cand = v5.sel.build_candidate_frame(df)
    time_cand = v5.build_time_factor_frame(df)
    equip_extra_cand = v5.build_equip_extra_frame(df)
    cand = pd.concat([env_equip_cand, time_cand, equip_extra_cand], axis=1)
    all_cols = v5.sel.CANDIDATE_COLUMNS + v5.TIME_FACTOR_COLUMNS + v5.EQUIP_EXTRA_COLUMNS

    target = df["plant_output_kw"]
    daylight = (df["solar_elevation_deg"] > 0).astype(float)
    baseline = v5.make_true_baseline(df)

    print(f"[0] 후보 {len(all_cols)}개, 사전 선형필터 없이 전부 사용")

    frame = baseline.copy()
    for c in all_cols:
        frame[c] = cand[c]
    frame["목표_발전출력_kW"] = target
    frame["목표_낮시간"] = daylight
    features = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
    f = frame.dropna(subset=features + ["목표_발전출력_kW"])
    f = f[f["목표_낮시간"] > 0]
    n = len(f)
    cut = int(n * 0.85)
    train, test = f.iloc[:cut], f.iloc[cut:]
    print(f"[1] 학습 {len(train)}행 · 시험 {len(test)}행 · 특성 {len(features)}개")

    model = LGBMRegressor(n_estimators=220, learning_rate=.04, num_leaves=31, min_child_samples=30,
                           subsample=.9, colsample_bytree=.9, reg_lambda=.3, random_state=SEED,
                           n_jobs=4, verbosity=-1)
    model.fit(train[features], train["목표_발전출력_kW"])

    print("[2] permutation importance 계산 중(20회 반복, MAE 기준)...")
    perm = permutation_importance(
        model, test[features], test["목표_발전출력_kW"],
        scoring="neg_mean_absolute_error", n_repeats=20, random_state=SEED, n_jobs=4,
    )
    imp_df = pd.DataFrame({
        "변수": features,
        "중요도_MAE증가량": perm.importances_mean,
        "표준편차": perm.importances_std,
        "분류": ["과거발전량(기준선)" if c not in all_cols else
                ("시간요인" if c in v5.TIME_FACTOR_COLUMNS else "환경·설비요인")
                for c in features],
    }).sort_values("중요도_MAE증가량", ascending=False)
    imp_df.to_csv(OUT_DIR / "특성중요도_permutation.csv", index=False, encoding="utf-8-sig")
    print(imp_df.to_string(index=False))

    # 09-03 v5와 동일 원칙: 과거발전량 baseline(9개)은 그대로 유지(자기회귀
    # 근간이라 중요도로 가지치기 대상 아님), 나머지 후보만 중요도>0인 것만 채택.
    baseline_cols = list(baseline.columns)
    selected = [
        row["변수"] for _, row in imp_df.iterrows()
        if row["변수"] not in baseline_cols and row["중요도_MAE증가량"] > 0
    ]
    dropped = [c for c in all_cols if c not in selected]
    print(f"\n[3] 중요도>0 채택 {len(selected)}개, 가지치기 {len(dropped)}개: {dropped}")

    def evaluate(name, cols):
        fr = baseline.copy()
        for c in cols:
            fr[c] = cand[c]
        fr["목표_발전출력_kW"] = target
        fr["목표_낮시간"] = daylight
        feats = [c for c in fr.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
        ff = fr.dropna(subset=feats + ["목표_발전출력_kW"])
        ff = ff[ff["목표_낮시간"] > 0]
        nn = len(ff); cc = int(nn * 0.85)
        tr, te = ff.iloc[:cc], ff.iloc[cc:]
        m = LGBMRegressor(n_estimators=220, learning_rate=.04, num_leaves=31, min_child_samples=30,
                           subsample=.9, colsample_bytree=.9, reg_lambda=.3, random_state=SEED,
                           n_jobs=4, verbosity=-1)
        m.fit(tr[feats], tr["목표_발전출력_kW"])
        pred = np.clip(m.predict(te[feats]), 0, capacity_kw)
        e = te["목표_발전출력_kW"].to_numpy() - pred
        return dict(구성=name, 특성수=len(feats), 학습=len(tr), 시험=len(te),
                    MAE_kW=round(float(np.abs(e).mean()), 3), RMSE_kW=round(float(np.sqrt((e**2).mean())), 3))

    print("\n[4] 최종 재학습 비교...")
    results = [
        evaluate("F_전체48개(필터없음)", all_cols),
        evaluate("G_importance기반 최종", selected),
    ]
    comp = pd.DataFrame(results)
    print(comp.to_string(index=False))

    comp.to_csv(OUT_DIR / "최종성능비교.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "요약.json").write_text(json.dumps({
        "선택": selected, "가지치기": dropped, "성능비교": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
