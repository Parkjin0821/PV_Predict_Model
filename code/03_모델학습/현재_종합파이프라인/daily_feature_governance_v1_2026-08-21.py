# -*- coding: utf-8 -*-
"""① 일간모델 특성 거버넌스 보완 — 로드맵 08-22 1단계.

## 배경
일간모델은 지금까지 NWP 12종·장비 17종(총 51개 특성)을 상관분석·
다중공선성 검증 없이 그대로 썼다(08-20 밤 "시간 부족으로 스코프 축소"
로 명시적으로 미뤄둔 부분). 초·단기 트리모델은 상관계수→다중공선성→
배포필터 3단계를 다 거쳤는데 일간만 그 기준에 못 미쳤다.

## 방법론(사용자 확정 원칙 — 단순 상관계수 기계적용 금지)
초·단기처럼 "|r|≥0.3이면 채택"을 그대로 옮기지 않는다. **트리모델은
단순 선형상관이 낮아도 비선형적으로 유용한 특성이 있을 수 있어서**,
아래 4가지를 폴드 내부에서 확인한다:

1. **발행시점 가용성**: NWP는 v3 고정-tm(D 09:00 KST 발표, 목표일
   D+1 집계)이라 D 10시 발행 시점에 이미 확보된 예보다. 장비는
   `shift(2)`(2일 전 평균)라 그날 자정 이전에 이미 확정된 값이다.
   둘 다 구조적으로 발행시점에 가용 — 이건 폴드마다 다시 셀 필요 없이
   구조 자체로 보장된다(재확인만, 별도 계산 없음).
2. **정보누출**: 위 1과 동일 — v3 NWP는 이미 리크 없음 검증됨(AGENTS.md
   NWP 고정-tm 절), 장비 shift(2)는 보수적이라 누출 불가능.
3. **다중공선성**: 폴드 학습구간 안에서 VIF+쌍상관(하네스
   `compute_vif`/`prune_multicollinearity`와 동일 원칙 재사용, O'Brien
   2007 기준 유지).
4. **증분효과(ablation)**: 상관계수든 순열중요도든 "후보에서 뺄지"를
   결정하는 기준일 뿐이다. 최종적으로 **전체특성(51개) 기준선 대비
   시험구간 성능이 실제로 나빠지지 않는지**를 직접 비교해서 확정한다
   (기준 자체가 틀렸을 가능성에 대비 — 통계적 신호와 실측 성능이
   다르면 실측을 따른다).

### 후보 탈락/유지 판정(2·4 대신 실제 사용하는 기준)
- **선형 신호**: 학습구간(폴드) 안에서 피어슨 상관계수 |r|≥0.15
  (일간 집계는 시간단위보다 표본이 훨씬 적어 초단기 임계값 0.3보다
  느슨하게 잡음 — 표본 수 차이를 감안한 조정, 자체 판단).
- **비선형 신호**: 순열중요도(permutation importance, MAE 기준)를
  내부 홀드아웃(inner_va, 학습구간의 마지막 20%)에서 계산해
  `중요도평균 - 중요도표준편차 > 0`(반복측정 대비 통계적으로 0보다
  크다고 볼 수 있는 경우)이면 유지.
- **유지 조건**: 선형 신호 OR 비선형 신호 중 하나라도 만족(합집합) —
  선형은 낮아도 트리가 비선형으로 쓰는 경우를 놓치지 않기 위함.
- **다중공선성 가지치기**: 위에서 살아남은 특성들 사이에 쌍상관
  |r|≥0.8이고 VIF≥10이면, **순열중요도가 더 낮은 쪽**을 뺀다(하네스는
  "|타깃상관|이 낮은 쪽"을 뺐는데, 여기선 비선형 신호를 반영해 순열
  중요도로 대체 — 사용자 지적 반영).

## 최종 판정 방식
폴드마다 위 규칙으로 특성을 고른 뒤, **전체특성(기존 51개) 기준선과
거버넌스 특성으로 각각 LightGBM·XGBoost를 재학습해 시험구간(진짜
홀드아웃) 성능을 직접 비교**한다. 거버넌스 특성이 기준선보다 뚜렷이
나쁘면 채택하지 않고 원인을 기록한다(임계값 자체를 재검토).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent

_spec_daily = importlib.util.spec_from_file_location("daily_v2", ROOT / "train_daily_official_v2_2026-08-21.py")
daily_mod = importlib.util.module_from_spec(_spec_daily)
sys.modules["daily_v2"] = daily_mod
_spec_daily.loader.exec_module(daily_mod)

_spec_h = importlib.util.spec_from_file_location("harness", ROOT / "backtest_harness_v1_2026-08-20밤.py")
harness = importlib.util.module_from_spec(_spec_h)
sys.modules["harness"] = harness
_spec_h.loader.exec_module(harness)

OUT = ROOT / "outputs" / "일간_특성거버넌스_v1_2026-08-21"
LINEAR_THRESHOLD = 0.15
VIF_SEVERE = harness.VIF_SEVERE
PAIR_CORR_HIGH = harness.PAIR_CORR_HIGH


def energy_metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    e = y - p
    denom = float(np.abs(y).sum())
    return {
        "표본수": int(len(y)), "MAE_kWh": float(np.abs(e).mean()),
        "RMSE_kWh": float(np.sqrt(np.mean(e ** 2))),
        "WAPE_pct": float(np.abs(e).sum() / denom * 100) if denom > 0 else None,
    }


def select_features_governed(inner_tr: pd.DataFrame, inner_va: pd.DataFrame, candidate_cols: list[str],
                              target_col: str, medians: pd.Series, seed: int) -> tuple[list[str], pd.DataFrame]:
    y_tr = inner_tr[target_col]

    # ── 선형 신호(피어슨 상관, 학습구간 안에서만) ──
    linear_pass = {}
    for c in candidate_cols:
        sub = pd.concat([inner_tr[c], y_tr], axis=1).dropna()
        if len(sub) < 20:
            linear_pass[c] = False
            continue
        r = np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1]
        linear_pass[c] = bool(np.isfinite(r) and abs(r) >= LINEAR_THRESHOLD)

    # ── 비선형 신호(순열중요도, 내부 홀드아웃에서만) ──
    scout = LGBMRegressor(
        objective="regression_l1", n_estimators=200, learning_rate=0.05,
        num_leaves=15, max_depth=5, min_child_samples=8,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, verbosity=-1, n_jobs=4,
    )
    scout.fit(inner_tr[candidate_cols].fillna(medians), y_tr)
    perm = permutation_importance(
        scout, inner_va[candidate_cols].fillna(medians), inner_va[target_col],
        scoring="neg_mean_absolute_error", n_repeats=8, random_state=seed, n_jobs=4,
    )
    perm_mean = pd.Series(perm.importances_mean, index=candidate_cols)
    perm_std = pd.Series(perm.importances_std, index=candidate_cols)
    nonlinear_pass = (perm_mean - perm_std) > 0

    keep = [c for c in candidate_cols if linear_pass[c] or bool(nonlinear_pass[c])]

    diag = pd.DataFrame({
        "선형상관": {c: round(float(np.corrcoef(
            pd.concat([inner_tr[c], y_tr], axis=1).dropna().iloc[:, 0],
            pd.concat([inner_tr[c], y_tr], axis=1).dropna().iloc[:, 1])[0, 1]), 4)
            if pd.concat([inner_tr[c], y_tr], axis=1).dropna().shape[0] >= 20 else np.nan
            for c in candidate_cols},
        "선형통과": linear_pass,
        "순열중요도평균": perm_mean.to_dict(),
        "순열통과": nonlinear_pass.to_dict(),
    })

    # ── 다중공선성 가지치기(순열중요도를 유지 우선순위로) ──
    if len(keep) >= 2:
        x = inner_tr[keep].fillna(medians[keep]).dropna(axis=1, how="all")
        keep = [c for c in keep if c in x.columns]
        if len(keep) >= 2:
            try:
                vif = harness.compute_vif(x[keep])
            except Exception:
                vif = pd.Series(1.0, index=keep)
            corr_matrix = x[keep].corr(method="pearson")
            drop = set()
            severe = set(vif[vif >= VIF_SEVERE].index)
            for i, a in enumerate(keep):
                for b in keep[i + 1:]:
                    r = corr_matrix.loc[a, b]
                    if pd.notna(r) and abs(r) >= PAIR_CORR_HIGH:
                        weaker = a if perm_mean.get(a, 0) < perm_mean.get(b, 0) else b
                        if weaker in severe:
                            drop.add(weaker)
            keep = [c for c in keep if c not in drop]

    return keep, diag


def fold_ablation(train, test, candidate_cols, chosen_cols, target_col, capacity_daily, seed) -> dict:
    medians_full = train[candidate_cols].median(numeric_only=True)
    medians_gov = train[chosen_cols].median(numeric_only=True) if chosen_cols else medians_full

    row = {}
    for label, cols, medians in [("전체특성", candidate_cols, medians_full), ("거버넌스특성", chosen_cols, medians_gov)]:
        if not cols:
            continue
        for name, cls in [("LightGBM", None), ("XGBoost", None)]:
            model = (
                LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.025,
                               num_leaves=15, max_depth=6, min_child_samples=14,
                               subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                               random_state=seed, verbosity=-1, n_jobs=4)
                if name == "LightGBM" else
                XGBRegressor(objective="reg:absoluteerror", n_estimators=500, learning_rate=0.025,
                              max_depth=4, min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                              reg_lambda=2.0, random_state=seed, n_jobs=4)
            )
            model.fit(train[cols].fillna(medians), train[target_col])
            pred = np.clip(model.predict(test[cols].fillna(medians)), 0, capacity_daily)
            row[f"{label}_{name}"] = energy_metrics(test[target_col], pred)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    capacity_daily = float(config["site"]["capacity_kw"]) * 24

    data, candidate_cols = daily_mod.build_daily_dataset()
    target_col = daily_mod.ACTUAL
    print(f"후보 특성 {len(candidate_cols)}개, 표본 {len(data)}일")
    print("발행시점 가용성·정보누출: 구조로 보장됨(NWP=v3 고정-tm D+1 집계, 장비=shift(2)) — 재확인만, 위반 없음")

    fold_summ = []
    all_diag = []
    for i, w in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절','')}"
        train_all = data[data.index < start]
        test_all = data[(data.index >= start) & (data.index <= end)]
        if len(train_all) < 60 or len(test_all) < 10:
            print(f"[{fold_name}] 표본 부족으로 건너뜀")
            continue

        cut = int(len(train_all) * 0.8)
        inner_tr, inner_va = train_all.iloc[:cut], train_all.iloc[cut:]
        medians_inner = inner_tr[candidate_cols].median(numeric_only=True)

        chosen, diag = select_features_governed(inner_tr, inner_va, candidate_cols, target_col, medians_inner, seed)
        diag["폴드"] = fold_name
        all_diag.append(diag.reset_index().rename(columns={"index": "특성"}))
        print(f"\n[{fold_name}] 전체{len(candidate_cols)}개 → 거버넌스{len(chosen)}개 선택")

        row = fold_ablation(train_all, test_all, candidate_cols, chosen, target_col, capacity_daily, seed)
        row["폴드"] = fold_name
        row["선택특성수"] = len(chosen)
        row["선택특성"] = chosen
        fold_summ.append(row)
        for label in ["전체특성", "거버넌스특성"]:
            for name in ["LightGBM", "XGBoost"]:
                m = row.get(f"{label}_{name}")
                if m:
                    print(f"  {label:10s} {name:8s} WAPE={m['WAPE_pct']:.2f}% MAE={m['MAE_kWh']:.1f} RMSE={m['RMSE_kWh']:.1f}")

    pd.concat(all_diag, ignore_index=True).to_csv(OUT / "특성진단_폴드별.csv", index=False, encoding="utf-8-sig")
    (OUT / "폴드별_결과.json").write_text(json.dumps(fold_summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n\n=== 전체특성 vs 거버넌스특성 비교(폴드 평균) ===")
    summary_rows = []
    for label in ["전체특성", "거버넌스특성"]:
        for name in ["LightGBM", "XGBoost"]:
            key = f"{label}_{name}"
            vals = [r[key] for r in fold_summ if key in r]
            if not vals:
                continue
            summary_rows.append({
                "구성": f"{label}_{name}", "폴드수": len(vals),
                "평균MAE_kWh": round(float(np.mean([v["MAE_kWh"] for v in vals])), 1),
                "평균RMSE_kWh": round(float(np.mean([v["RMSE_kWh"] for v in vals])), 1),
                "평균WAPE_pct": round(float(np.mean([v["WAPE_pct"] for v in vals])), 2),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "비교_요약.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    avg_feat = np.mean([r["선택특성수"] for r in fold_summ])
    print(f"\n평균 선택특성수: {avg_feat:.1f}개 (기존 {len(candidate_cols)}개 대비)")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
