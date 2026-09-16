# -*- coding: utf-8 -*-
"""광주·영광 초단기 - GHI(obs_ghi_wm2) 필요성 상관분석·다중공선성 규명(09-15).

## 배경
09-15 GHI제거 ablation에서 "GHI를 빼도 MAE가 거의 같거나 오히려 개선"이
나왔는데, 사용자 지적("상관분석을 통해서 초단기 검증에서 얘가 필요가
없는지 확실하게 알아야 한다")대로 **MAE 결과만으로 결론내지 않고 왜
불필요한지를 상관구조로 직접 규명**한다.

## 분석 항목(프로젝트 기존 방법론 재사용 - 부안 08-31/김제 09-01
`check_correlation_multicollinearity_*` 계열과 동일 기준)
1. 타깃 단변량 상관(|r|) - 폴드내부(leakage 없음)
2. VIF(분산팽창계수) - GHI가 다른 피처들로 얼마나 설명되는지
3. GHI ~ 나머지 피처 회귀 R² - **핵심 지표**(1에 가까울수록 GHI 정보가
   이미 다른 피처에 다 들어있다는 직접 증거)
4. 부분상관(partial correlation) - 나머지 피처를 통제한 뒤 GHI가
   타깃과 여전히 독립적 관련이 있는지(0에 가까우면 고유정보 없음)
5. 쌍상관 상위(GHI와 가장 강하게 겹치는 피처가 무엇인지)

모든 계산은 폴드 학습구간 내부에서만 수행(시험구간 정보 미사용).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
RV_SCRIPT = ROOT / "revalidate_all_horizons_issue_safe_v1_2026-09-08.py"
GWANGJU_DIR = ROOT / "광주_준비_2026-09-08"
YEONGGWANG_DIR = ROOT / "영광_준비_2026-09-03"
OUT_DIR = ROOT / "outputs" / "GHI_필요성_상관분석_v1_2026-09-15"

TARGET_FEATURE = "obs_ghi_wm2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def vif_for(x: pd.DataFrame, col: str) -> float:
    others = [c for c in x.columns if c != col]
    if not others:
        return float("nan")
    model = LinearRegression().fit(x[others], x[col])
    r2 = model.score(x[others], x[col])
    return float("inf") if r2 >= 1.0 else round(1.0 / (1.0 - r2), 2), round(r2, 4)


def partial_corr(x: pd.DataFrame, feature: str, y: pd.Series) -> float:
    """나머지 피처를 통제한 뒤 feature와 y의 상관(잔차 상관법)."""
    others = [c for c in x.columns if c != feature]
    if not others:
        return float(x[feature].corr(y))
    res_f = x[feature] - LinearRegression().fit(x[others], x[feature]).predict(x[others])
    res_y = y - LinearRegression().fit(x[others], y).predict(x[others])
    return round(float(np.corrcoef(res_f, res_y)[0, 1]), 4)


def analyze_region(region: str, module_name: str, module_path: Path, rv) -> dict:
    m = load_module(module_name, module_path)
    out = {}
    for h in m.LEAD_HOURS:
        d, target, features = rv.prepare_mod_frame(m, h)
        days = pd.DatetimeIndex(np.sort(d.issue_day.unique()))
        folds = rv.folds(days, m.INITIAL_TRAIN_DAYS, m.TEST_BLOCK_DAYS)

        fold_rows = []
        for i, (trd, _ted) in enumerate(folds, start=1):
            train = d[d.issue_day.isin(trd)].dropna(subset=features + [target])
            if len(train) < m.MIN_ROWS_PER_FOLD:
                continue
            x = train[features]
            y = train[target]
            if TARGET_FEATURE not in x.columns:
                continue
            vif, r2_from_others = vif_for(x, TARGET_FEATURE)
            pc = partial_corr(x, TARGET_FEATURE, y)
            pair = (x.corr()[TARGET_FEATURE].drop(TARGET_FEATURE)
                    .abs().sort_values(ascending=False).head(4).round(3).to_dict())
            fold_rows.append({
                "폴드": i, "학습행수": len(train),
                "GHI_타깃상관": round(float(x[TARGET_FEATURE].corr(y)), 3),
                "GHI_VIF": vif,
                "GHI_나머지피처로_설명된_R2": r2_from_others,
                "GHI_부분상관(나머지통제후)": pc,
                "GHI와_쌍상관_상위": pair,
            })

        if fold_rows:
            arr_r = np.array([r["GHI_타깃상관"] for r in fold_rows])
            arr_r2 = np.array([r["GHI_나머지피처로_설명된_R2"] for r in fold_rows])
            arr_pc = np.array([r["GHI_부분상관(나머지통제후)"] for r in fold_rows])
            arr_vif = np.array([r["GHI_VIF"] for r in fold_rows if np.isfinite(r["GHI_VIF"])])
            out[f"+{h}h"] = {
                "폴드수": len(fold_rows),
                "요약": {
                    "GHI_타깃상관_평균": round(float(arr_r.mean()), 3),
                    "GHI_나머지피처로_설명된_R2_평균": round(float(arr_r2.mean()), 4),
                    "GHI_부분상관_평균": round(float(arr_pc.mean()), 4),
                    "GHI_VIF_평균": round(float(arr_vif.mean()), 2) if len(arr_vif) else None,
                },
                "폴드별": fold_rows,
            }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rv = load_module("revalidate_all_horizons", RV_SCRIPT)

    result = {
        "_분석목적": "GHI(obs_ghi_wm2)가 초단기 모델에서 고유정보를 갖는지, 아니면 "
                  "다른 피처(태양고도·발전량lag·운량 등)에 이미 포함된 중복정보인지 규명",
        "_판정기준": {
            "중복_판정": "GHI_나머지피처로_설명된_R2가 높고(>=0.7) 부분상관이 0에 가까우면(<0.2) "
                      "고유정보 없음 = 제거해도 무방",
            "필요_판정": "부분상관이 유의하게 크면(>=0.3) 다른 피처로 대체 불가 = 유지 필요",
        },
        "광주": analyze_region("광주", "gwangju_corr", ROOT / "ultra_short_term_v1_gwangju_2026-09-08.py", rv),
        "영광": analyze_region("영광", "yeonggwang_corr",
                             YEONGGWANG_DIR / "ultra_short_term_v1_yeonggwang_2026-09-07.py", rv),
    }
    (OUT_DIR / "GHI_필요성_상관분석_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약만 출력
    summary = {r: {h: v["요약"] for h, v in result[r].items()} for r in ("광주", "영광")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
