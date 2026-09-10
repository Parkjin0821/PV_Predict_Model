# -*- coding: utf-8 -*-
"""영광 초단기(+1h~+4h) v3(09-07) - 날씨상태(regime) 분류 후 그룹별 모델
적용, 국내 학술근거 기반 검증.

## 배경(★국내 학술 근거 확인 후 착수, 사용자 확정: "그런 방식으로
가자" → "국내 학술 근거에 따라서 검증까지만 해볼래"★)
- 김백천·정승환·김민석·김종근·김성신(2021), "계절별 기상조건에 기반한
  태양광 발전량 예측에 관한 연구", *한국지능시스템학회 논문지* 31(2),
  102-108 - 계절·날씨로 데이터를 먼저 분류하고 그룹별로 모델을 따로
  적용하는 방식이 미분류 단일모델보다 우수.
- 김상진·유재혁·장병훈·우성민(2022), "머신러닝 기반의 예측 시장 참여를
  위한 태양광 발전량 예측 알고리즘 및 수익성에 관한 연구",
  *한국태양에너지학회 논문집* 42(6), 173-183 - 일사량 기준 예보데이터
  군집화(날씨상태 구분) 활용.

## 무엇을 검증하나
v1(`ultra_short_term_v1_yeonggwang_2026-09-07.py`)의 단일모델 대비,
**현재 시점(kt_now, 예측 시점에 이미 알고 있는 정보 - leakage 없음)의
청천지수로 "맑음/보통/흐림" 3개 규드로 분류한 뒤 그룹별로 별도
LightGBM을 학습**하는 게 nMAE를 줄이는지 직접 비교한다. 09-07 사후분석
(폴드1 상관대조)에서 영광은 구름상관이 김제보다 오히려 강했던 점
(obs_cloud_pct: -0.481 vs -0.373)에 착안 - "날씨상태별로 나누면 각
그룹 내에서는 더 예측이 쉬워질 것"이라는 가설.

## 방법(leakage 없음)
- regime 분류 기준: kt_now(현재 관측 kW ÷ Haurwitz 청천전력, v1에서
  이미 계산됨) - 예측 시점에 이미 알고 있는 값만 사용, 목표시각 정보
  전혀 안 씀.
- 맑음: kt_now>=0.7, 보통: 0.3<=kt_now<0.7, 흐림: kt_now<0.3
- 폴드별로 구조(structure)는 v1과 동일하게 "전체 학습데이터 기준"으로
  1번만 선택(내부 80/20 홀드아웃, 시험폴드 미접촉) - 그 구조를 그대로
  3개 regime 모델에 재사용(하이퍼파라미터 탐색을 3배로 늘리지 않음).
- regime별 학습표본이 200행 미만이면 그 regime은 "전체 학습데이터"로
  대체 학습(fallback) - 부안/영광 인버터분해 B방법의 fallback 관례와
  동일한 원칙.
- 시험행 예측은 그 행의 "자기 kt_now"로 정해진 regime의 모델을 사용.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V1_SPEC = importlib.util.spec_from_file_location(
    "ultra_short_yg_v1", HERE / "ultra_short_term_v1_yeonggwang_2026-09-07.py")
V1 = importlib.util.module_from_spec(V1_SPEC)
V1_SPEC.loader.exec_module(V1)  # type: ignore

OUT_DIR = HERE / "outputs" / "영광_초단기_v3_weather_regime_2026-09-07"
MIN_REGIME_TRAIN = 200


def assign_regime(kt: np.ndarray) -> np.ndarray:
    regime = np.full(kt.shape, "보통", dtype=object)
    regime[kt >= 0.7] = "맑음"
    regime[kt < 0.3] = "흐림"
    regime[np.isnan(kt)] = "보통"  # 정보 없으면 중간값 취급(안전한 기본값)
    return regime


def run_walkforward_regime(d: pd.DataFrame, target_col: str, features: list[str], folds: list[tuple]) -> dict:
    rng = np.random.RandomState(V1.SEED)
    all_true, all_model, all_baseline = [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출"
        need = features + [target_col, "power_lag_0min", "kt_now", "clearsky_power_target_kw"]
        train = d[d["day"].isin(train_days)].dropna(subset=features + [target_col])
        test = d[d["day"].isin(test_days)].dropna(subset=need)
        if len(train) < V1.MIN_ROWS_PER_FOLD or len(test) < 1:
            continue

        chosen = V1.select_structure_internal_holdout(train_days, d, target_col, features, rng)
        train_regime = assign_regime(train["kt_now"].to_numpy())
        test_regime = assign_regime(test["kt_now"].to_numpy())

        models, regime_train_n = {}, {}
        for r in ("맑음", "보통", "흐림"):
            sub = train[train_regime == r]
            used_fallback = len(sub) < MIN_REGIME_TRAIN
            fit_data = train if used_fallback else sub
            m = V1.LGBMRegressor(**V1.STRUCTURES[chosen], random_state=V1.SEED, n_jobs=-1, verbosity=-1)
            m.fit(fit_data[features], fit_data[target_col])
            models[r] = m
            regime_train_n[r] = {"표본수": int(len(sub)), "fallback_적용": bool(used_fallback)}

        pred_model = np.zeros(len(test))
        for r in ("맑음", "보통", "흐림"):
            mask = test_regime == r
            if mask.any():
                pred_model[mask] = np.clip(models[r].predict(test.loc[mask, features]), 0, None)

        y_true = test[target_col].to_numpy()
        pred_baseline = np.clip(test["kt_now"].to_numpy() * test["clearsky_power_target_kw"].to_numpy(), 0, None)

        all_true.append(y_true); all_model.append(pred_model); all_baseline.append(pred_baseline)
        fold_rows.append({
            "폴드": i, "학습일수": len(train_days), "시험일수": len(test_days),
            "선택된구조": chosen, "regime_학습표본": regime_train_n,
            "모델_MAE": V1.pooled_score(y_true, pred_model)["MAE_kW"],
            "스마트지속성_MAE": V1.pooled_score(y_true, pred_baseline)["MAE_kW"],
        })

    if not all_true:
        return {"폴드수": 0, "폴드별": [], "오류": "유효 폴드 없음"}
    y_true_all = np.concatenate(all_true)
    return {
        "폴드수": len(fold_rows), "폴드별": fold_rows,
        "pooled_모델": V1.pooled_score(y_true_all, np.concatenate(all_model)),
        "pooled_스마트지속성": V1.pooled_score(y_true_all, np.concatenate(all_baseline)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, meta = V1.load_base()

    def improve(model_mae, base_mae):
        return round((1 - model_mae / base_mae) * 100, 1) if base_mae else None

    horizon_results = {}
    for h in V1.LEAD_HOURS:
        d, target_col, features = V1.build_horizon_frame(base, h)
        days = pd.DatetimeIndex(np.sort(d["day"].unique()))
        folds = V1.expanding_folds_full_coverage(days, V1.INITIAL_TRAIN_DAYS, V1.TEST_BLOCK_DAYS)
        perf = run_walkforward_regime(d, target_col, features, folds)
        entry = {"리드타임": f"+{h}h", "일수_total": int(len(days)), "성능": perf}
        if perf.get("폴드수", 0) > 0:
            entry["성능"]["MAE_개선율_vs스마트지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_스마트지속성"]["MAE_kW"])
            entry["nMAE_pct"] = round(perf["pooled_모델"]["MAE_kW"] / V1.CAPACITY_KW * 100, 3)
        horizon_results[f"+{h}h"] = entry
        print(f"[+{h}h] v3(regime) 완료 - pooled MAE(모델)={perf.get('pooled_모델',{}).get('MAE_kW')}kW, "
              f"nMAE={entry.get('nMAE_pct')}%")

    result = {
        **meta, "capacity_kw_사용값": V1.CAPACITY_KW,
        "특징": "kt_now(현재 청천지수, leakage 없음) 기준 맑음/보통/흐림 3규드로 분류 후 그룹별 별도 LightGBM 학습. "
               "구조(하이퍼파라미터)는 v1과 동일하게 전체데이터 기준 1회만 선택해 재사용.",
        "regime_분류기준": {"맑음": "kt_now>=0.7", "보통": "0.3<=kt_now<0.7", "흐림": "kt_now<0.3"},
        "regime_학습표본_최소기준": MIN_REGIME_TRAIN,
        "평가대상": "각 리드타임의 대상시각 태양고도>0인 행만",
        "리드타임별_결과": horizon_results,
        "_방법론출처": "김백천 외(2021, 한국지능시스템학회 31(2)) 계절·날씨 분류+그룹별모델, "
                     "김상진 외(2022, 한국태양에너지학회 42(6)) 일사량기준 군집화 - 국내 학술근거 기반 검증.",
        "_판정": "잠정치 - v1(단일모델) 대비 regime분리모델의 개선효과 검증 목적. promote_to_official 대상 아님.",
    }
    (OUT_DIR / "영광_초단기_v3_regime_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "리드타임별_결과"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
