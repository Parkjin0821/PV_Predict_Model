# -*- coding: utf-8 -*-
"""부안 요인 재검증 - 광주 v6 방법론(사전 Pearson필터 없이 전체후보 →
permutation importance)을 부안에 적용.

## 배경
사용자 09-03 지시("②로 넘어가서 부안·김제·영광 요인 재검증 시작하자").
광주 09-03 6차 절에서 "Pearson 선형필터가 시간처럼 비선형/순환적으로
중요한 변수를 기계적으로 탈락시킨다"는 사용자 지적이 실측으로 증명돼
표준 방법론이 됨(permutation importance로 재선정, 결과 채택 확정).
부안은 지금 `total_output_weather_model_v5_방법론정정_2026-08-31.py`가
쓰는 FEATURES_15/19가 **처음부터 좁게 정해진 후보**(WEATHER8+ASOS4만,
사용 가능한 ASOS 12종·NWP 10종 중 일부만)라 광주처럼 "48개 전체 후보를
필터 없이 넣고 재선정"하는 과정 자체를 거친 적이 없다 - 이번이 처음.

## 재사용한 것 (재구현 안 함)
`total_output_weather_model_v5_방법론정정_2026-08-31.py`의
`build_dataset()`(결함구간 제외·품질필터·순환특성·1일전지속성 lag
전부 그대로)·`expanding_folds_full_coverage()`·`run_walkforward()`·
`pooled_score()`·`TARGET`·`INITIAL_TRAIN_DAYS`/`TEST_BLOCK_DAYS`를
그대로 import해서 쓴다. 새로 만드는 건 (1) 후보를 넓힌 특성프레임과
(2) permutation importance 재선정 로직뿐 - 광주 v6
(select_by_importance_v6_2026-09-03.py)와 동일한 절차.

## 확장한 후보(v5 대비 새로 추가)
- ASOS: 기존 4종(ASOS4)에서 **사용 가능한 12종 전부**로 확대(강수량·
  적설은 결측 많지만 LightGBM native missing으로 처리, 일사량 2종은
  100% 결측이라 원천 제외 - 켤 수 없는 스위치를 후보에 넣지 않음).
- NWP: 기존 8종(WEATHER8)에 DSWRFLX·DIFSWRF 추가(결측률 75%/68%로
  높지만 native missing+결측여부 플래그로 후보에는 포함 - 미리
  판단하지 않고 permutation importance가 직접 평가하게 둠, 이게 이번
  재검증의 핵심 취지).
- Lag/baseline: 기존 1개(1일전 동시각)에서 광주 baseline과 동일하게
  1·2·3·6·24시간전 + 6·24시간 이동평균·표준편차(총 9개)로 확대 -
  target_time_kst 기준 lookup으로 계산(미래 값 절대 안 씀).

## 절차(광주 v6와 동일)
1. 확장 후보 전부 + baseline 9개를 사전필터 없이 LightGBM 1회 학습
   (85/15 시간순 분할, 광주 v6와 동일).
2. permutation_importance(20회 반복, MAE 기준)로 재선정, 중요도>0만 채택.
3. 최종 선택셋을 **부안 자체 rolling-origin CV**(walk-forward, 기존
   평가 관례)로 재평가해 기존 FEATURES_15(pooled MAE 46.37kW·RMSE
   90.57kW, 08-31 방법론정정판 확정치)와 공정 비교.
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

HERE = Path(__file__).resolve().parent
V5_SCRIPT = HERE / "total_output_weather_model_v5_방법론정정_2026-08-31.py"
OUT_DIR = HERE / "outputs" / "요인재검증_v6_permutation_2026-09-03"
SEED = 42

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ASOS 12종(사용가능, 일사량 2종은 100%결측이라 원천제외) - issue시각 기준
ASOS_FULL = [
    "issue_asos_기온_C", "issue_asos_강수량_mm", "issue_asos_풍속_m_s",
    "issue_asos_풍향_deg", "issue_asos_상대습도_pct", "issue_asos_현지기압_hPa",
    "issue_asos_해면기압_hPa", "issue_asos_일조시간_hr", "issue_asos_적설_cm",
    "issue_asos_전운량_10분위", "issue_asos_전운량_pct", "issue_asos_지면온도_C",
]
NWP_FULL = ["forecast_DSWRF", "forecast_DSWRFLX", "forecast_DIFSWRF",
            "forecast_TCDC", "forecast_LCDC", "forecast_MCDC", "forecast_HCDC",
            "forecast_REH", "forecast_POP", "forecast_SKY"]
NATIVE_MISSING_OK = {"issue_asos_강수량_mm", "issue_asos_적설_cm",
                      "forecast_DSWRFLX", "forecast_DIFSWRF"}  # 결측률 높음 - LightGBM native로 처리, dropna 대상 제외
PHYSICAL = ["solar_elevation_deg", "physical_daylight"]
CYCLICAL = ["target_hour_sin", "target_hour_cos", "doy_sin", "doy_cos"]
# ★09-03 실측 발견★: 부안 결합데이터는 매시간이 아니라 **3시간 격자**
# (00/03/06/09/12/15/18/21 KST)다 - 광주와 다름. 처음에 광주식 lag
# [1,2,3,6,24]를 그대로 썼다가 1·2시간전이 격자에 존재하지 않는 시점이라
# 100% 결측이 나는 버그를 실측(디버그)으로 잡았다. 3의 배수만 실제
# 격자점과 만난다.
LAG_HOURS = [3, 6, 9, 24]
ROLL_HOURS = [6, 24]  # 시간기준 윈도우(아래 rolling('Nh'))로 계산 - 격자 간격과 무관하게 항상 정확


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_expanded_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """확장 baseline(lag/rolling) 추가 - target_time_kst 기준 lookup만 사용,
    미래 정보 없음(1일전지속성 lag와 동일한 lookup 패턴 재사용)."""
    out = df.copy()
    lookup = df.drop_duplicates("target_time_kst").set_index("target_time_kst")[target_col].sort_index()
    baseline_cols = []
    for h in LAG_HOURS:
        col = f"발전출력_{h}시간전_kW"
        query = out["target_time_kst"] - pd.Timedelta(hours=h)
        out[col] = query.map(lookup)
        baseline_cols.append(col)
    for h in ROLL_HOURS:
        # ★시간기준 윈도우★('{h}h') - 격자가 3시간이든 1시간이든 항상
        # "실제 h시간 구간"을 정확히 잡는다(정수 윈도우였다면 3시간격자
        # 에선 h행=3h시간이 돼버려 의미가 달라짐 - 이 버그도 같이 잡음).
        roll_mean = lookup.rolling(f"{h}h", min_periods=2).mean()
        roll_std = lookup.rolling(f"{h}h", min_periods=2).std()
        mean_col, std_col = f"발전출력_{h}시간이동평균_kW", f"발전출력_{h}시간이동표준편차_kW"
        query_lag = out["target_time_kst"] - pd.Timedelta(hours=LAG_HOURS[0])
        out[mean_col] = query_lag.map(roll_mean)
        out[std_col] = query_lag.map(roll_std)
        baseline_cols += [mean_col, std_col]
    # 결측 많은 NWP 2종은 native missing과 별개로 결측여부 플래그도 추가(관례)
    for c in ["forecast_DSWRFLX", "forecast_DIFSWRF"]:
        out[f"{c}_결측여부"] = out[c].isna().astype(float)
    return out, baseline_cols


def run_walkforward_native_missing(v5_mod, df: pd.DataFrame, folds: list[tuple],
                                    features: list[str], model_kind: str) -> dict:
    """v5.run_walkforward과 100% 동일 로직이나, NATIVE_MISSING_OK 컬럼은
    dropna 요구에서 제외(LightGBM native missing 그대로 활용) - 결측률
    높은 DSWRFLX·DIFSWRF·강수량·적설이 선택됐을 때 표본이 거의 다
    날아가는 걸 막는다. 그 외(폴드 구성·모델·지속성 대조·pooled 집계)는
    재구현 없이 v5 그대로."""
    required = [c for c in features if c not in NATIVE_MISSING_OK]
    all_true, all_pred, all_pers = [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤에 있음"
        train = df[df["issue_day"].isin(train_days)].dropna(subset=required + [v5_mod.TARGET])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=required + [v5_mod.TARGET])
        if len(train) < 50 or len(test) < 1:
            continue
        if model_kind == "lightgbm":
            model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                                  max_depth=4, min_child_samples=15, subsample=0.9,
                                  colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=SEED, n_jobs=-1, verbosity=-1)
        else:
            raise ValueError(model_kind)
        model.fit(train[features], train[v5_mod.TARGET])
        pred = np.clip(model.predict(test[features]), 0, None)
        y_true = test[v5_mod.TARGET].to_numpy()
        pers = test["lag_1day_same_slot_kw"].to_numpy()
        all_true.append(y_true); all_pred.append(pred); all_pers.append(pers)
        fold_rows.append({"폴드": i, "시험발행일수": len(test_days), "시험행수": len(test),
                          **{f"모델_{k}": v for k, v in v5_mod.pooled_score(y_true, pred).items()
                             if k != "n"}})
    y_true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    pers_all = np.concatenate(all_pers)
    return {"폴드수": len(fold_rows), "폴드별": fold_rows,
            "pooled_모델": v5_mod.pooled_score(y_true_all, pred_all),
            "pooled_지속성": v5_mod.pooled_score(y_true_all, pers_all)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v5 = _load_module("buan_v5_20260903", V5_SCRIPT)
    df, removed_targets = v5.build_dataset()

    frame, baseline_cols = build_expanded_frame(df, v5.TARGET)
    candidate_cols = (ASOS_FULL + NWP_FULL + PHYSICAL + CYCLICAL +
                      ["forecast_DSWRFLX_결측여부", "forecast_DIFSWRF_결측여부"])
    all_cols = baseline_cols + candidate_cols
    daylight = frame[frame["physical_daylight"] == 1].dropna(subset=[v5.TARGET])

    print(f"[0] 후보 {len(candidate_cols)}개(+baseline {len(baseline_cols)}개) "
          f"= 총 {len(all_cols)}개, 사전 선형필터 없이 전부 사용")

    required_cols = [c for c in all_cols if c not in NATIVE_MISSING_OK]
    f = daylight.dropna(subset=required_cols + [v5.TARGET]).sort_values("target_time_kst")
    n = len(f)
    cut = int(n * 0.85)
    train, test = f.iloc[:cut], f.iloc[cut:]
    print(f"[1] importance계산용 분할 - 학습 {len(train)}행 · 시험 {len(test)}행 · 특성 {len(all_cols)}개")

    model = LGBMRegressor(n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=SEED,
                          n_jobs=4, verbosity=-1)
    model.fit(train[all_cols], train[v5.TARGET])

    print("[2] permutation importance 계산 중(20회 반복, MAE 기준)...")
    perm = permutation_importance(
        model, test[all_cols], test[v5.TARGET],
        scoring="neg_mean_absolute_error", n_repeats=20, random_state=SEED, n_jobs=4,
    )
    imp_df = pd.DataFrame({
        "변수": all_cols,
        "중요도_MAE증가량": perm.importances_mean,
        "표준편차": perm.importances_std,
        "분류": ["과거발전량(baseline)" if c in baseline_cols else
                ("NWP예보" if c.startswith("forecast_") else
                 ("ASOS실측" if c.startswith("issue_asos_") else
                  ("순환시간" if c in CYCLICAL else "물리량")))
                for c in all_cols],
    }).sort_values("중요도_MAE증가량", ascending=False)
    imp_df.to_csv(OUT_DIR / "특성중요도_permutation.csv", index=False, encoding="utf-8-sig")
    print(imp_df.to_string(index=False))

    # 광주 v6와 동일 원칙: baseline은 유지, 나머지 후보만 중요도>0인 것 채택
    selected_candidates = [
        row["변수"] for _, row in imp_df.iterrows()
        if row["변수"] in candidate_cols and row["중요도_MAE증가량"] > 0
    ]
    dropped = [c for c in candidate_cols if c not in selected_candidates]
    final_features = baseline_cols + selected_candidates
    print(f"\n[3] 후보 {len(candidate_cols)}개 중 중요도>0 채택 {len(selected_candidates)}개, "
          f"가지치기 {len(dropped)}개: {dropped}")

    # [4] 기존 v5 walk-forward CV 관례로 공정 재평가(lag_1day_same_slot_kw는
    # 지속성 대조 계산에 필요하므로 원본 df에 이미 있음, 여기 frame에도 유지)
    final_features_with_persistence = list(dict.fromkeys(
        final_features + ["lag_1day_same_slot_kw"]))
    folds = v5.expanding_folds_full_coverage(
        pd.DatetimeIndex(np.sort(frame["issue_day"].unique())),
        v5.INITIAL_TRAIN_DAYS, v5.TEST_BLOCK_DAYS)

    result_new = run_walkforward_native_missing(
        v5, frame, folds, final_features_with_persistence, "lightgbm")
    result_old15 = run_walkforward_native_missing(v5, frame, folds, v5.FEATURES_15, "lightgbm")

    def improve(model_mae, pers_mae):
        return round((1 - model_mae / pers_mae) * 100, 1) if pers_mae else None

    summary = {
        "v6_importance기반": {
            **result_new,
            "특성수": len(final_features_with_persistence),
            "pooled_MAE_개선율_pct_vs지속성": improve(
                result_new["pooled_모델"]["MAE_kW"], result_new["pooled_지속성"]["MAE_kW"]),
        },
        "기존_v5_FEATURES_15": {
            **result_old15,
            "특성수": len(v5.FEATURES_15),
            "pooled_MAE_개선율_pct_vs지속성": improve(
                result_old15["pooled_모델"]["MAE_kW"], result_old15["pooled_지속성"]["MAE_kW"]),
        },
    }
    (OUT_DIR / "요약.json").write_text(json.dumps({
        "선택_baseline": baseline_cols, "선택_후보": selected_candidates,
        "가지치기": dropped, "성능비교": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== [4] 기존 walk-forward CV로 공정 재평가 ===")
    print(f"v6(importance기반, {len(final_features_with_persistence)}특성): "
          f"pooled MAE={result_new['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_new['pooled_모델']['RMSE_kW']}kW")
    print(f"기존 v5(FEATURES_15, {len(v5.FEATURES_15)}특성): "
          f"pooled MAE={result_old15['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_old15['pooled_모델']['RMSE_kW']}kW")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
