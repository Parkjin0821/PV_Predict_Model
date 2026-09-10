# -*- coding: utf-8 -*-
"""김제 요인 재검증 - 부안(09-03)과 동일 방법론을 김제에 적용(시간단위 재구축).

## 배경
사용자 지시("김제로 넘어가서 같은 방식으로 재검증하자"). 부안에서 확립한
절차(3시간격자 실측 확인 → 시간단위 재구축 → 광주식 방송(broadcast)
→ v6 permutation importance 재선정)를 그대로 김제에 적용한다.

## 사전 실측 확인(재사용 전 필수 게이트 - 이 프로젝트 습관)
- 김제 원천 NWP·GRID도 3시간 격자(00·03·06·09·12·15·18·21시)임을
  `kma_live_inputs_gimje_v1_2026-09-01/kma_live_inputs.sqlite3`에서
  직접 재확인(부안·광주와 동일 패턴 - 가정하지 않고 매번 실측).
- 김제는 **710일치**(2024-08-25~2026-08-04) 시간단위 원자료가 이미
  있음(부안은 236일) - 요인재검증에 훨씬 유리한 표본 조건.

## 재사용한 것(재구현 안 함)
- `total_output_weather_model_v1_gimje_2026-09-01.py`의
  `expanding_folds_full_coverage()`·`pooled_score()`·`TARGET`·
  `INITIAL_TRAIN_DAYS`(90)·`TEST_BLOCK_DAYS`(30)·LightGBM 하이퍼파라미터.
- 시간단위 타깃: 코덱스 산출물 `시간집계_v1_2026-08-31/
  김제_발전소_1시간_공식후보.parquet`(17,041행) 그대로.
- ASOS 시간단위 원본: `기상과거백필_v1_2026-08-31/ASOS/
  기상청_ASOS243_시간환경_20240825_20260804.csv`(17,030행) 그대로 -
  기존 3시간 결합테이블은 이 중 1/3만 쓰고 있었음(부안과 동일 문제).
- NWP·GRID 방송(broadcast) 원칙: 광주
  `train_ultra_short_official_v1_2026-08-21.py` 주석 그대로 재적용.
- 결함구간: 기존 결합parquet의 `is_defect_period`(2025-05-28~06-11,
  120행)에서 날짜범위만 그대로 가져옴(시간단위 산출물엔 이 플래그가
  없어 날짜상수로 재적용 - 새로 판정한 게 아니라 기존 확정값 재사용).
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
V1_SCRIPT = HERE / "total_output_weather_model_v1_gimje_2026-09-01.py"
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31"
    r"\김제_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet")
HOURLY_TARGET_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_발전소_1시간_공식후보.parquet")
ASOS_HOURLY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31"
    r"\ASOS\기상청_ASOS243_시간환경_20240825_20260804.csv")
OUT_DIR = HERE / "outputs" / "요인재검증_v6_hourly_2026-09-03"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
# 기존 결합parquet의 is_defect_period 그대로(실측 확인, 새 판정 아님)
DEFECT_START = pd.Timestamp("2025-05-28")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2025-06-12")

ASOS_RENAME = {
    "기온_C": "issue_asos_기온_C", "강수량_mm": "issue_asos_강수량_mm",
    "풍속_m_s": "issue_asos_풍속_m_s", "풍향_deg": "issue_asos_풍향_deg",
    "상대습도_pct": "issue_asos_상대습도_pct", "현지기압_hPa": "issue_asos_현지기압_hPa",
    "해면기압_hPa": "issue_asos_해면기압_hPa", "일조시간_hr": "issue_asos_일조시간_hr",
    "적설_cm": "issue_asos_적설_cm", "전운량_10분위": "issue_asos_전운량_10분위",
    "전운량_pct": "issue_asos_전운량_pct", "지면온도_C": "issue_asos_지면온도_C",
}
ASOS_FULL = list(ASOS_RENAME.values())
# ★09-03 실측 확인★: 김제 결합parquet엔 forecast_DSWRFLX·forecast_DIFSWRF
# 컬럼 자체가 없음(부안과 달리 처음부터 미수집 - AGENTS.md 08-26절
# "KIMR 상류결측이라 운영모델 8종도 이 두 변수를 쓰지 않는다"와 일치).
# 가정하지 않고 실측 확인 후 후보에서 제외.
NWP_FULL = ["forecast_DSWRF", "forecast_TCDC", "forecast_LCDC",
            "forecast_MCDC", "forecast_HCDC", "forecast_REH", "forecast_POP",
            "forecast_SKY"]
NATIVE_MISSING_OK = {"issue_asos_강수량_mm", "issue_asos_적설_cm"}
PHYSICAL = ["solar_elevation_deg", "physical_daylight"]
CYCLICAL = ["target_hour_sin", "target_hour_cos", "doy_sin", "doy_cos"]
LAG_HOURS = [1, 2, 3, 6, 24]
ROLL_HOURS = [6, 24]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _to_naive_kst(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_hourly_frame() -> tuple[pd.DataFrame, list[str]]:
    hourly = pd.read_parquet(HOURLY_TARGET_PARQUET)
    hourly["grid_time_kst"] = _to_naive_kst(hourly["grid_time_kst"])
    hourly = hourly.set_index("grid_time_kst").sort_index()
    hourly.loc[hourly["quality_status"] != "valid_ge9of12", TARGET] = np.nan

    hourly["target_day"] = hourly.index.normalize()
    in_defect = ((hourly["target_day"] >= DEFECT_START) &
                (hourly["target_day"] < DEFECT_END_EXCLUSIVE))
    hourly = hourly.loc[~in_defect].copy()

    asos = pd.read_csv(ASOS_HOURLY_CSV)
    asos["시각"] = _to_naive_kst(asos["시각"])
    asos = asos.rename(columns=ASOS_RENAME).set_index("시각").sort_index()
    hourly = hourly.join(asos[ASOS_FULL], how="left")

    join_df = pd.read_parquet(JOIN_PARQUET)
    join_df["target_time_kst"] = _to_naive_kst(join_df["target_time_kst"])
    nwp_grid_3h = (join_df.drop_duplicates("target_time_kst")
                   .set_index("target_time_kst")[NWP_FULL].sort_index())
    broadcast = nwp_grid_3h.reindex(hourly.index, method="ffill")
    for c in NWP_FULL:
        hourly[c] = broadcast[c]

    hour = hourly.index.hour
    doy = hourly.index.dayofyear
    hourly["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    hourly["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    hourly["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    hourly["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    power = hourly[TARGET]
    baseline_cols = []
    for h in LAG_HOURS:
        col = f"발전출력_{h}시간전_kW"
        hourly[col] = power.shift(h)
        baseline_cols.append(col)
    for h in ROLL_HOURS:
        mean_col, std_col = f"발전출력_{h}시간이동평균_kW", f"발전출력_{h}시간이동표준편차_kW"
        hourly[mean_col] = power.shift(1).rolling(h, min_periods=max(3, h // 2)).mean()
        hourly[std_col] = power.shift(1).rolling(h, min_periods=max(3, h // 2)).std()
        baseline_cols += [mean_col, std_col]
    hourly["lag_1day_same_slot_kw"] = power.shift(24)

    hourly = hourly.reset_index().rename(columns={"grid_time_kst": "target_time_kst"})
    hourly["issue_day"] = hourly["target_time_kst"].dt.normalize()
    return hourly, baseline_cols


def run_walkforward_native_missing(df: pd.DataFrame, folds: list[tuple],
                                    features: list[str], pooled_score_fn) -> dict:
    required = [c for c in features if c not in NATIVE_MISSING_OK]
    all_true, all_pred, all_pers = [], [], []
    fold_rows = []
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤에 있음"
        train = df[df["issue_day"].isin(train_days)].dropna(subset=required + [TARGET])
        test = df[df["issue_day"].isin(test_days)].dropna(subset=required + [TARGET])
        if len(train) < 100 or len(test) < 5:
            continue
        model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                              max_depth=4, min_child_samples=15, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[features], train[TARGET])
        pred = np.clip(model.predict(test[features]), 0, None)
        y_true = test[TARGET].to_numpy()
        pers = test["lag_1day_same_slot_kw"].to_numpy()
        all_true.append(y_true); all_pred.append(pred); all_pers.append(pers)
        fold_rows.append({"폴드": i, "시험발행일수": len(test_days), "시험행수": len(test),
                          **{f"모델_{k}": v for k, v in pooled_score_fn(y_true, pred).items()
                             if k != "n"}})
    y_true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    pers_all = np.concatenate(all_pers)
    return {"폴드수": len(fold_rows), "폴드별": fold_rows,
            "pooled_모델": pooled_score_fn(y_true_all, pred_all),
            "pooled_지속성": pooled_score_fn(y_true_all, pers_all)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1 = _load_module("gimje_v1_hourly_20260903", V1_SCRIPT)
    frame, baseline_cols = build_hourly_frame()

    candidate_cols = ASOS_FULL + NWP_FULL + PHYSICAL + CYCLICAL
    all_cols = baseline_cols + candidate_cols
    daylight = frame[frame["physical_daylight"] == 1].dropna(subset=[TARGET])
    print(f"[0] 시간단위 재구축 - 전체 {len(frame):,}행, 낮시간 {len(daylight):,}행")
    print(f"    후보 {len(candidate_cols)}개(+baseline {len(baseline_cols)}개) = 총 {len(all_cols)}개")

    required_cols = [c for c in all_cols if c not in NATIVE_MISSING_OK]
    f = daylight.dropna(subset=required_cols + [TARGET]).sort_values("target_time_kst")
    n = len(f)
    cut = int(n * 0.85)
    train, test = f.iloc[:cut], f.iloc[cut:]
    print(f"[1] importance계산용 분할 - 학습 {len(train):,}행 · 시험 {len(test):,}행")

    model = LGBMRegressor(n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=SEED,
                          n_jobs=4, verbosity=-1)
    model.fit(train[all_cols], train[TARGET])

    print("[2] permutation importance 계산 중(20회 반복, MAE 기준)...")
    perm = permutation_importance(
        model, test[all_cols], test[TARGET],
        scoring="neg_mean_absolute_error", n_repeats=20, random_state=SEED, n_jobs=4,
    )
    imp_df = pd.DataFrame({
        "변수": all_cols, "중요도_MAE증가량": perm.importances_mean,
        "표준편차": perm.importances_std,
        "분류": ["과거발전량(baseline)" if c in baseline_cols else
                ("NWP예보" if c.startswith("forecast_") else
                 ("ASOS실측" if c.startswith("issue_asos_") else
                  ("순환시간" if c in CYCLICAL else "물리량")))
                for c in all_cols],
    }).sort_values("중요도_MAE증가량", ascending=False)
    imp_df.to_csv(OUT_DIR / "특성중요도_permutation.csv", index=False, encoding="utf-8-sig")
    print(imp_df.to_string(index=False))

    selected_candidates = [row["변수"] for _, row in imp_df.iterrows()
                           if row["변수"] in candidate_cols and row["중요도_MAE증가량"] > 0]
    dropped = [c for c in candidate_cols if c not in selected_candidates]
    final_features = list(dict.fromkeys(baseline_cols + selected_candidates + ["lag_1day_same_slot_kw"]))
    print(f"\n[3] 후보 {len(candidate_cols)}개 중 채택 {len(selected_candidates)}개, "
          f"가지치기 {len(dropped)}개: {dropped}")

    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    folds = v1.expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
    result_new = run_walkforward_native_missing(frame, folds, final_features, v1.pooled_score)

    old15_features = list(dict.fromkeys(v1.FEATURES_15 + ["lag_1day_same_slot_kw"]))
    result_old15 = run_walkforward_native_missing(frame, folds, old15_features, v1.pooled_score)

    summary = {"v6_hourly_importance기반": {**result_new, "특성수": len(final_features)},
              "기존_v1_FEATURES_15_hourly재평가": {**result_old15, "특성수": len(old15_features)}}
    (OUT_DIR / "요약.json").write_text(json.dumps({
        "선택_baseline": baseline_cols, "선택_후보": selected_candidates,
        "가지치기": dropped, "성능비교": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== [4] 시간단위 walk-forward CV 비교 ===")
    print(f"v6(importance기반, {len(final_features)}특성, 시간단위 {n:,}행 풀): "
          f"pooled MAE={result_new['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_new['pooled_모델']['RMSE_kW']}kW, "
          f"지속성MAE={result_new['pooled_지속성']['MAE_kW']}kW")
    print(f"기존 v1(FEATURES_15, 같은 시간단위 풀로 재평가): "
          f"pooled MAE={result_old15['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_old15['pooled_모델']['RMSE_kW']}kW")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
