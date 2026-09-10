# -*- coding: utf-8 -*-
"""부안 요인 재검증 v2 - 시간단위(hourly)로 재구축.

## 배경(사용자 09-03 지적 2건, 둘 다 반영)
1차(factor_reverify_v6_permutation_buan_2026-09-03.py, 3시간 격자
1,584행)는 기존 v5 대비 이득이 없었다. 사용자가 두 가지를 지적:
1. "부안은 계속 데이터가 쌓이고 있으니 광주처럼 만들어두면 개선율이
   나중엔 증가할 것" - 표본 자체가 지금은 적어서(8개월) 손해였을
   수 있음. 데이터가 더 있으면 결과가 달라질 수 있는 구조로 만들어야
   한다.
2. "매시간 격자가 필요한 이유는 초단기(+1~4h) 때문" - 정확한 지적.
   실측 확인 결과 NWP·GRID 원본 자체가 3시간 격자인 건 광주도 동일
   (`kma_live_inputs.sqlite3`의 nwp_values/grid_forecast 둘 다 00·03·
   06·09·12·15·18·21시만 존재, 광주도 마찬가지로 실측 확인함) - 즉
   부안만의 한계가 아니었다. 광주가 이 문제를 실제로 어떻게 푸는지
   원본 코드(`train_ultra_short_official_v1_2026-08-21.py` 주석)에서
   확인: **보간(interpolation)이 아니라 방송(broadcast)** - "그 시간(3h
   구간) 안의 슬롯에는 가장 최근에 알려진 값을 그대로 쓴다. 억지
   보간이 아니라 실제로 그 시각에 모델이 아는 정보를 정확히 표현한
   것"(원문 그대로 재사용, 재구현 아님).

## 이번에 새로 한 것
- **타깃을 3시간 격자(1,584행)가 아니라 이미 존재하는 시간단위 집계
  (`시간집계_v1_2026-08-28/부안_발전소_1시간_공식후보.parquet`,
  5,652행)로 교체** - 코덱스가 이미 만들어둔 산출물 재사용, 신규 계산
  안 함.
- **ASOS는 원래 시간단위 원본**(`기상청_ASOS243_시간환경_...csv`,
  5,662행)이 있었는데 기존 3시간 결합 테이블이 그중 1/3만 쓰고
  있었다 - 원본을 직접 다시 읽어 전부 사용.
- **NWP·GRID만 3시간 격자 그대로**(원천이 3시간이라 늘릴 수 없음) -
  광주 방식 그대로 방송(`reindex(hourly_index, method='ffill')`).
- lag/rolling은 이제 진짜 시간단위라 광주와 동일한 [1,2,3,6,24]시간전
  + 6·24시간 이동평균/표준편차를 그대로 쓴다(3시간용 보정 불필요).

## 재사용한 것(재구현 안 함)
`total_output_weather_model_v5_방법론정정_2026-08-31.py`의
`expanding_folds_full_coverage()`·`pooled_score()`·결함구간 상수
(DEFECT_START/END)·LightGBM 하이퍼파라미터. 폴드 그룹 단위만 issue_day
대신 이 스크립트의 hourly grid_time_kst의 날짜(day)로 바꿔 쓴다(시간
단위 데이터라 "발행일" 개념 자체가 없어짐 - day-ahead 구조가 아니라
매시간이 독립 시험단위인 구조로 전환됐기 때문, 임의 변경 아니라
데이터 성격 변화에 따른 불가피한 대응).
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
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_v1_2026-08-28"
    r"\부안_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet")
HOURLY_TARGET_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_발전소_1시간_공식후보.parquet")
ASOS_HOURLY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\기상과거백필_v1_2026-08-28"
    r"\ASOS\기상청_ASOS243_시간환경_20251212_20260804.csv")
OUT_DIR = HERE / "outputs" / "요인재검증_v6_hourly_2026-09-03"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20

ASOS_RENAME = {
    "기온_C": "issue_asos_기온_C", "강수량_mm": "issue_asos_강수량_mm",
    "풍속_m_s": "issue_asos_풍속_m_s", "풍향_deg": "issue_asos_풍향_deg",
    "상대습도_pct": "issue_asos_상대습도_pct", "현지기압_hPa": "issue_asos_현지기압_hPa",
    "해면기압_hPa": "issue_asos_해면기압_hPa", "일조시간_hr": "issue_asos_일조시간_hr",
    "적설_cm": "issue_asos_적설_cm", "전운량_10분위": "issue_asos_전운량_10분위",
    "전운량_pct": "issue_asos_전운량_pct", "지면온도_C": "issue_asos_지면온도_C",
}
ASOS_FULL = list(ASOS_RENAME.values())
NWP_FULL = ["forecast_DSWRF", "forecast_DSWRFLX", "forecast_DIFSWRF",
            "forecast_TCDC", "forecast_LCDC", "forecast_MCDC", "forecast_HCDC",
            "forecast_REH", "forecast_POP", "forecast_SKY"]
NATIVE_MISSING_OK = {"issue_asos_강수량_mm", "issue_asos_적설_cm",
                      "forecast_DSWRFLX", "forecast_DIFSWRF"}
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


def build_hourly_frame(v5_mod) -> pd.DataFrame:
    # 1) 시간단위 실측 타깃(코덱스 산출물 재사용)
    hourly = pd.read_parquet(HOURLY_TARGET_PARQUET)
    hourly["grid_time_kst"] = _to_naive_kst(hourly["grid_time_kst"])
    hourly = hourly.set_index("grid_time_kst").sort_index()
    hourly.loc[hourly["quality_status"] != "valid_ge9of12", TARGET] = np.nan

    # 2) 결함구간 제외(v5와 동일 기준, target_time 기준)
    hourly["target_day"] = hourly.index.normalize()
    in_defect = ((hourly["target_day"] >= v5_mod.DEFECT_START) &
                (hourly["target_day"] < v5_mod.DEFECT_END_EXCLUSIVE))
    hourly = hourly.loc[~in_defect].copy()

    # 3) ASOS 시간단위 원본 그대로(3시간결합 테이블은 1/3만 썼었음 - 전부 사용)
    asos = pd.read_csv(ASOS_HOURLY_CSV)
    asos["시각"] = _to_naive_kst(asos["시각"])  # hourly 타깃과 tz 형식 통일(둘 다 KST 기준)
    asos = asos.rename(columns=ASOS_RENAME).set_index("시각").sort_index()
    hourly = hourly.join(asos[ASOS_FULL], how="left")

    # 4) NWP·GRID는 원천이 3시간 격자(광주도 동일 - 실측 확인) - 광주
    #    train_ultra_short_official_v1의 "방송"(broadcast) 원칙 그대로:
    #    보간 아님, 가장 최근에 알려진 값을 다음 갱신 전까지 그대로 씀.
    join_df = pd.read_parquet(JOIN_PARQUET)
    join_df["target_time_kst"] = _to_naive_kst(join_df["target_time_kst"])
    nwp_grid_3h = (join_df.drop_duplicates("target_time_kst")
                   .set_index("target_time_kst")[NWP_FULL].sort_index())
    broadcast = nwp_grid_3h.reindex(hourly.index, method="ffill")
    for c in NWP_FULL:
        hourly[c] = broadcast[c]
    for c in ["forecast_DSWRFLX", "forecast_DIFSWRF"]:
        hourly[f"{c}_결측여부"] = hourly[c].isna().astype(float)

    # 5) 순환시간 특성
    hour = hourly.index.hour
    doy = hourly.index.dayofyear
    hourly["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    hourly["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    hourly["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    hourly["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 6) lag/rolling - 이제 진짜 시간단위라 정수 lag가 그대로 정확함
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
    hourly["lag_1day_same_slot_kw"] = power.shift(24)  # 지속성 대조용(v5와 동일 정의)

    hourly = hourly.reset_index().rename(columns={"grid_time_kst": "target_time_kst"})
    hourly["issue_day"] = hourly["target_time_kst"].dt.normalize()  # 폴드 그룹핑용
    return hourly, baseline_cols


def expanding_folds_by_day(days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    n = len(days)
    folds = []
    end = initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((days[:end], days[end:test_end]))
        end = test_end
    return folds


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
    v5 = _load_module("buan_v5_hourly_20260903", V5_SCRIPT)
    frame, baseline_cols = build_hourly_frame(v5)

    candidate_cols = (ASOS_FULL + NWP_FULL + PHYSICAL + CYCLICAL +
                      ["forecast_DSWRFLX_결측여부", "forecast_DIFSWRF_결측여부"])
    all_cols = baseline_cols + candidate_cols
    daylight = frame[frame["physical_daylight"] == 1].dropna(subset=[TARGET])
    print(f"[0] 시간단위 재구축 - 전체 {len(frame):,}행(3시간격자 대비 목표 ~3.5배), "
          f"낮시간 {len(daylight):,}행")
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
    folds = expanding_folds_by_day(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
    result_new = run_walkforward_native_missing(frame, folds, final_features, v5.pooled_score)

    old15_features = list(dict.fromkeys(v5.FEATURES_15 + ["lag_1day_same_slot_kw"]))
    result_old15 = run_walkforward_native_missing(frame, folds, old15_features, v5.pooled_score)

    summary = {"v6_hourly_importance기반": {**result_new, "특성수": len(final_features)},
              "기존_v5_FEATURES_15_hourly재평가": {**result_old15, "특성수": len(old15_features)},
              "기존_v5_FEATURES_15_원래성능(3시간격자기준)": {"pooled_모델": {"MAE_kW": 46.37, "RMSE_kW": 90.57}}}
    (OUT_DIR / "요약.json").write_text(json.dumps({
        "선택_baseline": baseline_cols, "선택_후보": selected_candidates,
        "가지치기": dropped, "성능비교": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== [4] 시간단위 walk-forward CV 비교 ===")
    print(f"v6(importance기반, {len(final_features)}특성, 시간단위 {n:,}행 풀): "
          f"pooled MAE={result_new['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_new['pooled_모델']['RMSE_kW']}kW, "
          f"지속성MAE={result_new['pooled_지속성']['MAE_kW']}kW")
    print(f"기존 v5(FEATURES_15, 같은 시간단위 풀로 재평가): "
          f"pooled MAE={result_old15['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_old15['pooled_모델']['RMSE_kW']}kW")
    print(f"(참고) 기존 v5 원래 확정치(3시간격자 기준, 08-31): MAE=46.37kW, RMSE=90.57kW")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
