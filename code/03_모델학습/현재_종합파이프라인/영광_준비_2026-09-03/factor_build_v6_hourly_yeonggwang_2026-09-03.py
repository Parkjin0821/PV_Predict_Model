# -*- coding: utf-8 -*-
"""영광 총출력모델 최초 구축 - 부안·김제와 동일 v6 방법론(시간단위 + permutation importance).

## 배경
부안·김제는 "재검증"(기존 좁은 특성셋 v1/v5가 있었고 그걸 v6로 대체하는
비교)이었지만, 영광은 **기존 총출력모델 자체가 없다** - 데이터셋이
오늘(09-03) 막 만들어졌기 때문. 그래서 "재검증"이 아니라 "처음부터
v6 방식(사전 Pearson필터 없이 전체후보→permutation importance)으로
구축" - 비교 대상은 "예전 모델"이 아니라 **지속성(persistence)
베이스라인**이다(부안·김제도 자체적으로 이 비교를 이미 하고 있었음
- 새 관례 아님).

## 재사용한 것(재구현 안 함)
- 결합데이터: `combine_yeonggwang_history_power_asos_nwp_v1_2026-09-03.py`
  + `join_yeonggwang_grid_forecast_v1_2026-09-03.py`(코덱스 작성,
  사용자 09-03 실행 완료) 산출물 그대로.
- 시간단위 타깃: 코덱스 산출물 `시간집계_v1_2026-09-01/
  영광_발전소_1시간_공식후보.parquet`(17,008행) 그대로.
- ASOS 시간단위 원본: `기상과거백필_v1_2026-08-31/ASOS/
  기상청_ASOS252_시간환경_20240825_20260804.csv` 그대로.
- NWP·GRID 방송(broadcast) 원칙: 광주
  `train_ultra_short_official_v1_2026-08-21.py` 주석 그대로(부안·
  김제와 동일하게 재적용).
- walk-forward CV 구조·LightGBM 하이퍼파라미터: 부안 v5/김제 v1과
  동일(`INITIAL_TRAIN_DAYS`=90·`TEST_BLOCK_DAYS`=30 - 영광도 김제처럼
  710일치 긴 이력이 있어 김제 값 그대로 사용).

## 영광 고유 사항
- **결함구간 미감사**: `combine_yeonggwang_...` 실행 결과 자체가
  "★defect_period_audited★: false"로 명시(2026-09-03 사용자 실행
  확인) - 이 스크립트는 결함구간 제외를 아예 하지 않는다(없는 근거로
  날짜를 지어내지 않음). 인버터별 이력 감사는 별도 후속 작업으로
  남겨둔다.
- NWP는 김제와 동일하게 DSWRF 1종만(DSWRFLX·DIFSWRF 미수집) - NWP_FULL
  목록도 김제와 동일하게 축소.
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
JOIN_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_v1_2026-09-03"
    r"\영광_과거발전_ASOS_NWP_GRID_결합_v1_2026-09-03.parquet")
HOURLY_TARGET_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
    r"\영광_발전소_1시간_공식후보.parquet")
ASOS_HOURLY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31"
    r"\ASOS\기상청_ASOS252_시간환경_20240825_20260804.csv")
OUT_DIR = HERE / "outputs" / "요인구축_v6_hourly_2026-09-03"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30

ASOS_RENAME = {
    "기온_C": "issue_asos_기온_C", "강수량_mm": "issue_asos_강수량_mm",
    "풍속_m_s": "issue_asos_풍속_m_s", "풍향_deg": "issue_asos_풍향_deg",
    "상대습도_pct": "issue_asos_상대습도_pct", "현지기압_hPa": "issue_asos_현지기압_hPa",
    "해면기압_hPa": "issue_asos_해면기압_hPa", "일조시간_hr": "issue_asos_일조시간_hr",
    "적설_cm": "issue_asos_적설_cm", "전운량_10분위": "issue_asos_전운량_10분위",
    "전운량_pct": "issue_asos_전운량_pct", "지면온도_C": "issue_asos_지면온도_C",
}
ASOS_FULL = list(ASOS_RENAME.values())
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


def pooled_score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = y_true - pred
    return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
            "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}


def expanding_folds_full_coverage(days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    n = len(days)
    folds = []
    end = initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((days[:end], days[end:test_end]))
        end = test_end
    return folds


def build_hourly_frame() -> tuple[pd.DataFrame, list[str]]:
    hourly = pd.read_parquet(HOURLY_TARGET_PARQUET)
    hourly["grid_time_kst"] = _to_naive_kst(hourly["grid_time_kst"])
    hourly = hourly.set_index("grid_time_kst").sort_index()
    hourly.loc[hourly["quality_status"] != "valid_ge9of12", TARGET] = np.nan
    # ★결함구간 제외 없음★ - 09-03 결합 산출물이 "defect_period_audited: false"로
    # 명시(사용자 실행 결과 확인) - 없는 근거로 날짜를 지어내지 않는다.

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
                                    features: list[str]) -> dict:
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
                          **{f"모델_{k}": v for k, v in pooled_score(y_true, pred).items()
                             if k != "n"}})
    y_true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    pers_all = np.concatenate(all_pers)
    return {"폴드수": len(fold_rows), "폴드별": fold_rows,
            "pooled_모델": pooled_score(y_true_all, pred_all),
            "pooled_지속성": pooled_score(y_true_all, pers_all)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, baseline_cols = build_hourly_frame()

    candidate_cols = ASOS_FULL + NWP_FULL + PHYSICAL + CYCLICAL
    all_cols = baseline_cols + candidate_cols
    daylight = frame[frame["physical_daylight"] == 1].dropna(subset=[TARGET])
    print(f"[0] 영광 시간단위 최초 구축 - 전체 {len(frame):,}행, 낮시간 {len(daylight):,}행")
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
    folds = expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
    result_v6 = run_walkforward_native_missing(frame, folds, final_features)

    improve_mae = round((1 - result_v6["pooled_모델"]["MAE_kW"] / result_v6["pooled_지속성"]["MAE_kW"]) * 100, 1)
    improve_rmse = round((1 - result_v6["pooled_모델"]["RMSE_kW"] / result_v6["pooled_지속성"]["RMSE_kW"]) * 100, 1)

    (OUT_DIR / "요약.json").write_text(json.dumps({
        "비고": "영광은 기존 총출력모델이 없어 지속성 대비로만 판정(부안·김제식 v1대비 비교 아님)",
        "선택_baseline": baseline_cols, "선택_후보": selected_candidates,
        "가지치기": dropped, "v6_결과": result_v6,
        "MAE_개선율_vs지속성_pct": improve_mae, "RMSE_개선율_vs지속성_pct": improve_rmse,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== [4] 시간단위 walk-forward CV (지속성 대비) ===")
    print(f"v6(importance기반, {len(final_features)}특성, {n:,}행 풀): "
          f"pooled MAE={result_v6['pooled_모델']['MAE_kW']}kW, "
          f"RMSE={result_v6['pooled_모델']['RMSE_kW']}kW")
    print(f"지속성(persistence): MAE={result_v6['pooled_지속성']['MAE_kW']}kW, "
          f"RMSE={result_v6['pooled_지속성']['RMSE_kW']}kW")
    print(f"개선율: MAE {improve_mae}%, RMSE {improve_rmse}%")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
