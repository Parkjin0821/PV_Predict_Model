# -*- coding: utf-8 -*-
"""부안 초단기(+1~4h)급 순수 lag 기준모델 - 코드만 준비, 아직 실행 안 함
(08-31, 사용자 지시: "코드만 미리 작성해둬").

## 왜 지금 실행하지 않는가(실측 확인, 08-31)
부안 실시간 Blockdata DB(blockdata_live_v1_2026-08-28/blockdata_history.
sqlite3)를 직접 열어보니 **2026-08-28T14:40~2026-08-31T13:20, 52건(3일치,
그것도 5분 간격이 아니라 듬성듬성)뿐**이었다. 처음엔 "08-05부터 약
26일치"라고 잘못 말했었는데, 08-05는 Excel/API 경계를 명시적으로 보존해둔
시점일 뿐 실시간 수집기가 그때부터 안정적으로 돈 게 아니었다 - 실제
안정 가동은 08-28부터다. 3일 52건으로 학습/시험을 나누는 건 통계적으로
의미가 없다(가짜 성공 금지) - 그래서 이 스크립트는 **최소 데이터량
미달 시 스스로 실행을 거부**하도록 만들었다. 데이터가 쌓인 뒤 그대로
실행하면 된다.

## 설계
- 데이터: `plant_snapshots`(Blockdata 실시간 DB, plant_id=16783)의
  `snapshot_time`·`plant_ac_power_kw`만 읽기전용으로 사용. API 호출 없음.
- 타깃: 관측시각 t 기준 t+1h/+2h/+3h/+4h 실제 발전출력.
- 특성(전부 lag 기반, 기상특성 없음 - "순수 lag 기준모델"이라는 이름 그대로):
  t시점 값, t-15분·t-30분·t-1h 값, 최근 1시간 이동평균, 전일 동시각 값
  (있으면).
- 기준선 비교: "persistence"(t시점 값을 그대로 t+Nh 예측값으로 사용) -
  이게 진짜 기준선이고, lag특성 기반 LightGBM이 이걸 이기는지 보는 게
  핵심.
- 검증: 최소 MIN_DAYS(기본 14일) 미만이면 실행 자체를 거부(예외 발생,
  가짜 결과 생성 안 함). 14일 이상 쌓이면 expanding-window walk-forward
  (총출력모델과 같은 원칙, 초기 7일 학습 시작 - 초단기는 하루 안에도
  샘플이 많아 총출력모델보다 초기창을 짧게 잡음).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
LIVE_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3"
)
OUT_DIR = HERE / "outputs" / "초단기_lag기준모델_2026-08-31"
PLANT_ID = 16783
HORIZONS_MIN = [60, 120, 180, 240]  # +1h/+2h/+3h/+4h
MIN_DAYS = 14  # 이보다 적으면 실행 거부 - 사용자와 합의한 최소선(임의값, 재논의 가능)
INITIAL_TRAIN_DAYS = 7
TEST_BLOCK_DAYS = 7
SEED = 42


class InsufficientDataError(RuntimeError):
    """데이터가 최소 기준에 못 미치면 조용히 넘어가지 않고 여기서 멈춘다."""


def load_snapshots() -> pd.DataFrame:
    if not LIVE_DB.is_file():
        raise FileNotFoundError(f"실시간 DB 없음: {LIVE_DB}")
    conn = sqlite3.connect(LIVE_DB)
    try:
        df = pd.read_sql_query(
            "SELECT snapshot_time, plant_ac_power_kw, quality_status "
            "FROM plant_snapshots WHERE plant_id = ? ORDER BY snapshot_time",
            conn, params=[PLANT_ID],
        )
    finally:
        conn.close()
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=False)
    if df["snapshot_time"].dt.tz is not None:
        df["snapshot_time"] = df["snapshot_time"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return df.dropna(subset=["plant_ac_power_kw"]).sort_values("snapshot_time").reset_index(drop=True)


def check_min_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise InsufficientDataError("실시간 스냅샷이 0건 - 아직 아무 데이터도 없음.")
    span_days = (df["snapshot_time"].max() - df["snapshot_time"].min()).total_seconds() / 86400
    if span_days < MIN_DAYS:
        raise InsufficientDataError(
            f"실시간 데이터 기간이 {span_days:.1f}일로 최소기준({MIN_DAYS}일) 미달 - "
            f"실행을 거부한다(가짜 성공 금지). 더 쌓인 뒤 다시 실행할 것. "
            f"현재 {len(df)}건, {df['snapshot_time'].min()}~{df['snapshot_time'].max()}."
        )


def build_regular_grid(df: pd.DataFrame, freq_min: int = 5) -> pd.Series:
    """불규칙 스냅샷을 지정 간격 그리드로 최근접 보간(전방 채움 없음 - 결측은 결측으로 남김,
    직전 관측이 tolerance 안이면 그대로, 아니면 NaN)."""
    s = df.set_index("snapshot_time")["plant_ac_power_kw"]
    s = s[~s.index.duplicated(keep="last")]
    start, end = s.index.min().floor(f"{freq_min}min"), s.index.max().ceil(f"{freq_min}min")
    grid = pd.date_range(start, end, freq=f"{freq_min}min")
    aligned = s.reindex(grid, method="nearest", tolerance=pd.Timedelta(minutes=freq_min))
    return aligned


def build_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"value": s})
    df["lag_15m"] = s.shift(3)   # 5분그리드 기준 3칸=15분
    df["lag_30m"] = s.shift(6)
    df["lag_1h"] = s.shift(12)
    df["roll_1h_mean"] = s.rolling(12, min_periods=6).mean()
    df["lag_1day_same_time"] = s.shift(288)  # 5분*288=24시간
    return df


def expanding_folds(n_points: int, initial: int, block: int) -> list[tuple]:
    folds, end = [], initial
    while end < n_points:
        test_end = min(end + block, n_points)
        folds.append((end, test_end))
        end = test_end
    return folds


def run_for_horizon(feat: pd.DataFrame, target_series: pd.Series, horizon_min: int) -> dict:
    steps = horizon_min // 5
    y = target_series.shift(-steps)
    data = feat.copy()
    data["y"] = y
    data["persistence"] = data["value"]  # t시점 값을 그대로 t+H 예측값으로

    feature_cols = ["value", "lag_15m", "lag_30m", "lag_1h", "roll_1h_mean", "lag_1day_same_time"]
    data = data.dropna(subset=feature_cols + ["y"])
    n_days = (data.index.max() - data.index.min()).days if len(data) else 0
    initial_pts = INITIAL_TRAIN_DAYS * 288
    block_pts = TEST_BLOCK_DAYS * 288
    folds = expanding_folds(len(data), initial_pts, block_pts)

    all_true, all_pred, all_pers = [], [], []
    for s0, e0 in folds:
        train, test = data.iloc[:s0], data.iloc[s0:e0]
        if len(train) < 200 or len(test) < 1:
            continue
        model = LGBMRegressor(n_estimators=120, learning_rate=0.05, num_leaves=15,
                              max_depth=4, min_child_samples=20, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(train[feature_cols], train["y"])
        pred = np.clip(model.predict(test[feature_cols]), 0, None)
        all_true.append(test["y"].to_numpy())
        all_pred.append(pred)
        all_pers.append(test["persistence"].to_numpy())

    if not all_true:
        return {"horizon_min": horizon_min, "상태": "폴드 생성 불가(데이터 더 필요)"}

    yt, yp, yper = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_pers)
    mae_model = mean_absolute_error(yt, yp)
    rmse_model = float(np.sqrt(mean_squared_error(yt, yp)))
    mae_pers = mean_absolute_error(yt, yper)
    rmse_pers = float(np.sqrt(mean_squared_error(yt, yper)))
    return {
        "horizon_min": horizon_min, "n_시험표본": int(len(yt)), "폴드수": len(folds),
        "LightGBM_MAE_kW": round(mae_model, 2), "LightGBM_RMSE_kW": round(rmse_model, 2),
        "persistence_MAE_kW": round(mae_pers, 2), "persistence_RMSE_kW": round(rmse_pers, 2),
        "MAE_개선율_pct": round((1 - mae_model / mae_pers) * 100, 1) if mae_pers else None,
        "RMSE_개선율_pct": round((1 - rmse_model / rmse_pers) * 100, 1) if rmse_pers else None,
    }


def main() -> None:
    df = load_snapshots()
    check_min_data(df)  # 데이터 부족하면 여기서 예외로 멈춤 - 가짜 결과 생성 안 함

    grid = build_regular_grid(df)
    feat = build_features(grid)

    results = [run_for_horizon(feat, grid, h) for h in HORIZONS_MIN]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "데이터기간": {"시작": str(df["snapshot_time"].min()), "끝": str(df["snapshot_time"].max()),
                    "스냅샷건수": len(df)},
        "수평별_결과": results,
        "_주의": "순수 lag 기반 기준모델(기상특성 없음) - persistence를 이기는지가 핵심. "
              "초단기(+1~4h) tier의 첫 후보일 뿐 공식경로 대상 아님(promote_to_official 없음).",
    }
    (OUT_DIR / "부안_초단기_lag기준모델_결과.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
