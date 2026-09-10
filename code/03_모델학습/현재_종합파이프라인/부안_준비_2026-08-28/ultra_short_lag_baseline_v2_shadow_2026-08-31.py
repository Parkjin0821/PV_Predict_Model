# -*- coding: utf-8 -*-
"""부안 초단기 라이브 shadow 검증 v2(08-31, Codex 지적 5건 반영 - v1
대체). v1(ultra_short_lag_baseline_v1_2026-08-31.py)의 결함:
1. quality_status를 SELECT만 하고 실제 필터에 안 씀 -> v2는 quality_status
   가 'ok' 계열이 아니면 그 스냅샷 자체를 버린다(정확한 값 집합은 실제
   DB 데이터가 쌓이면 재확인 필요 - 지금은 sqlite CHECK 없이 온 값이
   비어있지 않은 경우만 통과시키는 최소 방어로 시작하고, 실제 값 쌓이면
   화이트리스트로 좁힐 것).
2. MIN_DAYS를 기간 길이(max-min)로만 검사해 데이터 밀도를 안 봤음 ->
   v2는 "기간 길이"와 "실제 5분 슬롯 커버리지(%)"를 둘 다 검사한다.
3. reindex(method='nearest', tolerance=...)가 하나의 관측을 두 슬롯에
   중복 귀속시킬 수 있음 -> v2는 정확히 grid에 맞는 시각만 쓰고(반올림
   후 중복 시각은 마지막 값만 keep='last'), nearest 매칭을 쓰지 않는다.
4. dropna() 이후 행수를 "하루 288개"로 가정해 폴드를 구성했음 -> v2는
   실제 달력일(date) 컬럼으로 그룹핑해 폴드를 만든다(과거백테스트
   스크립트와 동일 원칙).
5. 14일은 LightGBM 공식검증에는 너무 짧음 -> v2는 3단계 게이트를 둔다:
   <30일=실행거부, 30~89일=기술시험(technical_trial)로만 라벨링,
   90~149일=예비평가(preliminary), 150일+가을포함=공식판정 후보
   (official_candidate) - 어느 단계든 promote_to_official은 아니다.

## 여전히 실행 안 함(08-31 기준 실시간 데이터 2.9일) - 데이터 쌓이면
그대로 재실행.
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
OUT_DIR = HERE / "outputs" / "초단기_lag기준모델_v2_shadow_2026-08-31"
PLANT_ID = 16783
HORIZONS_MIN = [60, 120, 180, 240]
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20
SEED = 42

# ★정정5★ 3단계 게이트(사용자 로드맵 그대로).
GATE_TECHNICAL_TRIAL_DAYS = 30
GATE_PRELIMINARY_DAYS = 90
GATE_OFFICIAL_CANDIDATE_DAYS = 150

# ★정정2★ 기간 길이뿐 아니라 실제 커버리지(%)도 검사.
MIN_COVERAGE_RATIO = 0.5  # 이론적 5분슬롯 대비 실제 수신비율 최소선(임의값, 재논의 가능)

# 승인된 정상 상태값(★정정1★ - 실측 확인: 현재 DB에 'ok'(49건)·'warning'(3건)만
# 존재. 'ok'만 통과, 'warning'은 제외 - 값 종류가 늘어나면 이 목록을 다시 확인할 것).
OK_QUALITY_VALUES = {"ok"}


class InsufficientDataError(RuntimeError):
    pass


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

    # ★정정1★ quality_status를 실제로 필터에 쓴다.
    before = len(df)
    quality_lower = df["quality_status"].astype(str).str.strip().str.lower()
    ok_mask = quality_lower.isin({v.lower() for v in OK_QUALITY_VALUES})
    dropped_by_quality = int((~ok_mask).sum())
    df = df[ok_mask]
    print(f"[quality_status 필터] 전체 {before}건 중 {dropped_by_quality}건 제외, {len(df)}건 남음")

    return df.dropna(subset=["plant_ac_power_kw"]).sort_values("snapshot_time").reset_index(drop=True)


def determine_gate(span_days: float, coverage_ratio: float, season_days: dict) -> dict:
    if span_days < GATE_TECHNICAL_TRIAL_DAYS:
        raise InsufficientDataError(
            f"실시간 데이터 기간이 {span_days:.1f}일로 기술시험 최소선"
            f"({GATE_TECHNICAL_TRIAL_DAYS}일) 미달 - 실행을 거부한다(가짜 성공 금지)."
        )
    if coverage_ratio < MIN_COVERAGE_RATIO:
        raise InsufficientDataError(
            f"5분슬롯 커버리지 {coverage_ratio*100:.1f}%가 최소선"
            f"({MIN_COVERAGE_RATIO*100:.0f}%) 미달 - 기간은 충분해도 밀도가 부족해 실행을 거부한다."
        )
    if span_days < GATE_PRELIMINARY_DAYS:
        stage = "technical_trial"
    elif span_days < GATE_OFFICIAL_CANDIDATE_DAYS:
        stage = "preliminary"
    else:
        stage = "official_candidate"
    if "가을" not in season_days or season_days.get("가을", 0) == 0:
        stage_note = f"{stage}(가을 표본 없음 - 공식판정 근거로는 여전히 불충분)"
    else:
        stage_note = stage
    return {"stage": stage, "stage_note": stage_note, "span_days": round(span_days, 1),
           "coverage_ratio_pct": round(coverage_ratio * 100, 1)}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "겨울"
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    return "가을"


def build_regular_grid(df: pd.DataFrame, freq_min: int = 5) -> pd.Series:
    """★정정3★ nearest+tolerance 대신, 5분 배수로 반올림해 정확히 매칭되는
    시각만 쓴다(중복 귀속 방지) - 반올림 후 같은 슬롯에 여러 관측이
    몰리면 마지막 값만 남긴다."""
    s = df.set_index("snapshot_time")["plant_ac_power_kw"]
    rounded_index = s.index.round(f"{freq_min}min")
    s2 = pd.Series(s.to_numpy(), index=rounded_index)
    s2 = s2[~s2.index.duplicated(keep="last")]
    start, end = s2.index.min(), s2.index.max()
    grid = pd.date_range(start, end, freq=f"{freq_min}min")
    aligned = s2.reindex(grid)  # 정확히 일치하는 시각만 채워짐, 나머지는 NaN(보간 없음)
    return aligned, len(s2), len(grid)


def build_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"value": s})
    df["lag_15m"] = s.shift(3)
    df["lag_30m"] = s.shift(6)
    df["lag_1h"] = s.shift(12)
    df["roll_1h_mean"] = s.rolling(12, min_periods=12).mean()
    df["lag_1day_same_time"] = s.shift(288)
    df["date"] = df.index.normalize()  # ★정정4★ 실제 달력일 기준 폴드용
    return df


def expanding_folds_by_date(dates: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    n = len(dates)
    folds, end = [], initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((dates[:end], dates[end:test_end]))
        end = test_end
    return folds


def run_for_horizon(feat: pd.DataFrame, s: pd.Series, horizon_min: int) -> dict:
    steps = horizon_min // 5
    data = feat.copy()
    data["y"] = s.shift(-steps)
    data["persistence"] = data["value"]
    feature_cols = ["value", "lag_15m", "lag_30m", "lag_1h", "roll_1h_mean", "lag_1day_same_time"]
    data = data.dropna(subset=feature_cols + ["y"])

    dates = pd.DatetimeIndex(np.sort(data["date"].unique()))
    folds = expanding_folds_by_date(dates, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    all_true, all_pred, all_pers = [], [], []
    for train_dates, test_dates in folds:
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]
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
    mae_model = mean_absolute_error(yt, yp); rmse_model = float(np.sqrt(mean_squared_error(yt, yp)))
    mae_pers = mean_absolute_error(yt, yper); rmse_pers = float(np.sqrt(mean_squared_error(yt, yper)))
    return {
        "horizon_min": horizon_min, "n_시험표본": int(len(yt)), "폴드수": len(folds),
        "LightGBM_MAE_kW": round(mae_model, 2), "LightGBM_RMSE_kW": round(rmse_model, 2),
        "persistence_MAE_kW": round(mae_pers, 2), "persistence_RMSE_kW": round(rmse_pers, 2),
        "MAE_개선율_pct": round((1 - mae_model / mae_pers) * 100, 1) if mae_pers else None,
        "RMSE_개선율_pct": round((1 - rmse_model / rmse_pers) * 100, 1) if rmse_pers else None,
    }


def main() -> None:
    df = load_snapshots()
    if df.empty:
        raise InsufficientDataError("실시간 스냅샷이 0건(quality 필터 후) - 아직 데이터 없음.")

    grid, n_matched, n_grid_slots = build_regular_grid(df)
    span_days = (grid.index.max() - grid.index.min()).total_seconds() / 86400
    coverage_ratio = n_matched / n_grid_slots if n_grid_slots else 0.0
    season_days = pd.Series([_season(d.month) for d in
                             pd.DatetimeIndex(grid.dropna().index.normalize().unique())]).value_counts().to_dict()

    gate = determine_gate(span_days, coverage_ratio, season_days)  # 미달이면 여기서 예외로 멈춤

    feat = build_features(grid)
    results = [run_for_horizon(feat, grid, h) for h in HORIZONS_MIN]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "게이트": gate, "계절별_표본일수": season_days,
        "데이터기간": {"시작": str(grid.dropna().index.min()), "끝": str(grid.dropna().index.max()),
                    "매칭슬롯수": n_matched, "격자슬롯수": n_grid_slots},
        "수평별_결과": results,
        "_주의": f"stage={gate['stage_note']} - technical_trial/preliminary 단계에서는 참고용일 "
              f"뿐이고, official_candidate 단계에 도달해도 promote_to_official은 별도 결정 필요.",
    }
    (OUT_DIR / "부안_초단기_라이브shadow_결과.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
