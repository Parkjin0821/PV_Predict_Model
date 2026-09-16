# -*- coding: utf-8 -*-
"""부안 초단기(+1/+2/+3/+4h) 과거 백테스트(08-31, Codex 지적 반영: 라이브
DB만 볼 게 아니라 2년 가까운 과거 5분 인버터 자료로 지금 바로 만들 수
있다). 부안_인버터별_5분_야간0포함.parquet을 읽기전용으로 쓴다(API
호출 없음).

## 설계(지적 반영)
- 8대 완전가용 시각(5분 단위, 순간마다 8대 전부 ok)만 발전소 총출력으로
  합산 - 부분합계는 쓰지 않는다.
- 결함구간(2026-04-15~05-22) 제외.
- persistence·선형회귀·LightGBM을 +1h/+2h/+3h/+4h별로 비교.
- 검증: 날짜(달력일) 기준 expanding-window walk-forward(총출력모델과
  동일 원칙 - 초기60일학습→20일시험, 마지막 폴드가 남은 날 전부 흡수),
  pooled 지표.
- 특성·타깃 모두 "8대완전가용 시각"에서만 뽑는다 - 소스·타깃 시각
  둘 다 유효해야 함(하나라도 결측이면 그 표본은 버림, 보간 없음).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
SRC = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_라이브연계_v1_2026-09-14"
    r"\부안_인버터별_5분_야간0포함_라이브연계.parquet"
)
# ★09-14 변경★: 기존 08-28 정적 Excel 기반 parquet(08-05에서 하드리밋
# - 원천 자체 한계, 09-14 코덱스 확인)에서 코덱스가 신규 구축한
# 라이브연계 브릿지(엑셀+Blockdata 라이브 DB, 경계 중복 0건 확인,
# 2026-09-14까지 연장)로 교체. AGENTS.md 09-14 항목 참고. 이 파일은
# 공용 모듈이라 원본 SRC 교체가 곧 "공식" 경로에도 영향 - 재학습은
# 반드시 `retrain_buan_ultrashort_v1_2026-09-14.py`(bundle_version
# 인자로 별도 폴더 저장)를 통해서만 실행하고, 이 파일의 main()을
# 직접 돌려 기존 공식 번들을 덮어쓰지 않을 것.
OUT_DIR = HERE / "outputs" / "초단기_과거백테스트_2026-08-31"
DEFECT_START = pd.Timestamp("2026-04-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2026-05-23")
HORIZONS_MIN = [60, 120, 180, 240]
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20
SEED = 42


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "겨울"
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    return "가을"


def load_plant_5min_series() -> pd.Series:
    df = pd.read_parquet(SRC, columns=[
        "grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"])
    # 공통 품질정책의 현재 명칭과 기존 저장자료 명칭을 모두 읽는다.
    df["ok"] = df["quality_status_after_night"].isin(
        ["observed", "physical_zero_night", "night_zero_physical"]
    )
    ok = df[df["ok"]]
    count8 = ok.groupby("grid_time_kst")["inverter_number"].nunique()
    valid_times = count8[count8 == 8].index
    total = ok[ok["grid_time_kst"].isin(valid_times)].groupby("grid_time_kst")["ac_power_kw"].sum()

    total = total[(total.index.normalize() < DEFECT_START) | (total.index.normalize() >= DEFECT_END_EXCLUSIVE)]

    start, end = total.index.min().floor("5min"), total.index.max().ceil("5min")
    grid = pd.date_range(start, end, freq="5min")
    return total.reindex(grid)  # 격자 밖 시각은 그대로 NaN(보간 없음)


def build_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"value": s})
    df["lag_15m"] = s.shift(3)
    df["lag_30m"] = s.shift(6)
    df["lag_1h"] = s.shift(12)
    df["roll_1h_mean"] = s.rolling(12, min_periods=12).mean()
    df["lag_1day_same_time"] = s.shift(288)
    df["date"] = df.index.normalize()
    return df


def expanding_folds_full_coverage(days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    n = len(days)
    folds, end = [], initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((days[:end], days[end:test_end]))
        end = test_end
    return folds


def pooled_score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = y_true - pred
    return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
            "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}


def run_for_horizon(feat: pd.DataFrame, s: pd.Series, horizon_min: int) -> dict:
    steps = horizon_min // 5
    data = feat.copy()
    data["y"] = s.shift(-steps)
    data["persistence"] = data["value"]
    feature_cols = ["value", "lag_15m", "lag_30m", "lag_1h", "roll_1h_mean", "lag_1day_same_time"]
    data = data.dropna(subset=feature_cols + ["y"])

    days = pd.DatetimeIndex(np.sort(data["date"].unique()))
    folds = expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    def wf(model_kind: str) -> dict | None:
        all_true, all_pred = [], []
        for train_days, test_days in folds:
            train = data[data["date"].isin(train_days)]
            test = data[data["date"].isin(test_days)]
            if len(train) < 500 or len(test) < 1:
                continue
            if model_kind == "lightgbm":
                m = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                                  max_depth=4, min_child_samples=30, subsample=0.9,
                                  colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=SEED, n_jobs=-1, verbosity=-1)
            else:
                m = LinearRegression()
            m.fit(train[feature_cols], train["y"])
            pred = np.clip(m.predict(test[feature_cols]), 0, None)
            all_true.append(test["y"].to_numpy()); all_pred.append(pred)
        if not all_true:
            return None
        y = np.concatenate(all_true); p = np.concatenate(all_pred)
        return pooled_score(y, p)

    lgbm = wf("lightgbm")
    lin = wf("linear")
    pers_score = pooled_score(data["y"].to_numpy(), data["persistence"].to_numpy())

    def with_improve(score):
        if score is None:
            return None
        return {**score,
                "MAE_개선율_pct": round((1 - score["MAE_kW"] / pers_score["MAE_kW"]) * 100, 1) if pers_score["MAE_kW"] else None,
                "RMSE_개선율_pct": round((1 - score["RMSE_kW"] / pers_score["RMSE_kW"]) * 100, 1) if pers_score["RMSE_kW"] else None}

    return {
        "horizon_min": horizon_min, "표본일수": int(len(days)), "폴드수": len(folds),
        "persistence": pers_score,
        "LightGBM": with_improve(lgbm),
        "선형회귀": with_improve(lin),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = load_plant_5min_series()
    feat = build_features(s)
    season_days = pd.Series([_season(d.month) for d in pd.DatetimeIndex(feat["date"].dropna().unique())])

    results = [run_for_horizon(feat, s, h) for h in HORIZONS_MIN]

    out = {
        "데이터": {"8대완전가용_5분표본수": int(s.notna().sum()),
                "기간": [str(s.dropna().index.min()), str(s.dropna().index.max())]},
        "계절별_표본일수": season_days.value_counts().to_dict(),
        "수평별_결과": results,
        "_주의": "과거 아카이브 기반 백테스트 - 라이브 환경 재현성은 별도 shadow 검증 필요"
              "(최소30일 기술시험/90일 예비평가/150일+계절확보 후 공식판정, 사용자 로드맵대로). "
              "가을 표본 없음(부안 데이터 공통 한계). promote_to_official 대상 아님.",
    }
    (OUT_DIR / "부안_초단기_과거백테스트_결과.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
