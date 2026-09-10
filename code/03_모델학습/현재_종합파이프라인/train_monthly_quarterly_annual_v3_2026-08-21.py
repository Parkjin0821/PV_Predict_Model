# -*- coding: utf-8 -*-
"""중장기(월간·분기·연간) — v3, LightGBM vs XGBoost 명시 비교 추가.

## v2 대비 달라진 것 (사용자 요청, 08-21)
v2는 월간·분기 전부 **LightGBM만** 돌렸다. 같은 날 일간 계층조정에서
"모델 비교를 안 하고 하드코딩했다"는 문제가 드러났으므로(AGENTS.md
"후속 수정" 절), 중장기도 같은 기준으로 **XGBoost를 나란히 비교**한다.

- 월간 walk-forward에서 LightGBM·XGBoost 둘 다 예측 산출
- 분기·연간 bottom-up 합산도 두 모델 각각으로 계산
- 어느 쪽이 우세한지 표본수와 함께 명시(표본이 작아 "우위"를 단정하지
  않는다는 원칙은 유지)

## 주의 (v1·v2에서 이어지는 제약)
분기는 완전분기 5개, 연간은 OOF 백테스트 0개다. 여기서 나오는 모델
비교 결과는 **"어느 쪽이 낫다"는 확정이 아니라 지금 표본에서의 관찰**
이다. 데이터가 쌓이면 자동으로 표본이 늘어나는 구조이므로 주기적으로
재실행할 것.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "월간_분기_연간_v3_2026-08-21"
DAILY_ACTUAL_PARQUET = ROOT / "outputs" / "집계_일간_실제발전량_v2_인버터5부분복구_2026-08-21.parquet"

MIN_TRAIN_MONTHS = 6

MODEL_CANDIDATES = {
    "LightGBM": lambda seed: LGBMRegressor(
        objective="regression_l1", n_estimators=80, learning_rate=0.05,
        num_leaves=7, max_depth=3, min_child_samples=3,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, verbosity=-1, n_jobs=2,
    ),
    # 월간은 표본이 극히 적으므로(20여개) 일간모델의 500그루 설정을 그대로
    # 쓰면 과적합이 뻔하다 — LightGBM 쪽 소표본 설정(80그루·깊이3)과
    # 대등한 규모로 맞춰 공정 비교가 되게 한다.
    "XGBoost": lambda seed: XGBRegressor(
        objective="reg:absoluteerror", n_estimators=80, learning_rate=0.05,
        max_depth=3, min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=seed, n_jobs=2, verbosity=0,
    ),
}


def energy_metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return {"표본수": 0, "MAE_kWh": None, "RMSE_kWh": None, "WAPE_pct": None}
    e = y - p
    denom = float(np.abs(y).sum())
    return {
        "표본수": int(len(y)),
        "MAE_kWh": round(float(np.abs(e).mean()), 1),
        "RMSE_kWh": round(float(np.sqrt(np.mean(e ** 2))), 1),
        "WAPE_pct": round(float(np.abs(e).sum() / denom * 100), 2) if denom > 0 else None,
    }


def build_monthly_series() -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_ACTUAL_PARQUET).copy()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    monthly = daily["일간발전량_kWh"].resample("MS").agg(합계_kWh="sum", 유효일수="count")
    monthly["그달_전체일수"] = monthly.index.days_in_month
    monthly["완전월"] = monthly["유효일수"] >= monthly["그달_전체일수"]
    if "가용인버터수_낮시간평균" in daily.columns:
        monthly["가용인버터수_월평균"] = daily["가용인버터수_낮시간평균"].resample("MS").mean()
    return monthly


def build_feature_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    m = monthly.reindex(full_index)
    target = m["합계_kWh"].where(m["완전월"].fillna(False))

    feat = pd.DataFrame(index=full_index)
    feat["1개월전"] = target.shift(1)
    feat["2개월전"] = target.shift(2)
    feat["3개월전"] = target.shift(3)
    feat["12개월전(전년동월)"] = target.shift(12)
    feat["직전3개월평균"] = target.shift(1).rolling(3, min_periods=3).mean()
    feat["직전6개월평균"] = target.shift(1).rolling(6, min_periods=6).mean()
    feat["월_sin"] = np.sin(2 * np.pi * full_index.month / 12)
    feat["월_cos"] = np.cos(2 * np.pi * full_index.month / 12)
    feat["그달_일수"] = full_index.days_in_month
    if "가용인버터수_월평균" in m.columns:
        feat["가용인버터수_월평균"] = m["가용인버터수_월평균"]
    feat["목표_kWh"] = target
    return feat


def walk_forward(feat: pd.DataFrame, feature_cols: list[str], require_cols: list[str], seed: int) -> pd.DataFrame:
    """expanding-window walk-forward. 후보 모델 전부를 같은 분할에서 평가."""
    valid = feat.dropna(subset=require_cols + ["목표_kWh"])
    rows = []
    for i in range(MIN_TRAIN_MONTHS, len(valid)):
        train = valid.iloc[:i]
        test = valid.iloc[i:i + 1]
        row = {
            "월": test.index[0],
            "실제_kWh": float(test["목표_kWh"].iloc[0]),
            "전월지속성_kWh": float(test["1개월전"].iloc[0]),
            "전년동월_kWh": float(test["12개월전(전년동월)"].iloc[0]) if pd.notna(test["12개월전(전년동월)"].iloc[0]) else None,
            "직전3개월평균_kWh": float(test["직전3개월평균"].iloc[0]),
        }
        for name, make_model in MODEL_CANDIDATES.items():
            model = make_model(seed)
            model.fit(train[feature_cols], train["목표_kWh"])
            row[f"{name}_kWh"] = float(np.clip(model.predict(test[feature_cols])[0], 0, None))
        rows.append(row)
    return pd.DataFrame(rows)


PRED_COLUMNS = [
    ("LightGBM", "LightGBM_kWh"),
    ("XGBoost", "XGBoost_kWh"),
    ("전월지속성", "전월지속성_kWh"),
    ("직전3개월평균", "직전3개월평균_kWh"),
    ("전년동월/동기", "전년동월_kWh"),
]


def summarize(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in PRED_COLUMNS:
        sub = oof.dropna(subset=[col])
        if len(sub) == 0:
            continue
        rows.append({"구성": name, **energy_metrics(sub["실제_kWh"], sub[col])})
    return pd.DataFrame(rows)


def period_bottom_up(oof: pd.DataFrame, freq: str, n_months: int) -> pd.DataFrame:
    """월간 OOF를 분기/연간으로 상향식 합산. 해당 기간의 모든 달이 있어야 채택."""
    df = oof.copy()
    df["기간"] = pd.PeriodIndex(df["월"], freq=freq)
    sum_n = lambda s: s.sum(min_count=n_months)  # noqa: E731
    agg = {"월수": ("실제_kWh", "size"), "실제_kWh": ("실제_kWh", sum_n)}
    for _, col in PRED_COLUMNS:
        agg[col] = (col, sum_n)
    grouped = df.groupby("기간").agg(**agg)
    return grouped[grouped["월수"] == n_months].copy()


def report_period(label: str, table: pd.DataFrame, out_dir: Path) -> pd.DataFrame | None:
    table.to_csv(out_dir / f"{label}_bottomup_상세.csv", encoding="utf-8-sig")
    print(f"\n[{label}] 완전기간 표본수: {len(table)}")
    if len(table) == 0:
        print(f"  완전{label} 0개 — 성능 산출 불가")
        return None
    print(table.to_string())
    rows = []
    for name, col in PRED_COLUMNS:
        sub = table.dropna(subset=[col])
        if len(sub) == 0:
            continue
        rows.append({"구성": name, **energy_metrics(sub["실제_kWh"], sub[col])})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f"{label}_성능요약.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== {label} 성능(bottom-up, 완전{label} {len(table)}개 — 표본크기 주의) ===")
    print(summary_df.to_string(index=False))
    return summary_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])

    monthly = build_monthly_series()
    monthly.to_csv(OUT / "월간_원자료_완전월판정.csv", encoding="utf-8-sig")
    print(f"전체 {len(monthly)}개월 중 완전월 {int(monthly['완전월'].sum())}개")

    feat = build_feature_frame(monthly)
    feature_cols = ["1개월전", "2개월전", "3개월전", "12개월전(전년동월)",
                     "직전3개월평균", "직전6개월평균", "월_sin", "월_cos", "그달_일수"]
    if "가용인버터수_월평균" in feat.columns:
        feature_cols.append("가용인버터수_월평균")
    require_cols = ["1개월전"]

    oof = walk_forward(feat, feature_cols, require_cols, seed)
    oof.to_csv(OUT / "월간_OOF_상세.csv", index=False, encoding="utf-8-sig")
    monthly_summary = summarize(oof)
    monthly_summary.to_csv(OUT / "월간_성능요약.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 월간 성능(시험표본 {len(oof)}개월, expanding walk-forward) ===")
    print(monthly_summary.to_string(index=False))

    q = period_bottom_up(oof, "Q", 3)
    report_period("분기", q, OUT)

    y = period_bottom_up(oof, "Y", 12)
    report_period("연간", y, OUT)

    print("\n[연간 참고] OOF와 무관한 순수 실측 완전연도 총량")
    full_years = monthly[monthly["완전월"]].resample("YS").agg(
        연간실측_kWh=("합계_kWh", "sum"), 완전월수=("완전월", "sum"))
    calendar_full = full_years[full_years["완전월수"] == 12]
    if len(calendar_full):
        print(calendar_full.to_string())
        calendar_full.to_csv(OUT / "연간_실측_완전연도.csv", encoding="utf-8-sig")
    else:
        print("  완전한 실측 1개년 없음")

    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
