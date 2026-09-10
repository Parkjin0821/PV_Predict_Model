# -*- coding: utf-8 -*-
"""중장기(월간·분기·연간) 누적 발전량 예측 — v2, 5번 인버터 결측 복구 반영.

## v1 대비 달라진 것
v1(같은 날 앞서 작성) 이후 5번 인버터 통신다운(2025-08-15~11-16)과
6/18 소폭결측을 4/5(또는 그 이상) 인버터 부분합으로 복구했다
(`02_전처리/recover_inverter5_gap_v1_2026-08-21.py`, 원본 대비 정합성
검증·보정계수 적용까지 완료). 그 결과:
- 완전월이 25개월 중 18개 → **22개로 증가**.
- **2025년(1~12월) 전체가 처음으로 완전한 1개년이 됐다** — 연간 검증
  표본이 0개에서 1개(비록 1개뿐이지만)로 늘었다.
- 데이터 소스만 `집계_일간_실제발전량_v2_인버터5부분복구_2026-08-21.parquet`
  로 교체하고 나머지 방법론(월간 자기이력 walk-forward → 분기 bottom-up)
  은 v1과 동일하게 재사용한다.
- **신규**: 연간도 이제 "정식 모델은 아니지만 1개 표본짜리 bottom-up
  참고치"를 산출한다(v1은 이것조차 못 했음 — 표본 0개였으므로).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "월간_분기_연간_v2_2026-08-21"
DAILY_ACTUAL_PARQUET = ROOT / "outputs" / "집계_일간_실제발전량_v2_인버터5부분복구_2026-08-21.parquet"

MIN_TRAIN_MONTHS = 6


def energy_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return {"표본수": 0, "MAE_kWh": None, "RMSE_kWh": None, "WAPE_pct": None}
    e = y - p
    denom = float(np.abs(y).sum())
    return {
        "표본수": int(len(y)),
        "MAE_kWh": float(np.abs(e).mean()),
        "RMSE_kWh": float(np.sqrt(np.mean(e ** 2))),
        "WAPE_pct": float(np.abs(e).sum() / denom * 100) if denom > 0 else None,
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
    feat["월"] = full_index.month
    feat["월_sin"] = np.sin(2 * np.pi * full_index.month / 12)
    feat["월_cos"] = np.cos(2 * np.pi * full_index.month / 12)
    feat["그달_일수"] = full_index.days_in_month
    if "가용인버터수_월평균" in m.columns:
        feat["가용인버터수_월평균"] = m["가용인버터수_월평균"]  # 부분용량 달을 모델이 알아챌 수 있게
    feat["목표_kWh"] = target
    return feat


def walk_forward(feat: pd.DataFrame, feature_cols: list[str], require_cols: list[str], seed: int) -> pd.DataFrame:
    valid = feat.dropna(subset=require_cols + ["목표_kWh"])
    rows = []
    for i in range(MIN_TRAIN_MONTHS, len(valid)):
        train = valid.iloc[:i]
        test = valid.iloc[i:i + 1]
        model = LGBMRegressor(
            objective="regression_l1", n_estimators=80, learning_rate=0.05,
            num_leaves=7, max_depth=3, min_child_samples=3,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbosity=-1, n_jobs=2,
        )
        model.fit(train[feature_cols], train["목표_kWh"])
        pred_ml = float(np.clip(model.predict(test[feature_cols])[0], 0, None))
        actual = float(test["목표_kWh"].iloc[0])
        rows.append({
            "월": test.index[0], "실제_kWh": actual, "LightGBM_kWh": pred_ml,
            "전월지속성_kWh": float(test["1개월전"].iloc[0]),
            "전년동월_kWh": float(test["12개월전(전년동월)"].iloc[0]) if pd.notna(test["12개월전(전년동월)"].iloc[0]) else None,
            "직전3개월평균_kWh": float(test["직전3개월평균"].iloc[0]),
        })
    return pd.DataFrame(rows)


def summarize(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in [("LightGBM", "LightGBM_kWh"), ("전월지속성", "전월지속성_kWh"),
                       ("직전3개월평균", "직전3개월평균_kWh")]:
        m = energy_metrics(oof["실제_kWh"], oof[col])
        rows.append({"구성": name, **m})
    yoy = oof.dropna(subset=["전년동월_kWh"])
    if len(yoy) > 0:
        m = energy_metrics(yoy["실제_kWh"], yoy["전년동월_kWh"])
        rows.append({"구성": "전년동월", **m})
    return pd.DataFrame(rows)


def period_bottom_up(oof: pd.DataFrame, freq: str, n_months: int) -> pd.DataFrame:
    """월간 OOF를 분기(freq='Q')·연간(freq='A')으로 상향식 합산. 그 기간의
    모든 달이 OOF에 있어야만(부분기간 제외) 채택."""
    df = oof.copy()
    df["기간"] = pd.PeriodIndex(df["월"], freq=freq)
    sum_n = lambda s: s.sum(min_count=n_months)  # noqa: E731
    grouped = df.groupby("기간").agg(
        월수=("실제_kWh", "size"),
        실제_kWh=("실제_kWh", sum_n),
        LightGBM_kWh=("LightGBM_kWh", sum_n),
        전월지속성_kWh=("전월지속성_kWh", sum_n),
        직전3개월평균_kWh=("직전3개월평균_kWh", sum_n),
    )
    complete = grouped[grouped["월수"] == n_months].copy()
    yoy = df.dropna(subset=["전년동월_kWh"]).groupby("기간")["전년동월_kWh"].agg(["sum", "size"])
    if len(yoy):
        complete["전년동기_kWh"] = yoy.loc[yoy["size"] == n_months, "sum"]
    return complete


def report_period(name: str, table: pd.DataFrame, out_dir: Path) -> None:
    table.to_csv(out_dir / f"{name}_bottomup_상세.csv", encoding="utf-8-sig")
    print(f"\n[{name}] 완전기간 표본수: {len(table)}")
    if len(table) == 0:
        print(f"  완전{name} 0개 — 성능 산출 불가")
        return
    print(table.to_string())
    rows = []
    for label, col in [(f"LightGBM(합산)", "LightGBM_kWh"), (f"전월지속성(합산)", "전월지속성_kWh"),
                        (f"직전3개월평균(합산)", "직전3개월평균_kWh")]:
        m = energy_metrics(table["실제_kWh"], table[col])
        rows.append({"구성": label, **m})
    if "전년동기_kWh" in table.columns and table["전년동기_kWh"].notna().any():
        yq = table.dropna(subset=["전년동기_kWh"])
        m = energy_metrics(yq["실제_kWh"], yq["전년동기_kWh"])
        rows.append({"구성": "전년동기(합산)", **m})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f"{name}_성능요약.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== {name} 성능(bottom-up, 완전{name} {len(table)}개 — 표본크기 주의) ===")
    print(summary_df.to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])

    monthly = build_monthly_series()
    monthly.to_csv(OUT / "월간_원자료_완전월판정.csv", encoding="utf-8-sig")
    n_complete = int(monthly["완전월"].sum())
    print(f"전체 {len(monthly)}개월 중 완전월 {n_complete}개(v1: 18개였음)")
    print(monthly[["합계_kWh", "유효일수", "그달_전체일수", "완전월"]].to_string())

    feat = build_feature_frame(monthly)
    feature_cols = ["1개월전", "2개월전", "3개월전", "12개월전(전년동월)",
                     "직전3개월평균", "직전6개월평균", "월_sin", "월_cos", "그달_일수"]
    if "가용인버터수_월평균" in feat.columns:
        feature_cols.append("가용인버터수_월평균")
    require_cols = ["1개월전"]

    print(f"\n[월간] walk-forward 가능 표본: {len(feat.dropna(subset=require_cols + ['목표_kWh']))}개")

    oof = walk_forward(feat, feature_cols, require_cols, seed)
    oof.to_csv(OUT / "월간_OOF_상세.csv", index=False, encoding="utf-8-sig")
    monthly_summary = summarize(oof)
    monthly_summary.to_csv(OUT / "월간_성능요약.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 월간 성능(시험표본 {len(oof)}개월, expanding walk-forward) ===")
    print(monthly_summary.to_string(index=False))

    print("\n[분기] 월간 OOF 3개월 bottom-up 합산 — 완전분기만 채택")
    q = period_bottom_up(oof, "Q", 3)
    report_period("분기", q, OUT)

    print("\n[연간] 월간 OOF 12개월 bottom-up 합산 — 완전연도만 채택(v1은 0개였음)")
    y = period_bottom_up(oof, "Y", 12)
    report_period("연간", y, OUT)
    if len(y) == 0:
        print("  (참고) OOF 시험구간에 12개월이 전부 들어간 연도가 아직 없다는 뜻 —")
        print("  monthly walk-forward의 초기 학습구간(MIN_TRAIN_MONTHS)에 걸쳐있는")
        print("  달이 있으면 그 해는 bottom-up에서 제외된다. 아래 실제(비OOF) 연간")
        print("  총량은 참고용으로 별도 표기한다.")

    # OOF 여부와 무관하게, 실측만으로 계산 가능한 완전연도 총량은 참고치로 항상 남긴다
    print("\n[연간 참고] OOF 여부와 무관한 순수 실측 완전연도 총량")
    full_years = monthly[monthly["완전월"]].resample("YS").agg(연간실측_kWh=("합계_kWh", "sum"), 완전월수=("완전월", "sum"))
    calendar_full = full_years[full_years["완전월수"] == 12]
    if len(calendar_full):
        print(calendar_full.to_string())
        calendar_full.to_csv(OUT / "연간_실측_완전연도.csv", encoding="utf-8-sig")
    else:
        print("  완전한 실측 1개년이 여전히 없음")

    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
