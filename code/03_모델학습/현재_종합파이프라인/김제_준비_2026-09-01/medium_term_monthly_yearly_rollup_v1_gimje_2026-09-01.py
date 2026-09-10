# -*- coding: utf-8 -*-
"""김제 중장기 - 월간·연간 누적 집계(09-01).

새 모델 아님 - 이미 있는 일간 실적(김제_발전소_일간_공식후보.csv)을
월/연 단위로 합산만 한다. 프로젝트 확정 티어 정의("일/월/연 주기, 일간·
월간 누적")의 나머지 부분.

## 원칙(임의 추정 금지 재적용)
- 결측일(quality_status != valid_daylight_ge90pct)의 발전량은 절대 추정해서
  채우지 않는다 - 유효일만 그대로 합산하고, 그 합계가 "그 달의 진짜 총량"이
  아니라 "관측된 유효일만의 합계"임을 명시한다(부분합 스케일업 금지 원칙과
  동일 정신).
- 월/연이 시작·끝 경계에서 통째로 안 채워진 경우(예: 2024-08은 8/25부터,
  2026-08은 8/4까지만) "일수_해당월실제일수"와 "일수_보유일수"를 분리해서
  보여준다 - 달력상 31일인데 데이터가 7일뿐이어도 "31일 중 7일 관측"으로
  투명하게 남긴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DAILY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_발전소_일간_공식후보.csv"
)
DEFECT_CONFIG = HERE.parent.parent.parent / "02_전처리" / "김제" / "config" / "김제_전처리_규칙_v1_2026-08-31.json"
OUT_DIR = HERE / "outputs" / "김제_중장기_월간연간집계_v1_2026-09-01"

VALID_STATUS = "valid_daylight_ge90pct"
MONTH_COMPLETE_MIN_RATIO = 0.90  # 월 유효일 비율이 이 이상이면 "완결" 등급


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(DAILY_CSV)
    df["date_kst"] = pd.to_datetime(df["date_kst"])
    df["valid"] = df["quality_status"] == VALID_STATUS
    df["daily_energy_kwh_valid_only"] = df["daily_energy_kwh"].where(df["valid"])

    defect_cfg = json.loads(DEFECT_CONFIG.read_text(encoding="utf-8"))["defect_period"]
    defect_start = pd.Timestamp(defect_cfg["start_inclusive"])
    defect_end_excl = pd.Timestamp(defect_cfg["end_exclusive"])
    df["is_defect_period"] = (df["date_kst"] >= defect_start) & (df["date_kst"] < defect_end_excl)

    return df.sort_values("date_kst").reset_index(drop=True)


def calendar_days_in_month(year: int, month: int) -> int:
    return int(pd.Period(f"{year}-{month:02d}").days_in_month)


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = df["date_kst"].dt.to_period("M")
    rows = []
    for ym, g in df.groupby("year_month"):
        n_observed_days = len(g)
        n_valid_days = int(g["valid"].sum())
        n_defect_days = int(g["is_defect_period"].sum())
        n_calendar_days = calendar_days_in_month(ym.year, ym.month)
        valid_ratio = n_valid_days / n_calendar_days if n_calendar_days else np.nan
        energy_sum = float(g["daily_energy_kwh_valid_only"].sum(min_count=1)) if n_valid_days else None
        is_boundary_partial = n_observed_days < n_calendar_days  # 월 시작/끝이 데이터기간 경계에 걸림
        rows.append({
            "연월": str(ym),
            "달력상_일수": n_calendar_days,
            "관측보유_일수": n_observed_days,
            "유효_일수": n_valid_days,
            "결함구간_일수": n_defect_days,
            "유효율_pct": round(valid_ratio * 100, 1) if not np.isnan(valid_ratio) else None,
            "월간_유효일합계_kWh": round(energy_sum, 1) if energy_sum is not None else None,
            "등급": (
                "달력경계_부분월(해석주의)" if is_boundary_partial else
                "완결" if valid_ratio >= MONTH_COMPLETE_MIN_RATIO else
                "부분(유효일만 합산, 실제 총량보다 낮을 수 있음)"
            ),
        })
    return pd.DataFrame(rows)


def build_yearly(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    m["연"] = m["연월"].str[:4]
    rows = []
    for year, g in m.groupby("연"):
        has_boundary = (g["등급"] == "달력경계_부분월(해석주의)").any()
        rows.append({
            "연": year,
            "포함_월수": len(g),
            "관측보유_일수_합": int(g["관측보유_일수"].sum()),
            "유효_일수_합": int(g["유효_일수"].sum()),
            "결함구간_일수_합": int(g["결함구간_일수"].sum()),
            "연간_유효일합계_kWh": round(float(g["월간_유효일합계_kWh"].sum(skipna=True)), 1),
            "달력경계_부분월_포함여부": bool(has_boundary),
            "등급": "부분월포함(해석주의)" if has_boundary or len(g) < 12 else "완결(12개월)",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_daily()
    monthly = build_monthly(daily)
    yearly = build_yearly(monthly)

    monthly.to_csv(OUT_DIR / "김제_월간_집계.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT_DIR / "김제_연간_집계.csv", index=False, encoding="utf-8-sig")

    summary = {
        "원칙": "결측일 발전량은 추정하지 않음 - 유효일만 합산, 그 합계는 '관측된 유효일 합계'일 뿐 "
              "그 달/연의 확정 총량이 아닐 수 있음(부분합 스케일업 금지 원칙과 동일).",
        "데이터기간": f"{daily['date_kst'].min().date()} ~ {daily['date_kst'].max().date()}",
        "월_수": len(monthly),
        "완결_월수": int((monthly["등급"] == "완결").sum()),
        "부분_월수(경계+저유효율)": int((monthly["등급"] != "완결").sum()),
        "연_수": len(yearly),
        "월간_집계_파일": str(OUT_DIR / "김제_월간_집계.csv"),
        "연간_집계_파일": str(OUT_DIR / "김제_연간_집계.csv"),
        "_참고": "월간·연간 '예측'(미래 총량 forecast)은 별도다 - 김제 NWP는 D+1까지만 있어 "
               "월 단위 기상기반 예측은 불가, 계절climatology 기반 추정만 가능(필요시 별도 요청).",
    }
    (OUT_DIR / "김제_월간연간집계_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("=== 월간 집계(전체) ===")
    print(monthly.to_string(index=False))
    print()
    print("=== 연간 집계 ===")
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
