# -*- coding: utf-8 -*-
"""김제: 실제 사용 가능한 계절·시험행 수 감사 + 결함구간 후보 탐지.

부안 audit_buan_season_testrows_v1_2026-08-28.py와 완전히 동일한 방법론 -
build_gimje_time_aggregates_v1_2026-08-31.py가 만든 일간 공식후보 CSV를
읽기전용으로 읽어 판정한다. API 호출 없음, 원본 데이터 수정 없음, 기상
백필과 독립적(발전량 시간집계만 있으면 됨).

결함구간 후보가 나와도 여기서 바로 확정하지 않는다 - 부안 때도 자동감지
경계(05-25)와 실측 정밀경계(05-22)가 3일 차이났다. 사람이 일별 valid
플래그를 직접 대조해서 최종 경계를 정할 것(임의 축소/확대 금지).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DAILY_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_발전소_일간_공식후보.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "계절시험행_감사_2026-08-31"

VALID_STATUS = "valid_daylight_ge90pct"


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "겨울"
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    return "가을"


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(DAILY_CSV, encoding="utf-8-sig")
    df["date_kst"] = pd.to_datetime(df["date_kst"])
    df["valid"] = df["quality_status"] == VALID_STATUS
    df["month"] = df["date_kst"].dt.month
    df["계절"] = df["month"].apply(_season)
    return df.sort_values("date_kst").reset_index(drop=True)


def find_defect_streaks(df: pd.DataFrame, ratio_threshold: float = 0.9,
                        min_streak_days: int = 10) -> list[dict]:
    """valid_ratio가 threshold 미만인 날이 연속(최대 3일 정상값 허용)으로
    min_streak_days 이상 이어지는 구간을 결함후보로 잡는다 - 확정 아님,
    부안 defect_period(04-15~05-22)와 같은 역할을 할 김제 후보 탐지용."""
    bad = (df["daylight_valid_ratio"].fillna(0) < ratio_threshold).to_numpy()
    dates = df["date_kst"].to_numpy()
    streaks = []
    i = 0
    n = len(bad)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        good_gap = 0
        while j < n:
            if bad[j]:
                good_gap = 0
                j += 1
            elif good_gap < 3:
                good_gap += 1
                j += 1
            else:
                break
        length = j - i
        if length >= min_streak_days:
            streaks.append({
                "시작": str(pd.Timestamp(dates[i]).date()),
                "종료": str(pd.Timestamp(dates[j - 1]).date()),
                "일수": int(length),
                "구간내_valid_ratio_평균": float(df["daylight_valid_ratio"].iloc[i:j].fillna(0).mean()),
            })
        i = j
    return streaks


def run() -> dict:
    df = load_daily()
    total = len(df)
    valid_total = int(df["valid"].sum())

    monthly = (df.groupby(df["date_kst"].dt.strftime("%Y-%m"))
              .agg(총일=("valid", "size"), 유효일=("valid", "sum")))
    monthly_records = [{"연월": ym, "총일": int(r["총일"]), "유효일": int(r["유효일"])}
                       for ym, r in monthly.iterrows()]

    seasonal = df.groupby("계절").agg(총일=("valid", "size"), 유효일=("valid", "sum"))
    all_seasons = ["봄", "여름", "가을", "겨울"]
    seasonal_records = {}
    for s in all_seasons:
        if s in seasonal.index:
            row = seasonal.loc[s]
            seasonal_records[s] = {"총일": int(row["총일"]), "유효일": int(row["유효일"])}
        else:
            seasonal_records[s] = {"총일": 0, "유효일": 0}

    defect_candidates = find_defect_streaks(df)

    missing_seasons = [s for s in all_seasons if seasonal_records[s]["총일"] == 0]
    multi_year = bool(
        df["date_kst"].dt.year.nunique() >= 2
        and df.groupby("계절")["date_kst"].apply(lambda x: x.dt.year.nunique() >= 2).any()
    )

    gwangju_5season_feasible = bool((not missing_seasons) and multi_year)
    verdict = {
        "광주식_5계절_rolling_origin_CV_적용가능": gwangju_5season_feasible,
        "불가사유": [] if gwangju_5season_feasible else [
            *(["완전히 없는 계절: " + ",".join(missing_seasons)] if missing_seasons else []),
            *(["연도 반복 없음(단일 관측기간, 계절이 1회씩만 등장)"] if not multi_year else []),
        ],
        "권장": "위 불가사유가 비어있지 않으면 광주와 동일한 5계절 walk-forward "
              "구조 대신 축소된 시간순 holdout/expanding-window을 쓰고 그 "
              "한계를 명시할 것. 김제는 710일(부안 236일보다 훨씬 김)이라 "
              "부안보다는 계절 커버리지가 나을 가능성이 높음 - 실측으로 확인.",
    }

    result = {
        "데이터출처": str(DAILY_CSV),
        "총일수": total, "총유효일수": valid_total,
        "유효율_pct": round(valid_total / total * 100, 1) if total else None,
        "기간": {"시작": str(df["date_kst"].min().date()), "종료": str(df["date_kst"].max().date())},
        "월별": monthly_records,
        "계절별": seasonal_records,
        "결함구간_후보(확정아님)": defect_candidates,
        "판정": verdict,
    }
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "김제_계절시험행_감사.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
