"""Validate the completed 710-day TMP/SKY/REH Gwangju forecast dataset.

Read-only validator. It never edits the collected CSV.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\grid_forecast_710d_collection_v3_2026-08-19"
    r"\광주_격자예보_3시간단위_TMP_SKY_REH_710일.csv"
)
START = date(2024, 8, 25)
END = date(2026, 8, 4)
HOURS = (0, 3, 6, 9, 12, 15, 18, 21)
VARIABLES = ("TMP", "SKY", "REH")


def expected_dates() -> list[str]:
    result: list[str] = []
    current = START
    while current <= END:
        result.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return result


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.is_file():
        print(f"[WAIT] 아직 최종 파일이 없습니다: {csv_path}")
        return 2

    df = pd.read_csv(csv_path, dtype={"발표일": str})
    required = ["발표일"] + [
        f"{variable}_{hour:02d}h"
        for hour in HOURS
        for variable in VARIABLES
    ]
    missing_columns = [column for column in required if column not in df.columns]
    duplicated_dates = int(df["발표일"].duplicated().sum()) if "발표일" in df else -1
    expected = expected_dates()
    actual = set(df["발표일"].astype(str)) if "발표일" in df else set()
    missing_dates = [day for day in expected if day not in actual]
    unexpected_dates = sorted(actual - set(expected))
    null_cells = int(df[required[1:]].isna().sum().sum()) if not missing_columns else -1

    print(f"파일: {csv_path}")
    print(f"행 수: {len(df)} / 예상 {len(expected)}")
    print(f"기간: {df['발표일'].min()} ~ {df['발표일'].max()}")
    print(f"누락 열: {len(missing_columns)}")
    print(f"누락 날짜: {len(missing_dates)}")
    print(f"중복 날짜: {duplicated_dates}")
    print(f"범위 밖 날짜: {len(unexpected_dates)}")
    print(f"필수 값 결측 셀: {null_cells}")

    ok = (
        len(df) == len(expected)
        and not missing_columns
        and not missing_dates
        and duplicated_dates == 0
        and not unexpected_dates
        and null_cells == 0
    )
    print("[PASS] 1단계 검증 완료" if ok else "[FAIL] 위 항목을 확인하세요")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

