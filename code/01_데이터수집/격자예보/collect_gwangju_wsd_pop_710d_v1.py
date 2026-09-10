"""Collect WSD and POP forecasts for all 710 Gwangju issue dates.

This is a versioned wrapper around the existing authenticated UCUBE collector.
It does not copy, display, or log the KMA API key. Progress is saved daily and
the same command can be rerun to resume safely.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd


SOURCE_COLLECTOR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\grid_forecast_resume_20260606.py"
)
OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\grid_forecast_wsd_pop_710d_v4_2026-08-19"
)
OUTPUT_CSV = OUTPUT_DIR / "광주_격자예보_WSD_POP_710일.csv"
START = datetime(2024, 8, 25)
END = datetime(2026, 8, 4)
VARIABLES = ["WSD", "POP"]


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "ucube_grid_forecast_authenticated", SOURCE_COLLECTOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 인증 수집기를 불러올 수 없습니다: {SOURCE_COLLECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_output(module) -> None:
    if not OUTPUT_CSV.is_file():
        print(f"최종 파일이 아직 없습니다: {OUTPUT_CSV}")
        return

    df = pd.read_csv(OUTPUT_CSV, dtype={"발표일": str})
    expected_days = module.expected_days(START, END)
    required = [
        f"{variable}_{hour:02d}h"
        for hour in module.FCST_HOURS
        for variable in VARIABLES
    ]
    missing_columns = [column for column in required if column not in df.columns]
    missing_dates = sorted(set(expected_days) - set(df["발표일"].astype(str)))
    duplicates = int(df["발표일"].duplicated().sum())
    null_cells = int(df[required].isna().sum().sum()) if not missing_columns else -1

    print(f"행 수: {len(df)} / 예상 {len(expected_days)}")
    print(f"누락 열: {len(missing_columns)}")
    print(f"누락 날짜: {len(missing_dates)}")
    print(f"중복 날짜: {duplicates}")
    print(f"필수 값 결측 셀: {null_cells}")
    if (
        len(df) == len(expected_days)
        and not missing_columns
        and not missing_dates
        and duplicates == 0
        and null_cells == 0
    ):
        print("[PASS] 2단계 WSD·POP 710일 수집 검증 완료")
    else:
        print("[FAIL] 같은 명령을 다시 실행해 복구하거나 오류 로그를 확인하세요")


def main() -> int:
    if not SOURCE_COLLECTOR.is_file():
        raise FileNotFoundError(SOURCE_COLLECTOR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collector = load_collector()
    collector.START = START
    collector.END = END
    collector.VARS_TODO = VARIABLES
    collector.RECOVERY_CSV = OUTPUT_CSV

    # The WSD/POP file is a standalone source table. Do not merge it into the
    # TMP/SKY/REH dataset until both have passed their own validation.
    collector.build_final_csv = lambda: validate_output(collector)
    result = collector.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())

