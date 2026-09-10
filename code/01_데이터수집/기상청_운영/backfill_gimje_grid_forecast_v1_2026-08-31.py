"""김제 격자예보(동네예보) REH·POP·SKY 과거 백필(08-31).

부안 상관분석에서 REH·POP·SKY가 DSWRF 다음으로 강한 신호였던 걸 근거로
(AGENTS.md 08-31 절 참고) 김제도 처음부터 GRID 3종을 함께 백필한다 -
부안처럼 나중에 따로 추가하지 않고 5·6단계 모델 설계 전에 미리 확보.

backfill_buan_grid_forecast_v1_2026-08-31.py와 완전히 동일한 패턴 -
검증된 인증 수집기(grid_forecast_resume_20260606.py)를 그대로 재사용,
좌표·격자·기간만 김제 값으로 덮어쓴다.
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
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제"
    r"\기상과거백필_v1_2026-08-31\GRID"
)
OUTPUT_CSV = OUTPUT_DIR / "김제_격자예보_REH_POP_SKY_710일.csv"

# 김제(7018) 격자좌표 - 김제_기상수집경로_v1_2026-08-31.json과 동일
# (Claude가 KMA 표준 Lambert 변환식으로 직접 계산, 광주58/74·부안56/88 재현으로 검증완료).
GIMJE_NX = 58
GIMJE_NY = 88
GRID_W = 149  # 전국 격자 폭 - 지점에 따라 안 바뀜(광주·부안과 동일 상수)

# 김제 과거 Excel과 동일한 710일 범위(부안 236일보다 훨씬 긺 - 부안은 2025-12-12 시작이었지만
# 김제 Excel은 2024-08-25부터 있음).
START = datetime(2024, 8, 25)
END = datetime(2026, 8, 4)
VARIABLES = ["REH", "POP", "SKY"]


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "ucube_grid_forecast_authenticated_gimje", SOURCE_COLLECTOR
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
        print("[PASS] 김제 격자예보 REH·POP·SKY 710일 수집 검증 완료")
    else:
        print("[FAIL] 같은 명령을 다시 실행해 복구하거나 오류 로그를 확인하세요")


def main() -> int:
    if not SOURCE_COLLECTOR.is_file():
        raise FileNotFoundError(SOURCE_COLLECTOR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collector = load_collector()

    collector.NX = GIMJE_NX
    collector.NY = GIMJE_NY
    collector.IDX = (GIMJE_NY - 1) * GRID_W + (GIMJE_NX - 1)

    collector.START = START
    collector.END = END
    collector.VARS_TODO = VARIABLES
    collector.RECOVERY_CSV = OUTPUT_CSV

    collector.build_final_csv = lambda: validate_output(collector)
    result = collector.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
