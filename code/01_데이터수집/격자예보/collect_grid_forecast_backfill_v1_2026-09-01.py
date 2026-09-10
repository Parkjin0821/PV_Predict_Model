# -*- coding: utf-8 -*-
"""범용 격자예보(동네예보) 과거 백필 수집기 - 4개 발전소 공용, CLI 파라미터 방식.

09-01 "지역 공통 파이프라인" 논의 결과: 격자예보·NWP 라이브 수집기
2개는 사이트별 새 파일(래퍼) 대신 CLI 파라미터로 통일하기로 결정
(전면 통합은 안 함 - 모델링/상관분석/특성선택은 사이트별 물리적 차이가
실재해서 그대로 유지, AGENTS.md 해당 절 참고).

핵심 코어(`grid_forecast_resume_20260606.py`)는 절대 수정하지 않는다
- 그 파일의 "인자 없이 실행하면 광주 710일 TMP/SKY/REH 이력을 그대로
재현한다"는 기존 동작(과거 재현성)을 그대로 보존해야 하기 때문이다.
대신 이 스크립트가 그 모듈을 임포트한 뒤 전역변수(NX/NY/IDX/VARS_TODO/
START/END/RECOVERY_CSV/BASE_CSV_1/2/FINAL_CSV)를 CLI 인자로 덮어쓰는
override 패턴을 그대로 쓰되, 사이트마다 새 파일을 안 만들어도 되게
파라미터화했다 - 광주 60일 복구·WSD/POP 확장(08-19), 김제/부안 NWP
라이브(09-01), 영광 GRID 백필(09-01)에서 각각 따로 만들었던 래퍼들이
전부 이 한 스크립트로 흡수된다(단, 이미 실행된 과거 래퍼 파일 자체는
이력 보존을 위해 삭제하지 않고 파일 상단에 "대체됨" 표시만 추가).

사용 예(영광 GRID 백필 이어받기):
    python collect_grid_forecast_backfill_v1_2026-09-01.py \\
      --nx 52 --ny 78 --vars REH POP SKY \\
      --start-date 2024-08-25 --end-date 2026-08-04 \\
      --output-csv "C:\\...\\영광\\...\\영광_격자예보_REH_POP_SKY_710일.csv" \\
      --site-tag 영광

같은 명령을 다시 실행하면 기존 파일에 있는 발표일은 건너뛰고 이어서
수집한다(원본 `completed_recovery_days()` 로직 그대로 상속) - 새 파일이
아니라 지정한 `--output-csv`에 바로 이어붙인다(RECOVERY_CSV=FINAL_CSV=
그 경로로 지정하므로 별도 통합 단계 불필요, 08-31 WSD/POP 래퍼와
동일한 "독립 산출물" 취급 - 병합 검증만 자체 수행).

nx/ny 참고값(09-01 기준 확정): 광주(58,74) · 부안(56,88) · 김제(58,88)
· 영광(52,78).
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd


SOURCE_COLLECTOR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\grid_forecast_resume_20260606.py"
)


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "ucube_grid_forecast_authenticated_generic", SOURCE_COLLECTOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 인증 수집기를 불러올 수 없습니다: {SOURCE_COLLECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_output(module, output_csv: Path, start: datetime, end: datetime,
                     variables: list[str], site_tag: str) -> None:
    if not output_csv.is_file():
        print(f"최종 파일이 아직 없습니다: {output_csv}")
        return

    df = pd.read_csv(output_csv, dtype={"발표일": str})
    expected_days = module.expected_days(start, end)
    required = [
        f"{variable}_{hour:02d}h"
        for hour in module.FCST_HOURS
        for variable in variables
    ]
    missing_columns = [column for column in required if column not in df.columns]
    missing_dates = sorted(set(expected_days) - set(df["발표일"].astype(str)))
    duplicates = int(df["발표일"].duplicated().sum())
    null_cells = int(df[required].isna().sum().sum()) if not missing_columns else -1

    print(f"[{site_tag}] 행 수: {len(df)} / 예상 {len(expected_days)}")
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
        print(f"[PASS] {site_tag} 격자예보 {'/'.join(variables)} {len(expected_days)}일 수집 검증 완료")
    else:
        print("[FAIL] 같은 명령을 다시 실행해 복구하거나 오류 로그를 확인하세요")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="범용 격자예보(동네예보) 과거 백필 수집기")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--vars", nargs="+", required=True,
                         help="예: REH POP SKY / TMP SKY REH / WSD POP 등 - "
                              "dfs_shrt_grd API가 지원하는 동네예보 변수코드")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-csv", type=Path, required=True,
                         help="기존 부분수집 파일이 있으면 그 경로를 그대로 지정 - 이어붙임")
    parser.add_argument("--site-tag", default="", help="로그 표시용(선택)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    if end < start:
        raise SystemExit("--end-date는 --start-date보다 빠를 수 없습니다.")

    if not SOURCE_COLLECTOR.is_file():
        raise FileNotFoundError(SOURCE_COLLECTOR)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    collector = load_collector()

    collector.NX, collector.NY = args.nx, args.ny
    collector.IDX = (collector.NY - 1) * collector.GRID_W + (collector.NX - 1)

    collector.START = start
    collector.END = end
    collector.VARS_TODO = args.vars
    collector.RECOVERY_CSV = args.output_csv

    site_tag = args.site_tag or args.output_csv.stem
    collector.build_final_csv = lambda: validate_output(
        collector, args.output_csv, start, end, args.vars, site_tag
    )
    result = collector.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
