"""★09-01 저녁부로 대체(SUPERSEDED)★ - 더 이상 실행되지 않음, 이력 보존용으로만 남김.

범용 CLI 러너 `collect_grid_forecast_backfill_v1_2026-09-01.py`가 이
래퍼를 흡수했다. 앞으로는 그걸 쓸 것:

    python collect_grid_forecast_backfill_v1_2026-09-01.py \\
      --nx 52 --ny 78 --vars REH POP SKY \\
      --start-date 2024-08-25 --end-date 2026-08-04 \\
      --output-csv "...\\영광\\기상과거백필_v1_2026-08-31\\GRID\\영광_격자예보_REH_POP_SKY_710일.csv" \\
      --site-tag 영광

09-01 저녁 실측: 위 명령으로 기존 26일을 정확히 감지하고 27일째부터
이어받다가 동일한 403(할당량 소진)으로 안전 중단 - 이 래퍼와 동일하게
동작함을 확인.

--- 이하 원본 docstring(참고용, 새로 수정·재사용하지 말 것) ---

영광 김삿갓1(7912) 격자예보(REH/POP/SKY) 710일 수집 - 재개용 래퍼.

기존 8/31 실행이 API 일일 할당량 소진(HTTP 403)으로 26일째(2024-09-19)에서
멈췄다. 이 래퍼는 grid_forecast_resume_20260606.py를 그대로 재사용하고
(로직 변경 없음) 영광 격자좌표·변수·기간·출력경로만 주입한다 - 광주
collect_gwangju_wsd_pop_710d_v1.py와 동일한 override 패턴.

같은 명령을 다시 실행하면 기존 26행을 건너뛰고 이어서 수집한다
(completed_recovery_days()가 출력 CSV의 기존 발표일을 읽어 todo에서 제외).

좌표 출처: 영광_기상수집경로_v1_2026-08-31.json village_forecast_grid
  (nx=52, ny=78 - 김제 3단계와 동일한 Lambert Conformal Conic 변환, 기존
  확정값 재현 검증 후 대입, Claude 08-31).
변수 순서(REH/POP/SKY)·710일 범위(2024-08-25~2026-08-04)는 기존 26행
파일의 헤더·발표일과 정확히 일치시킴 - 새 파일 아님, 이어쓰기.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd


SOURCE_COLLECTOR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\grid_forecast_resume_20260606.py"
)
OUTPUT_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31"
    r"\GRID\영광_격자예보_REH_POP_SKY_710일.csv"
)
START = datetime(2024, 8, 25)
END = datetime(2026, 8, 4)
VARIABLES = ["REH", "POP", "SKY"]
NX, NY = 52, 78


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "ucube_grid_forecast_authenticated_yeonggwang", SOURCE_COLLECTOR
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
        print("[PASS] 영광 격자예보 REH/POP/SKY 710일 수집 검증 완료")
    else:
        print("[FAIL] 같은 명령을 다시 실행해 복구하거나 오류 로그를 확인하세요")


def main() -> int:
    if not SOURCE_COLLECTOR.is_file():
        raise FileNotFoundError(SOURCE_COLLECTOR)
    if not OUTPUT_CSV.is_file():
        raise FileNotFoundError(
            f"기존 영광 부분수집 파일이 없습니다(새 파일을 만들려던 게 아니라면 확인): {OUTPUT_CSV}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    collector = load_collector()

    # 영광 격자좌표로 교체 - IDX는 NX/NY에서 파생되므로 반드시 같이 재계산.
    collector.NX, collector.NY = NX, NY
    collector.IDX = (collector.NY - 1) * collector.GRID_W + (collector.NX - 1)

    collector.START = START
    collector.END = END
    collector.VARS_TODO = VARIABLES
    collector.RECOVERY_CSV = OUTPUT_CSV

    # 영광은 독립 산출물 - 광주 TMP/SKY/REH 통합본과 병합하지 않는다.
    collector.build_final_csv = lambda: validate_output(collector)
    result = collector.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
