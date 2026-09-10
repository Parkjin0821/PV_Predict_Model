# -*- coding: utf-8 -*-
"""영광 NWP·GRID 백필 재개 - 스케줄러용 오케스트레이터.

★09-03 오전 교체★: 원래 .bat 래퍼(resume_yeonggwang_backfill_v1_2026-09-02.bat)가
cmd.exe의 한글 경로 인코딩 버그로 00:10 실행이 완전 실패(로그도 못 씀,
LastTaskResult=255)한 걸 실측으로 확인. 파이썬은 유니코드 경로를 그대로
다뤄서 이 문제 자체가 없다 - 기존 UCUBE_* 스케줄러들과 동일한 패턴(파이썬
스크립트를 python.exe로 직접 호출)으로 교체.

재개 명령 2건은 AGENTS.md 09-01/09-02 절에 문서화된 것과 완전히 동일 -
로직 변경 없음, 완료일 자동감지로 안전하게 이어받는다(subprocess로 기존
검증된 스크립트를 그대로 호출할 뿐, 재구현 아님).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
PYTHON = r"C:\Users\u-cube\AppData\Local\Python\bin\python.exe"
LOG_PATH = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31"
    r"\resume_log_2026-09-03.txt"
)

NWP_SCRIPT = (
    r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
    r"\01_데이터수집\수치예보_일사운량"
    r"\collect_gwangju_nwp_dayahead_fixed_tm_710d_v2_2026-08-20.py"
)
NWP_ARGS = [
    "--vars", "DSWRF", "TCDC", "LCDC", "MCDC", "HCDC",
    "--latitude", "35.323668890815696", "--longitude", "126.486825627353",
    "--site-tag", "영광",
    "--output-dir",
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31\NWP",
]

GRID_SCRIPT = (
    r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
    r"\01_데이터수집\격자예보\collect_grid_forecast_backfill_v1_2026-09-01.py"
)
GRID_ARGS = [
    "--nx", "52", "--ny", "78", "--vars", "REH", "POP", "SKY",
    "--start-date", "2024-08-25", "--end-date", "2026-08-04",
    "--output-csv",
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31"
    r"\GRID\영광_격자예보_REH_POP_SKY_710일.csv",
    "--site-tag", "영광",
]


def run_and_log(label: str, script: str, args: list[str]) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n--- {label}: {datetime.now(tz=KST).isoformat()} ---\n")
        log.flush()
        result = subprocess.run(
            [PYTHON, script, *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        log.write(result.stdout)
        if result.stderr:
            log.write("\n[stderr]\n" + result.stderr)
        log.write(f"\n[exit_code={result.returncode}]\n")
        log.flush()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n===== 영광 백필 재개 시작: {datetime.now(tz=KST).isoformat()} =====\n")
    run_and_log("NWP 재개", NWP_SCRIPT, NWP_ARGS)
    run_and_log("GRID 재개", GRID_SCRIPT, GRID_ARGS)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n===== 영광 백필 재개 종료: {datetime.now(tz=KST).isoformat()} =====\n")
    print("완료, 로그:", LOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
