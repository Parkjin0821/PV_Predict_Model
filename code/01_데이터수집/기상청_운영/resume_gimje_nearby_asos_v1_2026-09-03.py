# -*- coding: utf-8 -*-
"""김제 인근관측소(전주146·정읍245) ASOS 백필 이어받기 - 스케줄러용.

## 배경
사용자 지시(09-03): "영광 끝났으니 오늘 남은 API 할당량으로 이어받되,
라이브 수집(광주·부안·김제 정기 스케줄) 쓸 최소량은 남겨둬라." 08-31
절 "API 할당량 사용순서" 3번 항목(영광 먼저 → 그날 API 잔여 할당량
확인되면 김제 이어받기, 처음부터 재호출 안 함)을 그대로 실행하는
스크립트다.

## 왜 "늦은 시간"에 도는가(사용자의 "최소는 남겨두고" 요구 반영)
실시간으로 "남은 할당량"을 조회하는 API가 없다(초과는 403으로만
감지됨). 그래서 하루 중 라이브 수집 정기 스케줄(15분·매시·매일 단위로
계속 호출)이 이미 그날 몫을 다 쓴 뒤인 **23:00 KST**에 이 스크립트를
돌려, 그 시점까지 남아있는 할당량만 이 백필이 쓰게 한다 - 라이브
수집을 방해하지 않는 가장 단순한 방법.

## 재사용한 것(재구현 안 함)
`backfill_buan_asos_apihub_auto_v1_2026-08-31.py`를 그대로
subprocess로 호출한다(완결성 실패 자동수용판, 이름은 '부안'이지만
`--station`으로 다른 지점도 그대로 처리 가능하게 설계된 범용
스크립트 - 08-31 절 "완결성 실패 자동수용판" 참고). 체크포인트
(`완료일.json`)가 이미 있어 **처음부터 재호출하지 않고 기존
완료일(전주146 329일·정읍245 318일, 09-01 기준) 다음부터 이어받는다**
(스크립트 자체 내장 로직, 여기서 새로 구현 안 함).

## 종료 조건
API 할당량 소진(403)이면 그 지점에서 조용히 멈추고(체크포인트는 이미
저장된 상태) 다음 스테이션으로 넘어간다 - 하루에 다 못 끝내도 다음날
같은 시각에 또 돌아서 이어간다. 두 지점 다 710일(목표 종료일
2026-08-04)에 도달하면 이 태스크는 그때 사람이 확인 후 스케줄러에서
삭제할 것(영광 ONETIME 완료 후 삭제한 것과 동일 패턴, 자동삭제는
안 함 - 완료 여부는 사람이 파일행수로 재확인하는 게 이 프로젝트 원칙).
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
PYTHON = r"C:\Users\u-cube\AppData\Local\Python\bin\python.exe"
HERE = Path(__file__).resolve().parent
BACKFILL_SCRIPT = HERE / "backfill_buan_asos_apihub_auto_v1_2026-08-31.py"
OUT_ROOT = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
)
LOG_PATH = OUT_ROOT / "resume_log_2026-09-03.txt"

# (지점번호, 출력폴더명) - 08-31절 기존 진행상황: 전주146=329/710일,
# 정읍245=318/710일(09-01 15:18 기준, 이후 미변경 09-03 13:56 확인).
STATIONS = [
    (146, "전주146"),
    (245, "정읍245"),
]
START_DATE = "2024-08-25"
END_DATE = "2026-08-04"


def run_and_log(label: str, station: int, out_dir: Path) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n--- {label}(station={station}) 재개: "
                  f"{datetime.now(tz=KST).isoformat()} ---\n")
        log.flush()
        result = subprocess.run(
            [PYTHON, str(BACKFILL_SCRIPT),
             "--start-date", START_DATE, "--end-date", END_DATE,
             "--station", str(station), "--output-dir", str(out_dir)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        log.write(result.stdout)
        if result.stderr:
            log.write("\n[stderr]\n" + result.stderr)
        log.write(f"\n[exit_code={result.returncode}]\n")
        log.flush()


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n===== 김제 인근관측소 백필 재개 시작: "
                  f"{datetime.now(tz=KST).isoformat()} =====\n")
    for station, folder in STATIONS:
        run_and_log(folder, station, OUT_ROOT / folder)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n===== 김제 인근관측소 백필 재개 종료: "
                  f"{datetime.now(tz=KST).isoformat()} =====\n")
    print("완료, 로그:", LOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
