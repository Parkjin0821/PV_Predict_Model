# -*- coding: utf-8 -*-
"""4지역(광주·부안·김제·영광) D+1+D+2(+48h) 구엔드포인트 710일 백필을
매시간 자동 재개한다.

09-10 사용자 요청: "00:00 되면 알아서 돌리고, 안되면 1시간 단위로
체킹해서 자동 진행". 별도 재시도 로직을 새로 만들지 않고, 기존
`collect_kma_nwp_d1d2_extended_v1_2026-09-08.py`가 이미 가진
"already_complete 스킵" 재개형 설계를 그대로 활용한다 - 이 스크립트를
매시간(00:00 자정 리셋 직후 슬롯 포함) 실행하면:
  - 아직 안 끝난 지역/날짜는 이어서 채워짐
  - 그날 일일한도(typ01, 20,000회 공유)가 아직 안 풀렸으면 이번 시간엔
    호출이 실패/0건일 뿐, 다음 시간 슬롯에 자동 재시도됨
  - 4지역 전부 완료되면 이후 호출은 전부 already_complete로 즉시 스킵
    (API 낭비 없음, 굳이 끄지 않아도 무해)

일일 예산은 4지역 합쳐 4000회로 제한(typ01 공유한도 20,000회 중
ASOS/GRID/D+1 등 다른 일일 수집에도 여유를 남기기 위함).
"""

from __future__ import annotations

import subprocess
import sys
import socket
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
HERE = Path(__file__).resolve().parent
PYTHON = r"C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
COLLECTOR = HERE / "collect_kma_nwp_d1d2_extended_v1_2026-09-08.py"
LOG_DIR = HERE / "logs" / "d1d2_backfill_hourly"

REGIONS = ["광주", "부안", "김제", "영광"]
START_DATE = "2024-08-25"
END_DATE = "2026-08-04"
PER_REGION_BUDGET = 1000  # 4지역 합 최대 4000회/시간 슬롯
CONNECT_TIMEOUT_SECONDS = 5
REGION_PROCESS_TIMEOUT_SECONDS = 180


def endpoint_reachable() -> tuple[bool, str]:
    """인증키를 전송하지 않고 API 호스트의 TCP 443 연결만 짧게 확인한다."""
    host = urlparse(src_url()).hostname
    if not host:
        return False, "collector URL hostname 확인 실패"
    errors = []
    for attempt in range(1, 3):
        try:
            with socket.create_connection((host, 443), timeout=CONNECT_TIMEOUT_SECONDS):
                return True, f"{host}:443 연결 성공({attempt}회차)"
        except OSError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return False, f"{host}:443 연결 2회 실패: {' | '.join(errors)}"


def src_url() -> str:
    """하위 수집기와 같은 모듈에서 실제 엔드포인트를 읽는다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("d1d2_collector", COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.src.URL


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=KST)
    log_path = LOG_DIR / f"{now.strftime('%Y%m%d')}.log"
    lines = [f"=== {now.isoformat(timespec='seconds')} ==="]
    any_fail = False

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        reachable, detail = endpoint_reachable()
        fh.write(f"[endpoint_preflight] reachable={reachable} detail={detail}\n")
        if not reachable:
            fh.write("API 연결 장애로 이번 슬롯을 즉시 종료; 다음 정시에 재시도\n\n")
            print(f"API 연결 사전점검 실패, 즉시 종료: {detail}")
            return 0
        for region in REGIONS:
            cmd = [
                PYTHON, str(COLLECTOR), "--region", region,
                "--start-date", START_DATE, "--end-date", END_DATE,
                "--live", "--max-api-calls", str(PER_REGION_BUDGET),
                "--max-retries", "1", "--timeout-seconds", "10",
                "--max-consecutive-day-failures", "2",
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=REGION_PROCESS_TIMEOUT_SECONDS,
                )
                fh.write(f"--- {region} (exit={proc.returncode}) ---\n")
                fh.write((proc.stdout or "")[-4000:] + "\n")
                if proc.stderr:
                    fh.write("[stderr] " + proc.stderr[-2000:] + "\n")
                if proc.returncode != 0:
                    any_fail = True
            except subprocess.TimeoutExpired as exc:
                fh.write(
                    f"--- {region} {REGION_PROCESS_TIMEOUT_SECONDS}초 초과로 중단; "
                    "다음 정시에 재시도 ---\n"
                )
                any_fail = True
            except Exception as exc:  # noqa: BLE001
                fh.write(f"--- {region} 실행오류: {type(exc).__name__}: {exc} ---\n")
                any_fail = True
        fh.write("\n")

    print(f"완료(로그: {log_path}), any_fail={any_fail}")
    return 0  # 부분 실패도 다음 시간에 자동 재시도되므로 태스크 자체는 항상 성공 처리


if __name__ == "__main__":
    sys.exit(main())
