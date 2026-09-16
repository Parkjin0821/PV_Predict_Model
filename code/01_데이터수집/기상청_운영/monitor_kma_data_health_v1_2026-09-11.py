# -*- coding: utf-8 -*-
"""KMA 데이터 건강상태 주기 모니터(09-11, Claude).

09-11 발견한 사고("오늘 00Z NC 모델 +15h 리드 파일이 KMA 서버에 아직
없음 - API 문제 아니라 순수 발행지연")를 계기로, 매번 수동으로 원인을
다시 파헤치지 않도록 4가지 핵심 신호를 30분마다 자동 기록한다:

  1. apihub.kma.go.kr 자체 접속 가능 여부(TCP 443)
  2. 4지역 ASOS 관측 최신시각·지연시간(시간)
  3. 4지역 GRID(동네예보) 최신 수신시각(원래 하루 1회라 지연 판단은
     "오늘 자정 수신 여부"만 확인)
  4. KIM NC(수치예보) 오늘 00Z 런의 실제 파일 존재 여부 - 라이브
     콜렉터를 흉내내지 않고 원본 API를 직접 한 번 찔러서 "file is not
     exist"인지 실제 값이 있는지 확인(광주 1지역만, 전체 호출량 절약)

결과는 사람이 바로 읽을 수 있는 한 줄 요약을 로그파일에 append하고,
스크립트 자체가 직접 파일에 쓰므로(print() 의존 없음) pythonw.exe로
예약해도 09-11 사고(콘솔 없는 환경에서 print 전용 스크립트가 죽는 문제)
와 무관하게 안전하다.
"""
from __future__ import annotations

import json
import socket
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs" / "kma_data_health"
LOG_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = LOG_DIR / "health_log.jsonl"

ASOS_PATHS = {
    "광주": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3",
    "부안": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3",
    "김제": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",
    "영광": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",
}

ASOS_STALE_THRESHOLD_H = 3.0  # 초단기 shadow가 "결측"으로 치는 기준과 동일

# ★09-16 추가★: GRID(동네예보 격자)는 당일 05시 발표분(tmfc 고정)만
# 수집한다. 수집기 collect_kma_asos_grid_live_v1_2026-08-25.py 의
# 발표전_호출생략 로직과 같은 기준이어야 한다 - 바꿀 땐 양쪽 같이 볼 것.
GRID_PUBLICATION_HOUR_KST = 5


def check_apihub() -> dict:
    try:
        with socket.create_connection(("apihub.kma.go.kr", 443), timeout=8):
            return {"reachable": True}
    except OSError as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def check_asos(now: datetime) -> dict:
    out = {}
    for region, path in ASOS_PATHS.items():
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            row = conn.execute("SELECT MAX(observation_time) FROM asos_hourly").fetchone()
            conn.close()
            latest = row[0]
            if latest:
                latest_dt = datetime.fromisoformat(latest)
                # DB에는 +09:00 오프셋이 붙어 저장돼 있다 - naive/aware를
                # 섞어 빼면 TypeError가 나므로 항상 KST-aware로 맞춘다.
                if latest_dt.tzinfo is None:
                    latest_dt = latest_dt.replace(tzinfo=KST)
                now_aware = now if now.tzinfo else now.replace(tzinfo=KST)
                lag_h = round((now_aware - latest_dt).total_seconds() / 3600, 2)
            else:
                lag_h = None
            out[region] = {"latest": latest, "lag_hours": lag_h,
                            "stale": (lag_h is not None and lag_h > ASOS_STALE_THRESHOLD_H)}
        except Exception as exc:  # noqa: BLE001
            out[region] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def check_grid(now: datetime) -> dict:
    today_str = now.strftime("%Y-%m-%d")
    out = {}
    for region, path in ASOS_PATHS.items():
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            row = conn.execute(
                "SELECT MAX(first_received_at) FROM grid_forecast WHERE first_received_at >= ?",
                (today_str,),
            ).fetchone()
            conn.close()
            out[region] = {"received_today": bool(row[0]), "latest_today": row[0]}
        except Exception as exc:  # noqa: BLE001
            out[region] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


NC_LIVE_LOG_DIR = HERE / "logs" / "kim_nc_live"


def check_nc_file(now: datetime) -> dict:
    """09-11 정정: 원래는 이 함수가 직접 광주 1건을 실제 API로 조회해
    확인했으나, 이것도 공유 할당량을 매 30분 깎아먹는 행위였다(할당량
    소진 의심 국면에서는 이 자체가 문제). 대신 UCUBE_KIM_NC_Live_30min이
    이미 자기 주기(같은 30분)로 실제 호출하고 남긴 로그를 그대로
    읽기만 한다 - 여기서는 API를 전혀 추가로 호출하지 않는다."""
    log_path = NC_LIVE_LOG_DIR / f"{now.strftime('%Y%m%d')}.log"
    if not log_path.exists():
        return {"status": "no_log_today"}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"status": "log_read_error", "error": f"{type(exc).__name__}: {exc}"}

    for line in reversed(lines):
        if "인증/권한 실패" in line or "HTTP 403" in line:
            return {"status": "auth_403", "detail": line.strip(), "source": "passive_log_read"}
        if "발행대기" in line or "publication_pending" in line:
            return {"status": "publication_pending", "detail": line.strip(), "source": "passive_log_read"}
        if "collector end exit=0" in line:
            return {"status": "ok_or_skip", "detail": line.strip(), "source": "passive_log_read"}
        if "FAIL" in line and "광주" in line:
            return {"status": "other_fail", "detail": line.strip(), "source": "passive_log_read"}
    return {"status": "no_recent_entry_found", "source": "passive_log_read"}


def main() -> int:
    now = datetime.now(tz=KST)
    now_naive = now.replace(tzinfo=None)

    result = {
        "checked_at_kst": now.isoformat(timespec="seconds"),
        "apihub": check_apihub(),
        "asos": check_asos(now),
        "grid": check_grid(now),
        "nc_gwangju_probe": check_nc_file(now),
    }

    with JSONL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 사람이 바로 훑어볼 수 있는 한 줄 요약
    asos_stale = [r for r, v in result["asos"].items() if v.get("stale")]
    # ★09-16 수정★: GRID는 당일 05시 발표분(tmfc 고정)이라 05시 전엔 자료가
    # 존재할 수 없다. 09-16에 수집기(collect_kma_asos_grid_live)에 발표전
    # 호출생략을 넣으면서, 00~05시엔 4지역 전부 "오늘 미수신"으로 찍혀
    # 매일 밤 5시간짜리 상시 오탐이 된다. 발표 전이면 '발표전(정상)'으로
    # 구분해서 남긴다(판정 로직·수집엔 영향 없음, 요약 문구만).
    grid_before_publication = now.hour < GRID_PUBLICATION_HOUR_KST
    grid_missing = [r for r, v in result["grid"].items() if not v.get("received_today")]
    result["grid_before_publication"] = grid_before_publication
    if grid_before_publication:
        grid_text = f"발표전(정상, {GRID_PUBLICATION_HOUR_KST}시 발표분 대기)"
    else:
        grid_text = str(grid_missing or "없음")
    nc_status = result["nc_gwangju_probe"].get("status")
    apihub_ok = result["apihub"].get("reachable")

    summary = (
        f"[{now.strftime('%Y-%m-%d %H:%M')}] "
        f"apihub={'OK' if apihub_ok else 'DOWN'} | "
        f"ASOS결측지역={asos_stale or '없음'} | "
        f"GRID오늘미수신={grid_text} | "
        f"NC(광주)={nc_status}"
    )

    text_log_path = LOG_DIR / f"{now.strftime('%Y%m%d')}.log"
    with text_log_path.open("a", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
