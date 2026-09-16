# -*- coding: utf-8 -*-
"""부안·김제·영광 NWP D+1+D+2 확장 수집기(재개형) - +48h 모델 구축용.

## 배경(★09-08 Claude 정정 이후 착수★)
"D+1 자료가 +35h까지밖에 없어서 +48h는 원천 불가능"이라던 이전 판단은
틀렸다(코덱스 정정: KMA KIM 전구모델은 표준화 자료 기준 +135h까지
공식 지원, 지금 +35h인 건 기존 수집기가 D+1 8개 시각만 요청하도록
스스로 제한해놨을 뿐). 이 스크립트는 기존 D+1 전용 수집기
(`collect_kma_nwp_live_v1_2026-08-25.py` 및 그 지역별 런처)를 건드리지
않고, **D+1 8개 시각 + D+2 8개 시각 = 총 16개 시각**을 별도 DB에
추가로 수집한다.

## 재사용(재구현 안 함)
`collect_gwangju_nwp_dayahead_fixed_tm_710d_v2_2026-08-20.py`(src)의
`load_auth_key()`·`VAR_MODEL`·`URL`·`parse_response()`·
`fixed_run_for_issue_day()`를 그대로 가져다 쓴다. 요청 파라미터 구성
(`tm`/`tmef1`/`tmef2`/`int`/`lat`/`lon`/`authKey`)과 응답 파싱 로직은
전혀 새로 만들지 않았다 - 대상시각 목록만 D+1+D+2로 확장했다.

## 안전장치
- **기존 라이브 DB(`kma_live_inputs.sqlite3`)를 건드리지 않는다** - 이미
  스케줄러가 돌고 있는 D+1 전용 파이프라인과 완전히 분리된 새 DB
  (`kma_nwp_d1d2_live_v1_2026-09-08.sqlite3`)에만 쓴다.
- **`--dry-run`이 기본값(True)** - 실제 HTTP 요청 없이 파라미터 구성·
  배치 분할(6개씩)·DB 스키마·재개(resume) 로직만 검증한다. 실제 API
  호출은 `--live` 플래그를 명시해야만 나간다(현재 KMA 원천결측 상황이라
  무의미한 호출을 최소화하려는 코덱스의 계획과 동일 원칙).
- 요청당 최대 6개 시각 제한(기존 D+1 수집기와 동일 근거)을 지켜
  16개를 [6,6,4]로 분할한다.
- `nwp_run_status`에 `issue_date`별 완료 여부를 기록해 재실행 시
  이미 끝난 발행일은 건너뛴다(재개형).

## 사용법
```
# 1) 문법·무API 테스트(기본값)
python collect_kma_nwp_d1d2_extended_v1_2026-09-08.py --region 영광 --issue-date 2026-09-08

# 2) 실제 API 시험(하루만, KMA 상태 확인 후)
python collect_kma_nwp_d1d2_extended_v1_2026-09-08.py --region 영광 --issue-date 2026-09-08 --live
```
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import random
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "01_데이터수집" / "수치예보_일사운량" / "collect_gwangju_nwp_dayahead_fixed_tm_710d_v2_2026-08-20.py"
KST_OFFSET = timedelta(hours=9)

# 09-07 확정: KIMR(DSWRFLX/DIFSWRF)은 영구중단 공식회신 받음 - 요청 안 함.
DEFAULT_VARS = ["DSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]
# ★09-15 변경(사용자 지시 - "API 여유분 충분하니 시간해상도 늘려")★:
# KIM 모델 자체가 1시간 단위로 출력함을 실측 확인(hf=16 등 3의 배수가
# 아닌 값도 정상 응답, 파일명도 ".1hr." - 기존 3시간 간격은 API/모델
# 제약이 아니라 이 스크립트의 선택이었다). 8개(3h간격)→24개(1h간격)로
# 확장 - 하루 기대값이 지역당 80(5변수×16시각)→240(5변수×48시각)으로
# 3배 증가. 하루 API한도 20,000회 중 사용량 1,999회(09-15 15시대
# 사용자 확인) 대비 충분한 여유. 변경 전 8개 버전은
# .backup_before_hourly_20260915에 보존.
FCST_HOURS = list(range(24))  # KST 대상시각(각 day-offset 공통, 0~23시 전체)
MAX_PER_REQUEST = 6

REGIONS = {
    "부안": {"latitude": 35.7874617462152, "longitude": 126.73000042548799,
            "out": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_nwp_d1d2_live_v1_2026-09-08")},
    "김제": {"latitude": 35.80026670423991, "longitude": 126.851859588009,
            "out": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_nwp_d1d2_live_v1_2026-09-08")},
    "영광": {"latitude": 35.323668890815696, "longitude": 126.486825627353,
            "out": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_nwp_d1d2_live_v1_2026-09-08")},
    # ★09-10 추가★: 광주는 애초에 이 710일 D+1+D+2 소급백필 대상에서
    # 빠져있었다(자체 D+1 전용 파이프라인이 있다고 보고 제외된 것으로
    # 추정 - 09-10 사용자 질문으로 발견). NC는 최근 ~70일만 조회 가능해
    # 710일 archive를 절대 못 만드므로, 부안·김제·영광과 같은 방식(구
    # 엔드포인트 소급)으로 뒤늦게 합류시킨다. out 경로는 NC 수집기가
    # 이미 쓰고 있는 DB와 동일하게 맞춰 같은 테이블에 합쳐지게 한다
    # (구간이 겹치지 않아 충돌 없음 - PRIMARY KEY(issue_date,variable,
    # target_time_kst) upsert).
    "광주": {"latitude": 35.15971540328101, "longitude": 126.851512233363,
            "out": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_d1d2_live_v1_2026-09-09")},
}


class LiveNwpError(RuntimeError):
    pass


class LiveNwpAuthError(LiveNwpError):
    pass


class ApiDailyBudgetReached(LiveNwpError):
    """호출 예산에 도달했을 때의 정상적인 중단 신호.

    이미 완료된 issue_date는 run_status가 보존하므로, 다음 날 같은 명령을
    실행하면 미완료 날짜부터 안전하게 이어받는다.
    """
    pass


class ApiCallBudget:
    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0

    def reserve(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise ApiDailyBudgetReached(
                f"일일 API 안전 예산({self.limit:,}회)에 도달했습니다"
            )
        self.used += 1


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"소스 모듈을 불러올 수 없습니다: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


src = load_module("ucube_nwp_d1d2_source", SOURCE)

# ★09-15 추가(Claude, 사용자 요청 - 백필용 별도 인증키 분리)★: D1D2 백필은
# 실시간 수집(ASOS/GRID/NWP D1/NC live)과 같은 authKey를 공유해왔는데,
# 이 둘이 하루 20,000회 한도를 경쟁해 이번 주 내내 403이 반복됐다(AGENTS.md
# 09-11~09-14 항목). 사용자가 백필 전용 신규 authKey를 별도 발급받아,
# 이 파일(4지역 D1D2 백필이 전부 거치는 유일한 진입점)에서만 새 키를 쓰고
# 나머지 실시간 수집기는 기존 키 그대로 두도록 분리한다.
# 새 키는 절대 코드에 평문으로 넣지 않는다 - 환경변수 KMA_AUTH_KEY_BACKFILL을
# 설정해야만 적용되고, 설정 안 하면(기본값) 기존과 완전히 동일하게 동작한다
# (안전한 기본값 - 사용자가 아직 env var를 안 넣었어도 백필이 멈추지 않음).
_original_load_auth_key = src.load_auth_key


def _load_backfill_auth_key() -> str:
    override = os.environ.get("KMA_AUTH_KEY_BACKFILL", "").strip()
    return override if override else _original_load_auth_key()


src.load_auth_key = _load_backfill_auth_key


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def iso_kst(value: datetime) -> str:
    return value.astimezone(KST).isoformat(timespec="seconds")


def kst_targets_d1_d2(issue_day_text: str) -> list[tuple[int, int, datetime]]:
    """(day_offset, KST시, target_utc) 16개 - day_offset 1=D+1, 2=D+2."""
    d_kst_midnight = datetime.strptime(issue_day_text, "%Y%m%d")
    out = []
    for day_offset in (1, 2):
        for hour in FCST_HOURS:
            target_kst = d_kst_midnight + timedelta(days=day_offset, hours=hour)
            target_utc = target_kst - KST_OFFSET
            out.append((day_offset, hour, target_utc))
    return out


def batch_windows(targets: list[tuple[int, int, datetime]], size: int = MAX_PER_REQUEST):
    for i in range(0, len(targets), size):
        yield targets[i:i + size]


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS nwp_d1d2_requests (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          issue_date TEXT NOT NULL,
          variable TEXT NOT NULL,
          window_start_utc TEXT NOT NULL,
          window_end_utc TEXT NOT NULL,
          requested_at TEXT NOT NULL,
          received_at TEXT,
          status TEXT NOT NULL,
          http_status INTEGER,
          attempt_count INTEGER,
          latency_ms INTEGER,
          response_hash TEXT,
          error_type TEXT,
          error_message TEXT,
          dry_run INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nwp_d1d2_values (
          issue_date TEXT NOT NULL,
          variable TEXT NOT NULL,
          day_offset INTEGER NOT NULL,
          target_time_kst TEXT NOT NULL,
          target_time_utc TEXT NOT NULL,
          lead_hours REAL NOT NULL,
          value REAL,
          is_missing INTEGER NOT NULL,
          first_received_at TEXT NOT NULL,
          response_hash TEXT NOT NULL,
          dry_run INTEGER NOT NULL,
          PRIMARY KEY(issue_date, variable, target_time_kst)
        );
        CREATE TABLE IF NOT EXISTS nwp_d1d2_run_status (
          issue_date TEXT PRIMARY KEY,
          completed_at TEXT NOT NULL,
          expected_value_count INTEGER NOT NULL,
          stored_value_count INTEGER NOT NULL,
          missing_value_count INTEGER NOT NULL,
          dry_run INTEGER NOT NULL,
          status_message TEXT NOT NULL
        );
        """
    )
    return conn


def already_done(conn: sqlite3.Connection, issue_day: date, dry_run: bool) -> bool:
    issue_text = issue_day.strftime("%Y%m%d")
    row = conn.execute(
        "SELECT dry_run, expected_value_count, stored_value_count "
        "FROM nwp_d1d2_run_status WHERE issue_date=?",
        (issue_text,),
    ).fetchone()
    if row is None:
        return False
    # dry-run 완료는 실제(live) 재실행을 막지 않는다. live도 80/80 유효값이
    # 모두 저장된 날짜만 완료로 본다. native missing이 남은 날짜는 다음
    # 예약 주기에서 결측 셀만 다시 요청한다.
    return ((row[0] == 0 and row[2] >= row[1]) or dry_run)


def pending_targets(conn: sqlite3.Connection, issue_text: str, var: str,
                    targets: list[tuple[int, int, datetime]], dry_run: bool):
    """이미 저장된 유효 셀을 제외하고 재요청이 필요한 셀만 반환한다."""
    valid_utc = {
        row[0] for row in conn.execute(
            "SELECT target_time_utc FROM nwp_d1d2_values "
            "WHERE issue_date=? AND variable=? AND dry_run=? "
            "AND is_missing=0 AND value IS NOT NULL",
            (issue_text, var, int(dry_run)),
        )
    }
    return [target for target in targets if target[2].isoformat() not in valid_utc]


def request_window_live(session, auth_key, run_tm, var, window, latitude, longitude,
                        max_retries: int, timeout: float, budget: ApiCallBudget):
    utc_instants = [utc for _, _, utc in window]
    params = {
        "nwp": src.VAR_MODEL[var], "varn": var,
        "tm": run_tm.strftime("%Y%m%d%H%M"),
        "tmef1": utc_instants[0].strftime("%Y%m%d%H%M"),
        "tmef2": utc_instants[-1].strftime("%Y%m%d%H%M"),
        "int": "3", "lat": f"{latitude:.8f}", "lon": f"{longitude:.8f}", "authKey": auth_key,
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            budget.reserve()
            response = session.get(src.URL, params=params, timeout=timeout)
            latency_ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in (401, 403):
                raise LiveNwpAuthError(f"NWP 인증/권한 실패(HTTP {response.status_code})")
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = LiveNwpError(f"일시 HTTP 오류 {response.status_code}")
                if attempt < max_retries:
                    time.sleep(min(10 * attempt, 60) + random.random() * 0.25)
                    continue
                raise last_error
            if 400 <= response.status_code < 500:
                raise LiveNwpError(f"NWP 요청 오류(HTTP {response.status_code})")
            response.raise_for_status()
            raw_text = response.content.decode("utf-8", errors="replace")
            by_utc = src.parse_response(raw_text, run_tm, utc_instants)
            values = {utc: by_utc[utc] for utc in utc_instants}
            return values, raw_text, response.status_code, latency_ms, attempt
        except (LiveNwpAuthError, LiveNwpError, src.StopCollection):
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(10 * attempt, 60) + random.random() * 0.25)
                continue
    raise LiveNwpError(f"NWP 요청 실패: {type(last_error).__name__ if last_error else 'unknown'}")


def synth_dry_run_response(window) -> tuple[dict, str]:
    """--dry-run용 가짜 응답 - 실제 API를 절대 호출하지 않는다. 파라미터
    구성·배치분할·DB저장·재개 로직만 검증하는 게 목적이라 값은 전부
    NaN(결측)으로 채운다(가짜 성공 방지 - 진짜처럼 보이는 값을 절대
    만들어내지 않음)."""
    utc_instants = [utc for _, _, utc in window]
    values = {utc: math.nan for utc in utc_instants}
    return values, "DRY_RUN_NO_HTTP_CALL"


def collect_one_issue_day(region: str, issue_day: date, vars_: list[str],
                          dry_run: bool, max_retries: int, timeout: float,
                          budget: ApiCallBudget | None = None) -> dict[str, Any]:
    cfg = REGIONS[region]
    out = cfg["out"]
    conn = open_db(out / "kma_nwp_d1d2_live.sqlite3")
    issue_text = issue_day.strftime("%Y%m%d")
    if already_done(conn, issue_day, dry_run):
        conn.close()
        return {"region": region, "issue_date": issue_text, "status": "already_complete_skip"}

    run_tm = src.fixed_run_for_issue_day(issue_text)
    all_targets = kst_targets_d1_d2(issue_text)
    session = requests.Session() if not dry_run else None
    auth_key = None if dry_run else src.load_auth_key()
    budget = budget or ApiCallBudget(None)
    # ★리드타임 기준시각은 반드시 계획된 발행시각(D 10:00 KST)이어야 한다★
    # - 스크립트를 실행한 시각(now_kst())을 기준으로 삼으면 실행 타이밍에
    # 따라 lead_hours가 흔들려서 09-08에 정정한 "리드타임은 항상 실제
    # 발행시각 기준으로 정확히 기록" 원칙을 어기게 된다.
    planned_issue = datetime(issue_day.year, issue_day.month, issue_day.day, 10, 0, tzinfo=KST)

    stored, missing = 0, 0
    for var in vars_:
        var_targets = pending_targets(conn, issue_text, var, all_targets, dry_run)
        for window in batch_windows(var_targets):
            requested_at = now_kst()
            try:
                if dry_run:
                    values, raw_text, http_status, latency_ms, attempts = (
                        *synth_dry_run_response(window), 200, 0, 0)
                else:
                    values, raw_text, http_status, latency_ms, attempts = request_window_live(
                        session, auth_key, run_tm, var, window,
                        cfg["latitude"], cfg["longitude"], max_retries, timeout, budget)
            except Exception as exc:
                with conn:
                    conn.execute(
                        "INSERT INTO nwp_d1d2_requests(issue_date,variable,window_start_utc,"
                        "window_end_utc,requested_at,status,error_type,error_message,dry_run) "
                        "VALUES (?,?,?,?,?,'failure',?,?,?)",
                        (issue_text, var, window[0][2].isoformat(), window[-1][2].isoformat(),
                         iso_kst(requested_at), type(exc).__name__, str(exc)[:1000], int(dry_run)),
                    )
                conn.close()
                raise
            digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            with conn:
                conn.execute(
                    "INSERT INTO nwp_d1d2_requests(issue_date,variable,window_start_utc,"
                    "window_end_utc,requested_at,received_at,status,http_status,attempt_count,"
                    "latency_ms,response_hash,dry_run) VALUES (?,?,?,?,?,?,'success',?,?,?,?,?)",
                    (issue_text, var, window[0][2].isoformat(), window[-1][2].isoformat(),
                     iso_kst(requested_at), iso_kst(now_kst()), http_status, attempts,
                     latency_ms, digest, int(dry_run)),
                )
                for day_offset, hour, target_utc in window:
                    target_kst = target_utc.replace(tzinfo=UTC).astimezone(KST)
                    lead_h = (target_kst - planned_issue).total_seconds() / 3600
                    value = values[target_utc]
                    is_missing = int(value is None or (isinstance(value, float) and math.isnan(value)))
                    stored += 0 if is_missing else 1
                    missing += 1 if is_missing else 0
                    conn.execute(
                        "INSERT OR REPLACE INTO nwp_d1d2_values(issue_date,variable,day_offset,"
                        "target_time_kst,target_time_utc,lead_hours,value,is_missing,"
                        "first_received_at,response_hash,dry_run) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (issue_text, var, day_offset, target_kst.isoformat(), target_utc.isoformat(),
                         round(lead_h, 2), None if is_missing else float(value), is_missing,
                         iso_kst(now_kst()), digest, int(dry_run)),
                    )

    expected = len(vars_) * len(all_targets)
    stored, missing = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN is_missing=0 AND value IS NOT NULL THEN 1 ELSE 0 END),0), "
        "COALESCE(SUM(CASE WHEN is_missing=1 OR value IS NULL THEN 1 ELSE 0 END),0) "
        "FROM nwp_d1d2_values WHERE issue_date=? AND dry_run=?",
        (issue_text, int(dry_run)),
    ).fetchone()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO nwp_d1d2_run_status(issue_date,completed_at,"
            "expected_value_count,stored_value_count,missing_value_count,dry_run,status_message) "
            "VALUES (?,?,?,?,?,?,?)",
            (issue_text, iso_kst(now_kst()), expected, stored, missing, int(dry_run),
             "dry_run_ok" if dry_run else ("complete_with_native_missing" if missing else "complete")),
        )
    conn.close()
    return {"region": region, "issue_date": issue_text, "dry_run": dry_run,
            "expected": expected, "stored": stored, "missing": missing,
            "lead_hours_covered(vs_10am_issue)": sorted({
                round((t[2].replace(tzinfo=UTC).astimezone(KST) - planned_issue).total_seconds() / 3600, 1)
                for t in all_targets})}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="부안·김제·영광 NWP D+1+D+2 확장 수집기(재개형)")
    p.add_argument("--region", choices=list(REGIONS), help="한 지역만 실행")
    p.add_argument("--all-regions", action="store_true", help="부안·김제·영광 순차 실행")
    p.add_argument("--issue-date", default=None, help="YYYY-MM-DD, 기본값=오늘")
    p.add_argument("--start-date", default=None, help="백필 시작일 YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="백필 종료일 YYYY-MM-DD")
    p.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    p.add_argument("--live", action="store_true", help="명시해야만 실제 API 호출(기본은 dry-run)")
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--timeout-seconds", type=float, default=60)
    p.add_argument("--max-api-calls", type=int, default=None,
                   help="이번 실행 전체의 실제 HTTP 호출 안전 상한. dry-run에는 무관")
    p.add_argument("--max-consecutive-day-failures", type=int, default=None,
                   help="연속 날짜 실패가 이 횟수에 도달하면 조기 중단(기본: 제한 없음)")
    # ★09-15 추가(사용자 지시)★: 시간해상도를 실행 인자로 분리한다.
    # 09-15 낮에 FCST_HOURS를 1시간 간격(24개/일)으로 올렸는데, 하루치
    # 비용이 80→240회로 3배가 되면서 "과거 날짜를 많이 채운다"는 백필
    # 본래 목적의 처리량이 정확히 1/3로 떨어졌다(20,000회/일 기준
    # 250일분 → 83일분). 실시간 수집은 1시간(운영 유연성), 과거 백필은
    # 3시간(처리량 우선)으로 분리해서 둘 다 살린다.
    p.add_argument("--hours-step", type=int, default=1, choices=(1, 3),
                   help="대상시각 간격(시간). 1=매시(기본, 실시간용 240개/일), "
                        "3=3시간 간격(과거 백필용 80개/일, 처리량 3배)")
    p.add_argument("--newest-first", action="store_true",
                   help="백필 범위를 최신 날짜부터 처리해 Shadow 검증용 최근 구간을 우선 확보")
    args = p.parse_args()
    if bool(args.region) == bool(args.all_regions):
        p.error("--region 하나 또는 --all-regions 중 하나를 지정해야 합니다")
    if args.issue_date and (args.start_date or args.end_date):
        p.error("--issue-date와 --start-date/--end-date는 함께 쓸 수 없습니다")
    if bool(args.start_date) != bool(args.end_date):
        p.error("--start-date와 --end-date는 함께 지정해야 합니다")
    if args.max_api_calls is not None and args.max_api_calls <= 0:
        p.error("--max-api-calls는 양수여야 합니다")
    if args.max_consecutive_day_failures is not None and args.max_consecutive_day_failures <= 0:
        p.error("--max-consecutive-day-failures는 양수여야 합니다")
    return args


def main() -> None:
    args = parse_args()
    # ★09-15★ 해상도를 인자대로 적용(기본 1시간). 모듈 전역을 바꾸므로
    # kst_targets_d1_d2()·_expected_value_count() 등 하위 참조가 전부
    # 이 값을 따라간다.
    global FCST_HOURS
    FCST_HOURS = list(range(0, 24, args.hours_step))
    if args.start_date:
        start_day = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_day = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        start_day = (datetime.strptime(args.issue_date, "%Y-%m-%d").date()
                     if args.issue_date else now_kst().date())
        end_day = start_day
    if end_day < start_day:
        raise ValueError("--end-date는 --start-date보다 빠를 수 없습니다")

    regions = list(REGIONS) if args.all_regions else [args.region]
    if args.all_regions and len(regions) > 1:
        # ★★09-16 신규(사용자 제안 - 라운드로빈)★★: REGIONS 삽입순서가
        # 항상 부안→김제→영광→광주라, 예산(--max-api-calls)이 모자라는
        # 실행에서는 매번 순서상 마지막인 광주만 굶었다(09-16 실측:
        # KIM NC 당일발행 순간 4지역이 동시에 대량요청으로 전환되며
        # 500예산을 소진, 광주만 ApiDailyBudgetReached - AGENTS.md 참고).
        # 예산을 2000으로 올려 근본원인은 없앴지만, 순서 고정 자체가
        # 구조적 불공평이라 방어적으로 라운드로빈을 추가한다.
        # 실행 시각(30분 슬롯 번호)로 시작 지점을 매번 돌려, 예산이
        # 부족해지는 어떤 상황에서도 특정 지역만 상시 불리해지지 않게 한다.
        now = now_kst()
        slot = (now.hour * 60 + now.minute) // 30
        offset = slot % len(regions)
        regions = regions[offset:] + regions[:offset]
    budget = ApiCallBudget(args.max_api_calls if args.live else None)
    results: list[dict[str, Any]] = []
    stopped_reason = None
    consecutive_day_failures = 0
    stop_for_failures = False
    day_count = (end_day - start_day).days + 1
    days = [start_day + timedelta(days=i) for i in range(day_count)]
    if args.newest_first:
        days.reverse()
    try:
        for day in days:
            for region in regions:
                # ★09-10 정정★: 이전엔 하루라도 예외(예: NC 대상기간 밖
                # "file is not exist")가 나면 while 루프 전체가 죽어서
                # 뒤 날짜들을 하나도 시도조차 못 했다(광주 백필 2회
                # 연속 조기중단으로 발견). 날짜 하나의 실패가 나머지를
                # 막지 않도록 격리하고, 결과 목록엔 실패 기록을 남긴다.
                try:
                    one_result = collect_one_issue_day(
                        region, day, args.vars, dry_run=not args.live,
                        max_retries=args.max_retries, timeout=args.timeout_seconds,
                        budget=budget,
                    )
                    results.append(one_result)
                    consecutive_day_failures = 0
                except ApiDailyBudgetReached:
                    raise
                except Exception as exc:  # noqa: BLE001
                    results.append({
                        "region": region, "issue_date": day.isoformat(),
                        "status": "day_failed", "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300],
                    })
                    consecutive_day_failures += 1
                    if (args.max_consecutive_day_failures is not None and
                            consecutive_day_failures >= args.max_consecutive_day_failures):
                        stopped_reason = (
                            f"연속 날짜 실패 {consecutive_day_failures}회로 조기 중단; "
                            "다음 예약 실행에서 자동 재개"
                        )
                        stop_for_failures = True
                        break
            if stop_for_failures:
                break
    except ApiDailyBudgetReached as exc:
        stopped_reason = str(exc)
    result = {
        "status": "early_stop_resume_next_run" if stopped_reason else "complete",
        "regions": regions,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "completed_or_skipped_issue_region_count": len(results),
        "api_calls_used": budget.used if args.live else 0,
        "api_call_limit": args.max_api_calls if args.live else None,
        "stopped_reason": stopped_reason,
        "last_result": results[-1] if results else None,
    }
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
