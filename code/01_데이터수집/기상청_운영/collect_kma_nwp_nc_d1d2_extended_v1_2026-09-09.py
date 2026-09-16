# -*- coding: utf-8 -*-
"""KIM NC 표준화 API 기반 4지역 D+1+D+2 수집기.

기존 09-08 확장 수집기의 저장 스키마·재개·호출 예산 안전장치는 그대로
재사용하고, 결측이 계속된 산업특화 텍스트 API만 공식 KIM NC 지점 API로
교체한다. 기존 행은 삭제하지 않으며, NC 80/80 비결측 확인 뒤에만 해당
발행일을 완료로 건너뛴다.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈 로드 실패: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


legacy = load("ucube_d1d2_legacy", HERE / "collect_kma_nwp_d1d2_extended_v1_2026-09-08.py")
nc = load("ucube_d1d2_nc", HERE / "collect_kma_nwp_nc_live_v1_2026-09-09.py")
NC_CUTOVER_KST = "2026-09-09T13:45:00+09:00"

# +48h 체계도 광주를 포함한 4지역으로 통일한다.
legacy.REGIONS["광주"] = {
    "latitude": 35.15971540328101,
    "longitude": 126.851512233363,
    "out": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_d1d2_live_v1_2026-09-09"),
}


# ★09-15 변경★: 하드코딩 80 대신 legacy.FCST_HOURS/DEFAULT_VARS에서
# 동적으로 계산 - 09-15 시간해상도 확장(8→24시각/일)으로 하루 기대값이
# 80→240이 됐고, 앞으로 또 바뀌어도 이 값이 자동으로 따라가게 함.
def _expected_value_count() -> int:
    return len(legacy.FCST_HOURS) * 2 * len(legacy.DEFAULT_VARS)


def nc_done(conn, issue_day: date, dry_run: bool) -> bool:
    if dry_run:
        return False
    row = conn.execute(
        "SELECT dry_run, expected_value_count, stored_value_count, missing_value_count, status_message, completed_at "
        "FROM nwp_d1d2_run_status WHERE issue_date=?", (issue_day.strftime("%Y%m%d"),)
    ).fetchone()
    if row is None:
        return False
    # ★09-15 재수정★: 해상도를 실행 인자로 분리(--hours-step)하면서, 같은
    # 발행일이라도 3시간(80개)로 받힌 날과 1시간(240개)로 받힌 날이 공존한다.
    # 그래서 "현재 실행의 기대값"과 비교하면 안 되고(그러면 80개 완결일을
    # 미완료로 오판해 240개로 통째 재수집 = 할당량 낭비), **그 행 자체가
    # 내부적으로 완결인지**(stored==expected, missing==0)만 본다.
    stored_ok = int(row[1]) > 0 and int(row[1]) == int(row[2]) and int(row[3]) == 0
    source_confirmed = str(row[4]) == "complete_nc" or str(row[5]) >= NC_CUTOVER_KST
    return bool(int(row[0]) == 0 and stored_ok and source_confirmed)


def nc_request_window(session, auth_key, run_tm, var, window, latitude, longitude,
                      max_retries, timeout, budget):
    """NC point API는 hf 한 개씩 조회하므로 3시간 대상시각을 개별 요청한다."""
    nc_name, scale = nc.VAR_MAP[var]
    values, raw_parts, total_ms, attempts_total = {}, [], 0, 0
    for _, _, target_utc in window:
        hf = round((target_utc - run_tm).total_seconds() / 3600)
        params = {
            "group": "KIMG", "nwp": "NE57", "data": "U", "name": nc_name,
            "tmfc": run_tm.strftime("%Y%m%d%H"), "hf": str(hf), "disp": "A", "help": "1",
            "lat": f"{latitude:.8f}", "lon": f"{longitude:.8f}", "authKey": auth_key,
        }
        for attempt in range(1, max_retries + 1):
            started = time.perf_counter()
            try:
                budget.reserve()
                response = session.get(nc.NC_URL, params=params, timeout=timeout)
                total_ms += round((time.perf_counter() - started) * 1000)
                attempts_total += 1
                if response.status_code in (401, 403):
                    raise legacy.LiveNwpAuthError(f"KIM NC 인증/권한 실패(HTTP {response.status_code})")
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    raise legacy.LiveNwpError(f"KIM NC 일시 HTTP 오류 {response.status_code}")
                response.raise_for_status()
                text = nc.decode_text(response.content)
                values[target_utc] = nc.parse_one(text, run_tm, target_utc, nc_name) * scale
                raw_parts.append(text)
                break
            except legacy.LiveNwpAuthError:
                raise
            except (requests.RequestException, legacy.LiveNwpError) as exc:
                if attempt == max_retries:
                    raise legacy.LiveNwpError(f"KIM NC 요청 실패: {type(exc).__name__}") from exc
                time.sleep(min(5 * attempt, 20))
    return values, "\n".join(raw_parts), 200, total_ms, attempts_total


_legacy_collect = legacy.collect_one_issue_day


def collect_one_issue_day(region, issue_day, vars_, dry_run, max_retries, timeout, budget=None):
    cfg = legacy.REGIONS[region]
    conn = legacy.open_db(cfg["out"] / "kma_nwp_d1d2_live.sqlite3")
    if nc_done(conn, issue_day, dry_run):
        conn.close()
        return {"region": region, "issue_date": issue_day.strftime("%Y%m%d"),
                "status": "already_complete_nc_skip", "source": "KIM_NC"}
    conn.close()
    result = _legacy_collect(region, issue_day, vars_, dry_run, max_retries, timeout, budget)
    if not dry_run:
        conn = legacy.open_db(cfg["out"] / "kma_nwp_d1d2_live.sqlite3")
        try:
            # 해상도(80/240)와 무관하게, 이번 실행이 기대한 개수를 결측 없이
            # 다 채웠으면 완결로 본다(--hours-step 분리 이후 재수정).
            status = ("complete_nc" if result.get("missing", 1) == 0
                       and result.get("stored") == result.get("expected") else "complete_nc_with_missing")
            with conn:
                conn.execute("UPDATE nwp_d1d2_run_status SET status_message=? WHERE issue_date=?",
                             (status, issue_day.strftime("%Y%m%d")))
        finally:
            conn.close()
    result["source"] = "KIM_NC"
    return result


legacy.already_done = nc_done
legacy.request_window_live = nc_request_window
legacy.collect_one_issue_day = collect_one_issue_day


if __name__ == "__main__":
    legacy.main()
