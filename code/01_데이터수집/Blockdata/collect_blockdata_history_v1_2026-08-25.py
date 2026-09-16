# -*- coding: utf-8 -*-
"""Blockdata 실시간 스냅샷 이력 수집기.

광주 발전소(기본 ID 6715)의 ``GET /data/{plant_id}`` 응답을 주기적으로
읽어 다음 세 층으로 보존한다.

1. API 원문 JSON: 동일 응답은 SHA-256 기준 한 번만 파일/DB에 저장
2. 인버터별 측정값: (plant_id, inverter_number, measurement_time) 중복 방지
3. 발전소 합계: 각 응답의 최신 측정시각 기준 합계와 품질 플래그

API 키는 환경변수 ``BLOCKDATA_API_KEY`` 또는 기존 키 파일에서 읽으며
출력·로그·DB에 절대 저장하지 않는다. 기본 실행은 1회 수집이다. Windows
작업 스케줄러에서 5분마다 1회 실행하는 방식을 권장한다. ``--loop``는
개발/임시 운영용이다.

오프라인 검증 예시::

    python collect_blockdata_history_v1_2026-08-25.py \
      --input-json "...\\data_6715_snapshot.json" --output-dir "...\\test"

실제 1회 수집::

    python collect_blockdata_history_v1_2026-08-25.py --once
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests


KST = ZoneInfo("Asia/Seoul")
DEFAULT_PLANT_ID = 6715
DEFAULT_EXPECTED_INVERTERS = 5
# ★09-16 수정★: 수집기의 인버터 구성 검증은 발전소 API 정격 240이 아니라
# 현재 인버터 5대 등록용량 합계(49.5+49.5+43.5+56.5+41.58)를 기준으로
# 삼는다. 공식 설비용량·nMAE 분모 240kW와 역할을 분리한다.
# 이 기본값은 광주(DEFAULT_PLANT_ID=6715) 전용이다 - 부안·김제·영광은
# 예약작업에서 --expected-capacity-kw를 명시적으로 넘기므로 영향 없음.
DEFAULT_EXPECTED_CAPACITY_KW = 241.58
DEFAULT_KEY_FILE = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일\API_KeyInfo.txt"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25"
)
BASE_URL = "https://api.blockdata.kr/data/{plant_id}"

INVERTER_NUMERIC_FIELDS = (
    "capacity",
    "dc_volt",
    "dc_current",
    "dc_power",
    "ac_volt_r",
    "ac_volt_s",
    "ac_volt_t",
    "ac_current_r",
    "ac_current_s",
    "ac_current_t",
    "ac_power",
    "pf",
    "freq",
    "daily_energy",
    "total_energy",
)


class BlockdataCollectorError(RuntimeError):
    """수집·응답 검증 실패."""


class BlockdataAuthError(BlockdataCollectorError):
    """401/403 인증 실패. 반복 호출하지 않고 즉시 중단한다."""


@dataclass(frozen=True)
class FetchResult:
    payload: dict[str, Any]
    raw_text: str
    http_status: int
    latency_ms: int
    attempts: int


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def iso_kst(value: datetime) -> str:
    return value.astimezone(KST).isoformat(timespec="seconds")


def parse_api_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    parsed: datetime | None = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            pass
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise BlockdataCollectorError(f"측정시각 형식을 해석할 수 없습니다: {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_api_key(key_file: Path) -> str:
    env_key = os.environ.get("BLOCKDATA_API_KEY", "").strip()
    if env_key:
        return env_key
    if not key_file.is_file():
        raise FileNotFoundError(
            "Blockdata API 키가 없습니다. BLOCKDATA_API_KEY 환경변수 또는 "
            f"키 파일을 준비하세요: {key_file}"
        )
    key = key_file.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise BlockdataCollectorError(f"API 키 파일이 비어 있습니다: {key_file}")
    return key


def fetch_live(
    session: requests.Session,
    plant_id: int,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
) -> FetchResult:
    url = BASE_URL.format(plant_id=plant_id)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            response = session.get(
                url,
                headers={"key": api_key, "Accept": "application/json"},
                timeout=timeout_seconds,
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in (401, 403):
                raise BlockdataAuthError(
                    f"Blockdata 인증 실패(HTTP {response.status_code}). 키를 확인하세요."
                )
            if response.status_code == 404:
                raise BlockdataCollectorError(f"발전소 또는 API 경로 없음(HTTP 404): {plant_id}")
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt - 1), 30)
                last_error = BlockdataCollectorError(f"일시 HTTP 오류 {response.status_code}")
                if attempt < max_retries:
                    time.sleep(delay + random.random() * 0.25)
                    continue
                raise last_error
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise BlockdataCollectorError("Blockdata 응답이 JSON이 아닙니다.") from exc
            if not isinstance(payload, dict):
                raise BlockdataCollectorError("Blockdata 응답 최상위가 객체(dict)가 아닙니다.")
            return FetchResult(payload, response.text, response.status_code, latency_ms, attempt)
        except BlockdataAuthError:
            raise
        except BlockdataCollectorError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 30) + random.random() * 0.25)
                continue
    raise BlockdataCollectorError(
        f"Blockdata 호출 실패({max_retries}회): {type(last_error).__name__ if last_error else 'unknown'}"
    )


def load_fixture(path: Path) -> FetchResult:
    raw_text = path.read_text(encoding="utf-8-sig")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise BlockdataCollectorError("fixture 최상위가 객체(dict)가 아닙니다.")
    return FetchResult(payload, raw_text, 0, 0, 0)


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_snapshots (
            response_hash TEXT PRIMARY KEY,
            plant_id INTEGER NOT NULL,
            first_collected_at TEXT NOT NULL,
            last_collected_at TEXT NOT NULL,
            source_latest_measurement_at TEXT,
            seen_count INTEGER NOT NULL DEFAULT 1,
            raw_file TEXT,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS inverter_measurements (
            plant_id INTEGER NOT NULL,
            inverter_number INTEGER NOT NULL,
            measurement_time TEXT NOT NULL,
            first_collected_at TEXT NOT NULL,
            last_collected_at TEXT NOT NULL,
            seen_count INTEGER NOT NULL DEFAULT 1,
            response_hash TEXT NOT NULL,
            capacity REAL, dc_volt REAL, dc_current REAL, dc_power REAL,
            ac_volt_r REAL, ac_volt_s REAL, ac_volt_t REAL,
            ac_current_r REAL, ac_current_s REAL, ac_current_t REAL,
            ac_power REAL, pf REAL, freq REAL,
            daily_energy REAL, total_energy REAL,
            quality_status TEXT NOT NULL,
            quality_flags_json TEXT NOT NULL,
            raw_inverter_json TEXT NOT NULL,
            PRIMARY KEY (plant_id, inverter_number, measurement_time),
            FOREIGN KEY (response_hash) REFERENCES raw_snapshots(response_hash)
        );

        CREATE TABLE IF NOT EXISTS plant_snapshots (
            plant_id INTEGER NOT NULL,
            snapshot_time TEXT NOT NULL,
            first_collected_at TEXT NOT NULL,
            last_collected_at TEXT NOT NULL,
            seen_count INTEGER NOT NULL DEFAULT 1,
            response_hash TEXT NOT NULL,
            plant_name TEXT,
            plant_address TEXT,
            plant_capacity REAL,
            expected_inverter_count INTEGER NOT NULL,
            received_inverter_count INTEGER NOT NULL,
            valid_ac_power_count INTEGER NOT NULL,
            registered_capacity_sum_kw REAL,
            plant_ac_power_kw REAL,
            plant_dc_power_kw REAL,
            plant_daily_energy_kwh REAL,
            plant_total_energy_kwh REAL,
            measurement_spread_seconds REAL,
            quality_status TEXT NOT NULL,
            quality_flags_json TEXT NOT NULL,
            PRIMARY KEY (plant_id, snapshot_time),
            FOREIGN KEY (response_hash) REFERENCES raw_snapshots(response_hash)
        );

        CREATE TABLE IF NOT EXISTS poll_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT NOT NULL,
            plant_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            http_status INTEGER,
            latency_ms INTEGER,
            attempts INTEGER,
            response_hash TEXT,
            new_inverter_rows INTEGER NOT NULL DEFAULT 0,
            duplicate_inverter_rows INTEGER NOT NULL DEFAULT 0,
            quality_status TEXT,
            error_type TEXT,
            error_message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_inverter_time
          ON inverter_measurements(measurement_time);
        CREATE INDEX IF NOT EXISTS idx_plant_time
          ON plant_snapshots(snapshot_time);
        CREATE INDEX IF NOT EXISTS idx_poll_time
          ON poll_attempts(collected_at);
        """
    )
    return conn


def validate_payload(payload: dict[str, Any], requested_plant_id: int) -> list[dict[str, Any]]:
    payload_plant_id = payload.get("plant_id")
    if payload_plant_id is not None and int(payload_plant_id) != requested_plant_id:
        raise BlockdataCollectorError(
            f"요청 발전소 ID({requested_plant_id})와 응답 ID({payload_plant_id})가 다릅니다."
        )
    inverters = payload.get("inverter")
    if not isinstance(inverters, list) or not inverters:
        raise BlockdataCollectorError("응답에 비어 있지 않은 inverter[] 배열이 없습니다.")
    if not all(isinstance(item, dict) for item in inverters):
        raise BlockdataCollectorError("inverter[] 원소 중 객체(dict)가 아닌 값이 있습니다.")
    return inverters


def inverter_quality_flags(
    item: dict[str, Any], collected_at: datetime, stale_minutes: float
) -> tuple[datetime, list[str]]:
    flags: list[str] = []
    measurement_time = parse_api_datetime(item.get("datetime"))
    if measurement_time is None:
        raise BlockdataCollectorError("인버터 측정값에 datetime이 없습니다.")
    if item.get("number") in (None, ""):
        raise BlockdataCollectorError("인버터 측정값에 number가 없습니다.")
    age_minutes = (collected_at - measurement_time).total_seconds() / 60.0
    if age_minutes > stale_minutes:
        flags.append(f"stale_measurement>{stale_minutes:g}min")
    if age_minutes < -5:
        flags.append("future_measurement>5min")
    ac_power = as_float(item.get("ac_power"))
    capacity = as_float(item.get("capacity"))
    if ac_power is None:
        flags.append("missing_ac_power")
    elif ac_power < 0:
        flags.append("negative_ac_power")
    if ac_power is not None and capacity is not None and capacity > 0 and ac_power > capacity * 1.2:
        flags.append("ac_power_above_120pct_capacity")
    if as_float(item.get("daily_energy")) is None:
        flags.append("missing_daily_energy")
    if as_float(item.get("total_energy")) is None:
        flags.append("missing_total_energy")
    return measurement_time, flags


def write_raw_file(output_dir: Path, raw_text: str, response_hash: str, snapshot_time: datetime) -> Path:
    raw_dir = output_dir / "raw" / snapshot_time.strftime("%Y") / snapshot_time.strftime("%m") / snapshot_time.strftime("%d")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"blockdata_{snapshot_time.strftime('%Y%m%dT%H%M%S%z')}_{response_hash[:12]}.json"
    if not path.exists():
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(raw_text, encoding="utf-8")
        temp_path.replace(path)
    return path


def persist_snapshot(
    conn: sqlite3.Connection,
    output_dir: Path,
    result: FetchResult,
    requested_plant_id: int,
    collected_at: datetime,
    expected_inverters: int,
    expected_capacity_kw: float,
    stale_minutes: float,
) -> dict[str, Any]:
    payload = result.payload
    inverters = validate_payload(payload, requested_plant_id)
    response_hash = hashlib.sha256(result.raw_text.encode("utf-8")).hexdigest()

    parsed: list[tuple[dict[str, Any], datetime, list[str]]] = []
    numbers: list[int] = []
    for item in inverters:
        measurement_time, flags = inverter_quality_flags(item, collected_at, stale_minutes)
        number = int(item["number"])
        numbers.append(number)
        parsed.append((item, measurement_time, flags))
    if len(set(numbers)) != len(numbers):
        raise BlockdataCollectorError("한 응답 안에 같은 인버터 number가 중복됩니다.")

    snapshot_time = max(row[1] for row in parsed)
    measurement_spread_seconds = (snapshot_time - min(row[1] for row in parsed)).total_seconds()
    plant_flags: list[str] = []
    if len(inverters) != expected_inverters:
        plant_flags.append(f"inverter_count={len(inverters)}_expected={expected_inverters}")
    if measurement_spread_seconds > 300:
        plant_flags.append("measurement_spread>300sec")

    capacity_values = [as_float(item.get("capacity")) for item, _, _ in parsed]
    capacity_sum = sum(v for v in capacity_values if v is not None)
    # ★★09-16 수정(사용자 승인)★★: 기존 허용오차 0.011kW는 지나치게 엄격해서,
    # 발전소가 인버터 등록용량을 소폭 조정하기만 해도(광주: 09-01·09-09 두 번
    # 변경, 합계 241.58 vs API 정격 240.0 = 1.58kW 차) **모든 스냅샷이 영구
    # warning**이 됐다. 광주는 2,745건 연속 warning이라 진짜 경보가 묻혔다.
    #
    # expected_capacity_kw는 인버터 구성 감시용 기준이다. 모델 평가용
    # 공식 설비용량·nMAE 분모는 별도 240kW를 사용한다.
    capacity_tolerance_kw = max(2.0, expected_capacity_kw * 0.01)
    capacity_gap = capacity_sum - expected_capacity_kw
    if abs(capacity_gap) > capacity_tolerance_kw:
        # 경보 의미를 "용량 불일치" 하나로 뭉치지 않고 원인별로 나눈다.
        plant_flags.append(
            f"capacity_sum_shift={capacity_sum:g}_expected={expected_capacity_kw:g}"
            f"_gap={capacity_gap:+.2f}_tol={capacity_tolerance_kw:g}"
        )
    for item, _, flags in parsed:
        if flags:
            plant_flags.append(f"inverter_{int(item['number'])}:" + ",".join(flags))

    quality_status = "ok" if not plant_flags else "warning"
    raw_file = write_raw_file(output_dir, result.raw_text, response_hash, snapshot_time)
    collected_iso = iso_kst(collected_at)
    snapshot_iso = iso_kst(snapshot_time)
    new_rows = 0
    duplicate_rows = 0

    with conn:
        conn.execute(
            """
            INSERT INTO raw_snapshots(
              response_hash, plant_id, first_collected_at, last_collected_at,
              source_latest_measurement_at, seen_count, raw_file, raw_json
            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(response_hash) DO UPDATE SET
              last_collected_at=excluded.last_collected_at,
              seen_count=raw_snapshots.seen_count+1
            """,
            (response_hash, requested_plant_id, collected_iso, collected_iso,
             snapshot_iso, str(raw_file), result.raw_text),
        )

        for item, measurement_time, flags in parsed:
            key = (requested_plant_id, int(item["number"]), iso_kst(measurement_time))
            exists = conn.execute(
                "SELECT 1 FROM inverter_measurements WHERE plant_id=? AND inverter_number=? AND measurement_time=?",
                key,
            ).fetchone() is not None
            values = [as_float(item.get(field)) for field in INVERTER_NUMERIC_FIELDS]
            conn.execute(
                f"""
                INSERT INTO inverter_measurements(
                  plant_id, inverter_number, measurement_time,
                  first_collected_at, last_collected_at, seen_count, response_hash,
                  {', '.join(INVERTER_NUMERIC_FIELDS)},
                  quality_status, quality_flags_json, raw_inverter_json
                ) VALUES ({', '.join(['?'] * (7 + len(INVERTER_NUMERIC_FIELDS) + 3))})
                ON CONFLICT(plant_id, inverter_number, measurement_time) DO UPDATE SET
                  last_collected_at=excluded.last_collected_at,
                  seen_count=inverter_measurements.seen_count+1,
                  response_hash=excluded.response_hash,
                  quality_status=excluded.quality_status,
                  quality_flags_json=excluded.quality_flags_json,
                  raw_inverter_json=excluded.raw_inverter_json
                """,
                (*key, collected_iso, collected_iso, 1, response_hash, *values,
                 "ok" if not flags else "warning", json.dumps(flags, ensure_ascii=False),
                 json.dumps(item, ensure_ascii=False, separators=(",", ":"))),
            )
            if exists:
                duplicate_rows += 1
            else:
                new_rows += 1

        def sum_field(field: str) -> float | None:
            values = [as_float(item.get(field)) for item, _, _ in parsed]
            valid = [value for value in values if value is not None]
            return sum(valid) if valid else None

        valid_ac_count = sum(as_float(item.get("ac_power")) is not None for item, _, _ in parsed)
        plant_exists = conn.execute(
            "SELECT 1 FROM plant_snapshots WHERE plant_id=? AND snapshot_time=?",
            (requested_plant_id, snapshot_iso),
        ).fetchone() is not None
        conn.execute(
            """
            INSERT INTO plant_snapshots(
              plant_id, snapshot_time, first_collected_at, last_collected_at,
              seen_count, response_hash, plant_name, plant_address, plant_capacity,
              expected_inverter_count, received_inverter_count, valid_ac_power_count,
              registered_capacity_sum_kw, plant_ac_power_kw, plant_dc_power_kw,
              plant_daily_energy_kwh, plant_total_energy_kwh,
              measurement_spread_seconds, quality_status, quality_flags_json
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(plant_id, snapshot_time) DO UPDATE SET
              last_collected_at=excluded.last_collected_at,
              seen_count=plant_snapshots.seen_count+1,
              response_hash=excluded.response_hash,
              valid_ac_power_count=excluded.valid_ac_power_count,
              registered_capacity_sum_kw=excluded.registered_capacity_sum_kw,
              plant_ac_power_kw=excluded.plant_ac_power_kw,
              plant_dc_power_kw=excluded.plant_dc_power_kw,
              plant_daily_energy_kwh=excluded.plant_daily_energy_kwh,
              plant_total_energy_kwh=excluded.plant_total_energy_kwh,
              measurement_spread_seconds=excluded.measurement_spread_seconds,
              quality_status=excluded.quality_status,
              quality_flags_json=excluded.quality_flags_json
            """,
            (
                requested_plant_id, snapshot_iso, collected_iso, collected_iso,
                response_hash, payload.get("plant_name"), payload.get("plant_address"),
                as_float(payload.get("plant_capacity")), expected_inverters, len(inverters),
                valid_ac_count, capacity_sum, sum_field("ac_power"), sum_field("dc_power"),
                sum_field("daily_energy"), sum_field("total_energy"), measurement_spread_seconds,
                quality_status, json.dumps(plant_flags, ensure_ascii=False),
            ),
        )

        conn.execute(
            """
            INSERT INTO poll_attempts(
              collected_at, plant_id, status, http_status, latency_ms, attempts,
              response_hash, new_inverter_rows, duplicate_inverter_rows, quality_status
            ) VALUES (?, ?, 'success', ?, ?, ?, ?, ?, ?, ?)
            """,
            (collected_iso, requested_plant_id, result.http_status, result.latency_ms,
             result.attempts, response_hash, new_rows, duplicate_rows, quality_status),
        )

    summary = {
        "collected_at": collected_iso,
        "plant_id": requested_plant_id,
        "snapshot_time": snapshot_iso,
        "response_hash": response_hash,
        "http_status": result.http_status,
        "attempts": result.attempts,
        "latency_ms": result.latency_ms,
        "new_inverter_rows": new_rows,
        "duplicate_inverter_rows": duplicate_rows,
        "plant_snapshot_duplicate": plant_exists,
        "received_inverters": len(inverters),
        "registered_capacity_sum_kw": capacity_sum,
        "plant_ac_power_kw": sum_field("ac_power"),
        "quality_status": quality_status,
        "quality_flags": plant_flags,
        "raw_file": str(raw_file),
    }
    write_status_file(output_dir, summary)
    return summary


def record_failure(
    conn: sqlite3.Connection,
    plant_id: int,
    collected_at: datetime,
    exc: Exception,
) -> None:
    message = str(exc)
    if len(message) > 1000:
        message = message[:1000]
    with conn:
        conn.execute(
            """
            INSERT INTO poll_attempts(
              collected_at, plant_id, status, error_type, error_message
            ) VALUES (?, ?, 'failure', ?, ?)
            """,
            (iso_kst(collected_at), plant_id, type(exc).__name__, message),
        )


def write_status_file(output_dir: Path, summary: dict[str, Any]) -> None:
    path = output_dir / "collector_status.json"
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def export_latest_csv(conn: sqlite3.Connection, output_dir: Path, plant_id: int) -> None:
    latest = conn.execute(
        "SELECT snapshot_time FROM plant_snapshots WHERE plant_id=? ORDER BY snapshot_time DESC LIMIT 1",
        (plant_id,),
    ).fetchone()
    if latest is None:
        return
    snapshot_time = latest[0]
    rows = conn.execute(
        """
        SELECT plant_id, inverter_number, measurement_time, capacity, dc_volt,
               dc_current, dc_power, ac_volt_r, ac_volt_s, ac_volt_t,
               ac_current_r, ac_current_s, ac_current_t, ac_power, pf, freq,
               daily_energy, total_energy, quality_status, quality_flags_json
        FROM inverter_measurements
        WHERE plant_id=? AND measurement_time>=datetime(?, '-10 minutes')
        ORDER BY inverter_number
        """,
        (plant_id, snapshot_time),
    ).fetchall()
    columns = [description[0] for description in conn.execute(
        """
        SELECT plant_id, inverter_number, measurement_time, capacity, dc_volt,
               dc_current, dc_power, ac_volt_r, ac_volt_s, ac_volt_t,
               ac_current_r, ac_current_s, ac_current_t, ac_power, pf, freq,
               daily_energy, total_energy, quality_status, quality_flags_json
        FROM inverter_measurements LIMIT 0
        """
    ).description]
    path = output_dir / "latest_inverters.csv"
    temp_path = path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blockdata 실시간 이력 수집기")
    parser.add_argument("--plant-id", type=int, default=DEFAULT_PLANT_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--input-json", type=Path, help="API 대신 저장 JSON으로 오프라인 테스트")
    parser.add_argument("--once", action="store_true", help="1회 수집(기본 동작)")
    parser.add_argument("--loop", action="store_true", help="내부 루프로 반복 수집")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--expected-inverters", type=int, default=DEFAULT_EXPECTED_INVERTERS)
    parser.add_argument("--expected-capacity-kw", type=float, default=DEFAULT_EXPECTED_CAPACITY_KW)
    parser.add_argument("--stale-minutes", type=float, default=15.0)
    args = parser.parse_args()
    if args.once and args.loop:
        parser.error("--once와 --loop는 동시에 사용할 수 없습니다.")
    if args.input_json and args.loop:
        parser.error("--input-json은 --loop와 함께 사용할 수 없습니다.")
    if args.interval_seconds < 60:
        parser.error("--interval-seconds는 API 보호를 위해 60초 이상이어야 합니다.")
    if args.max_retries < 1:
        parser.error("--max-retries는 1 이상이어야 합니다.")
    return args


def collect_once(args: argparse.Namespace, conn: sqlite3.Connection, session: requests.Session) -> bool:
    collected_at = now_kst()
    try:
        if args.input_json:
            result = load_fixture(args.input_json)
        else:
            api_key = load_api_key(args.key_file)
            result = fetch_live(
                session, args.plant_id, api_key, args.timeout_seconds, args.max_retries
            )
        summary = persist_snapshot(
            conn, args.output_dir, result, args.plant_id, collected_at,
            args.expected_inverters, args.expected_capacity_kw, args.stale_minutes,
        )
        export_latest_csv(conn, args.output_dir, args.plant_id)
        print(
            f"[OK] plant={args.plant_id} snapshot={summary['snapshot_time']} "
            f"new={summary['new_inverter_rows']} duplicate={summary['duplicate_inverter_rows']} "
            f"ac_power={summary['plant_ac_power_kw']:.3f}kW quality={summary['quality_status']}"
        )
        return True
    except Exception as exc:
        record_failure(conn, args.plant_id, collected_at, exc)
        print(f"[FAIL] plant={args.plant_id} {type(exc).__name__}: {exc}", file=sys.stderr)
        if isinstance(exc, BlockdataAuthError):
            raise
        return False


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "blockdata_history.sqlite3"
    conn = open_database(db_path)
    session = requests.Session()
    try:
        if not args.loop:
            return 0 if collect_once(args, conn, session) else 1
        print(
            f"[START] plant={args.plant_id} interval={args.interval_seconds}s db={db_path}"
        )
        while True:
            started = time.monotonic()
            try:
                collect_once(args, conn, session)
            except BlockdataAuthError:
                return 2
            elapsed = time.monotonic() - started
            time.sleep(max(1.0, args.interval_seconds - elapsed))
    except KeyboardInterrupt:
        print("[STOP] 사용자 중단 — 저장된 이력은 보존됩니다.")
        return 0
    finally:
        session.close()
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
