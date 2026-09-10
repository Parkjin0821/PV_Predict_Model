# -*- coding: utf-8 -*-
"""광주 대시보드용 JSON 생성기 - 저장자료 전용, API 호출 없음.

09-02 사용자 요청("대시보드 실데이터 연결기") 구현. 원칙:
1. 기존 3개 SQLite를 **읽기전용**(`?mode=ro`)으로만 연다 - 절대 쓰지 않는다.
2. Shadow가 시작되지 않았으므로 예측값(forecastEnergyKwh, 수평별 예측)은
   **절대 채우지 않는다** - 계속 null + "SHADOW · 공식 운영 전".
3. 과거 백테스트 지표(Home.tsx의 GWANGJU_VERIFIED_METRICS)는 이 스크립트가
   건드리지 않는다 - 라이브 지표와 분리 유지.
4. 데이터가 없거나 오래되면(각 소스별 임계값) 숫자를 그대로 내보내지 않고
   null + 사유를 남긴다 - 임의 채움 금지.
5. 인증키·요청 URL 전체·raw 파일 절대경로는 출력 JSON에 넣지 않는다.
6. 출력은 임시파일 작성 후 os.replace로 원자적 교체.

실행:
    python generate_gwangju_dashboard_json_v1_2026-09-02.py
API 호출 없이 로컬 SQLite만 읽으므로 몇 초 내 끝난다. 크론/스케줄러에
5~15분 간격으로 등록해 재실행해도 안전(멱등, 부작용 없음).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
SITE_LATITUDE = 35.14428133
SITE_LONGITUDE = 126.84058771

# 09-03 날씨카드 개편(사용자 지시): 태양고도를 재구현하지 않고 검증된
# pv_pipeline.solar_position() 그대로 재사용(치평동_대표좌표_출처.json
# 확정 좌표, config.json과 동일).
_pv_pipeline_spec = importlib.util.spec_from_file_location(
    "pv_pipeline_for_dashboard",
    Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
         r"\03_모델학습\현재_종합파이프라인\pv_pipeline.py"),
)
_pv_pipeline = importlib.util.module_from_spec(_pv_pipeline_spec)
_pv_pipeline_spec.loader.exec_module(_pv_pipeline)


def current_solar_elevation_deg(ref_now: datetime) -> float:
    idx = pd.DatetimeIndex([ref_now.astimezone(KST).replace(tzinfo=None)])
    elev, _azimuth = _pv_pipeline.solar_position(idx, SITE_LATITUDE, SITE_LONGITUDE)
    return round(float(elev[0]), 1)

BLOCKDATA_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25"
    r"\blockdata_history.sqlite3"
)
WEATHER_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25"
    r"\kma_live_inputs.sqlite3"
)
OUTPUT_JSON = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\태양광 발전 예측 대시보드\public\data\gwangju-dashboard.json"
)

# 09-02 저녁 Blockdata 감사(P0-1) 사용자 승인 반영: 4개 용량 필드를 독립적으로
# 유지한다. 값은 config.json의 capacity_policy(같은 프로젝트,
# 03_모델학습/현재_종합파이프라인/config.json)와 반드시 동일해야 하며, 이 스크립트가
# 그 config를 직접 읽지 않는 원칙(로컬 DB 2개만 읽음)을 지키기 위해 값만 그대로
# 복제해온다 - 두 값이 벌어지면 이 주석부터 의심할 것.
# ★09-02 밤 갱신★: Blockdata 실시간 API(인버터별 capacity 필드)가 09-01
# 18:04를 기점으로 219.0(AC 인버터 등록합) → 240.58(DC 모듈 용량 추정,
# 1.0985배 - 08-24 이미 확인된 DC:AC 오버사이징비 1.096과 거의 일치)로
# 바뀐 걸 실측 확인. 실제 AC 출력 실측 최대치는 옛 상한(3번 22.8kW·4번
# 26.6kW, 옛 등록값 30/39kW 안 넘음) 그대로라 설비 증설은 아니라고 판단.
# **사용자 지시로 modelReferenceCapacityKw를 219→240.58로 전환**(config.json
# capacity_policy.active_profile도 동일하게 blockdata_live_inverter_sum_240_58로
# 갱신 - 모델이 capacity_kw를 학습 특성으로 안 써서 재학습 불필요함을 코드로
# 확인 후 진행). 블록데이터 측 API 필드 전환 공식 확인은 아직 대기 중.
CAPACITY_PROFILE = {
    "plantRegisteredCapacityKw": {
        "valueKw": 240.0,
        "source": "Blockdata GET /plant/6715 plant_capacity",
        "definition": (
            "Blockdata 발전소 정보 API의 정적 필드(08-19부터 불변). 아래 "
            "inverterRegisteredCapacitySumKw(09-01 이후 240.58)와는 다른 API "
            "엔드포인트·필드 - 우연히 비슷한 수치일 뿐 같은 출처 아님."
        ),
    },
    "inverterRegisteredCapacitySumKw": {
        "valueKw": 240.58,
        "source": (
            "Blockdata GET /data/6715 인버터 5대 실시간 capacity 필드 합계"
            "(49.5+49.5+43.5+56.5+41.58kW, 09-01 18:04부터)"
        ),
        "definition": (
            "09-01 18:04 이전에는 이 필드가 219.0(50+50+30+39+50, AC 인버터 정격)"
            "이었으나, 이후 240.58(DC 모듈 용량 추정)로 전환됨을 실측 확인. "
            "실제 AC 출력 상한은 옛 219 기준 그대로 유지(설비 증설 아님). "
            "현장 명판·설비대장으로는 아직 대조되지 않음."
        ),
    },
    "modelReferenceCapacityKw": {
        "valueKw": 240.58,
        "source": "config.json capacity_policy.active_profile = blockdata_live_inverter_sum_240_58",
        "definition": (
            "PR·설비이용률 등 이 대시보드 계산에 실제로 사용 중인 기준값. "
            "09-02 밤 사용자 지시로 219→240.58로 전환(모델은 capacity_kw를 "
            "학습 특성으로 쓰지 않아 재학습 불필요 확인 후 진행). 정책"
            "(active_profile)이 다시 바뀌면 이 값도 함께 달라진다."
        ),
    },
    "nameplateCapacityVerified": {
        "value": False,
        "source": "현장 명판·설비대장 미대조(2026-09-02 감사 기준)",
        "definition": (
            "위 세 값은 전부 Blockdata API 응답이며, 현장 명판 정격과 아직 대조되지 "
            "않았다. Blockdata 측 API 필드 전환(09-01) 공식 확인 및 배만수 부장 "
            "정의 확인 대기 중 - 확인 결과에 따라 기준값이 재조정될 수 있다."
        ),
    },
}

# PR·이용률 계산은 modelReferenceCapacityKw를 그대로 따른다(위 프로필과 단일 출처 유지).
PLANT_CAPACITY_KW = CAPACITY_PROFILE["modelReferenceCapacityKw"]["valueKw"]

# 라이브 값이 지금의 모델기준(240.58)에서 다시 벗어나면(예: 블록데이터가
# 또 필드를 바꾸거나 219로 되돌리면) 알아채기 위한 감시 기준값.
CAPACITY_LIVE_BASELINE_KW = PLANT_CAPACITY_KW
CAPACITY_LIVE_TOLERANCE_KW = 0.5


def load_capacity_live_change_notice() -> dict[str, Any] | None:
    """Blockdata 라이브 API가 지금 실제로 보고 중인 registered_capacity_sum_kw가
    modelReferenceCapacityKw(현재 240.58)와 다르면, 언제부터·무엇으로
    바뀌었는지 실측해 참고정보로 반환한다. 같으면 None(알릴 것 없음)."""
    conn = open_readonly(BLOCKDATA_DB)
    try:
        latest = conn.execute(
            "SELECT registered_capacity_sum_kw, snapshot_time FROM plant_snapshots "
            "ORDER BY snapshot_time DESC LIMIT 1"
        ).fetchone()
        if latest is None or latest[0] is None:
            return None
        current_kw = float(latest[0])
        if abs(current_kw - CAPACITY_LIVE_BASELINE_KW) <= CAPACITY_LIVE_TOLERANCE_KW:
            return None  # 모델기준과 일치 - 알릴 변화 없음

        # 지금 값이 언제부터 이어지고 있는지: 모델기준이 마지막으로 관측된
        # 시각 바로 다음 스냅샷을 전환 시점으로 잡는다(계단전환 1회 가정 -
        # 여러 번 왔다갔다 하면 다시 봐야 함).
        last_baseline_row = conn.execute(
            "SELECT MAX(snapshot_time) FROM plant_snapshots WHERE ABS(registered_capacity_sum_kw - ?) <= ?",
            (CAPACITY_LIVE_BASELINE_KW, CAPACITY_LIVE_TOLERANCE_KW),
        ).fetchone()
        last_baseline_at = last_baseline_row[0] if last_baseline_row else None
        if last_baseline_at:
            changed_row = conn.execute(
                "SELECT MIN(snapshot_time) FROM plant_snapshots WHERE snapshot_time > ?",
                (last_baseline_at,),
            ).fetchone()
            changed_at = changed_row[0] if changed_row else None
        else:
            changed_at = None

        return {
            "observedValueKw": round(current_kw, 2),
            "observedAtKst": latest[1],
            "changedFromKw": CAPACITY_LIVE_BASELINE_KW,
            "changedSinceKst": changed_at,
            "note": (
                f"Blockdata 실시간 API가 {changed_at or '알 수 없는 시점'}부터 인버터 "
                f"등록용량 합계를 모델기준({CAPACITY_LIVE_BASELINE_KW:.2f}kW) 대신 "
                f"{current_kw:.2f}kW로 보고 중 - 다시 확인 필요."
            ),
        }
    finally:
        conn.close()

# 신선도 임계값(분) - 이보다 오래되면 값을 null로 내리고 사유를 남긴다.
POWER_STALE_MIN = 20        # Blockdata 5분 주기 폴링 기준 4배 여유
ASOS_STALE_MIN = 150        # ASOS는 시간자료(1시간 주기) + 지연 여유
GRID_STALE_HOURS = 27       # 격자예보 발행 주기(3시간)+지연 여유, 최소 하루+3시간


def open_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"원본 DB가 없습니다(읽기전용 전제 위반 방지): {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.execute("PRAGMA query_only = ON")
    return conn


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def parse_kst(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def minutes_since(ts: datetime | None, ref: datetime) -> float | None:
    if ts is None:
        return None
    return round((ref - ts).total_seconds() / 60.0, 1)


# ───────────────────── 발전량·인버터 (Blockdata) ─────────────────────

def load_power_and_inverters(ref_now: datetime) -> dict[str, Any]:
    conn = open_readonly(BLOCKDATA_DB)
    try:
        row = conn.execute(
            """
            SELECT snapshot_time, plant_ac_power_kw, plant_daily_energy_kwh,
                   quality_status, quality_flags_json,
                   expected_inverter_count, received_inverter_count
            FROM plant_snapshots ORDER BY snapshot_time DESC LIMIT 1
            """
        ).fetchone()

        current_power_kw: float | None = None
        today_energy_kwh: float | None = None
        power_note = "발전량 이력 없음"
        power_age_min: float | None = None

        if row:
            snap_time = parse_kst(row[0])
            power_age_min = minutes_since(snap_time, ref_now)
            if power_age_min is not None and power_age_min <= POWER_STALE_MIN:
                current_power_kw = row[1]
                today_energy_kwh = row[2]
                power_note = f"{power_age_min:.0f}분 전 갱신"
            else:
                power_note = (
                    f"{power_age_min:.0f}분 전이 마지막 수신(지연)"
                    if power_age_min is not None
                    else "수신 시각 해석 불가"
                )

        # 최근 10분 내 인버터별 최신값(측정시각이 인버터마다 조금씩 다를 수 있어 중복 제거)
        inverters: list[dict[str, Any]] = []
        for inv_row in conn.execute(
            """
            SELECT inverter_number, MAX(measurement_time) AS mt, ac_power, dc_power, quality_status, freq
            FROM inverter_measurements
            WHERE measurement_time >= datetime(
              (SELECT MAX(measurement_time) FROM inverter_measurements), '-15 minutes'
            )
            GROUP BY inverter_number ORDER BY inverter_number
            """
        ):
            inv_number, mt, ac_power, dc_power, q_status, freq_raw = inv_row
            efficiency = None
            if ac_power is not None and dc_power is not None and dc_power > 0:
                efficiency = round(ac_power / dc_power * 100, 1)

            # 09-02 저녁 Blockdata 감사(P0-2) 사용자 승인 반영: freq=0Hz는 710일
            # 이력 전체 + 오늘 재확인(934/934, 915/915)에서 발전 중에도 100%
            # 재현되는 패턴이라 "고장"이 아니라 "그 인버터에서는 계통주파수를 실제로
            # 읽어오지 못하는 비유효 원천값"으로 취급한다. 인버터 번호를 하드코딩해
            # 분기하지 않고, 관측된 raw 값 자체로 판정한다 - 다른 인버터가 나중에
            # 같은 패턴을 보이면 자동으로 같이 분류된다.
            if freq_raw is None:
                freq_quality = "unavailable"
                freq_hz = None
            elif freq_raw == 0:
                freq_quality = "known_invalid_telemetry"
                freq_hz = None
            else:
                freq_quality = "ok"
                freq_hz = freq_raw

            inverters.append(
                {
                    "inverterNumber": inv_number,
                    "qualityStatus": q_status if q_status in ("ok", "warning") else "offline",
                    "acPowerKw": ac_power,
                    "efficiencyPct": efficiency,
                    "lastUpdatedKst": mt,
                    "frequencyHz": freq_hz,
                    "frequencyRawHz": freq_raw,
                    "frequencyQualityStatus": freq_quality,
                    # 이 시스템은 freq를 설비고장 판정에 쓰지 않는다(Blockdata
                    # quality_status 자체가 freq=0인 3·4번도 계속 "ok"로 보고함,
                    # 09-02 실측 확인) - 항상 False로 명시해 프런트가 임의로 알람에
                    # 연결하지 않도록 한다.
                    "frequencyUsedForAlarm": False,
                }
            )

        # 오늘 시간대별 실제발전량 - 각 시간대 마지막 스냅샷을 대표값으로 사용.
        today_str = ref_now.strftime("%Y-%m-%d")
        series: list[dict[str, Any]] = []
        for hour in range(24):
            hour_start = f"{today_str}T{hour:02d}:00:00"
            hour_end = f"{today_str}T{hour:02d}:59:59"
            hour_row = conn.execute(
                """
                SELECT plant_ac_power_kw FROM plant_snapshots
                WHERE snapshot_time >= ? AND snapshot_time <= ?
                ORDER BY snapshot_time DESC LIMIT 1
                """,
                (hour_start, hour_end),
            ).fetchone()
            series.append({"hourKst": f"{hour:02d}", "actualPowerKw": hour_row[0] if hour_row else None})

        return {
            "currentPowerKw": current_power_kw,
            "todayEnergyKwh": today_energy_kwh,
            "powerNote": power_note,
            "inverters": inverters,
            "todayPowerSeries": series,
        }
    finally:
        conn.close()


# ───────────────────── 발전성능(PR·이용률·효율) ─────────────────────

def load_daily_energy_kwh(conn: sqlite3.Connection, date_str: str) -> float | None:
    """그 날짜(KST) 마지막 plant_daily_energy_kwh 값 - 일일 총 발전량으로 취급."""
    row = conn.execute(
        """
        SELECT plant_daily_energy_kwh FROM plant_snapshots
        WHERE snapshot_time LIKE ? AND plant_daily_energy_kwh IS NOT NULL
        ORDER BY snapshot_time DESC LIMIT 1
        """,
        (f"{date_str}%",),
    ).fetchone()
    return row[0] if row else None


def load_daylight_irradiance_kwh_m2(conn: sqlite3.Connection, date_str: str) -> tuple[float | None, int]:
    """그 날짜 ASOS 시간별 일사량(W/m²)을 1시간 폭으로 적산 -> kWh/m². (개수도 함께 반환해 완결성 판단)"""
    rows = conn.execute(
        "SELECT solar_w_m2 FROM asos_hourly WHERE observation_time LIKE ?",
        (f"{date_str}%",),
    ).fetchall()
    values = [r[0] for r in rows if r[0] is not None]
    if not values:
        return None, 0
    total_wh_m2 = sum(values) * 1.0  # 각 값이 1시간 평균 W/m² -> ×1h = Wh/m²
    return round(total_wh_m2 / 1000.0, 3), len(values)


def compute_performance(ref_now: datetime, power: dict[str, Any], weather: dict[str, Any]) -> dict[str, Any]:
    # 이용률·평균효율은 지금 이 순간(실시간) 기준 - 노이즈 논쟁이 적다.
    utilization = None
    if power["currentPowerKw"] is not None:
        utilization = round(power["currentPowerKw"] / PLANT_CAPACITY_KW * 100, 1)
    eff_values = [i["efficiencyPct"] for i in power["inverters"] if i["efficiencyPct"] is not None]
    avg_eff = round(sum(eff_values) / len(eff_values), 1) if eff_values else None

    # PR(성능비)은 IEC 61724 정의(PR = Yf/Yr)를 그대로 따르되, 실시간/당일치가 아니라
    # "가장 최근 완결된 하루"만 계산한다 - 부분일 PR은 구름 변동성 때문에 노이즈가
    # 커서(문헌 확인, 09-02) 채택하지 않는다. 온도보정(WCPR)도 지금은 적용 안 함 -
    # 그 사실을 note에 명시해 단순 PR임을 숨기지 않는다.
    pr_pct: float | None = None
    pr_date: str | None = None
    pr_note = "완결일 자료 부족 - 계산 불가"
    # 09-02 UX 감사(요구사항 21번, "완결률 표시") 반영: PR 계산에 이미 쓰던
    # irr_hours(ASOS 비결측 시간수)를 새로 계산하지 않고 그대로 노출만 한다.
    # 24시간 대비 비율 - 일조시간이 아니라 "관측 데이터가 결측 없이 확보된
    # 시간 비율"이라는 뜻을 note에 그대로 명시(일조시간으로 오인 방지).
    pr_completeness_pct: float | None = None
    pr_irr_hours: int | None = None

    conn = open_readonly(WEATHER_DB)
    conn_bd = open_readonly(BLOCKDATA_DB)
    try:
        for back in range(1, 4):  # 어제부터 최대 3일 전까지 완결일 탐색
            candidate = (ref_now - timedelta(days=back)).strftime("%Y-%m-%d")
            energy_kwh = load_daily_energy_kwh(conn_bd, candidate)
            irr_kwh_m2, irr_hours = load_daylight_irradiance_kwh_m2(conn, candidate)
            if energy_kwh is None or irr_kwh_m2 is None or irr_kwh_m2 <= 0 or irr_hours < 10:
                continue
            yf = energy_kwh / PLANT_CAPACITY_KW  # kWh/kW
            yr = irr_kwh_m2 / 1.0  # kWh/m² ÷ 1kW/m²(STC) = 동일 수치, 단위: 등가 피크시간
            pr_pct = round(yf / yr * 100, 1) if yr > 0 else None
            pr_date = candidate
            pr_irr_hours = irr_hours
            pr_completeness_pct = round(irr_hours / 24 * 100, 1)
            pr_note = (
                f"IEC 61724 단순 PR(온도보정 없음), 모델기준 용량 {PLANT_CAPACITY_KW:.0f}kW, "
                f"ASOS 수평일사(경사면 미보정) {irr_hours}시간 적산 기준"
            )
            break
    finally:
        conn.close()
        conn_bd.close()

    return {
        "capacityUtilizationPct": utilization,
        "averageEfficiencyPct": avg_eff,
        "performanceRatioPct": pr_pct,
        "performanceRatioDateKst": pr_date,
        "performanceRatioNote": pr_note,
        "performanceRatioCompletenessPct": pr_completeness_pct,
        "performanceRatioIrrHours": pr_irr_hours,
    }


# ───────────────────── 기상(ASOS·GRID·NWP) ─────────────────────

def heat_index_c(temp_c: float, humidity_pct: float) -> float:
    """미국 NWS Rothfusz 근사(화씨 기반) - 27도 이상에서만 의미있음."""
    t_f = temp_c * 9 / 5 + 32
    hi_f = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * humidity_pct
        - 0.22475541 * t_f * humidity_pct - 0.00683783 * t_f ** 2
        - 0.05481717 * humidity_pct ** 2 + 0.00122874 * t_f ** 2 * humidity_pct
        + 0.00085282 * t_f * humidity_pct ** 2 - 0.00000199 * t_f ** 2 * humidity_pct ** 2
    )
    return round((hi_f - 32) * 5 / 9, 1)


def wind_chill_c(temp_c: float, wind_ms: float) -> float:
    """환경부/기상청 체감온도 공식(풍속 km/h 환산)."""
    wind_kmh = wind_ms * 3.6
    if wind_kmh < 4.8:
        return round(temp_c, 1)
    wc = (
        13.12 + 0.6215 * temp_c - 11.37 * wind_kmh ** 0.16
        + 0.3965 * temp_c * wind_kmh ** 0.16
    )
    return round(wc, 1)


def compute_feels_like(temp_c: float | None, humidity_pct: float | None, wind_ms: float | None) -> float | None:
    if temp_c is None:
        return None
    if temp_c >= 27 and humidity_pct is not None:
        return heat_index_c(temp_c, humidity_pct)
    if temp_c <= 10 and wind_ms is not None:
        return wind_chill_c(temp_c, wind_ms)
    return round(temp_c, 1)


def load_weather_and_feeds(ref_now: datetime) -> dict[str, Any]:
    conn = open_readonly(WEATHER_DB)
    try:
        # --- ASOS ---
        asos_row = conn.execute(
            "SELECT observation_time, temperature_c, humidity_pct, wind_speed_m_s, solar_w_m2, cloud_pct "
            "FROM asos_hourly ORDER BY observation_time DESC LIMIT 1"
        ).fetchone()
        # 09-03 UX 감사 반영: 대시보드 날씨 카드를 "태양광에 영향 큰 요인"으로
        # 고정(사용자 지시) - 09-03 아침 상관분석(select_and_check_all_factors_v5)
        # 결과 기준 |r| 랭킹: 일사량(0.82)·일조시간(0.64)·습도(-0.58)·전운량
        # (-0.43~-0.47)·기온(0.29) 순, 풍속(0.15)·강수확률(POP, -0.36 - 그나마
        # 이 중엔 중간이지만 "예보"값이라 실측 계열과 성격이 다름)은 하위권.
        # 그래서 강수확률 대신 운량(cloudCoverPct)을 새로 추가 - 실측 기반
        # 랭킹으로 교체(임의 선택 아님).
        weather = {
            "temperatureC": None, "feelsLikeC": None, "irradianceWm2": None,
            "humidityPct": None, "windSpeedMs": None, "precipitationProbabilityPct": None,
            "cloudCoverPct": None,
            # 09-03: 천문 계산값이라 관측 결측과 무관하게 항상 계산 가능(관측 대기 X).
            "solarElevationDeg": current_solar_elevation_deg(ref_now),
        }
        asos_age_min: float | None = None
        asos_note = "ASOS 이력 없음"
        asos_time_str: str | None = None
        if asos_row:
            asos_time_str, temp, hum, wind, solar, cloud = asos_row
            asos_time = parse_kst(asos_time_str)
            asos_age_min = minutes_since(asos_time, ref_now)
            if asos_age_min is not None and asos_age_min <= ASOS_STALE_MIN:
                weather["temperatureC"] = temp
                weather["humidityPct"] = hum
                weather["windSpeedMs"] = wind
                weather["irradianceWm2"] = solar
                weather["cloudCoverPct"] = cloud
                weather["feelsLikeC"] = compute_feels_like(temp, hum, wind)
                asos_note = f"{asos_age_min:.0f}분 전 관측"
            else:
                asos_note = f"{asos_age_min:.0f}분 전이 마지막 관측(지연)" if asos_age_min is not None else "관측 시각 해석 불가"

        # --- GRID: 가장 가까운 미래/현재 3시간 슬롯의 POP를 강수확률로 사용 ---
        grid_issue_row = conn.execute("SELECT MAX(issue_date) FROM grid_forecast").fetchone()
        grid_latest_issue = grid_issue_row[0] if grid_issue_row else None
        grid_row_count = conn.execute(
            "SELECT COUNT(*) FROM grid_forecast WHERE issue_date = ?", (grid_latest_issue,)
        ).fetchone()[0] if grid_latest_issue else 0
        grid_note = "GRID 이력 없음"
        grid_stale = True
        if grid_latest_issue:
            issue_dt = datetime.strptime(grid_latest_issue, "%Y-%m-%d").replace(tzinfo=KST)
            grid_age_hours = (ref_now - issue_dt).total_seconds() / 3600
            grid_stale = grid_age_hours > GRID_STALE_HOURS
            grid_note = f"{grid_latest_issue} 발행분 {grid_row_count}행" + ("(지연)" if grid_stale else "")

        pop_row = conn.execute(
            """
            SELECT value FROM grid_forecast
            WHERE variable='POP' AND target_time_kst >= ?
            ORDER BY target_time_kst ASC LIMIT 1
            """,
            (ref_now.strftime("%Y-%m-%dT%H:%M:%S"),),
        ).fetchone()
        if pop_row and pop_row[0] is not None and not grid_stale:
            weather["precipitationProbabilityPct"] = pop_row[0]

        # --- NWP: 최신 발표일의 완결성(진짜 값 채워졌는지까지) ---
        nwp_row = conn.execute(
            "SELECT issue_date, complete, complete_before_issue, stored_value_count, "
            "expected_value_count, missing_value_count, status_message, completed_at "
            "FROM nwp_run_status ORDER BY issue_date DESC LIMIT 1"
        ).fetchone()
        nwp_note = "NWP 이력 없음"
        nwp_complete = False
        nwp_stale = True
        nwp_last = None
        nwp_expected = nwp_missing = nwp_received = None
        if nwp_row:
            (issue_date, complete, complete_before, stored, expected, missing, status_msg, completed_at) = nwp_row
            nwp_last = completed_at
            nwp_expected, nwp_missing = expected, missing
            nwp_received = stored - missing if stored is not None and missing is not None else None
            completed_dt = parse_kst(completed_at)
            issue_age_days = (ref_now.date() - datetime.strptime(issue_date, "%Y-%m-%d").date()).days
            nwp_stale = issue_age_days > 1
            all_missing = expected is not None and missing == expected
            if all_missing:
                nwp_note = f"{issue_date}런 특성 전부 결측(API native missing) - 완결성 실패"
                nwp_complete = False
            elif missing and missing > 0:
                nwp_note = f"{issue_date}런 {missing}/{expected} 결측(예: DSWRFLX·DIFSWRF 상시결측 포함) - 부분 완결성"
                nwp_complete = False
            elif complete and complete_before:
                nwp_note = f"{issue_date}런 전특성 정상수신"
                nwp_complete = True
            else:
                nwp_note = status_msg or "상태 불명"
                nwp_complete = False

        return {
            "weather": weather,
            "feeds": {
                "asos": {
                    "lastReceivedKst": asos_time_str,
                    "ageMinutes": asos_age_min,
                    "expectedCount": None, "receivedCount": None, "missingCount": None,
                    "complete": asos_age_min is not None and asos_age_min <= ASOS_STALE_MIN,
                    "stale": asos_age_min is None or asos_age_min > ASOS_STALE_MIN,
                    "note": asos_note,
                },
                "grid": {
                    "lastReceivedKst": grid_latest_issue,
                    "ageMinutes": None,
                    "expectedCount": None, "receivedCount": grid_row_count or None, "missingCount": None,
                    "complete": grid_row_count > 0 and not grid_stale,
                    "stale": grid_stale,
                    "note": grid_note,
                },
                "nwp": {
                    "lastReceivedKst": nwp_last,
                    "ageMinutes": minutes_since(parse_kst(nwp_last), ref_now),
                    "expectedCount": nwp_expected, "receivedCount": nwp_received, "missingCount": nwp_missing,
                    "complete": nwp_complete,
                    "stale": nwp_stale,
                    "note": nwp_note,
                },
            },
        }
    finally:
        conn.close()


# ───────────────────── 조립·저장 ─────────────────────

def build_snapshot() -> dict[str, Any]:
    ref_now = now_kst()
    power = load_power_and_inverters(ref_now)
    weather_feeds = load_weather_and_feeds(ref_now)
    performance = compute_performance(ref_now, power, weather_feeds)

    source_label = f"LOCAL DB · SHADOW 공식 운영 전 · {power['powerNote']}"

    return {
        "plantKey": "gwangju",
        "generatedAtKst": ref_now.isoformat(timespec="seconds"),
        "sourceLabel": source_label,
        "currentPowerKw": power["currentPowerKw"],
        "todayEnergyKwh": power["todayEnergyKwh"],
        # Shadow 미시작 - 절대 값 채우지 않음(사용자 지시 4번).
        "forecastEnergyKwh": None,
        "weather": weather_feeds["weather"],
        "inverters": power["inverters"],
        "dataCollection": weather_feeds["feeds"],
        "todayPowerSeries": power["todayPowerSeries"],
        "performance": performance,
        "capacity": {**CAPACITY_PROFILE, "liveChangeNotice": load_capacity_live_change_notice()},
    }


def validate(snapshot: dict[str, Any]) -> None:
    required_top = {
        "plantKey", "generatedAtKst", "sourceLabel", "currentPowerKw", "todayEnergyKwh",
        "forecastEnergyKwh", "weather", "inverters", "dataCollection",
        "todayPowerSeries", "performance", "capacity",
    }
    missing = required_top - snapshot.keys()
    if missing:
        raise RuntimeError(f"필수 필드 누락: {missing}")
    if snapshot["forecastEnergyKwh"] is not None:
        raise RuntimeError("Shadow 미시작인데 forecastEnergyKwh가 채워짐 - 중단(가짜 예측 방지)")
    if snapshot.get("plantKey") != "gwangju":
        raise RuntimeError("plantKey 불일치")


def atomic_write(snapshot: dict[str, Any], path: Path) -> None:
    """★09-03 추가★: Windows에서 이 파일을 서빙하는 Vite 개발서버(HMR
    파일감시·정적서빙)가 순간적으로 핸들을 잡고 있으면 os.replace가
    PermissionError(WinError 5)로 실패하는 걸 실측으로 확인했다(대시보드
    앱을 계속 띄워둔 채로 5분 스케줄러가 도는 구조라 반복 발생 - 서버를
    끄는 건 서비스 중단이라 금지, 대신 재시도로 흡수한다). 최대 5회,
    100ms 간격 재시도 - 그래도 안 되면 진짜 문제이니 그대로 예외를
    올린다(조용히 실패 삼키지 않음)."""
    validate(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(
        f"gwangju-dashboard.json 쓰기 5회 재시도 후에도 실패(다른 프로세스가 "
        f"계속 잠그고 있음 - 09-03 사건 참고): {last_error}"
    ) from last_error


def main() -> int:
    snapshot = build_snapshot()
    atomic_write(snapshot, OUTPUT_JSON)
    print(json.dumps({
        "status": "ok",
        "generatedAtKst": snapshot["generatedAtKst"],
        "currentPowerKw": snapshot["currentPowerKw"],
        "todayEnergyKwh": snapshot["todayEnergyKwh"],
        "inverterCount": len(snapshot["inverters"]),
        "nwp_note": snapshot["dataCollection"]["nwp"]["note"],
        "grid_note": snapshot["dataCollection"]["grid"]["note"],
        "asos_note": snapshot["dataCollection"]["asos"]["note"],
        "output": str(OUTPUT_JSON),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
