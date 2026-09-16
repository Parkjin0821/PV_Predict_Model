# -*- coding: utf-8 -*-
"""라이브 수집 현황 요약 JSON 생성기 - 저장자료 전용, API 호출 없음.

09-02 밤 사용자 요청("영광 진행률 사이트처럼, 수집 현황도 눈으로 보고
싶다")로 신규 작성. 4개 지역(광주·부안·김제·영광)의 Blockdata·ASOS/GRID/
NWP 8개 라이브 SQLite + 광주 shadow_predictions.sqlite3까지, 전부
읽기전용(`?mode=ro`)으로만 열어 행수·최신시각·신선도를 요약한다.

산출물: collection_status.json (같은 폴더의 collection_status.html이
5초마다 이 파일을 다시 읽어 화면에 그린다).

실행: python generate_collection_status_v1_2026-09-02.py
크론/스케줄러에 5분 간격으로 재실행해도 안전(멱등, 부작용 없음, 원자적
os.replace 교체).

★09-08 정리★(사용자 지시: "완료된 것들 빼도 되지 않냐 + 영광 추가"):
- 영광 라이브 3종(Blockdata·ASOS/GRID·NWP, 코덱스가 09-08 신규 구축)을
  REGIONS에 추가.
- 완료돼서 더는 "진행 현황"이 아닌 3개 섹션 제거: 예약작업 현황(09-03
  1회성 항목들, ONETIME 3개는 이미 삭제됨), 위성(GK2A) 수집·추출(수집
  자체가 09-04부로 중단 상태, AGENTS.md에 별도 기록됨), 김제 인근관측소
  백필(710/710일 100% 완료), 요인재검증 2단계(4개 지역 전부 완료 +
  이 페이지가 보여주던 수치는 이후 갱신된 공식수치보다 낡음 - 최신
  수치는 AGENTS.md/Notion 참고). 이 4개는 원래 "진행 중인지 보려고"
  만든 일회성 추적용이라 완료 후엔 대시보드에 남겨둘 이유가 없다.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

OUTPUT_DIR = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\수집현황_v1_2026-09-02")
OUTPUT_JSON = OUTPUT_DIR / "collection_status.json"

REGIONS = {
    "광주": {
        "blockdata": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25\blockdata_history.sqlite3",
        "weather": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3",
    },
    "부안": {
        "blockdata": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3",
        "weather": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3",
    },
    "김제": {
        "blockdata": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\blockdata_live_history_gimje_v1_2026-09-01\blockdata_history.sqlite3",
        "weather": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",
    },
    # ★09-08 신규★: 코덱스가 오늘 구축(작업지시서 1·2순위) - Blockdata
    # plant_id=7912, ASOS252·GRID(52,78), NWP 5종 매일 14:20.
    "영광": {
        "blockdata": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\blockdata_live_history_yeonggwang_v1_2026-09-08\blockdata_history.sqlite3",
        "weather": r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",
    },
}

SHADOW_DB = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인"
    r"\outputs\shadow_predictions\shadow_predictions.sqlite3"
)

# ★09-08 신규★: 부안·김제·영광 초단기(+1~4h) live shadow predict
# (live_feature_assembler_ultrashort_v1_2026-09-08.py, 5분 예약작업)의
# 상태·성능을 대시보드에서도 눈으로 확인할 수 있게 추가.
ULTRASHORT_PIPELINE_DIR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인"
)
ULTRASHORT_SHADOW_DB = ULTRASHORT_PIPELINE_DIR / "shadow_predictions_ultrashort.sqlite3"

# ★09-09 신규★: Blockdata "발전소정보"(정적 설비정보 - 방위각·설치각도·
# 모듈제조사 등) 편입. fetch_blockdata_plant_info_v1.py로 08월에 이미
# 받아뒀으나 그동안 어느 DB·모델·대시보드에도 안 쓰이고 있던 걸 사용자
# 질문으로 발견 - plant_equipment_info_v1_2026-09-09.json으로 정리한 뒤
# 여기서도 보여준다(AGENTS.md 09-09 절 참고).
PLANT_EQUIPMENT_JSON = ULTRASHORT_PIPELINE_DIR / "plant_equipment_info_v1_2026-09-09.json"


def load_plant_equipment() -> dict[str, Any]:
    if not PLANT_EQUIPMENT_JSON.is_file():
        return {}
    try:
        data = json.loads(PLANT_EQUIPMENT_JSON.read_text(encoding="utf-8"))
        return data.get("지역별_설비정보", {})
    except Exception:  # noqa: BLE001 - 대시보드는 이 파일 문제로 전체가 죽으면 안 됨
        return {}


# ★09-09 신규★: 사용자 질문("발전소 실시간 정보도 지금 수집이 되고
# 있는지 궁금해")에 답하려고 확인한 결과, `inverter_measurements`(5분
# 주기 Blockdata 실시간 계측)는 4지역 다 정상 수집 중이었다. 이 실측을
# 대시보드에도 그대로 보여준다 - 정적 설비정보(load_plant_equipment)와는
# 완전히 다른 테이블·수집기임을 구분한다.
def load_realtime_power(ref_now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region, paths in REGIONS.items():
        conn = open_readonly(paths["blockdata"])
        if conn is None:
            rows.append({"region": region, "missing": True})
            continue
        try:
            latest = conn.execute(
                "SELECT MAX(measurement_time) FROM inverter_measurements"
            ).fetchone()[0]
            if latest is None:
                rows.append({"region": region, "missing": False, "noData": True})
                continue
            # 인버터마다 초 단위로 측정시각이 조금씩 달라(예: 8호기
            # 14:38:26, 7호기 14:38:24) 정확히 같은 measurement_time으로
            # 필터하면 인버터 1대만 잡힌다(09-09 실측으로 발견·수정).
            # 인버터별 "자기 최신값"을 먼저 뽑은 뒤 그걸 합산한다.
            agg = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(ac_power),0), COALESCE(SUM(dc_power),0), "
                "COALESCE(SUM(daily_energy),0), "
                "SUM(CASE WHEN quality_status != 'ok' THEN 1 ELSE 0 END) "
                "FROM inverter_measurements im WHERE im.measurement_time = ("
                "  SELECT MAX(m2.measurement_time) FROM inverter_measurements m2"
                "  WHERE m2.inverter_number = im.inverter_number)"
            ).fetchone()
            inverter_n, ac_kw, dc_kw, daily_kwh, bad_quality = agg
            # ★09-09★ 사용자 질문("이상신호 뜨는데 이유를 몰라") - quality_flags_json에
            # 이미 사유가 저장돼 있으므로(예: stale_measurement>15min), 개수만
            # 세지 말고 실제 사유·인버터번호까지 같이 뽑아 대시보드에 보여준다.
            bad_detail = conn.execute(
                "SELECT inverter_number, quality_status, quality_flags_json, measurement_time "
                "FROM inverter_measurements im WHERE im.measurement_time = ("
                "  SELECT MAX(m2.measurement_time) FROM inverter_measurements m2"
                "  WHERE m2.inverter_number = im.inverter_number)"
                "  AND quality_status != 'ok'"
            ).fetchall()
            bad_reasons = []
            for inv_num, q_status, flags_json, m_time in bad_detail:
                try:
                    flags = json.loads(flags_json) if flags_json else []
                except (json.JSONDecodeError, TypeError):
                    flags = []
                bad_reasons.append({
                    "inverterNumber": inv_num, "qualityStatus": q_status,
                    "flags": flags, "measurementKst": m_time,
                })
            rows.append({
                "region": region, "missing": False, "noData": False,
                "latestKst": latest, "ageMinutes": minutes_since(parse_kst(latest), ref_now),
                "inverterCount": int(inverter_n or 0),
                "acPowerKw": round(float(ac_kw or 0), 1),
                "dcPowerKw": round(float(dc_kw or 0), 1),
                "dailyEnergyKwh": round(float(daily_kwh or 0), 1),
                "badQualityCount": int(bad_quality or 0),
                "badQualityDetail": bad_reasons,
            })
        finally:
            conn.close()
    return rows


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def open_readonly(path: str | Path) -> sqlite3.Connection | None:
    p = Path(path)
    if not p.is_file():
        return None
    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA query_only = ON")
    return conn


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


def table_stat(conn: sqlite3.Connection, table: str, time_col: str,
                where: str = "") -> dict[str, Any]:
    """table의 전체 행수 + 최신 time_col 값을 반환. 테이블 없으면 count=None."""
    try:
        clause = f"WHERE {where}" if where else ""
        row = conn.execute(
            f"SELECT COUNT(*), MAX({time_col}) FROM {table} {clause}"
        ).fetchone()
        return {"count": row[0], "latestKst": row[1]}
    except sqlite3.OperationalError:
        return {"count": None, "latestKst": None}


def feed_entry(ref_now: datetime, count: int | None, latest_raw: str | None,
               stale_min: float) -> dict[str, Any]:
    latest = parse_kst(latest_raw)
    age = minutes_since(latest, ref_now)
    return {
        "count": count,
        "latestKst": latest.isoformat(timespec="seconds") if latest else None,
        "ageMinutes": age,
        "stale": age is not None and age > stale_min,
        "missing": count is None,
    }


def load_region(name: str, paths: dict[str, str], ref_now: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {"region": name}

    block_conn = open_readonly(paths["blockdata"])
    latest_ac_power = None
    if block_conn is None:
        result["blockdata"] = feed_entry(ref_now, None, None, 20)
    else:
        try:
            s = table_stat(block_conn, "plant_snapshots", "snapshot_time")
            result["blockdata"] = feed_entry(ref_now, s["count"], s["latestKst"], 20)
            # ★09-15 추가★: 야간 비가동 오탐 방지용 - 최신 스냅샷의 출력값을
            # 같이 읽어둔다(아래 verdict 계산에서 사용). 실측 확인: 광주·영광은
            # 인버터가 밤에 완전히 꺼져 21:00경~05:05경 보고 자체를 중단하고
            # (09-14 기준 485분 공백), 부안·김제는 밤에도 0값을 계속 보고한다
            # - 설비 특성 차이지 장애가 아님.
            row = block_conn.execute(
                "SELECT plant_ac_power_kw FROM plant_snapshots "
                "ORDER BY snapshot_time DESC LIMIT 1"
            ).fetchone()
            if row is not None and row[0] is not None:
                latest_ac_power = float(row[0])
        finally:
            block_conn.close()
    result["blockdata"]["latestAcPowerKw"] = latest_ac_power

    weather_conn = open_readonly(paths["weather"])
    if weather_conn is None:
        result["asos"] = feed_entry(ref_now, None, None, 150)
        result["grid"] = feed_entry(ref_now, None, None, 27 * 60)
        result["nwp"] = feed_entry(ref_now, None, None, 27 * 60)
    else:
        try:
            asos = table_stat(weather_conn, "asos_hourly", "observation_time")
            result["asos"] = feed_entry(ref_now, asos["count"], asos["latestKst"], 150)
            grid = table_stat(weather_conn, "grid_forecast", "run_time_kst")
            result["grid"] = feed_entry(ref_now, grid["count"], grid["latestKst"], 27 * 60)
            nwp = table_stat(
                weather_conn, "nwp_values", "first_received_at",
                where="is_missing=0",
            )
            result["nwp"] = feed_entry(ref_now, nwp["count"], nwp["latestKst"], 27 * 60)
        finally:
            weather_conn.close()

    # ★09-15 신규(Claude, 사용자 요청 - "정상/이상 판정이 없어서 매번 물어보게
    # 된다")★: 4개 피드(blockdata/asos/grid/nwp)의 missing/stale을 한 줄
    # 판정으로 요약. 새 진단을 만드는 게 아니라 이미 있는 feed_entry의
    # missing·stale 불리언을 그대로 집계만 한다.
    feeds = ["blockdata", "asos", "grid", "nwp"]
    feed_labels = {"blockdata": "Blockdata", "asos": "ASOS", "grid": "GRID", "nwp": "NWP"}
    missing_feeds = [f for f in feeds if result.get(f, {}).get("missing")]
    stale_feeds = [f for f in feeds if not result.get(f, {}).get("missing") and result.get(f, {}).get("stale")]

    # ★09-15 추가(야간 비가동 오탐 방지)★: Blockdata가 "지연"인데 마지막
    # 스냅샷 출력이 사실상 0이면, 이건 장애가 아니라 발전이 끝나 인버터가
    # 보고를 멈춘 정상 상태다(광주·영광 실측: 매일 21:00경~05:05경 보고 중단,
    # 09-14·09-15 동일 패턴 확인). 이 경우에만 Blockdata를 지연 목록에서
    # 빼고 별도 사유로 표기한다 - 출력이 유의미한데 끊긴 경우는 그대로 경보.
    night_idle = False
    if ("blockdata" in stale_feeds
            and result.get("blockdata", {}).get("latestAcPowerKw") is not None
            and abs(result["blockdata"]["latestAcPowerKw"]) < 1.0):
        stale_feeds = [f for f in stale_feeds if f != "blockdata"]
        night_idle = True

    if missing_feeds:
        verdict, reason = "결측", f"{'·'.join(feed_labels[f] for f in missing_feeds)} DB 자체를 못 찾음"
    elif stale_feeds:
        verdict, reason = "지연", f"{'·'.join(feed_labels[f] for f in stale_feeds)} 기준 지연시간 초과"
    elif night_idle:
        verdict, reason = "정상", "발전 종료(출력 0)로 Blockdata 보고 중단 - 야간 정상, 나머지 피드 이상 없음"
    else:
        verdict, reason = "정상", "4개 피드 전부 기준 지연시간 이내"
    result["verdict"] = verdict
    result["verdictReason"] = reason
    result["blockdataNightIdle"] = night_idle

    return result


def load_kma_infrastructure_status(regions: list[dict[str, Any]]) -> dict[str, Any]:
    """모델 상태와 분리해 표시할 KMA 공통 입력 장애를 판정한다.

    단일 지역 결측을 전역 장애로 과장하지 않도록 4지역 ASOS가 모두
    180분을 넘긴 경우에만 활성화한다. 외부 네트워크를 새로 호출하지 않고
    이미 저장된 DB 최신시각만 사용한다.
    """
    affected = [
        row["region"] for row in regions
        if row.get("asos", {}).get("ageMinutes") is not None
        and row["asos"]["ageMinutes"] > 180
    ]
    active = len(affected) == len(REGIONS)
    return {
        "active": active,
        "scope": affected,
        "basis": "4지역 ASOS 최신값이 모두 3시간 초과" if active else "공통 장애 판정 기준 미충족",
        "gatePolicy": "장애 원인 대기행은 모델 생성률과 분리; 공식 판정은 일시정지" if active else "정상 판정",
    }


def load_shadow(ref_now: datetime) -> dict[str, Any]:
    conn = open_readonly(SHADOW_DB)
    if conn is None:
        return {"total": None, "success": None, "latestKst": None, "ageMinutes": None,
                "byTier": {}}
    try:
        total_row = conn.execute(
            "SELECT COUNT(*), MAX(predicted_at) FROM shadow_predictions"
        ).fetchone()
        success_row = conn.execute(
            "SELECT COUNT(*) FROM shadow_predictions WHERE status='success'"
        ).fetchone()
        by_tier: dict[str, dict[str, int]] = {}
        for tier, status, cnt in conn.execute(
            "SELECT tier, status, COUNT(*) FROM shadow_predictions "
            "GROUP BY tier, status"
        ):
            by_tier.setdefault(tier, {})[status] = cnt
        latest = parse_kst(total_row[1])
        return {
            "total": total_row[0],
            "success": success_row[0],
            "latestKst": latest.isoformat(timespec="seconds") if latest else None,
            "ageMinutes": minutes_since(latest, ref_now),
            "byTier": by_tier,
        }
    finally:
        conn.close()


def _load_ultrashort_evaluate_module():
    """03_모델학습/현재_종합파이프라인의 evaluate_ultrashort_shadow를
    재사용(재구현 안 함) - MAE/RMSE/nMAE 계산 로직을 여기서 다시 안 짠다."""
    import importlib.util
    import sys
    path = ULTRASHORT_PIPELINE_DIR / "evaluate_ultrashort_shadow_v1_2026-09-08.py"
    spec = importlib.util.spec_from_file_location("ultrashort_evaluate_for_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ultrashort_shadow(ref_now: datetime) -> dict[str, Any]:
    conn = open_readonly(ULTRASHORT_SHADOW_DB)
    if conn is None:
        return {"total": None, "success": None, "waiting": None, "failed": None,
                "latestKst": None, "ageMinutes": None, "byRegionHorizon": [], "metrics": []}
    try:
        recent_start = (ref_now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        total_row = conn.execute(
            "SELECT COUNT(*), MAX(run_at_kst) FROM shadow_ultrashort_predictions "
            "WHERE issue_time_kst >= ?", (recent_start,)
        ).fetchone()
        counts = dict(conn.execute(
            "SELECT status, COUNT(*) FROM shadow_ultrashort_predictions "
            "WHERE issue_time_kst >= ? GROUP BY status", (recent_start,)
        ).fetchall())
        by_rh = [
            {"region": region, "horizon": h, "status": status, "count": cnt}
            for region, h, status, cnt in conn.execute(
                "SELECT region, horizon_h, status, COUNT(*) FROM shadow_ultrashort_predictions "
                "WHERE issue_time_kst >= ? GROUP BY region, horizon_h, status "
                "ORDER BY region, horizon_h", (recent_start,)
            )
        ]
        latest_row = conn.execute(
            "SELECT MAX(run_at_kst) FROM shadow_ultrashort_predictions"
        ).fetchone()
        latest = parse_kst(latest_row[0])
        waiting = counts.get("대기", 0)
        night_wait = conn.execute(
            "SELECT COUNT(*) FROM shadow_ultrashort_predictions WHERE issue_time_kst >= ? "
            "AND status='대기' AND reason LIKE '%야간%'", (recent_start,)
        ).fetchone()[0]
        infrastructure_wait = conn.execute(
            "SELECT COUNT(*) FROM shadow_ultrashort_predictions WHERE issue_time_kst >= ? "
            "AND status='대기' AND reason NOT LIKE '%야간%' AND reason LIKE '%ASOS%'",
            (recent_start,),
        ).fetchone()[0]
        input_wait = max(0, waiting - night_wait - infrastructure_wait)
        operational_total = counts.get("성공", 0) + input_wait + counts.get("실패", 0)
        excluded_wait = night_wait + infrastructure_wait
        cumulative = dict(conn.execute(
            "SELECT status, COUNT(*) FROM shadow_ultrashort_predictions GROUP BY status"
        ).fetchall())
        metrics: list[dict[str, Any]] = []
        try:
            df = _load_ultrashort_evaluate_module().evaluate()
            if df is not None and not df.empty:
                metrics = df.to_dict("records")
                # 09-09: evaluate()의 R2가 계산불가(분산≈0, 예: 야간 실측이
                # 전부 0kW)일 때 None을 넣어도, dict 목록이 pd.DataFrame을
                # 거치며 float 컬럼이면 None이 NaN으로 조용히 바뀐다.
                # json.dumps는 NaN을 (표준 아닌) 리터럴 NaN으로 그대로
                # 써버려 브라우저 JSON.parse가 깨진다 - 여기서 명시적으로
                # None으로 되돌린다(math.isnan은 float에만 쓸 수 있어
                # isinstance로 먼저 걸러야 문자열 값에서 안 터진다).
                for row in metrics:
                    for k, v in row.items():
                        if isinstance(v, float) and math.isnan(v):
                            row[k] = None
        except Exception as exc:  # noqa: BLE001 - 대시보드는 평가 실패해도 나머지는 계속 보여줘야 함
            metrics = [{"error": str(exc)}]
        return {
            "total": total_row[0], "success": counts.get("성공", 0),
            "waiting": waiting, "failed": counts.get("실패", 0),
            "operationalTotal": operational_total, "excludedWait": excluded_wait,
            "periodLabel": "최근 24시간",
            "waitBreakdown": {"night": night_wait, "infrastructure": infrastructure_wait,
                              "input": input_wait},
            "cumulative": cumulative,
            "latestKst": latest.isoformat(timespec="seconds") if latest else None,
            "ageMinutes": minutes_since(latest, ref_now),
            "byRegionHorizon": by_rh, "metrics": metrics,
        }
    finally:
        conn.close()


# ★09-09 신규★: 4지역 초단기 Shadow 운영사고 2건(배터리·스케줄러 꼬임)
# 이후 사용자 판단으로 "7일 운영 신뢰성 사전게이트" 시계를 두 버그를
# 다 고친 시점부터 재시작하기로 확정(AGENTS.md 09-09 절). 그 판정
# 기준·진행상황을 대시보드에서도 눈으로 볼 수 있게 여기서 계산한다 -
# 기준선(성공률 95%+)만 재사용하고 판정 자체를 새로 만들지 않는다.
# ★★09-16 최종 재시작★★: 광주 power_lag 최소 완화에 이어 용량 역할을
# 인버터 등록합계(물리 clip 241.58kW)와 공식용량(nMAE 240.0kW)으로
# 분리했다. 최종 입력·출력 정책이 반영된 첫 정기 실행부터 7일 검증한다.
# 직전 창은 삭제하지 않고 `previousWindow`로 참고치 보존한다.
GATE_START_KST = datetime(2026, 9, 16, 17, 50, 0, tzinfo=KST)
GATE_JUDGE_KST = datetime(2026, 9, 23, 17, 50, 0, tzinfo=KST)
# 수정 전 참고치(삭제 금지 - 무엇이 얼마나 나아졌는지 비교 근거)
GATE_PREV_START_KST = datetime(2026, 9, 16, 17, 10, 0, tzinfo=KST)
GATE_PREV_JUDGE_KST = datetime(2026, 9, 23, 17, 10, 0, tzinfo=KST)
GATE_SLOT_MINUTES = 5
GATE_SUCCESS_THRESHOLD_PCT = 95.0
GATE_ACTIVE_START_MINUTE = 4 * 60 + 30
GATE_ACTIVE_END_MINUTE = 20 * 60 + 30


def _expected_gate_slots(window_end: datetime) -> int:
    """실제 예약창(04:30~20:30)에 포함되는 5분 슬롯만 센다."""
    cursor = GATE_START_KST
    count = 0
    while cursor <= window_end:
        minute_of_day = cursor.hour * 60 + cursor.minute
        if GATE_ACTIVE_START_MINUTE <= minute_of_day <= GATE_ACTIVE_END_MINUTE:
            count += 1
        cursor += timedelta(minutes=GATE_SLOT_MINUTES)
    return max(1, count)


def _first_snapshot_by_date(region: str, start_str: str, end_str: str) -> dict[str, datetime]:
    """지역별 '그 날 발전소가 보고를 재개한 첫 시각'을 날짜별로 반환.

    ★09-16 신규★: 광주(6715)·영광(7912) 인버터는 야간(대략 21:00~05:05)에
    보고를 아예 멈춘다. 재가동 직후에는 power_lag_15/30/60을 만들 이력이
    물리적으로 없어서, 재가동 후 60분 동안은 초단기 예측이 **원초적으로
    불가능**하다(모델·수집 문제가 아님). 야간게이트와 같은 성격이므로
    게이트 분모에서 빼기 위해 기준시각을 구한다.
    """
    conn = open_readonly(REGIONS[region]["blockdata"])
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT substr(snapshot_time,1,10) d, MIN(snapshot_time) "
            "FROM plant_snapshots "
            "WHERE substr(replace(snapshot_time,'T',' '),1,19) >= ? "
            "AND substr(replace(snapshot_time,'T',' '),1,19) <= ? "
            "GROUP BY d",
            (start_str, end_str),
        ).fetchall()
    except Exception:  # noqa: BLE001 - 게이트 표시가 여기서 죽으면 안 됨
        return {}
    finally:
        conn.close()
    out: dict[str, datetime] = {}
    for day, first in rows:
        try:
            out[day] = datetime.fromisoformat(first).replace(tzinfo=None)
        except Exception:  # noqa: BLE001
            continue
    return out


# 재가동 후 이 시간 동안은 power_lag_60min을 만들 수 없다(가장 긴 lag 기준).
WARMUP_MINUTES = 60


def load_gate_status(ref_now: datetime) -> dict[str, Any]:
    """09-16 사전게이트 진행상황을 shadow DB 실측으로 계산.

    "예약실행 성공률"은 5분 슬롯 대비 실제 issue_time이 기록된(상태 무관-
    성공/대기/실패 전부 포함, 스케줄러가 최소한 로그는 남겼다는 뜻) 비율로
    근사한다 - 조립기가 아예 못 뜬 경우(오늘 있었던 것처럼)는 로그 자체가
    없어 정확히 못 잡지만, 그 경우는 "미기록 슬롯"으로 그대로 드러난다.
    "예측생성률"은 기록된 issue_time 중 성공 상태 비율이다.
    """
    elapsed_min = (ref_now - GATE_START_KST).total_seconds() / 60.0
    if elapsed_min < 0:
        return {"status": "미시작", "startKst": GATE_START_KST.isoformat(timespec="seconds"),
                "judgeKst": GATE_JUDGE_KST.isoformat(timespec="seconds"), "regions": []}
    window_min = min(elapsed_min, (GATE_JUDGE_KST - GATE_START_KST).total_seconds() / 60.0)
    window_end = GATE_START_KST + timedelta(minutes=window_min)
    expected_slots = _expected_gate_slots(window_end)

    conn = open_readonly(ULTRASHORT_SHADOW_DB)
    regions: list[dict[str, Any]] = []
    if conn is not None:
        try:
            start_str = GATE_START_KST.strftime("%Y-%m-%d %H:%M:%S")
            end_str = window_end.strftime("%Y-%m-%d %H:%M:%S")
            active_clause = (
                "substr(replace(run_at_kst, 'T', ' '), 1, 19) >= ? "
                "AND substr(replace(run_at_kst, 'T', ' '), 1, 19) <= ? "
                "AND substr(replace(run_at_kst, 'T', ' '), 12, 8) "
                "BETWEEN '04:30:00' AND '20:30:59'"
            )
            for region in ["광주", "부안", "김제", "영광"]:
                logged = conn.execute(
                    "SELECT COUNT(DISTINCT issue_time_kst) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                success = conn.execute(
                    "SELECT COUNT(DISTINCT issue_time_kst) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status='성공' AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                failed_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '실패' AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                bad_pred = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '성공' AND {active_clause} "
                    "AND (predicted_kw IS NULL OR predicted_kw < 0)",
                    (region, start_str, end_str),
                ).fetchone()[0]
                # ★★09-16 정정★★: 기존 run_rate = logged(distinct issue_time) / 5분슬롯 은
                # **스케줄러 실행률이 아니다**. `issue_time`은 발전소가 준 마지막
                # 스냅샷 시각이고, 조립기는 `INSERT OR IGNORE` + UNIQUE(region,
                # horizon_h, issue_time)라 **같은 issue_time이면 행을 안 쓴다**.
                # 즉 발전소가 새 스냅샷을 덜 주면 조립기가 정상 실행돼도
                # run_rate가 자동으로 떨어진다.
                # 실측(09-15 16:00~): 신규 스냅샷 부안 221·김제 213·광주 118·영광 104
                #                     기록 사이클 부안 125·김제 121·광주 113·영광  99
                # → 영광 "실행률 78.5%"의 정체는 **야간 인버터 정지로 스냅샷이
                #   적은 것**이지 조립기 결함이 아니었다.
                # 정정: 분모를 "그 구간에 실제로 새 스냅샷이 있었던 횟수"로 바꿔
                # **데이터가 있었는데 조립기가 안 돌았는가**만 재게 한다.
                snap_conn = open_readonly(REGIONS[region]["blockdata"])
                new_snapshots = None
                if snap_conn is not None:
                    try:
                        new_snapshots = snap_conn.execute(
                            "SELECT COUNT(*) FROM plant_snapshots "
                            "WHERE substr(replace(snapshot_time,'T',' '),1,19) >= ? "
                            "AND substr(replace(snapshot_time,'T',' '),1,19) <= ?",
                            (start_str, end_str),
                        ).fetchone()[0]
                    except Exception:  # noqa: BLE001
                        new_snapshots = None
                    finally:
                        snap_conn.close()
                run_denominator = min(new_snapshots, expected_slots) if new_snapshots else expected_slots
                run_rate = (round(min(logged, run_denominator) / run_denominator * 100, 1)
                            if run_denominator else None)
                # ★09-09 정정★: 기존 gen_rate = success(issue_time distinct)/logged(issue_time
                # distinct, status별 합산)는 두 가지 문제가 있었다.
                # ①같은 issue_time이 수평(h1~h4)별로 상태가 갈리면(예: h1·h2는 성공,
                #   h3·h4는 야간이라 대기) status별 GROUP BY·distinct count 합산 과정에서
                #   그 issue_time이 성공 쪽과 대기 쪽에 중복 집계돼 분모가 부풀려진다.
                # ②"야간 target 태양고도<=0" 게이트로 대기 처리된 행(설계대로 정상 차단,
                #   실패 아님)이 분모에 그대로 들어가 있어, 하루가 지날수록(야간 슬롯이
                #   누적될수록) 지표가 계속 낮아지는 것처럼 보였다("계속 낮아지는 것 같다"는
                #   사용자 관찰과 일치).
                # 정정: 행(row) 단위로, 야간게이트 대기행은 "해당없음"으로 분모에서 제외하고
                # 계산한다. 실측 재계산 결과 광주 88.7%->98.3%, 부안 89.5%->100.0%,
                # 김제 83.1%->93.2%, 영광 90.8%->100.0%로 대부분 이미 통과권이었다.
                total_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                success_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '성공' AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                night_gated_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '대기' AND reason LIKE '%야간%' AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                # "ASOS(3h 이내 없음)"은 과거 조립기에서 실제 ASOS 전체
                # 누락뿐 아니라 GHI 단독 NULL까지 같은 문구로 기록했다.
                # 화면에서는 missing_features를 함께 보고 세 원인을 분리한다.
                asos_all_missing_pattern = (
                    '%"obs_temp_c"%"obs_rh_pct"%"obs_cloud_pct"%'
                    '"obs_wind_ms"%"obs_ghi_wm2"%'
                )
                actual_asos_wait_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '대기' AND reason LIKE '%ASOS(3h 이내 없음)%' "
                    "AND missing_features LIKE ? "
                    f"AND {active_clause}",
                    (region, asos_all_missing_pattern, start_str, end_str),
                ).fetchone()[0]
                ghi_only_wait_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '대기' AND reason LIKE '%ASOS(3h 이내 없음)%' "
                    "AND missing_features = '[\"obs_ghi_wm2\"]' "
                    f"AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0]
                power_ghi_wait_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '대기' AND reason LIKE '%ASOS(3h 이내 없음)%' "
                    "AND missing_features LIKE '%power_lag%' "
                    "AND missing_features LIKE '%obs_ghi_wm2%' "
                    "AND missing_features NOT LIKE ? "
                    f"AND {active_clause}",
                    (region, asos_all_missing_pattern, start_str, end_str),
                ).fetchone()[0]
                # ★09-16 신규(사용자 승인: "원초적으로 안 되는 시간대면 제외")★:
                # 야간 인버터 정지 후 재가동 직후 60분은 power_lag를 만들 이력이
                # 물리적으로 없어 예측이 불가능하다. 야간게이트와 동일한 성격의
                # 구조적 불가 구간이므로 분모에서 뺀다.
                # 판정은 "lag만 결측(obs_* 결측 없음)" + "재가동 후 60분 이내"
                # 두 조건을 모두 만족하는 행으로 좁힌다 - 낮 시간대 스냅샷
                # 한두 건 누락으로 생긴 lag 결측(모델이 실제로 못 돈 경우)은
                # 제외 대상이 아니다.
                warmup_rows = 0
                first_snap = _first_snapshot_by_date(region, start_str, end_str)
                if first_snap:
                    cands = conn.execute(
                        "SELECT issue_time_kst, missing_features FROM shadow_ultrashort_predictions "
                        f"WHERE region = ? AND status = '대기' "
                        "AND missing_features LIKE '%power_lag%' "
                        "AND missing_features NOT LIKE '%obs_%' "
                        f"AND {active_clause}",
                        (region, start_str, end_str),
                    ).fetchall()
                    for issue_time, _mf in cands:
                        try:
                            t = datetime.fromisoformat(str(issue_time)).replace(tzinfo=None)
                        except Exception:  # noqa: BLE001
                            continue
                        base = first_snap.get(t.strftime("%Y-%m-%d"))
                        if base is not None and t < base + timedelta(minutes=WARMUP_MINUTES):
                            warmup_rows += 1
                # ★09-16(사용자 승인)★: 지표를 **두 개로 분리**한다.
                #  ① 모델 자체 생성률 : 야간·재가동 워밍업을 뺀 "모델이 돌
                #     수 있었어야 하는 구간"에서의 성공률. 모델/피처 품질 판정용.
                #  ② 운영 종단간 생성률 : 발전소 스냅샷 누락까지 전부 포함한
                #     실제 운영 성공률. "고객이 체감하는" 수치.
                # 발전소 스냅샷 누락으로 생긴 lag 결측(워밍업 창 밖)은 모델
                # 결함이 아니지만 운영 품질에서는 엄연한 실패다 - 완전히 빼지
                # 않고 `sourceOutageWaitRows`로 **별도 집계**해 둘 다 보이게 한다.
                source_outage_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    f"WHERE region = ? AND status = '대기' "
                    "AND missing_features LIKE '%power_lag%' "
                    "AND missing_features NOT LIKE '%obs_%' "
                    f"AND {active_clause}",
                    (region, start_str, end_str),
                ).fetchone()[0] - warmup_rows
                source_outage_rows = max(source_outage_rows, 0)

                # ② 운영 종단간: 야간(물리적으로 예측 대상 아님)만 제외.
                e2e_denominator = total_rows - night_gated_rows
                e2e_gen_rate = (round(success_rows / e2e_denominator * 100, 1)
                                if e2e_denominator else None)
                # ① 모델 자체: 야간 + 워밍업(재가동 60분, 구조적 불가) 제외.
                gen_denominator = e2e_denominator - warmup_rows
                gen_rate = round(success_rows / gen_denominator * 100, 1) if gen_denominator else None
                # 모델가능률에서 제외하는 공통 인프라 장애는 실제 ASOS
                # 전체 누락만이다. GHI 단독/발전 lag 결측은 모델 입력 문제다.
                model_denominator = gen_denominator - actual_asos_wait_rows
                model_gen_rate = (
                    round(success_rows / model_denominator * 100, 1)
                    if model_denominator > 0 else None
                )
                regions.append({
                    "region": region, "expectedSlots": expected_slots, "loggedIssueTimes": logged,
                    # ★09-16★ run_rate 분모 정정 근거를 화면에서도 보이게
                    "newSnapshots": new_snapshots,
                    "runRateDenominator": run_denominator,
                    "runRateNote": ("분모=그 구간 발전소 신규 스냅샷 수(5분 슬롯 아님) - "
                                    "발전소가 데이터를 안 주면 조립기가 정상이어도 "
                                    "행이 안 쌓이므로"),
                    "successIssueTimes": success, "runRatePct": run_rate, "genRatePct": gen_rate,
                    "genRateNote": (f"야간게이트 대기({night_gated_rows}행)"
                                    + (f" + 재가동 워밍업({warmup_rows}행)" if warmup_rows else "")
                                    + " 제외 - 행기준 성공률"),
                    "warmupWaitRows": warmup_rows,
                    # ★09-16★ 지표 2종 분리 + 원천 데이터 장애 별도 집계
                    "modelOwnGenRatePct": model_gen_rate,
                    "e2eGenRatePct": e2e_gen_rate,
                    "sourceOutageWaitRows": source_outage_rows,
                    "metricNote": ("모델자체=야간·워밍업 제외 / "
                                   "운영종단간=야간만 제외(발전소 스냅샷 누락 포함)"),
                    "infrastructureWaitRows": actual_asos_wait_rows,
                    "ghiOnlyWaitRows": ghi_only_wait_rows,
                    "powerGhiWaitRows": power_ghi_wait_rows,
                    "modelEligibleGenRatePct": model_gen_rate,
                    "failedRows": failed_rows, "badPredictions": bad_pred,
                    "pass": (run_rate >= GATE_SUCCESS_THRESHOLD_PCT
                             and (model_gen_rate is None or model_gen_rate >= GATE_SUCCESS_THRESHOLD_PCT)
                             and failed_rows == 0 and bad_pred == 0),
                })
        finally:
            conn.close()

    days_elapsed = round(elapsed_min / 1440, 1)
    status = "판정완료" if ref_now >= GATE_JUDGE_KST else "진행중"
    return {
        # ★09-16★: 최종 창 재시작 사실과 그 이유를 화면에서도 알 수 있게 명시.
        "windowRestartedKst": GATE_START_KST.isoformat(timespec="seconds"),
        "windowRestartReason": (
            "광주 power_lag 최소 완화와 용량 역할 분리(물리 clip 241.58kW, "
            "공식/nMAE 240.0kW)를 모두 반영한 09-16 17:50 정기 실행부터 "
            "7일 재검증 - 이전 창은 참고치로 보존"),
        "previousWindow": {
            "startKst": GATE_PREV_START_KST.isoformat(timespec="seconds"),
            "judgeKst": GATE_PREV_JUDGE_KST.isoformat(timespec="seconds"),
            "note": "수정 전 참고치(삭제 금지)",
        },
        "status": status, "startKst": GATE_START_KST.isoformat(timespec="seconds"),
        "judgeKst": GATE_JUDGE_KST.isoformat(timespec="seconds"),
        "daysElapsed": days_elapsed, "daysTotal": 7.0,
        "thresholdPct": GATE_SUCCESS_THRESHOLD_PCT, "regions": regions,
    }


NWP_D1D2_DB = {
    "광주": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_d1d2_live_v1_2026-09-09\kma_nwp_d1d2_live.sqlite3"),
    "부안": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_nwp_d1d2_live_v1_2026-09-08\kma_nwp_d1d2_live.sqlite3"),
    "김제": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_nwp_d1d2_live_v1_2026-09-08\kma_nwp_d1d2_live.sqlite3"),
    "영광": Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_nwp_d1d2_live_v1_2026-09-08\kma_nwp_d1d2_live.sqlite3"),
}
NWP_D1D2_BACKFILL_TARGET_DAYS = 710  # 2024-08-25~2026-08-04, 프로젝트 표준기간
NC_VARS = ("DSWRF", "TCDC", "LCDC", "MCDC", "HCDC")

# ★09-09★ 사용자 질문("광주꺼는 원래 백필 완료 아니었어?")에 대한 답:
# 광주는 08-20에 D+1 전용(710일) 백필을 이미 별도 DB로 완료했고, 이 D+1+D+2
# (+48h) 확장 DB는 오늘(09-09) 광주가 처음 합류하며 새로 만들어진 별개 DB다.
# 부안/김제/영광은 어제(09-08)부터 이 확장 백필 대상이었다. 대시보드에서
# "1/710"이 마치 기존 완료분이 퇴보한 것처럼 보이는 오해를 막기 위해 표시.
NWP_D1D2_LATE_JOIN_NOTE = {
    "광주": "신규(09-09 합류) - 08-20에 끝난 D+1 전용 710일 백필과는 별개 DB, 퇴보 아님",
}


# ★09-16 삭제(사용자 지시 - "필요없는거 삭제")★: load_nwp_nc_transition/
# load_nwp_nc_shadow_observation 제거. 09-09 "D+1 발행시각 10시→14시"
# 전환 여부를 위한 임시 관측용이었는데 7일이 지나도록 결정이 안 났고,
# nwp_values를 정규 D+1 수집기와 구분없이 겹쳐 써서 실제로는 무엇을
# 관측하는지도 불분명했다(09-16 실제로 "부안 발행시각 재설계 필요"
# 오탐 발생 - 원인은 NC가 아니라 그날 버그수정 재실행 잔재였음).
# 백업: generate_collection_status_v1_2026-09-02.py.backup_before_remove_nctransition_shadow_20260916




def load_nwp_d1d2(ref_now: datetime) -> list[dict[str, Any]]:
    """★09-08 신규★: +48h 모델 구축용 D+1+D+2 확장 NWP 수집기
    (`collect_kma_nwp_d1d2_extended_v1_2026-09-08.py`) 현황. dry-run 행은
    제외하고 실제(live) 호출만 집계한다.

    ★09-08 추가★: 코덱스에게 710일 백필을 맡기면서, 진행률(710일 중
    완료된 issue_date 수 - live 기준, dry-run 제외)도 같이 보여준다."""
    rows = []
    for region, path in NWP_D1D2_DB.items():
        conn = open_readonly(path)
        if conn is None:
            rows.append({"region": region, "missing": True, "backfillDays": 0,
                        "backfillTarget": NWP_D1D2_BACKFILL_TARGET_DAYS})
            continue
        try:
            backfill_count = conn.execute(
                "SELECT COUNT(DISTINCT issue_date) FROM nwp_d1d2_run_status WHERE dry_run=0"
            ).fetchone()[0]
            valid_count = conn.execute(
                "SELECT COUNT(DISTINCT issue_date) FROM nwp_d1d2_run_status "
                "WHERE dry_run=0 AND missing_value_count=0"
            ).fetchone()[0]
            r = conn.execute(
                "SELECT issue_date, expected_value_count, stored_value_count, "
                "missing_value_count, completed_at, status_message FROM nwp_d1d2_run_status "
                "WHERE dry_run=0 ORDER BY issue_date DESC LIMIT 1"
            ).fetchone()
            if r is None:
                rows.append({"region": region, "missing": False, "noLiveRun": True,
                            "note": "실제(live) 실행 이력 없음(dry-run만 있음)",
                            "backfillDays": 0, "backfillValidDays": 0,
                            "backfillTarget": NWP_D1D2_BACKFILL_TARGET_DAYS,
                            "lateJoinNote": NWP_D1D2_LATE_JOIN_NOTE.get(region)})
                continue
            completed = parse_kst(r[4])
            # ★09-15 신규(사용자 요청 - "계속 수집 안 하냐, 오늘 끝난거냐,
            # 계속 물어보게 하지 말고 대시보드에 표시해달라")★: 진행률
            # 숫자만으론 "오늘 이게 끝난 상태인지 아직 도는 중인지"가
            # 안 보였음 - 오늘자(issue_date=오늘) 기준 완료/진행중/대기
            # 한 줄 판정을 추가한다. 기존 필드는 전혀 안 건드림(추가만).
            today_str = ref_now.strftime("%Y%m%d")
            if r[0] == today_str and r[3] == 0:
                today_status, today_reason = "오늘 완료", f"{today_str} 발행분 {r[2]}/{r[1]} 전부 수신 완료"
            elif r[0] == today_str and r[3] > 0:
                today_status, today_reason = "오늘 진행중", f"{today_str} 발행분 {r[2]}/{r[1]}만 수신, {r[3]}개 아직 미발행"
            else:
                days_behind = None
                try:
                    from datetime import date as _date
                    d0 = _date(int(r[0][:4]), int(r[0][4:6]), int(r[0][6:8]))
                    d1 = _date(int(today_str[:4]), int(today_str[4:6]), int(today_str[6:8]))
                    days_behind = (d1 - d0).days
                except Exception:
                    pass
                # ★09-16 수정★: 위 문구가 매일 00:00~발행시각(대략 13~14시)
                # 까지 13시간 내내 "계속 실패 중"으로 떠서 사용자가 매번
                # 확인해야 하는 상시 오탐이었다. KIM NC는 issue_date=오늘자
                # 파일이 기상청 서버에 오후에야 올라오고, 그 전까지 수집기는
                # 정상적으로 "KIM NC 발행대기: file is not exist"만 기록한다
                # (exit=0, status=complete). 실제 장애(인증/쿼터/네트워크)와
                # 구분해서 표시한다.
                pub_wait = None
                try:
                    q = conn.execute(
                        "SELECT requested_at, COALESCE(error_message,'') "
                        "FROM nwp_d1d2_requests WHERE issue_date=? AND dry_run=0 "
                        "ORDER BY requested_at DESC LIMIT 1",
                        (today_str,),
                    ).fetchone()
                    if q is not None:
                        pub_wait = ("발행대기" in q[1]) or ("file is not exist" in q[1])
                        last_try, last_msg = q[0], q[1]
                    else:
                        last_try, last_msg = None, ""
                except Exception:
                    last_try, last_msg = None, ""
                # ★09-16 수정(사용자 지적 - "너무 기니까 간략하게")★: 한눈에
                # 정상/이상만 구분되면 되는 자리라 사유를 한 줄로 압축.
                # 상세 근거(재시도 시각·에러 원문)가 필요하면 로그에서 확인.
                if pub_wait:
                    today_status = "오늘 발행대기(정상)"
                    today_reason = "기상청 미발행 - 통상 오후 확정, 30분마다 재시도 중"
                elif last_try is not None:
                    today_status = "오늘 실패"
                    today_reason = f"재시도 실패({days_behind}일 정체) - {last_msg[:40]}"
                else:
                    today_status = "오늘 미착수"
                    today_reason = f"{days_behind}일간 수집 시도 없음"
            # ★09-16 수정(사용자 지적 - "19.1시간전 이렇게 떠있어 헷갈린다")★:
            # 이 ageMinutes는 "최신 완료 issue_date가 끝난 시각으로부터 경과"다.
            # D1D2는 하루 1회(보통 13~14시) 갱신되는 지표라, 매일 발행 직전에는
            # 최대 약 24~30시간까지 나오는 게 정상이다(30분 주기 실시간
            # 피드와 같은 잣대로 보면 안 됨). 그런데 라벨이 숫자만 노출해서
            # 옆의 todayStatus("오늘 발행대기(정상)")와 모순돼 보였다.
            # 30시간(하루+여유 6시간, 다음날도 또 미발행이면 진짜 이상)을
            # 넘을 때만 실제 경보로 분리한다.
            age_min = minutes_since(completed, ref_now)
            NC_STALE_HOURS = 30.0
            if age_min is not None and age_min / 60.0 > NC_STALE_HOURS:
                latest_status = "이상(장기미발행)"
                latest_note = f"발행주기 초과({age_min/60:.0f}h) - 확인 필요"
            else:
                latest_status = "정상"
                latest_note = "하루 1회 발행 - 정상 범위"
            rows.append({
                "region": region, "missing": False, "noLiveRun": False, "issueDate": r[0],
                "backfillDays": backfill_count, "backfillValidDays": valid_count,
                "backfillTarget": NWP_D1D2_BACKFILL_TARGET_DAYS,
                "expected": r[1], "stored": r[2], "nativeMissing": r[3],
                "statusMessage": r[5],
                "ageMinutes": age_min,
                "latestStatus": latest_status, "latestNote": latest_note,
                "lateJoinNote": NWP_D1D2_LATE_JOIN_NOTE.get(region),
                "todayStatus": today_status, "todayReason": today_reason,
            })
        finally:
            conn.close()
    return rows


# ★09-15 신규(Claude, 사용자 요청 - "가을 표본 쌓이는거 헷갈리니까 대시보드로
# 보고싶다")★: D+1 일간모델의 "가을(9~11월) 신규 완전표본" 축적 현황을
# 광주·부안·김제·영광 4지역 항상 같이 보여준다(지역별로 있다/없다 헷갈리는
# 걸 방지). 부안·김제·영광은 walk-forward 후보 스크립트(재구현 없음,
# importlib로 build_daily_dataset() 그대로 재사용)로 "기상피처+타깃 전부
# 채워진 실제 학습/평가 가능 표본"을 센다(09-15 낮에 확립한 엄밀한 정의 -
# 단순 실적치 존재가 아니라 walk-forward가 실제 쓸 수 있는 행만 인정).
# 광주는 구조가 달라(라이브 shadow DB, offline 후보스크립트 없음)
# shadow_predictions에서 tier=일간·status=성공 행을 직접 센다.
FALL_START = "2026-09-01"
FALL_SAMPLE_THRESHOLD = 5

D1_DAILY_MEDIUM_SCRIPTS = {
    "부안": Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
                r"\03_모델학습\현재_종합파이프라인\부안_준비_2026-08-28\medium_term_daily_v1_buan_2026-09-14.py"),
    "김제": Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
                r"\03_모델학습\현재_종합파이프라인\김제_준비_2026-09-01\medium_term_daily_v1_gimje_2026-09-01.py"),
    "영광": Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
                r"\03_모델학습\현재_종합파이프라인\영광_준비_2026-09-03\medium_term_daily_v1_yeonggwang_2026-09-07.py"),
}


def _load_medium_daily_module(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_d1_daily_fall_samples(ref_now: datetime) -> list[dict[str, Any]]:
    order = ["광주", "부안", "김제", "영광"]
    rows: list[dict[str, Any]] = []

    for region, path in D1_DAILY_MEDIUM_SCRIPTS.items():
        try:
            m = _load_medium_daily_module(path)
            df, _meta = m.build_daily_dataset()
            complete = df.dropna(subset=m.CANDIDATE_FEATURES + [m.TARGET])
            fall = complete[complete["target_day"] >= FALL_START]
            count = int(len(fall))
            latest = fall["target_day"].max() if count else None
            rows.append({
                "region": region, "error": None, "newFallSampleDays": count,
                "threshold": FALL_SAMPLE_THRESHOLD, "reached": count >= FALL_SAMPLE_THRESHOLD,
                "latestSampleDay": latest.strftime("%Y-%m-%d") if latest is not None else None,
                "method": "walk-forward 완전표본(기상피처+타깃 전부 유효)",
            })
        except Exception as exc:  # noqa: BLE001 - 대시보드는 한 지역 실패해도 나머지는 계속 보여줘야 함
            rows.append({"region": region, "error": str(exc), "newFallSampleDays": None,
                        "threshold": FALL_SAMPLE_THRESHOLD, "reached": False,
                        "latestSampleDay": None, "method": None})

    conn = open_readonly(SHADOW_DB)
    if conn is None:
        rows.append({"region": "광주", "error": "shadow DB 없음", "newFallSampleDays": None,
                    "threshold": FALL_SAMPLE_THRESHOLD, "reached": False,
                    "latestSampleDay": None, "method": None})
    else:
        try:
            r = conn.execute(
                "SELECT COUNT(DISTINCT target_time), MAX(target_time) FROM shadow_predictions "
                "WHERE tier='일간' AND status='성공' AND target_time >= ?", (FALL_START,)
            ).fetchone()
            count = int(r[0] or 0)
            recent_status = dict(conn.execute(
                "SELECT status, COUNT(*) FROM shadow_predictions WHERE tier='일간' "
                "AND target_time >= ? GROUP BY status", (FALL_START,)
            ).fetchall())
            note = None
            if count == 0 and recent_status:
                note = f"9월 이후 전부 미성공(status별 건수: {recent_status}) - 09-14 WSD 수정 이후에도 재발, 별도 확인 필요"
            rows.append({
                "region": "광주", "error": None, "newFallSampleDays": count,
                "threshold": FALL_SAMPLE_THRESHOLD, "reached": count >= FALL_SAMPLE_THRESHOLD,
                "latestSampleDay": r[1], "method": "라이브 shadow 성공행(tier=일간)", "note": note,
            })
        finally:
            conn.close()

    return sorted(rows, key=lambda x: order.index(x["region"]))


def atomic_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> int:
    ref_now = now_kst()
    regions = [load_region(name, paths, ref_now) for name, paths in REGIONS.items()]
    # ★09-15 신규★: 4지역 verdict를 한 줄로 종합 - 페이지 맨 위에서 "지금 전체적으로
    # 괜찮은가"를 바로 답하기 위함(각 지역 카드까지 안 내려가도 됨).
    overall_bad = [r for r in regions if r["verdict"] != "정상"]
    overall_status = {
        "allOk": len(overall_bad) == 0,
        "badRegions": [{"region": r["region"], "verdict": r["verdict"], "reason": r["verdictReason"]} for r in overall_bad],
    }
    kma_infrastructure = load_kma_infrastructure_status(regions)
    gate_status = load_gate_status(ref_now)
    gate_status["infrastructureHold"] = kma_infrastructure["active"]
    snapshot = {
        "generatedAtKst": ref_now.isoformat(timespec="seconds"),
        "regions": regions,
        "overallStatus": overall_status,
        "kmaInfrastructure": kma_infrastructure,
        "shadowGwangju": load_shadow(ref_now),
        "ultraShortShadow": load_ultrashort_shadow(ref_now),
        "gateStatus": gate_status,
        "plantEquipment": load_plant_equipment(),
        "realtimePower": load_realtime_power(ref_now),
        "nwpD1D2": load_nwp_d1d2(ref_now),
        "d1DailyFallSamples": load_d1_daily_fall_samples(ref_now),
    }
    atomic_write(snapshot, OUTPUT_JSON)
    print(json.dumps({"status": "ok", "output": str(OUTPUT_JSON)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
