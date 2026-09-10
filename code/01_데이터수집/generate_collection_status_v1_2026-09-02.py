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
from datetime import datetime
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
    if block_conn is None:
        result["blockdata"] = feed_entry(ref_now, None, None, 20)
    else:
        try:
            s = table_stat(block_conn, "plant_snapshots", "snapshot_time")
            result["blockdata"] = feed_entry(ref_now, s["count"], s["latestKst"], 20)
        finally:
            block_conn.close()

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
        total_row = conn.execute(
            "SELECT COUNT(*), MAX(run_at_kst) FROM shadow_ultrashort_predictions"
        ).fetchone()
        counts = dict(conn.execute(
            "SELECT status, COUNT(*) FROM shadow_ultrashort_predictions GROUP BY status"
        ).fetchall())
        by_rh = [
            {"region": region, "horizon": h, "status": status, "count": cnt}
            for region, h, status, cnt in conn.execute(
                "SELECT region, horizon_h, status, COUNT(*) FROM shadow_ultrashort_predictions "
                "GROUP BY region, horizon_h, status ORDER BY region, horizon_h"
            )
        ]
        latest = parse_kst(total_row[1])
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
            "waiting": counts.get("대기", 0), "failed": counts.get("실패", 0),
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
GATE_START_KST = datetime(2026, 9, 9, 9, 19, 0, tzinfo=KST)
GATE_JUDGE_KST = datetime(2026, 9, 16, 9, 19, 0, tzinfo=KST)
GATE_SLOT_MINUTES = 5
GATE_SUCCESS_THRESHOLD_PCT = 95.0


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
    expected_slots = max(1, int(window_min // GATE_SLOT_MINUTES) + 1)

    conn = open_readonly(ULTRASHORT_SHADOW_DB)
    regions: list[dict[str, Any]] = []
    if conn is not None:
        try:
            start_str = GATE_START_KST.strftime("%Y-%m-%d %H:%M:%S")
            for region in ["광주", "부안", "김제", "영광"]:
                rows = conn.execute(
                    "SELECT status, COUNT(DISTINCT issue_time_kst) FROM shadow_ultrashort_predictions "
                    "WHERE region = ? AND issue_time_kst >= ? GROUP BY status",
                    (region, start_str),
                ).fetchall()
                by_status = dict(rows)
                logged = sum(by_status.values())
                success = by_status.get("성공", 0)
                failed_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    "WHERE region = ? AND issue_time_kst >= ? AND status = '실패'",
                    (region, start_str),
                ).fetchone()[0]
                bad_pred = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    "WHERE region = ? AND issue_time_kst >= ? AND status = '성공' "
                    "AND (predicted_kw IS NULL OR predicted_kw < 0)",
                    (region, start_str),
                ).fetchone()[0]
                run_rate = round(min(logged, expected_slots) / expected_slots * 100, 1)
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
                    "WHERE region = ? AND issue_time_kst >= ?",
                    (region, start_str),
                ).fetchone()[0]
                success_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    "WHERE region = ? AND issue_time_kst >= ? AND status = '성공'",
                    (region, start_str),
                ).fetchone()[0]
                night_gated_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    "WHERE region = ? AND issue_time_kst >= ? AND status = '대기' AND reason LIKE '%야간%'",
                    (region, start_str),
                ).fetchone()[0]
                infrastructure_wait_rows = conn.execute(
                    "SELECT COUNT(*) FROM shadow_ultrashort_predictions "
                    "WHERE region = ? AND issue_time_kst >= ? AND status = '대기' "
                    "AND reason LIKE '%ASOS(3h 이내 없음)%'",
                    (region, start_str),
                ).fetchone()[0]
                gen_denominator = total_rows - night_gated_rows
                gen_rate = round(success_rows / gen_denominator * 100, 1) if gen_denominator else None
                model_denominator = gen_denominator - infrastructure_wait_rows
                model_gen_rate = (
                    round(success_rows / model_denominator * 100, 1)
                    if model_denominator > 0 else None
                )
                regions.append({
                    "region": region, "expectedSlots": expected_slots, "loggedIssueTimes": logged,
                    "successIssueTimes": success, "runRatePct": run_rate, "genRatePct": gen_rate,
                    "genRateNote": f"야간게이트 대기({night_gated_rows}행) 제외 - 행기준 성공률",
                    "infrastructureWaitRows": infrastructure_wait_rows,
                    "modelEligibleGenRatePct": model_gen_rate,
                    "failedRows": failed_rows, "badPredictions": bad_pred,
                    "pass": (run_rate >= GATE_SUCCESS_THRESHOLD_PCT
                             and (gen_rate is None or gen_rate >= GATE_SUCCESS_THRESHOLD_PCT)
                             and failed_rows == 0 and bad_pred == 0),
                })
        finally:
            conn.close()

    days_elapsed = round(elapsed_min / 1440, 1)
    status = "판정완료" if ref_now >= GATE_JUDGE_KST else "진행중"
    return {
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


def load_nwp_nc_transition(ref_now: datetime) -> list[dict[str, Any]]:
    """09-09 KIM NC 대체경로 시험 현황.

    DB의 기존 received_before_issue 플래그는 과거 결측행 갱신 시 승계될 수
    있어 사용하지 않고, NC 재수집의 last_received_at과 계획 발행시각을
    직접 비교한다. 운영 전환 완료로 오표시하지 않는다.
    """
    rows = []
    marks = ",".join("?" for _ in NC_VARS)
    for region, paths in REGIONS.items():
        conn = open_readonly(paths["weather"])
        if conn is None:
            rows.append({"region": region, "missing": True})
            continue
        try:
            latest = conn.execute(
                f"SELECT issue_date, requested_tm_utc FROM nwp_values "
                f"WHERE variable IN ({marks}) ORDER BY issue_date DESC LIMIT 1", NC_VARS
            ).fetchone()
            if latest is None:
                rows.append({"region": region, "missing": False, "noRun": True})
                continue
            issue_date, run_tm = latest
            stat = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(is_missing),0), "
                f"MAX(first_received_at), MAX(last_received_at) "
                f"FROM nwp_values WHERE issue_date=? AND requested_tm_utc=? "
                f"AND variable IN ({marks})", (issue_date, run_tm, *NC_VARS)
            ).fetchone()
            # 발행시각 판정은 재조회마다 밀릴 수 있는 last_received_at이
            # 아니라, 40개 필수값이 모두 처음 확보된 시각(MAX first_received_at)을
            # 사용한다. last_received_at은 단순 최근 갱신 참고값으로만 보존한다.
            first_received = parse_kst(stat[2])
            last_received = parse_kst(stat[3])
            planned = datetime.fromisoformat(issue_date).replace(hour=10, tzinfo=KST)
            complete = int(stat[0] - stat[1]) == 40 and int(stat[1]) == 0
            before_issue = bool(first_received and first_received <= planned)
            rows.append({
                "region": region, "missing": False, "noRun": False,
                "issueDate": issue_date, "expected": 40, "rows": int(stat[0]),
                "valid": int(stat[0] - stat[1]), "nativeMissing": int(stat[1]),
                "firstReceivedKst": first_received.isoformat(timespec="seconds") if first_received else None,
                "lastReceivedKst": last_received.isoformat(timespec="seconds") if last_received else None,
                "ageMinutes": minutes_since(first_received, ref_now),
                "receivedBeforeIssueByActualTime": before_issue,
                "timingStatus": "발행 전 수신" if before_issue else "발행 후 수신",
                "judgement": ("D+1 사용 가능" if complete and before_issue
                               else "발행시각 재설계 필요" if complete
                               else "결측·보류"),
                "transitionStatus": "검증중(수신시각·5종 집계 보정 후 스케줄러 전환)",
            })
        finally:
            conn.close()
    return rows


def load_nwp_nc_shadow_observation(ref_now: datetime) -> list[dict[str, Any]]:
    """14시 NC Shadow용 수신시각 관측치.

    issue_date 라벨이 아니라 실제 last_received_at을 14:00 KST와 비교한다.
    예측을 만들지 않고, 5일 관측창의 지역별 유효일수·발행 전 수신일수만
    표시한다.
    """
    rows: list[dict[str, Any]] = []
    for region, path in NWP_D1D2_DB.items():
        conn = open_readonly(path)
        if conn is None:
            rows.append({"region": region, "observedDays": 0, "before14Days": 0,
                         "status": "DB 없음"})
            continue
        try:
            data = conn.execute(
                "SELECT issue_date, COUNT(*) AS n, COALESCE(SUM(is_missing),0) AS miss, "
                "MAX(first_received_at) AS last_received FROM nwp_d1d2_values "
                "WHERE issue_date >= '20260909' AND variable IN (?,?,?,?,?) "
                "AND dry_run=0 GROUP BY issue_date ORDER BY issue_date",
                NC_VARS,
            ).fetchall()
        finally:
            conn.close()
        valid_days = 0
        before14 = 0
        latest_received = None
        for issue_date, n, miss, last_received in data:
            if int(n) < 80 or int(miss) != 0:
                continue
            valid_days += 1
            received = parse_kst(last_received)
            if received:
                latest_received = max(latest_received, received) if latest_received else received
                cutoff = datetime.fromisoformat(str(issue_date)).replace(hour=14, tzinfo=KST)
                if received <= cutoff:
                    before14 += 1
        rows.append({
            "region": region, "observedDays": valid_days, "before14Days": before14,
            "latestReceivedKst": latest_received.isoformat(timespec="seconds") if latest_received else None,
            "status": "관측중(예측 미생성)",
        })
    return rows


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
            rows.append({
                "region": region, "missing": False, "noLiveRun": False, "issueDate": r[0],
                "backfillDays": backfill_count, "backfillValidDays": valid_count,
                "backfillTarget": NWP_D1D2_BACKFILL_TARGET_DAYS,
                "expected": r[1], "stored": r[2], "nativeMissing": r[3],
                "statusMessage": r[5],
                "ageMinutes": minutes_since(completed, ref_now),
                "lateJoinNote": NWP_D1D2_LATE_JOIN_NOTE.get(region),
            })
        finally:
            conn.close()
    return rows


def atomic_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> int:
    ref_now = now_kst()
    regions = [load_region(name, paths, ref_now) for name, paths in REGIONS.items()]
    kma_infrastructure = load_kma_infrastructure_status(regions)
    gate_status = load_gate_status(ref_now)
    gate_status["infrastructureHold"] = kma_infrastructure["active"]
    snapshot = {
        "generatedAtKst": ref_now.isoformat(timespec="seconds"),
        "regions": regions,
        "kmaInfrastructure": kma_infrastructure,
        "shadowGwangju": load_shadow(ref_now),
        "ultraShortShadow": load_ultrashort_shadow(ref_now),
        "gateStatus": gate_status,
        "plantEquipment": load_plant_equipment(),
        "realtimePower": load_realtime_power(ref_now),
        "nwpNcTransition": load_nwp_nc_transition(ref_now),
        "nwpNcShadowObservation": load_nwp_nc_shadow_observation(ref_now),
        "nwpD1D2": load_nwp_d1d2(ref_now),
    }
    atomic_write(snapshot, OUTPUT_JSON)
    print(json.dumps({"status": "ok", "output": str(OUTPUT_JSON)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
