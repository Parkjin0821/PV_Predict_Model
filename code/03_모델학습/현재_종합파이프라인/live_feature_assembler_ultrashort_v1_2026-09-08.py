# -*- coding: utf-8 -*-
"""부안·김제·영광 초단기(+1~4h) live feature assembler + shadow predict.

## 배경
09-08 확인 결과 초단기 티어(+1~4h)는 D+1·+24h·중장기와 달리 **NWP
예보 특성을 하나도 쓰지 않는다**(발전량 lag + ASOS 실측 + 태양기하·
시간 순환특성뿐). 3지역 전부 라이브 발전소 실측(Blockdata)·ASOS 실측
수집이 이미 정상 작동 중이므로, 전국단위 NWP 결측 게이트와 무관하게
지금 바로 조립·shadow 예측을 시작할 수 있다(사용자 확인 후 착수).

## 재사용(재구현 안 함)
- 특성 정의는 09-08에 저장한 배포용 번들(`공식_초단기_issue_safe_v1_
  2026-09-08/+{h}h/model.joblib`)의 manifest.json `features` 목록을
  그대로 따른다 - 이 파일이 특성을 다시 정의하지 않는다.
- 태양고도 계산은 각 지역 전처리 스크립트(`02_전처리/{지역}/build_*_
  time_aggregates_v1_*.py`)의 `solar_elevation_deg()`를 좌표까지
  그대로 import해서 쓴다(광주 NOAA 근사식, 좌표만 지역별).

## 원칙(가짜예측 금지)
필수 특성이 하나라도 결측이면 절대 예측하지 않고 상태 `대기`로만
기록한다(임의대체·보간 없음). 발전량 완전가용 판정은 지역별 인버터
대수(부안8·김제10·영광13)가 전부 유효할 때만 채택한다(부분합계 금지 -
09-08 이전 정식 파이프라인과 동일 원칙).

## ★알려진 단순화(정직하게 기록)★
학습 시 쓰인 `quality_status_after_night`(야간0 보정을 거친 후처리
카테고리)는 라이브 원천 DB엔 없다. 라이브 완전가용 판정은
`valid_ac_power_count==expected_inverter_count`(전 인버터 유효값 확보)로
한다 - `quality_status`는 'ok'/'warning' 둘 다 허용하는데, 09-08 실측
확인 결과 부안·김제 표본 전부 'warning'=`measurement_spread>300sec`
(폴링 타이밍 경고일 뿐)이었고 이때도 valid_ac_power_count는 항상
expected와 같았다(즉 전력값 자체는 유효). 'error' 등 미확인 상태만
화이트리스트에서 배제한다. 임의값을 채우는 규칙은 아니다.

## 저장
`shadow_predictions_ultrashort.sqlite3`(이 폴더)에 매 실행 시도를
전부 기록한다(성공·대기·실패 전부, 가짜 성공 방지를 위해 실패도 남김).
실측 도착 후 성능평가는 이 로그와 각 지역 plant_snapshots를
issue_time+target_time으로 조인해서 별도 스크립트로 진행한다(다음 과제).
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# ★09-16 추가★: pythonw 실행이라 stdout이 사라지므로 예외를 파일로 남긴다.
LOG_DIR = ROOT / "logs" / "ultrashort_assembler"
PROJECT_ROOT = ROOT.parent.parent  # 광주_PV_예측모델_통합_v1_2026-08-19
KST = ZoneInfo("Asia/Seoul")
SHADOW_DB = ROOT / "shadow_predictions_ultrashort.sqlite3"
HORIZONS = [1, 2, 3, 4]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BUAN_SOLAR = load_module(
    "buan_solar", ROOT / "부안_준비_2026-08-28" / "ultra_short_term_v1_buan_2026-09-08.py"
)  # 09-08 저녁: power_only→full 전환(아래 REGIONS 주석 참고)에 따라 로딩으로 변경
GIMJE_SOLAR = load_module(
    "gimje_solar", PROJECT_ROOT / "02_전처리" / "김제" / "build_gimje_time_aggregates_v1_2026-08-31.py"
)
YEONGGWANG_SOLAR = load_module(
    "yeonggwang_solar", PROJECT_ROOT / "02_전처리" / "영광" / "build_yeonggwang_time_aggregates_v1_2026-09-01.py"
)
# 광주는 02_전처리에 별도 태양고도 스크립트가 없으므로(기존 08-26
# 파이프라인은 clean_nwp_direct_diffuse_bsrn_qc_v1.solar_position을
# 다른 시그니처로 씀) 09-08 신규 학습 스크립트 자신이 노출하는
# solar_elevation_deg()를 그대로 재사용한다(같은 NOAA식, 같은 좌표).
GWANGJU_SOLAR = load_module(
    "gwangju_solar", ROOT / "ultra_short_term_v1_gwangju_2026-09-08.py"
)

REGIONS = {
    "부안": dict(
        # 09-08 저녁: "부안만 오차 크다" 감사로 power_only→full 전환.
        # 부안용 라이브 ASOS 수집(station 243)은 이미 정상 동작 중이었음을
        # 실측 확인(kma_live_inputs_v1_2026-08-28, 20:21 최신행 존재) -
        # 조립기 연결만 빠져 있었던 것. bundle_dir을 날씨피처 신규 번들
        # (공식_초단기_v2_날씨피처_2026-09-08)로 교체.
        mode="full",
        plant_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3",
        plant_id=16783, expected_inverters=8, capacity=998.715,
        bundle_dir=ROOT / "부안_준비_2026-08-28" / "outputs" / "공식_초단기_v2_날씨피처_2026-09-08",
        solar=BUAN_SOLAR,
        asos_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3",
        asos_station=243,
    ),
    "김제": dict(
        mode="full",
        plant_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\blockdata_live_history_gimje_v1_2026-09-01\blockdata_history.sqlite3",
        plant_id=7018, expected_inverters=10, capacity=999.005,
        bundle_dir=ROOT / "김제_준비_2026-09-01" / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08",
        solar=GIMJE_SOLAR,
        asos_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",
        asos_station=243,
    ),
    "영광": dict(
        mode="full",
        plant_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\blockdata_live_history_yeonggwang_v1_2026-09-08\blockdata_history.sqlite3",
        plant_id=7912, expected_inverters=13, capacity=639.94,
        bundle_dir=ROOT / "영광_준비_2026-09-03" / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08",
        solar=YEONGGWANG_SOLAR,
        asos_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",
        asos_station=252,
    ),
    "광주": dict(
        mode="full",
        plant_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25\blockdata_history.sqlite3",
        # 용량 역할 분리: 공식 설비용량 240은 nMAE 분모, 인버터 등록합
        # 241.58은 물리적 예측 clip 상한으로 사용한다.
        plant_id=6715, expected_inverters=5, capacity=240.0,
        clip_capacity=241.58,
        # 09-16 실측: 광주 벤더가 새 5분 스냅샷을 한 번 건너뛸 때
        # lag 목표와 직전/직후 완전 스냅샷의 차이가 4분대가 된다.
        # 명목 수집주기 한 칸(5분)까지만 허용하고 그 초과 공백은 계속
        # 결측으로 차단한다. 다른 지역은 기존 기본값 4분을 유지한다.
        power_lag_tolerance_minutes=5.0,
        bundle_dir=ROOT / "광주_준비_2026-09-08" / "outputs" / "공식_초단기_issue_safe_v1_2026-09-08",
        solar=GWANGJU_SOLAR,
        asos_db=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3",
        asos_station=156,
    ),
}

ASOS_MAP = {
    "temperature_c": "obs_temp_c", "humidity_pct": "obs_rh_pct",
    "cloud_pct": "obs_cloud_pct", "wind_speed_m_s": "obs_wind_ms",
    "solar_w_m2": "obs_ghi_wm2",
}


def init_shadow_db() -> None:
    """09-08 실측 감사(사용자 지적)로 발견: 5분 스케줄러와 수동 실행이
    겹치면 동일한 (region, horizon_h, issue_time_kst) 예측이 그대로
    중복 저장됐다(48행 중 32성공/16대기였는데 16개 그룹이 중복). 원인은
    유니크 제약이 없었기 때문 - UNIQUE 인덱스 + INSERT OR IGNORE로
    막는다(같은 issue_time 재확인 시도는 조용히 무시, 최초 1건만 유지)."""
    con = sqlite3.connect(SHADOW_DB, timeout=30)
    con.execute("""
        CREATE TABLE IF NOT EXISTS shadow_ultrashort_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            horizon_h INTEGER NOT NULL,
            issue_time_kst TEXT,
            target_time_kst TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            predicted_kw REAL,
            missing_features TEXT,
            model_sha256 TEXT,
            run_at_kst TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_shadow_region_horizon_issue
        ON shadow_ultrashort_predictions(region, horizon_h, issue_time_kst)
    """)
    con.commit()
    con.close()


def log_attempt(row: dict) -> bool:
    """반환값 True=신규 기록, False=이미 있던 (region,horizon_h,issue_time)
    이라 조용히 건너뜀(issue_time_kst가 NULL인 행 - 발전량 시리즈 자체가
    없을 때 - 은 SQLite UNIQUE 규칙상 서로 다른 값으로 취급돼 매번 새로
    쌓인다는 점은 알려진 제약; 실제 유의미한 issue_time이 잡힌 뒤부터는
    중복이 막힌다)."""
    con = sqlite3.connect(SHADOW_DB, timeout=30)
    cur = con.execute(
        """INSERT OR IGNORE INTO shadow_ultrashort_predictions
           (region, horizon_h, issue_time_kst, target_time_kst, status, reason,
            predicted_kw, missing_features, model_sha256, run_at_kst)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (row["region"], row["horizon_h"], row.get("issue_time_kst"), row.get("target_time_kst"),
         row["status"], row.get("reason"), row.get("predicted_kw"),
         json.dumps(row.get("missing_features", []), ensure_ascii=False),
         row.get("model_sha256"), datetime.now(KST).isoformat()),
    )
    inserted = cur.rowcount > 0
    con.commit()
    con.close()
    return inserted


def load_power_series(plant_db: str, plant_id: int, expected_inverters: int,
                      lookback_hours: int = 30) -> pd.Series:
    """5분 그리드 발전소 총출력 시리즈. 전 인버터 유효한 스냅샷만 채택,
    나머지 그리드 슬롯은 그대로 NaN(보간·부분합계 없음)."""
    con = sqlite3.connect(plant_db, timeout=30)
    since = (datetime.now(KST) - timedelta(hours=lookback_hours)).astimezone(KST).replace(tzinfo=None).isoformat()
    df = pd.read_sql_query(
        """SELECT snapshot_time, plant_ac_power_kw, quality_status,
                  expected_inverter_count, valid_ac_power_count
           FROM plant_snapshots
           WHERE plant_id = ? AND snapshot_time >= ?
           ORDER BY snapshot_time""",
        con, params=(plant_id, since),
    )
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"]).dt.tz_localize(None)
    # 실측 확인(09-08): quality_status='warning'은 전부
    # measurement_spread>300sec(8/10/13대 폴링이 5분 넘게 걸렸다는 타이밍
    # 경고)일 뿐 - 부안·김제 표본 전부 warning에서도 valid_ac_power_count
    # ==expected_inverter_count(전 인버터 유효값 확보)였다. 'error' 등
    # 다른 미확인 상태만 배제하도록 화이트리스트로 좁혀서 받는다(완전가용
    # 판정 자체는 여전히 valid_ac_power_count==expected로만 함).
    ok = (
        df["quality_status"].isin(["ok", "warning"])
        & (df["valid_ac_power_count"] == expected_inverters)
        & (df["expected_inverter_count"] == expected_inverters)
    )
    df = df[ok].copy()
    if df.empty:
        return pd.Series(dtype=float)
    # 5분 격자로 floor 후 reindex하지 않는다 - 09-08 실측 확인 결과 실제
    # 폴링 시각이 격자 경계 부근에서 흔들려서(예: 13:59:56은 14:00 격자가
    # 아니라 13:55 격자로 floor됨) 고정격자 reindex는 실제로 값이 있는데도
    # NaN으로 잘못 비는 인위적 공백을 만든다. 대신 원시 시각 그대로 두고
    # lookup_near()로 허용오차 내 최근접 탐색을 한다.
    return df.sort_values("snapshot_time").drop_duplicates("snapshot_time", keep="last").set_index("snapshot_time")["plant_ac_power_kw"]


def lookup_near(series: pd.Series, target_time: pd.Timestamp, tol_minutes: float = 4.0) -> float:
    """target_time에 가장 가까운 값을 허용오차(tol_minutes) 내에서만 반환.
    범위 밖이면 NaN(임의보간 없음)."""
    if series.empty:
        return np.nan
    pos = series.index.searchsorted(target_time)
    candidates = []
    if pos < len(series):
        candidates.append(series.index[pos])
    if pos > 0:
        candidates.append(series.index[pos - 1])
    if not candidates:
        return np.nan
    best = min(candidates, key=lambda t: abs((t - target_time).total_seconds()))
    if abs((best - target_time).total_seconds()) <= tol_minutes * 60:
        return series.loc[best]
    return np.nan


def load_latest_asos(asos_db: str, station: int, at_time: pd.Timestamp,
                     tolerance_hours: float = 3.0) -> dict:
    con = sqlite3.connect(asos_db, timeout=30)
    row = con.execute(
        """SELECT observation_time, temperature_c, humidity_pct, cloud_pct,
                  wind_speed_m_s, solar_w_m2
           FROM asos_hourly
           WHERE station = ? AND observation_time <= ?
           ORDER BY observation_time DESC LIMIT 1""",
        (station, at_time.isoformat()),
    ).fetchone()
    con.close()
    if row is None:
        return {}
    obs_time = pd.Timestamp(row[0]).tz_localize(None) if pd.Timestamp(row[0]).tzinfo is None else pd.Timestamp(row[0]).tz_convert(KST).tz_localize(None)
    if (at_time.tz_localize(None) if at_time.tzinfo else at_time) - obs_time > timedelta(hours=tolerance_hours):
        return {}
    cols = ["observation_time", "temperature_c", "humidity_pct", "cloud_pct", "wind_speed_m_s", "solar_w_m2"]
    raw = dict(zip(cols, row))
    return {ASOS_MAP[k]: raw[k] for k in ASOS_MAP if k in raw}


def solar_elev(solar_mod, ts: pd.Timestamp) -> float:
    arr = solar_mod.solar_elevation_deg(pd.DatetimeIndex([ts]))
    return float(arr[0])


def load_bundle(bundle_dir: Path, horizon: int) -> dict | None:
    p = bundle_dir / f"+{horizon}h" / "model.joblib"
    if not p.exists():
        return None
    payload = joblib.load(p)
    manifest = json.loads((bundle_dir / f"+{horizon}h" / "manifest.json").read_text(encoding="utf-8"))
    payload["_sha256"] = manifest.get("model_sha256")
    return payload


def run_buan(cfg: dict) -> None:
    series = load_power_series(cfg["plant_db"], cfg["plant_id"], cfg["expected_inverters"])
    for h in HORIZONS:
        bundle = load_bundle(cfg["bundle_dir"], h)
        if bundle is None:
            log_attempt({"region": "부안", "horizon_h": h, "status": "실패", "reason": "번들 없음"})
            continue
        if series.empty:
            log_attempt({"region": "부안", "horizon_h": h, "status": "대기",
                        "reason": "완전가용 발전량 시리즈 없음", "model_sha256": bundle["_sha256"]})
            continue
        issue_time = series.index[-1]
        target_time = issue_time + pd.Timedelta(hours=h)
        if solar_elev(cfg["solar"], target_time) <= 0:
            log_attempt({"region": "부안", "horizon_h": h, "issue_time_kst": str(issue_time),
                        "target_time_kst": str(target_time), "status": "대기",
                        "reason": "야간 target 태양고도<=0; 학습 도메인 밖 예측 차단",
                        "model_sha256": bundle["_sha256"]})
            continue
        window = series.loc[issue_time - pd.Timedelta(minutes=62):issue_time]
        row = {
            "value": lookup_near(series, issue_time, tol_minutes=1),
            "lag_15m": lookup_near(series, issue_time - pd.Timedelta(minutes=15)),
            "lag_30m": lookup_near(series, issue_time - pd.Timedelta(minutes=30)),
            "lag_1h": lookup_near(series, issue_time - pd.Timedelta(hours=1)),
            # 09-08 지적(사용자) 반영: 학습 정의(`s.rolling(12, min_periods=
            # 12).mean()`, 5분×12=1시간)와 정확히 맞춘다 - 완화(8개)는
            # replay로 타당성 검증 전까지 쓰지 않는다. 라이브 폴링이 격자에
            # 안 맞아 62분 창에 12개가 안 잡히면 정직하게 결측(대기) 처리.
            "roll_1h_mean": window.mean() if window.count() >= 12 else np.nan,
            "lag_1day_same_time": lookup_near(series, issue_time - pd.Timedelta(days=1)),
        }
        missing = [f for f in bundle["features"] if pd.isna(row.get(f))]
        if missing:
            log_attempt({"region": "부안", "horizon_h": h, "issue_time_kst": str(issue_time),
                        "target_time_kst": str(target_time), "status": "대기",
                        "reason": "필수 특성 결측", "missing_features": missing,
                        "model_sha256": bundle["_sha256"]})
            continue
        X = pd.DataFrame([row])[bundle["features"]]
        pred = float(np.clip(
            bundle["model"].predict(X)[0], 0,
            cfg.get("clip_capacity", cfg["capacity"]),
        ))
        log_attempt({"region": "부안", "horizon_h": h, "issue_time_kst": str(issue_time),
                    "target_time_kst": str(target_time), "status": "성공",
                    "predicted_kw": round(pred, 4), "model_sha256": bundle["_sha256"]})


def run_full(region: str, cfg: dict) -> None:
    series = load_power_series(cfg["plant_db"], cfg["plant_id"], cfg["expected_inverters"])
    if series.empty:
        for h in HORIZONS:
            log_attempt({"region": region, "horizon_h": h, "status": "대기",
                        "reason": "완전가용 발전량 시리즈 없음"})
        return
    issue_time = series.index[-1]
    asos = load_latest_asos(cfg["asos_db"], cfg["asos_station"], issue_time)
    solar_now = solar_elev(cfg["solar"], issue_time)
    hour = issue_time.hour + issue_time.minute / 60.0
    doy = issue_time.dayofyear
    power_lag_tolerance_minutes = float(cfg.get("power_lag_tolerance_minutes", 4.0))
    base_row = {
        "power_lag_0min": lookup_near(series, issue_time, tol_minutes=1),
        "power_lag_15min": lookup_near(
            series, issue_time - pd.Timedelta(minutes=15),
            tol_minutes=power_lag_tolerance_minutes,
        ),
        "power_lag_30min": lookup_near(
            series, issue_time - pd.Timedelta(minutes=30),
            tol_minutes=power_lag_tolerance_minutes,
        ),
        "power_lag_60min": lookup_near(
            series, issue_time - pd.Timedelta(hours=1),
            tol_minutes=power_lag_tolerance_minutes,
        ),
        "solar_elevation_now": solar_now,
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        **asos,
    }
    for h in HORIZONS:
        bundle = load_bundle(cfg["bundle_dir"], h)
        if bundle is None:
            log_attempt({"region": region, "horizon_h": h, "status": "실패", "reason": "번들 없음"})
            continue
        target_time = issue_time + pd.Timedelta(hours=h)
        row = dict(base_row)
        row["solar_elevation_target"] = solar_elev(cfg["solar"], target_time)
        if row["solar_elevation_target"] <= 0:
            log_attempt({"region": region, "horizon_h": h, "issue_time_kst": str(issue_time),
                        "target_time_kst": str(target_time), "status": "대기",
                        "reason": "야간 target 태양고도<=0; 학습 도메인 밖 예측 차단",
                        "model_sha256": bundle["_sha256"]})
            continue
        missing = [f for f in bundle["features"] if pd.isna(row.get(f, np.nan))]
        if missing:
            # 09-09 정정(사용자 지적 "왔다갔다 하네" 조사 중 발견): 예전엔
            # 원인과 무관하게 "ASOS 3h 이내 없음"으로 고정 표기해 김제의
            # 실제 원인(발전량 lag 간헐결측, Blockdata 소스 자체 공백)을
            # 가렸다. missing_features를 실제로 분류해서 표기한다.
            missing_kinds = set()
            for f in missing:
                if f.startswith("power_lag"):
                    missing_kinds.add("발전량lag(Blockdata)")
                elif f.startswith("obs_") or "asos" in f.lower():
                    missing_kinds.add("ASOS(3h 이내 없음)")
                else:
                    missing_kinds.add(f)
            log_attempt({"region": region, "horizon_h": h, "issue_time_kst": str(issue_time),
                        "target_time_kst": str(target_time), "status": "대기",
                        "reason": f"필수 특성 결측({', '.join(sorted(missing_kinds))})",
                        "missing_features": missing,
                        "model_sha256": bundle["_sha256"]})
            continue
        X = pd.DataFrame([row])[bundle["features"]]
        pred = float(np.clip(
            bundle["model"].predict(X)[0], 0,
            cfg.get("clip_capacity", cfg["capacity"]),
        ))
        log_attempt({"region": region, "horizon_h": h, "issue_time_kst": str(issue_time),
                    "target_time_kst": str(target_time), "status": "성공",
                    "predicted_kw": round(pred, 4), "model_sha256": bundle["_sha256"]})


def main() -> None:
    init_shadow_db()
    # 09-08 저녁: run_buan()(power_only, lag만)에서 run_full()(날씨피처
    # 포함)로 전환 - run_buan()은 롤백용으로 함수 자체는 남겨둔다.
    # ★09-10 정정★: 지역별 호출에 try/except가 없어 한 지역(영광)에서
    # 예외가 터지면 순서상 그 뒤(광주)까지 통째로 실행 자체가 안 되고
    # 있었다(밤사이 부안·김제는 계속 기록되는데 영광·광주만 9시간
    # 동반결측 - 실측으로 원인 확정). 지역별로 격리해 한 곳이 죽어도
    # 나머지는 계속 돌도록 수정.
    # ★★09-16 추가★★: 이 작업은 **pythonw.exe**로 실행돼 stdout이 통째로
    # 버려진다. 그래서 아래 except의 print가 아무 데도 안 남고, 영광이
    # 간헐적으로 실패해도(새 게이트 창에서 영광 97사이클 vs 부안 123사이클,
    # 26회 누락) **원인을 볼 방법이 전혀 없었다**. 파일로 직접 남긴다.
    err_log = LOG_DIR / f"assembler_{datetime.now(KST).strftime('%Y%m%d')}.log"
    for region in ["부안", "김제", "영광", "광주"]:
        try:
            run_full(region, REGIONS[region])
        except Exception as exc:  # noqa: BLE001 - 한 지역 예외가 나머지를 막으면 안 됨
            msg = (f"[{datetime.now(KST).isoformat(timespec='seconds')}] "
                   f"[{region} 실행오류] {type(exc).__name__}: {exc}\n"
                   + traceback.format_exc())
            print(msg)
            try:
                err_log.parent.mkdir(parents=True, exist_ok=True)
                with err_log.open("a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass
    con = sqlite3.connect(SHADOW_DB, timeout=30)
    latest = pd.read_sql_query(
        "SELECT region, horizon_h, status, issue_time_kst, target_time_kst, predicted_kw, reason "
        "FROM shadow_ultrashort_predictions ORDER BY id DESC LIMIT 12", con)
    con.close()
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
