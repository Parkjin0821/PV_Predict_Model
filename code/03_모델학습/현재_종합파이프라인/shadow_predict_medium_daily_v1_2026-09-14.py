# -*- coding: utf-8 -*-
"""부안·김제·영광 중장기(D+1 일간총량) 라이브 예측 드라이버 - 신규.

## 배경(09-14 발견)
`shadow_predict_일간_v1_2026-08-27.py`(`UCUBE_ShadowPredict_Daily`)는
**광주 전용**(e2e_retrain_v5 파이프라인 사용)이다. 김제·영광은 09-08에
`build_medium()`으로 배포용 번들(`공식_중장기_issue_safe_v1_2026-09-08/
model.joblib`)까지는 만들어놨지만, 그 번들로 실제 라이브 예측을 뽑는
드라이버가 지금까지 하나도 없었다(코드 전체 검색으로 확인 - 번들을
읽는 스크립트는 build 스크립트 자신뿐). 09-14에 부안도 같은 방식으로
번들이 생겼으므로, 이 3지역용 공용 드라이버를 신규 작성한다.

## 재구현 없음 - 기존 조각 재사용
- 특성 정의(WEATHER8 mean/max, daily_energy_lag1_kwh, doy_sin/cos)는
  `medium_term_daily_v1_{region}_*.py`가 이미 정의한 그대로 가져다 씀
  (여기서 특성 목록을 다시 안 만듦 - bundle["features"] 순서 그대로).
- 라이브 발전량 조회는 `live_feature_assembler_단기_v2_pooled_2026-09-10.py`
  의 `power()`와 동일한 품질기준(valid_ac_power_count==expected_
  inverter_count) 재사용.

## 가짜성공 방지
bundle 특성 중 하나라도 NaN이면 "대기" 반환, 예측 계산 안 함(다른
shadow_predict_*와 동일 원칙).

## DB
전용 신규 DB(`shadow_predictions_medium_daily.sqlite3`, region 컬럼
포함) - 광주 전용 기존 DB(`shadow_predictions.sqlite3`, region 컬럼
없음)는 안 건드림.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "shadow_predictions_medium_daily.sqlite3"

NWP_VARS = ["DSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]
GRID_VARS = ["REH", "POP", "SKY"]

REGIONS = {
    "부안": dict(
        plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3",
        kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3",
        bundle_dir=ROOT / "부안_준비_2026-08-28" / "outputs" / "공식_중장기_issue_safe_v1_2026-09-08",
        capacity=998.715, inv=8,
    ),
    "김제": dict(
        plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\blockdata_live_history_gimje_v1_2026-09-01\blockdata_history.sqlite3",
        kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",
        bundle_dir=ROOT / "김제_준비_2026-09-01" / "outputs" / "공식_중장기_issue_safe_v1_2026-09-08",
        capacity=999.005, inv=10,
    ),
    "영광": dict(
        plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\blockdata_live_history_yeonggwang_v1_2026-09-08\blockdata_history.sqlite3",
        kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",
        bundle_dir=ROOT / "영광_준비_2026-09-03" / "outputs" / "공식_중장기_issue_safe_v1_2026-09-08",
        capacity=639.94, inv=13,
    ),
}


def ensure_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_medium_daily_predictions(
            id INTEGER PRIMARY KEY,
            region TEXT NOT NULL,
            issue_time_kst TEXT NOT NULL,
            target_day TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            predicted_kwh REAL,
            missing_features TEXT,
            model_sha256 TEXT,
            run_at_kst TEXT NOT NULL,
            UNIQUE(region, issue_time_kst, target_day)
        )
    """)
    conn.commit()
    return conn


def load_bundle(bundle_dir: Path) -> dict | None:
    p = bundle_dir / "model.joblib"
    if not p.is_file():
        return None
    payload = joblib.load(p)
    manifest_path = bundle_dir / "manifest.json"
    sha = None
    if manifest_path.is_file():
        import json
        sha = json.loads(manifest_path.read_text(encoding="utf-8")).get("model_sha256")
    payload["_sha256"] = sha
    return payload


def yesterday_actual_kwh(plant_db: str, inv: int, target_day: pd.Timestamp) -> float:
    """target_day 기준 전일(=target_day-1)의 실제 일간 발전량(kWh) -
    live_feature_assembler_단기_v2_pooled_2026-09-10.py의 power() 품질기준과
    동일(5대/N대 완전가용 스냅샷만) 재사용, 5분 원시값을 직접 kWh로 합산.
    ★09-14 수정★: 처음엔 1시간 리샘플 후 "24시간 중 18시간 이상 유효"를
    요구했는데, 태양광은 야간엔 스냅샷 자체가 없는 게 정상이라(발전 안 함)
    이 기준을 항상 못 넘겨 전부 "대기"로 잘못 빠졌다(부안 실측 268/268건
    유효인데도 결측 처리됨). 야간 결측을 요구하지 않고, 5분 원시 스냅샷
    "개수"만으로 최소 커버리지를 확인하도록 고침(야간엔 원래 0건이어도
    정상)."""
    lag1_day = target_day - pd.Timedelta(days=1)
    start = lag1_day.isoformat()
    end = (lag1_day + pd.Timedelta(days=1)).isoformat()
    conn = sqlite3.connect(plant_db)
    try:
        d = pd.read_sql_query(
            "SELECT snapshot_time, plant_ac_power_kw, valid_ac_power_count, expected_inverter_count "
            "FROM plant_snapshots WHERE snapshot_time>=? AND snapshot_time<?",
            conn, params=(start, end))
    finally:
        conn.close()
    if d.empty:
        return float("nan")
    d["t"] = pd.to_datetime(d.snapshot_time).dt.tz_localize(None)
    d = d[(d.valid_ac_power_count == inv) & (d.expected_inverter_count == inv)]
    # 낮시간(약 12~13시간, 5분 간격이면 최대 ~150~160건) 기준 최소 절반은
    # 있어야 신뢰 - 너무 적으면(수집 중단 등) 결측으로 남긴다.
    if len(d) < 80:
        return float("nan")
    five_min_kwh = d["plant_ac_power_kw"] * (5.0 / 60.0)
    return float(five_min_kwh.sum())


def nwp_grid_features(kma_db: str, target_day: pd.Timestamp) -> dict:
    day_start = target_day.isoformat()
    day_end = (target_day + pd.Timedelta(days=1)).isoformat()
    conn = sqlite3.connect(kma_db)
    try:
        nwp = pd.read_sql_query(
            "SELECT variable, target_time_kst, value FROM nwp_values "
            "WHERE variable IN ({}) AND target_time_kst>=? AND target_time_kst<? "
            "AND is_missing=0 ORDER BY first_received_at".format(
                ",".join("?" * len(NWP_VARS))),
            conn, params=(*NWP_VARS, day_start, day_end))
        grid = pd.read_sql_query(
            "SELECT variable, target_time_kst, value FROM grid_forecast "
            "WHERE variable IN ({}) AND target_time_kst>=? AND target_time_kst<? "
            "AND is_missing=0 ORDER BY first_received_at".format(
                ",".join("?" * len(GRID_VARS))),
            conn, params=(*GRID_VARS, day_start, day_end))
    finally:
        conn.close()
    feats: dict = {}
    for v in NWP_VARS:
        vals = nwp.loc[nwp.variable == v, "value"]
        feats[f"forecast_{v}_mean"] = float(vals.mean()) if len(vals) else np.nan
        feats[f"forecast_{v}_max"] = float(vals.max()) if len(vals) else np.nan
    for v in GRID_VARS:
        vals = grid.loc[grid.variable == v, "value"]
        feats[f"forecast_{v}_mean"] = float(vals.mean()) if len(vals) else np.nan
        feats[f"forecast_{v}_max"] = float(vals.max()) if len(vals) else np.nan
    return feats


def predict_one(region: str, cfg: dict, now: pd.Timestamp) -> dict:
    bundle = load_bundle(cfg["bundle_dir"])
    if bundle is None:
        return {"region": region, "status": "대기", "reason": "번들 없음"}

    issue = now.floor("h")
    target_day = (now.tz_localize(None) + pd.Timedelta(days=1)).normalize()

    row: dict = {}
    row.update(nwp_grid_features(cfg["kma"], target_day))
    row["daily_energy_lag1_kwh"] = yesterday_actual_kwh(cfg["plant"], cfg["inv"], target_day)
    doy = target_day.dayofyear
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    missing = [f for f in bundle["features"] if f not in row or pd.isna(row[f])]
    result = {
        "region": region, "issue_time_kst": issue.isoformat(),
        "target_day": target_day.date().isoformat(), "model_sha256": bundle.get("_sha256"),
    }
    if missing:
        result.update(status="대기", reason="필수특성 결측", missing=missing)
    else:
        X = pd.DataFrame([row])[bundle["features"]]
        pred = float(np.clip(bundle["model"].predict(X)[0], 0, cfg["capacity"] * 24))
        result.update(status="성공", predicted_kwh=round(pred, 2))
    return result


def save(conn: sqlite3.Connection, r: dict, run_at: str) -> None:
    import json
    conn.execute(
        "INSERT OR REPLACE INTO shadow_medium_daily_predictions"
        "(region,issue_time_kst,target_day,status,reason,predicted_kwh,missing_features,model_sha256,run_at_kst) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (r["region"], r.get("issue_time_kst"), r.get("target_day"), r["status"], r.get("reason"),
         r.get("predicted_kwh"), json.dumps(r.get("missing", []), ensure_ascii=False),
         r.get("model_sha256"), run_at),
    )


def main() -> int:
    now = pd.Timestamp.now(tz=KST)
    conn = ensure_db()
    results = []
    for region, cfg in REGIONS.items():
        r = predict_one(region, cfg, now)
        results.append(r)
        save(conn, r, now.isoformat())
    conn.commit()
    conn.close()
    import json
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
