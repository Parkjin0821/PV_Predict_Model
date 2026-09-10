# -*- coding: utf-8 -*-
"""일간 D+1 라이브 운영 준비도 게이트.

현재 일간 번들은 30일 이동통계와 Blockdata 라이브 스키마에 없는
``mean_communication_ok``를 요구한다. 이를 임의 대체하지 않고, 실제 추론
전 준비조건을 감사해 shadow DB에 ``blocked``/``warming_up``으로 기록한다.
API 호출은 0건이다.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "outputs" / "운영모델_v2_2026-08-25" / "운영모델_일간_D+1.joblib"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


asm = _load("daily_ready_common", "live_feature_assembler_단기_v1_2026-08-26.py")
shadow = _load("daily_ready_shadow", "shadow_predict_단기_v1_2026-08-26.py")


def audit() -> dict:
    bundle = joblib.load(BUNDLE)
    now = pd.Timestamp.now(tz=asm.KST)
    with sqlite3.connect(asm.BLOCK_DB) as conn:
        t0, t1, n = conn.execute(
            "SELECT min(measurement_time), max(measurement_time), count(*) "
            "FROM inverter_measurements").fetchone()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(inverter_measurements)")}
    start = pd.to_datetime(t0, utc=True) if t0 else pd.NaT
    stop = pd.to_datetime(t1, utc=True) if t1 else pd.NaT
    history_days = float((stop - start).total_seconds() / 86400) if pd.notna(start) and pd.notna(stop) else 0.0

    target_day = (now + pd.Timedelta(days=1)).date().isoformat()
    with sqlite3.connect(asm.KMA_DB) as conn:
        nwp = conn.execute(
            "SELECT count(DISTINCT target_time_kst), "
            "sum(CASE WHEN is_missing=0 THEN 1 ELSE 0 END) FROM nwp_values "
            "WHERE substr(target_time_kst,1,10)=? AND first_received_at<=?",
            (target_day, now.isoformat()),
        ).fetchone()

    blockers = []
    if history_days < 32.0:
        blockers.append(f"Blockdata 이력 {history_days:.2f}일/최소 32일")
    if any(f.endswith("mean_communication_ok") for f in bundle["features"]) and "communication_ok" not in cols:
        blockers.append("일간 번들이 mean_communication_ok를 요구하지만 라이브 API 스키마에 원천필드 없음")
    if not nwp or int(nwp[0] or 0) < 8:
        blockers.append(f"목표일 D+1 NWP 시각 {int((nwp or (0,0))[0] or 0)}/8")
    hard = any("원천필드 없음" in x for x in blockers)
    return {
        "상태": "blocked" if hard else ("warming_up" if blockers else "ready"),
        "점검시각": now.isoformat(), "예측대상일": target_day,
        "Blockdata_이력일수": round(history_days, 3),
        "목표일_NWP_시각수": int((nwp or (0, 0))[0] or 0),
        "차단사유": blockers,
        "조치": "mean_communication_ok 제거 후보를 동일 5계절로 재검증 후 운영번들 재학습" if hard else None,
    }


def main() -> None:
    result = audit()
    conn = shadow.ensure_db()
    now = result["점검시각"]
    issue = pd.Timestamp(now).floor("D") + pd.Timedelta(hours=10)
    reason = "; ".join(result["차단사유"])
    conn.execute(
        "UPDATE shadow_predictions SET status='superseded' "
        "WHERE tier='일간' AND status IN ('warming_up','blocked')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO shadow_predictions "
        "(predicted_at,tier,horizon_h,issue_time,target_time,status,reason,bundle_version) "
        "VALUES (?, '일간', 24, ?, ?, ?, ?, 'v2_2026-08-25')",
        (now, str(issue.tz_localize(None)), result["예측대상일"], result["상태"], reason),
    )
    conn.commit(); conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
