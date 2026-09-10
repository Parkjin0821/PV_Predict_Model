# -*- coding: utf-8 -*-
"""★shadow 예측 실행·저장 — 단기(+1h/+24h/+48h)★

`live_feature_assembler_단기_v1_2026-08-26.py`로 조립·예측한 결과를
**Blockdata에 절대 전송하지 않고** 내부 SQLite에만 쌓는다(사용자와
합의한 shadow 원칙 — 실제 발전량이 나중에 도착하면 그때 오차를 계산).

## 실행 주기(Codex 스케줄러 등록용)
- +1h: 매시 정각 실행 권장(모델 자체는 15분 단위지만, 라이브 원자료가
  현재 1시간 그레인이라 1시간 주기로 시작 — 15분 단위로 올리려면
  이 파일의 `build_live_hourly`를 15분 그레인으로 바꿔야 함, 다음 작업).
- +24h·+48h: 매일 1회(D-1 10시 전후 권장, KPX 발행 관례와 동일).

## 실행 예시
```
python shadow_predict_단기_v1_2026-08-26.py            # 3개 수평 전부 1회 실행
python shadow_predict_단기_v1_2026-08-26.py --horizon 1  # +1h만
```
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
KST = ZoneInfo("Asia/Seoul")
BUNDLE_DIR = ROOT / "outputs" / "운영모델_v2_2026-08-25"
SHADOW_DB = ROOT / "outputs" / "shadow_predictions" / "shadow_predictions.sqlite3"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


asm = _load("shadow_asm", "live_feature_assembler_단기_v1_2026-08-26.py")


def ensure_db() -> sqlite3.Connection:
    SHADOW_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SHADOW_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            tier TEXT NOT NULL,
            horizon_h INTEGER NOT NULL,
            issue_time TEXT,
            target_time TEXT,
            predicted_kw REAL,
            n_missing_features INTEGER,
            bundle_version TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            actual_kw REAL,
            evaluated_at TEXT,
            UNIQUE(tier, horizon_h, issue_time)
        )
    """)
    conn.commit()
    return conn


def run_one(conn: sqlite3.Connection, horizon: int) -> dict:
    bundle_path = BUNDLE_DIR / f"운영모델_단기_h{horizon}.joblib"
    bundle = joblib.load(bundle_path)
    result = asm.assemble_and_predict(bundle, horizon=horizon)
    now = pd.Timestamp.now(tz=KST).isoformat()

    if result["상태"] == "성공":
        # 08-28 실배선 추적성 보완: assemble_and_predict()가 반환하는
        # "야간0채움특성"(있으면)을 reason 컬럼에 남긴다 - 안 남기면 이
        # 필드가 DB에 저장 안 되고 그대로 사라져서, 나중에(08-31 야간
        # 실행 검증) "왜 대기->성공으로 바뀌었는지"를 로그로 설명할 방법이
        # 없어진다(사용자 지시 4번째 확인항목).
        night_filled = result.get("야간0채움특성")
        reason = ("야간0채움: " + ",".join(night_filled)) if night_filled else None
        conn.execute(
            "INSERT OR REPLACE INTO shadow_predictions "
            "(predicted_at, tier, horizon_h, issue_time, target_time, predicted_kw, "
            "n_missing_features, bundle_version, status, reason) "
            "VALUES (?, '단기', ?, ?, ?, ?, ?, ?, 'success', ?)",
            (now, horizon, result["발행시각"], result["대상시각"], result["예측_kW"],
             result["특성결측수"], result.get("번들버전"), reason),
        )
    elif result["상태"] == "대기":
        conn.execute(
            "INSERT OR REPLACE INTO shadow_predictions "
            "(predicted_at, tier, horizon_h, issue_time, n_missing_features, "
            "bundle_version, status, reason) "
            "VALUES (?, '단기', ?, ?, ?, ?, 'warming_up', ?)",
            (now, horizon, result.get("발행시각"), result.get("특성결측수"),
             result.get("번들버전"), result.get("사유", "운영 입력 미충족")),
        )
    else:
        conn.execute(
            "INSERT INTO shadow_predictions "
            "(predicted_at, tier, horizon_h, issue_time, status, reason) "
            "VALUES (?, '단기', ?, ?, 'failed', ?)",
            (now, horizon, result.get("발행시각"), result.get("사유", "알 수 없음")),
        )
    conn.commit()
    return result


def evaluate_pending() -> int:
    """대상시각이 이미 지난, 아직 채점 안 된 예측을 Blockdata 실측과 대조한다.
    실측 자체를 보내는 게 아니라 우리 DB 안에서만 비교 — Blockdata 전송과 무관."""
    if not asm.BLOCK_DB.is_file():
        return 0
    conn = ensure_db()
    pending = pd.read_sql_query(
        "SELECT id, target_time FROM shadow_predictions WHERE status='success' AND actual_kw IS NULL",
        conn,
    )
    if pending.empty:
        conn.close()
        return 0
    block_conn = sqlite3.connect(asm.BLOCK_DB)
    updated = 0
    for _, row in pending.iterrows():
        target = pd.Timestamp(row["target_time"])
        if target > pd.Timestamp.now(tz=KST).tz_localize(None):
            continue  # 아직 그 시각이 안 지남 — 다음 실행에서 재시도
        # 대상시각과 가장 가까운(±20분 이내) 실측 스냅샷을 찾는다.
        window_start = (target - pd.Timedelta(minutes=20)).tz_localize("Asia/Seoul").tz_convert("UTC").tz_localize(None).isoformat()
        window_end = (target + pd.Timedelta(minutes=20)).tz_localize("Asia/Seoul").tz_convert("UTC").tz_localize(None).isoformat()
        actual = block_conn.execute(
            "SELECT plant_ac_power_kw FROM plant_snapshots "
            "WHERE snapshot_time BETWEEN ? AND ? ORDER BY snapshot_time LIMIT 1",
            (window_start, window_end),
        ).fetchone()
        if actual is None:
            continue
        conn.execute(
            "UPDATE shadow_predictions SET actual_kw=?, evaluated_at=? WHERE id=?",
            (actual[0], pd.Timestamp.now(tz=KST).isoformat(), int(row["id"])),
        )
        updated += 1
    conn.commit()
    conn.close()
    block_conn.close()
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[1, 24, 48], default=None)
    args = parser.parse_args()

    conn = ensure_db()
    horizons = [args.horizon] if args.horizon else [1, 24, 48]
    results = {}
    for h in horizons:
        try:
            results[f"+{h}h"] = run_one(conn, h)
        except Exception as e:  # noqa: BLE001 — 개별 수평 실패가 다른 수평까지 막지 않게
            results[f"+{h}h"] = {"상태": "예외", "사유": f"{type(e).__name__}: {e}"}
            conn.execute(
                "INSERT INTO shadow_predictions (predicted_at, tier, horizon_h, status, reason) "
                "VALUES (?, '단기', ?, 'error', ?)",
                (pd.Timestamp.now(tz=KST).isoformat(), h, str(e)),
            )
            conn.commit()
    conn.close()

    n_evaluated = evaluate_pending()
    print(json.dumps({"실행결과": results, "이번에_채점된_과거예측": n_evaluated,
                      "저장위치": str(SHADOW_DB)}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
