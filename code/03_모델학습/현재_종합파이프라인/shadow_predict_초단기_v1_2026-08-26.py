# -*- coding: utf-8 -*-
"""초단기 +1~+4h shadow 예측. 외부 전송/API 호출 없음."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "outputs" / "운영모델_v2_2026-08-25"
SHADOW_DB = ROOT / "outputs" / "shadow_predictions" / "shadow_predictions.sqlite3"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


asm = _load("shadow_ultra_asm", "live_feature_assembler_초단기_v1_2026-08-26.py")
short_shadow = _load("shadow_short_db", "shadow_predict_단기_v1_2026-08-26.py")


def run_one(conn: sqlite3.Connection, horizon: int) -> dict:
    bundle = joblib.load(BUNDLE_DIR / f"운영모델_초단기_h{horizon}.joblib")
    result = asm.assemble_and_predict(bundle, horizon)
    now = pd.Timestamp.now(tz=asm.short.KST).isoformat()
    status = {"성공": "success", "대기": "warming_up", "실패": "failed"}.get(
        result["상태"], "error")
    # 08-28 실배선 추적성 보완: 성공이어도 "야간0채움특성"이 있으면 reason에
    # 남긴다(shadow_predict_단기와 동일 이유 - 안 남기면 사라져서 08-31
    # 검증 때 대기->성공 전환 사유를 로그로 설명할 수 없다).
    night_filled = result.get("야간0채움특성")
    if status == "success":
        reason = ("야간0채움: " + ",".join(night_filled)) if night_filled else None
    else:
        reason = result.get("사유")
    conn.execute(
        "INSERT OR REPLACE INTO shadow_predictions "
        "(predicted_at,tier,horizon_h,issue_time,target_time,predicted_kw,"
        "n_missing_features,bundle_version,status,reason) VALUES "
        "(?, '초단기', ?, ?, ?, ?, ?, ?, ?, ?)",
        (now, horizon, result.get("발행시각"), result.get("대상시각"),
         result.get("예측_kW"), result.get("특성결측수"), result.get("번들버전"),
         status, reason),
    )
    conn.commit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    conn = short_shadow.ensure_db()
    results = {}
    for h in ([args.horizon] if args.horizon else [1, 2, 3, 4]):
        try:
            results[f"+{h}h"] = run_one(conn, h)
        except Exception as exc:  # 개별 수평 격리
            results[f"+{h}h"] = {"상태": "예외", "사유": f"{type(exc).__name__}: {exc}"}
    conn.close()
    evaluated = short_shadow.evaluate_pending()
    print(json.dumps({"실행결과": results, "이번에_채점된_과거예측": evaluated,
                      "저장위치": str(SHADOW_DB)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
