# -*- coding: utf-8 -*-
"""광주 중장기(D+1) shadow 예측 실측 대조 - MAE/RMSE.

`shadow_predict_일간_v1_2026-08-27.py`(`UCUBE_ShadowPredict_Daily`)가
08-27부터 광주 전용 `outputs/shadow_predictions/shadow_predictions.
sqlite3`(tier='일간')에 예측을 쌓아왔지만, 지금까지 이걸 실측과
대조해서 채점하는 스크립트가 하나도 없었다(부안·김제·영광과 동일한
공백이었음 - 09-14 사용자 지적으로 발견).

## 재구현 없음
실측 조회는 `shadow_predict_medium_daily_v1_2026-09-14.py`의
`yesterday_actual_kwh()`를 그대로 가져다 쓴다(품질기준 동일 -
valid_ac_power_count==expected_inverter_count인 5분 스냅샷만 kWh 합산).
그 함수는 지역별 plant_db/inv만 받으면 되므로 광주 값만 넣으면 된다.

## 단위
부안·김제·영광 평가 스크립트와 동일하게 MAE_kWh/RMSE_kWh(%로 정규화
안 함 - 기존 "일간 D+1은 kWh 단위" 관례).
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SHADOW_DB = ROOT / "outputs" / "shadow_predictions" / "shadow_predictions.sqlite3"
PLANT_DB = r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25\blockdata_history.sqlite3"
GWANGJU_INV = 5
MIN_INDEPENDENT_DAYS = 5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


predictor = load_module("medium_daily_predictor_for_gwangju",
                        ROOT / "shadow_predict_medium_daily_v1_2026-09-14.py")


def actual_kwh_for_day(day: pd.Timestamp) -> float:
    """predictor.yesterday_actual_kwh()는 (target_day-1)의 실측을 반환하므로
    day+1을 넣어 원하는 day 자체의 실측을 얻는다(재구현 안 함)."""
    return predictor.yesterday_actual_kwh(PLANT_DB, GWANGJU_INV, day + pd.Timedelta(days=1))


def evaluate() -> pd.DataFrame:
    if not SHADOW_DB.is_file():
        return pd.DataFrame()
    conn = sqlite3.connect(SHADOW_DB)
    try:
        preds = pd.read_sql_query(
            "SELECT id, target_time, predicted_kw FROM shadow_predictions "
            "WHERE tier='일간' AND status='성공'", conn)
    finally:
        conn.close()
    if preds.empty:
        return pd.DataFrame()

    preds["target_time"] = pd.to_datetime(preds["target_time"])
    today = pd.Timestamp.now().normalize()
    due = preds[preds["target_time"] < today].copy()
    if due.empty:
        return pd.DataFrame()

    due = due.sort_values("id").drop_duplicates(["target_time"], keep="last")
    due["actual_kwh"] = due["target_time"].apply(actual_kwh_for_day)
    evaluable = due.dropna(subset=["actual_kwh"])
    if evaluable.empty:
        return pd.DataFrame()

    err = evaluable["predicted_kw"] - evaluable["actual_kwh"]  # 컬럼명은 kw지만 실제 저장값은 kWh(일간 총량)
    return pd.DataFrame([{
        "region": "광주", "n": len(evaluable),
        "independent_days": evaluable["target_time"].nunique(),
        "MAE_kWh": round(float(err.abs().mean()), 1),
        "RMSE_kWh": round(float(np.sqrt((err ** 2).mean())), 1),
        "예비치_경고": evaluable["target_time"].nunique() < MIN_INDEPENDENT_DAYS,
    }])


def main() -> int:
    import json
    df = evaluate()
    if df.empty:
        print(json.dumps({"status": "평가가능표본없음",
                          "reason": "target_time이 아직 도래하지 않았거나 실측 미확보"},
                         ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(df.to_dict("records"), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
