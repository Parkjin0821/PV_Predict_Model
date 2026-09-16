# -*- coding: utf-8 -*-
"""부안·김제·영광 중장기(D+1) shadow 예측 실측 대조 - MAE/RMSE.

`shadow_predict_medium_daily_v1_2026-09-14.py`가 쌓아온
`shadow_predictions_medium_daily.sqlite3`의 `성공` 예측 중 target_day가
이미 지난 것만, 그 날 실제 발전량과 대조한다(`evaluate_ultrashort_
shadow_v1_2026-09-08.py`와 동일 원칙 - target 미도래 표본은 억지로
평가 안 하고 정직하게 "아직 없음" 보고).

## 단위(기존 관례 그대로)
이 티어는 하루 총량(kWh)이 타깃이라 %(nMAE) 대신 **MAE_kWh/RMSE_kWh**로
보고한다 - `build_medium()`이 만든 09-08 김제·영광 번들도 이미 이
단위로 판정했고("일간 D+1 번들은 MAE/RMSE로 산정되어 %지표와 다른
단위"), 여기서 새 정규화 방식을 만들지 않는다.

## 실측 소스
전일지속성 lag과 동일하게 blockdata_history.sqlite3에서 직접 계산
(`shadow_predict_medium_daily_v1_2026-09-14.py`의 `yesterday_actual_
kwh()`와 동일 품질기준 재사용 - 재구현 안 함, 대상일만 임의로 바꿔
호출).
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
SHADOW_DB = ROOT / "shadow_predictions_medium_daily.sqlite3"
MIN_INDEPENDENT_DAYS = 5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


predictor = load_module("medium_daily_predictor", ROOT / "shadow_predict_medium_daily_v1_2026-09-14.py")
REGIONS = predictor.REGIONS


def actual_kwh_for_day(plant_db: str, inv: int, day: pd.Timestamp) -> float:
    """predictor.yesterday_actual_kwh()를 재사용 - target_day+1을 넣으면
    그 함수 내부에서 다시 -1일 해서 원하는 day의 실측을 얻는다(재구현 안 함)."""
    return predictor.yesterday_actual_kwh(plant_db, inv, day + pd.Timedelta(days=1))


def evaluate() -> pd.DataFrame:
    conn = sqlite3.connect(SHADOW_DB)
    try:
        preds = pd.read_sql_query(
            "SELECT id, region, issue_time_kst, target_day, predicted_kwh "
            "FROM shadow_medium_daily_predictions WHERE status='성공'", conn)
    finally:
        conn.close()
    if preds.empty:
        return pd.DataFrame()

    preds["target_day"] = pd.to_datetime(preds["target_day"])
    today = pd.Timestamp.now().normalize()
    due = preds[preds["target_day"] < today].copy()
    if due.empty:
        return pd.DataFrame()

    due = due.sort_values("id").drop_duplicates(["region", "target_day"], keep="last")
    due["actual_kwh"] = due.apply(
        lambda r: actual_kwh_for_day(REGIONS[r["region"]]["plant"], REGIONS[r["region"]]["inv"], r["target_day"]),
        axis=1)
    evaluable = due.dropna(subset=["actual_kwh"])
    if evaluable.empty:
        return pd.DataFrame()

    rows = []
    for region, g in evaluable.groupby("region"):
        err = g["predicted_kwh"] - g["actual_kwh"]
        pers_err = g["actual_kwh"].shift(1) - g["actual_kwh"]  # 참고용, 독립일수 적으면 NaN 많음
        rows.append({
            "region": region, "n": len(g),
            "independent_days": g["target_day"].nunique(),
            "MAE_kWh": round(float(err.abs().mean()), 1),
            "RMSE_kWh": round(float(np.sqrt((err ** 2).mean())), 1),
            "예비치_경고": g["target_day"].nunique() < MIN_INDEPENDENT_DAYS,
        })
    return pd.DataFrame(rows)


def main() -> int:
    import json
    df = evaluate()
    if df.empty:
        print(json.dumps({"status": "평가가능표본없음",
                          "reason": "target_day가 아직 도래하지 않았거나 실측 미확보"},
                         ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(df.to_dict("records"), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
