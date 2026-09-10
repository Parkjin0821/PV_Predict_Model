# -*- coding: utf-8 -*-
"""초단기 shadow 예측 실측 대조 - MAE/RMSE/nMAE + 수평별 유효 표본 수.

`live_feature_assembler_ultrashort_v1_2026-09-08.py`가 쌓아온
`shadow_predictions_ultrashort.sqlite3`의 `성공` 예측 중 target_time이
이미 지난 것만, 그 시각 실제 발전량(`plant_snapshots`, 완전가용 기준은
조립기와 동일)과 대조한다. target_time 근방(±tol_minutes) 실측이 없으면
그 표본은 평가에서 제외한다(임의보간 없음) - "아직 평가 가능한 표본이
없다"는 상태를 정직하게 보고하며, 억지로 표본을 만들지 않는다.

API 호출 없음(전부 저장된 라이브 DB 읽기). 실행할 때마다 그 시점까지
확정된 결과를 다시 계산한다(별도 저장 없이 매번 최신화 - 표본이 계속
늘어나는 값이라 스냅샷을 고정할 이유가 아직 없음).
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHADOW_DB = ROOT / "shadow_predictions_ultrashort.sqlite3"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


assembler = load_module("ultra_assembler", ROOT / "live_feature_assembler_ultrashort_v1_2026-09-08.py")
REGIONS = assembler.REGIONS
CAPACITY = {r: c["capacity"] for r, c in REGIONS.items()}
MIN_INDEPENDENT_DAYS = 5


def evaluate(tol_minutes: float = 4.0, min_independent_days: int = MIN_INDEPENDENT_DAYS,
             representative_issues_per_day: int = 2) -> pd.DataFrame:
    con = sqlite3.connect(SHADOW_DB)
    preds = pd.read_sql_query(
        "SELECT id, region, horizon_h, issue_time_kst, target_time_kst, predicted_kw "
        "FROM shadow_ultrashort_predictions WHERE status='성공'", con)
    con.close()
    if preds.empty:
        return pd.DataFrame()
    preds["target_time_kst"] = pd.to_datetime(preds["target_time_kst"])
    now = pd.Timestamp.now(tz=assembler.KST).tz_localize(None)
    due = preds[preds["target_time_kst"] <= now].copy()
    if due.empty:
        return pd.DataFrame()
    due = (due.sort_values("id")
              .drop_duplicates(["region", "horizon_h", "issue_time_kst", "target_time_kst"], keep="last"))
    due["issue_date"] = due["issue_time_kst"].str[:10]
    due["issue_rank_in_day"] = (pd.to_datetime(due["issue_time_kst"])
                                  .groupby([due["region"], due["horizon_h"], due["issue_date"]])
                                  .rank(method="first"))
    due_rep = due[due["issue_rank_in_day"] <= representative_issues_per_day].copy()

    actual_series = {
        r: assembler.load_power_series(cfg["plant_db"], cfg["plant_id"], cfg["expected_inverters"],
                                       lookback_hours=48)
        for r, cfg in REGIONS.items()
    }

    def lookup_actual(row):
        s = actual_series.get(row["region"])
        if s is None or s.empty:
            return np.nan
        return assembler.lookup_near(s, row["target_time_kst"], tol_minutes=tol_minutes)

    due["actual_kw"] = due.apply(lookup_actual, axis=1)
    due_rep["actual_kw"] = due_rep.apply(lookup_actual, axis=1)
    due["issue_actual_kw"] = due.apply(lambda r: lookup_actual(pd.Series({"region": r["region"], "target_time_kst": pd.to_datetime(r["issue_time_kst"])})), axis=1)
    due_rep["issue_actual_kw"] = due_rep.apply(lambda r: lookup_actual(pd.Series({"region": r["region"], "target_time_kst": pd.to_datetime(r["issue_time_kst"])})), axis=1)
    evaluable = due.dropna(subset=["actual_kw"]).copy()
    representative = due_rep.dropna(subset=["actual_kw"]).copy()
    if evaluable.empty:
        return pd.DataFrame()

    evaluable["error_kw"] = evaluable["predicted_kw"] - evaluable["actual_kw"]
    solar_modules = {"부안": assembler.BUAN_SOLAR, "김제": assembler.GIMJE_SOLAR,
                     "영광": assembler.YEONGGWANG_SOLAR, "광주": assembler.GWANGJU_SOLAR}
    evaluable["target_elevation"] = evaluable.apply(
        lambda r: assembler.solar_elev(solar_modules[r["region"]], r["target_time_kst"]), axis=1)
    evaluable["period"] = np.where(evaluable["target_elevation"] > 0, "day", "night")
    evaluable["target_date"] = evaluable["target_time_kst"].dt.date
    representative["target_elevation"] = representative.apply(
        lambda r: assembler.solar_elev(solar_modules[r["region"]], r["target_time_kst"]), axis=1)
    representative["period"] = np.where(representative["target_elevation"] > 0, "day", "night")
    representative["target_date"] = representative["target_time_kst"].dt.date
    evaluable["persistence_kw"] = evaluable["issue_actual_kw"]
    target_clear = np.maximum(np.sin(np.deg2rad(evaluable["target_elevation"])), 0.0)
    issue_elev = evaluable.apply(lambda r: assembler.solar_elev(solar_modules[r["region"]], pd.to_datetime(r["issue_time_kst"])), axis=1)
    issue_clear = np.maximum(np.sin(np.deg2rad(issue_elev)), 0.0)
    evaluable["clear_sky_persistence_kw"] = np.where(issue_clear > 1e-6, evaluable["issue_actual_kw"] * target_clear / issue_clear, 0.0)
    rows = []
    for (region, h, period), g in evaluable.groupby(["region", "horizon_h", "period"]):
        mse = float((g["error_kw"] ** 2).mean())
        mae = float(g["error_kw"].abs().mean())
        rmse = float(np.sqrt(mse))
        cap = CAPACITY[region]
        # R² = 1 - SS_res/SS_tot. n이 작거나(예비치 구간 다수) 실측값 분산이
        # 거의 0(예: 야간 실측이 전부 0.x kW 근처)이면 분모가 0에 가까워져
        # 값이 무의미해진다 - 억지로 숫자를 만들지 않고 null로 정직하게 둔다.
        ss_res = float((g["error_kw"] ** 2).sum())
        ss_tot = float(((g["actual_kw"] - g["actual_kw"].mean()) ** 2).sum())
        r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 1e-6 else None
        rep_g = representative[(representative["region"] == region) &
                               (representative["horizon_h"] == h) &
                               (representative["period"] == period)]
        rows.append({
            "region": region, "horizon_h": h, "period": period, "n": len(g),
            "independent_days": int(g["target_date"].nunique()),
            "representative_n": int(len(rep_g)),
            "effective_N": int(rep_g["target_date"].nunique()),
            "representative_issues_per_day": representative_issues_per_day,
            "evaluation_status": "예비치" if g["target_date"].nunique() < min_independent_days else "평가가능",
            "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3), "MSE_kW2": round(mse, 3),
            "R2": r2,
            "nMAE_pct": round(mae / cap * 100, 3),
            "persistence_MAE_kW": round(float((g["persistence_kw"] - g["actual_kw"]).abs().mean()), 3),
            "clear_sky_persistence_MAE_kW": round(float((g["clear_sky_persistence_kw"] - g["actual_kw"]).abs().mean()), 3),
        })
    return pd.DataFrame(rows).sort_values(["region", "horizon_h"])


def main() -> None:
    result = evaluate()
    if result.empty:
        con = sqlite3.connect(SHADOW_DB)
        pending = pd.read_sql_query(
            "SELECT MIN(target_time_kst) AS earliest_due "
            "FROM shadow_ultrashort_predictions WHERE status='성공'", con)
        con.close()
        earliest = pending["earliest_due"].iloc[0] if not pending.empty else None
        print(f"아직 target_time이 도래한 평가 가능 표본이 없습니다"
             f"(가장 이른 target_time: {earliest}). 나중에 다시 실행하세요.")
        return
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
