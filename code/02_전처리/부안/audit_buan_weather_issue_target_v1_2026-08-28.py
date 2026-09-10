"""부안 KMA 수집결과를 발행시각 기준으로 감사한다(API 호출 없음).

저장 성공과 당시 예측에 사용 가능했는지를 분리한다. planned issue time
이후에 받은 예보는 DB에는 보존하지만 해당 발행 예측의 입력으로 승격하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
EXPECTED_GRID_VARS = {"TMP", "SKY", "REH", "WSD", "POP", "VEC"}
EXPECTED_NWP_VALUES = 56
EXPECTED_NATIVE_MISSING = {"DSWRFLX", "DIFSWRF"}
DEFAULT_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3"
)
DEFAULT_OUT = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\기상_발행대상시각감사_v1_2026-08-28"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--issue-date", help="YYYY-MM-DD; 기본 DB 최신 발표일")
    p.add_argument("--planned-issue-hour", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def read(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def kst_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(KST).dt.tz_localize(None)


def main() -> None:
    a = parse_args()
    if not a.db.exists():
        raise FileNotFoundError(a.db)
    conn = sqlite3.connect(f"file:{a.db.as_posix()}?mode=ro", uri=True)
    try:
        latest = conn.execute("SELECT MAX(issue_date) FROM nwp_values").fetchone()[0]
        issue_date = a.issue_date or latest
        if not issue_date:
            raise RuntimeError("NWP issue_date가 없음")
        asos = read(conn, "SELECT * FROM asos_hourly")
        grid = read(conn, "SELECT * FROM grid_forecast WHERE issue_date=?", (issue_date,))
        nwp = read(conn, "SELECT * FROM nwp_values WHERE issue_date=?", (issue_date,))
    finally:
        conn.close()

    issue_day = datetime.strptime(issue_date, "%Y-%m-%d").date()
    planned = datetime.combine(issue_day, time(a.planned_issue_hour), tzinfo=KST).replace(tzinfo=None)
    if asos.empty or grid.empty or nwp.empty:
        raise RuntimeError("ASOS·GRID·NWP 중 하나 이상 비어 있음")

    station_set = sorted(pd.to_numeric(asos["station"], errors="coerce").dropna().astype(int).unique())
    if station_set != [243]:
        raise RuntimeError(f"부안 DB에 ASOS 243 외 지점 혼입: {station_set}")
    grid_vars = set(grid["variable"].astype(str))
    if grid_vars != EXPECTED_GRID_VARS:
        raise RuntimeError(f"동네예보 변수 불일치: {sorted(grid_vars)}")
    if len(nwp) != EXPECTED_NWP_VALUES:
        raise RuntimeError(f"NWP 저장값 56개가 아님: {len(nwp)}")

    asos["observation_kst"] = kst_naive(asos["observation_time"])
    grid["target_kst"] = kst_naive(grid["target_time_kst"])
    grid["run_kst"] = kst_naive(grid["run_time_kst"])
    grid["received_kst"] = kst_naive(grid["first_received_at"])
    nwp["target_kst"] = kst_naive(nwp["target_time_kst"])
    nwp["run_kst"] = pd.to_datetime(nwp["requested_tm_utc"], utc=True).dt.tz_convert(KST).dt.tz_localize(None)
    nwp["received_kst"] = kst_naive(nwp["first_received_at"])

    asos_eligible = asos[asos["observation_kst"] <= planned]
    grid_eligible = grid[(grid["run_kst"] <= planned) & (grid["received_kst"] <= planned)]
    nwp_eligible = nwp[(nwp["run_kst"] <= planned) & (nwp["received_kst"] <= planned)]
    missing_vars = set(nwp.loc[nwp["is_missing"].astype(bool), "variable"].astype(str))
    unexpected_missing = sorted(missing_vars - EXPECTED_NATIVE_MISSING)
    if unexpected_missing:
        raise RuntimeError(f"예상 밖 NWP 결측 변수: {unexpected_missing}")

    targets = sorted(set(grid["target_kst"]) | set(nwp["target_kst"]))
    rows=[]
    for target in targets:
        ge=grid_eligible[grid_eligible["target_kst"]==target]
        ne=nwp_eligible[nwp_eligible["target_kst"]==target]
        rows.append({
            "prediction_issue_time_kst": planned,
            "target_time_kst": target,
            "grid_stored_variable_count": int((grid["target_kst"]==target).sum()),
            "grid_eligible_variable_count": int(len(ge)),
            "nwp_stored_variable_count": int((nwp["target_kst"]==target).sum()),
            "nwp_eligible_variable_count": int(len(ne)),
            "eligible_for_issue": bool(len(ge)==6 and len(ne)==7),
        })
    detail=pd.DataFrame(rows)
    eligible_targets=int(detail["eligible_for_issue"].sum())
    summary={
        "issue_date":issue_date, "planned_issue_time_kst":str(planned),
        "storage_validation":{
            "asos_station_243":True, "asos_rows":int(len(asos)),
            "grid_rows":int(len(grid)), "grid_missing":int(grid["is_missing"].sum()),
            "nwp_rows":int(len(nwp)), "nwp_missing":int(nwp["is_missing"].sum()),
            "nwp_missing_variables":sorted(missing_vars),
        },
        "as_of_issue_validation":{
            "latest_asos_observation_kst":str(asos_eligible["observation_kst"].max()) if len(asos_eligible) else None,
            "eligible_grid_rows":int(len(grid_eligible)),
            "eligible_nwp_rows":int(len(nwp_eligible)),
            "eligible_target_count":eligible_targets,
            "status":"ready" if eligible_targets==len(targets) else "blocked_late_collection",
            "reason":"저장은 정상이나 계획 발행시각 이후 수신자료는 해당 발행 예측에 사용 금지" if eligible_targets<len(targets) else None,
        },
        "future_leak_rows":int((grid_eligible["received_kst"]>planned).sum()+(nwp_eligible["received_kst"]>planned).sum()),
    }
    if summary["future_leak_rows"]:
        raise RuntimeError("미래수신 예보가 eligibility를 통과함")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.output_dir/"부안_발행대상시각_적격성.csv",index=False,encoding="utf-8-sig")
    (a.output_dir/"부안_기상수집_발행기준감사.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
