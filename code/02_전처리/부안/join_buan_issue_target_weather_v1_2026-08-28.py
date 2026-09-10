"""부안 발전시각에 ASOS·동네예보·NWP의 발행/대상시각을 누출 없이 연결.

API 호출은 하지 않는다. 부안 전용 KMA SQLite가 수집된 뒤 실행한다.
예보는 대상시각만 맞추지 않고, 예측 발행시각 이전에 실제 수신된 런만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_POWER = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\Excel_API_연결_v1_2026-08-28\부안_발전소_5분_Excel_API연결.parquet"
)
DEFAULT_WEATHER_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3"
)
DEFAULT_NWP_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3"
)
DEFAULT_OUT = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안"
    r"\기상_발행대상시각결합_v1_2026-08-28"
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--power", type=Path, default=DEFAULT_POWER)
    p.add_argument("--weather-db", type=Path, default=DEFAULT_WEATHER_DB)
    p.add_argument("--nwp-db", type=Path, default=DEFAULT_NWP_DB)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--issue-lag-hours", type=int, default=1,
                   help="과거자료용 보수적 예측발행시각=대상시각-lag; 운영예측은 실제 발행시각을 별도 입력")
    return p.parse_args()


def read_sql_ro(path: Path, sql: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def to_kst_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)


def main() -> None:
    a = args()
    power = pd.read_parquet(a.power)
    power["target_time_kst"] = pd.to_datetime(power["grid_time_kst"], errors="raise")
    # 사전 데이터셋 조립용 발행시각. 실제 운영에서는 드라이버의 실제 issue_time을 사용한다.
    power["prediction_issue_time_kst"] = power["target_time_kst"] - pd.Timedelta(hours=a.issue_lag_hours)

    asos = read_sql_ro(a.weather_db, "SELECT * FROM asos_hourly WHERE station=243")
    grid = read_sql_ro(a.weather_db, "SELECT * FROM grid_forecast")
    nwp = read_sql_ro(a.nwp_db, "SELECT * FROM nwp_values")
    if asos.empty or grid.empty or nwp.empty:
        raise RuntimeError("부안 ASOS·동네예보·NWP 중 하나 이상이 비어 있음")

    asos["observation_time_kst"] = to_kst_naive(asos["observation_time"])
    power = pd.merge_asof(
        power.sort_values("prediction_issue_time_kst"),
        asos.sort_values("observation_time_kst"),
        left_on="prediction_issue_time_kst", right_on="observation_time_kst",
        direction="backward", tolerance=pd.Timedelta("3h"), suffixes=("", "_asos"),
    )

    grid["run_time_kst"] = to_kst_naive(grid["run_time_kst"])
    grid["target_time_kst"] = to_kst_naive(grid["target_time_kst"])
    grid["last_received_at_kst"] = to_kst_naive(grid["last_received_at"])
    eligible_grid = grid.copy()
    # long→wide 전, 동일 대상·변수에서 가장 최신 런을 보존한다. 운영 시에는 아래에서
    # prediction_issue_time별 eligibility를 다시 검사하므로 수신 후 미래누출이 없다.
    joined_grid=[]
    for row in power[["target_time_kst", "prediction_issue_time_kst"]].drop_duplicates().itertuples(index=False):
        q=eligible_grid[(eligible_grid["target_time_kst"]==row.target_time_kst) &
                        (eligible_grid["run_time_kst"]<=row.prediction_issue_time_kst) &
                        (eligible_grid["last_received_at_kst"]<=row.prediction_issue_time_kst)]
        if q.empty: continue
        q=q.sort_values("run_time_kst").drop_duplicates("variable", keep="last").copy()
        q["prediction_issue_time_kst"]=row.prediction_issue_time_kst
        joined_grid.append(q)
    grid_long=pd.concat(joined_grid, ignore_index=True) if joined_grid else pd.DataFrame()

    nwp["target_time_kst"] = to_kst_naive(nwp["target_time_kst"])
    nwp["first_received_at_kst"] = to_kst_naive(nwp["first_received_at"])
    nwp["run_time_kst"] = pd.to_datetime(nwp["requested_tm_utc"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    joined_nwp=[]
    for row in power[["target_time_kst", "prediction_issue_time_kst"]].drop_duplicates().itertuples(index=False):
        q=nwp[(nwp["target_time_kst"]==row.target_time_kst) &
              (nwp["run_time_kst"]<=row.prediction_issue_time_kst) &
              (nwp["first_received_at_kst"]<=row.prediction_issue_time_kst)]
        if q.empty: continue
        q=q.sort_values("run_time_kst").drop_duplicates("variable", keep="last").copy()
        q["prediction_issue_time_kst"]=row.prediction_issue_time_kst
        joined_nwp.append(q)
    nwp_long=pd.concat(joined_nwp, ignore_index=True) if joined_nwp else pd.DataFrame()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    power.to_parquet(a.output_dir / "부안_발전_ASOS_발행기준결합.parquet", index=False)
    grid_long.to_parquet(a.output_dir / "부안_동네예보_발행대상시각_long.parquet", index=False)
    nwp_long.to_parquet(a.output_dir / "부안_NWP_발행대상시각_long.parquet", index=False)
    summary={
        "power_rows":int(len(power)), "asos_joined_rows":int(power["observation_time_kst"].notna().sum()),
        "grid_eligible_rows":int(len(grid_long)), "nwp_eligible_rows":int(len(nwp_long)),
        "future_leak_grid_rows":int((grid_long["last_received_at_kst"]>grid_long["prediction_issue_time_kst"]).sum()) if len(grid_long) else 0,
        "future_leak_nwp_rows":int((nwp_long["first_received_at_kst"]>nwp_long["prediction_issue_time_kst"]).sum()) if len(nwp_long) else 0,
        "note":"운영에서는 실제 prediction_issue_time을 사용; issue_lag는 과거 결합 감사용",
    }
    if summary["future_leak_grid_rows"] or summary["future_leak_nwp_rows"]:
        raise RuntimeError("발행/수신시각 미래누출 발견")
    (a.output_dir/"부안_기상결합_요약.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
