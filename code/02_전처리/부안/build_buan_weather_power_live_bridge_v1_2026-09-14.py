"""부안 D+1 재검증용 발전량·ASOS·NWP·GRID 라이브 연계 후보 생성.

기존 08-31 결합본을 보존하고, 이후 라이브 DB의 발행일만 같은 스키마로
이어붙인다. 예측 입력에는 first_received_at이 prediction_issue_time_kst
이하인 값만 허용한다. 대상시각 ASOS는 reference 접두어로만 보존한다.
API 호출은 하지 않는다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안")
DEFAULT_ARCHIVE = BASE / r"과거발전_기상결합_v1_2026-08-28\부안_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"
DEFAULT_POWER = BASE / r"시간집계_라이브연계_v1_2026-09-14\부안_발전소_1시간_라이브연계.parquet"
DEFAULT_DAILY = BASE / r"시간집계_라이브연계_v1_2026-09-14\부안_발전소_일간_라이브연계.csv"
DEFAULT_KMA_DB = BASE / r"kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3"
DEFAULT_OUTPUT = BASE / r"과거발전_기상결합_라이브연계_v1_2026-09-14"
AGG_MODULE = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\02_전처리\부안\build_buan_time_aggregates_v1_2026-08-28.py")
OUT_NAME = "부안_과거발전_ASOS_NWP_GRID_결합_라이브연계_v1_2026-09-14.parquet"
LIVE_NAME = "부안_라이브추가분_ASOS_NWP_GRID_결합_v1_2026-09-14.parquet"
DAILY_NAME = "부안_발전소_일간_라이브연계_D1검증용.csv"

NWP_VARS = ("DSWRF", "DSWRFLX", "DIFSWRF", "TCDC", "LCDC", "MCDC", "HCDC")
GRID_VARS = ("REH", "POP", "SKY")
WEATHER8 = ("DSWRF", "TCDC", "LCDC", "MCDC", "HCDC", "REH", "POP", "SKY")

ASOS_MAP = {
    "station": "지점번호",
    "observation_time": "시각",
    "temperature_c": "기온_C",
    "rainfall_mm": "강수량_mm",
    "wind_speed_m_s": "풍속_m_s",
    "wind_direction_deg": "풍향_deg",
    "humidity_pct": "상대습도_pct",
    "local_pressure_hpa": "현지기압_hPa",
    "sea_pressure_hpa": "해면기압_hPa",
    "sunshine_hr": "일조시간_hr",
    "solar_mj_m2": "일사량_MJ_m2",
    "solar_w_m2": "일사량_W_m2",
    "snow_cm": "적설_cm",
    "cloud_tenths": "전운량_10분위",
    "cloud_pct": "전운량_pct",
    "ground_temperature_c": "지면온도_C",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--power", type=Path, default=DEFAULT_POWER)
    p.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    p.add_argument("--kma-db", type=Path, default=DEFAULT_KMA_DB)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def load_aggregate_module():
    spec = importlib.util.spec_from_file_location("buan_time_aggregate_for_daily_gate", AGG_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"시간집계 모듈 로드 실패: {AGG_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correct_daily_denominator(path: Path) -> pd.DataFrame:
    """관측된 슬롯이 아니라 날짜별 전체 일광 5분 슬롯을 분모로 사용한다."""
    daily = pd.read_csv(path)
    daily["date_kst"] = pd.to_datetime(daily["date_kst"], errors="raise")
    agg = load_aggregate_module()
    expected = []
    for day in daily["date_kst"]:
        full = pd.date_range(day, day + pd.Timedelta(days=1) - pd.Timedelta(minutes=5), freq="5min")
        expected.append(int((agg.solar_elevation_deg(full) > 0).sum()))
    daily["daylight_expected_slots"] = expected
    daily["daylight_valid_ratio"] = daily["daylight_valid_slots"].div(
        daily["daylight_expected_slots"].replace(0, np.nan)
    )
    good = daily["daylight_valid_ratio"].ge(agg.DAYLIGHT_DAILY_MIN_RATIO)
    daily.loc[~good, "daily_energy_kwh"] = np.nan
    daily["quality_status"] = np.where(
        good, "valid_daylight_ge90pct", "invalid_daylight_lt90pct"
    )
    return daily


def parse_kst(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(
        "Asia/Seoul"
    ).dt.tz_localize(None)


def read_sql_ro(path: Path, sql: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def build_issue_asos(issues: pd.DataFrame, asos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for issue in issues["prediction_issue_time_kst"]:
        eligible = asos[
            (asos["observation_time_kst"] <= issue)
            & (asos["first_received_at_kst"] <= issue)
            & (asos["observation_time_kst"] >= issue - pd.Timedelta("3h"))
        ]
        rec = {"prediction_issue_time_kst": issue}
        if not eligible.empty:
            src = eligible.sort_values("observation_time_kst").iloc[-1]
            for db_col, old_col in ASOS_MAP.items():
                rec[f"issue_asos_{old_col}"] = src[db_col]
            rec["observation_time_kst"] = src["observation_time_kst"]
        rows.append(rec)
    return pd.DataFrame(rows)


def build_reference_asos(targets: pd.DataFrame, asos: pd.DataFrame) -> pd.DataFrame:
    src = asos.copy()
    src["target_time_kst"] = src["observation_time_kst"]
    keep = ["target_time_kst"]
    rename = {}
    for db_col, old_col in ASOS_MAP.items():
        keep.append(db_col)
        rename[db_col] = f"reference_target_asos_{old_col}"
    src = src[keep].rename(columns=rename).drop_duplicates("target_time_kst", keep="last")
    return targets.merge(src, on="target_time_kst", how="left", validate="many_to_one")


def main() -> None:
    a = args()
    archive = pd.read_parquet(a.archive)
    power = pd.read_parquet(a.power).rename(columns={"grid_time_kst": "target_time_kst"})
    daily_candidate = correct_daily_denominator(a.daily)
    power["target_time_kst"] = pd.to_datetime(power["target_time_kst"], errors="raise")

    nwp = read_sql_ro(a.kma_db, "SELECT * FROM nwp_values")
    grid = read_sql_ro(a.kma_db, "SELECT * FROM grid_forecast")
    asos = read_sql_ro(a.kma_db, "SELECT * FROM asos_hourly WHERE station=243")
    if nwp.empty or grid.empty or asos.empty:
        raise RuntimeError("부안 라이브 기상 DB의 필수 테이블이 비어 있음")

    nwp["prediction_issue_time_kst"] = parse_kst(nwp["planned_issue_at_kst"])
    nwp["target_time_kst"] = parse_kst(nwp["target_time_kst"])
    nwp["first_received_at_kst"] = parse_kst(nwp["first_received_at"])
    if nwp[["prediction_issue_time_kst", "target_time_kst", "first_received_at_kst"]].isna().any().any():
        raise RuntimeError("NWP 시각 파싱 실패")
    if nwp.duplicated(["issue_date", "variable", "target_time_kst"]).any():
        raise RuntimeError("NWP 발행일·변수·대상시각 중복")
    nwp["available_at_issue"] = nwp["first_received_at_kst"] <= nwp["prediction_issue_time_kst"]
    nwp["safe_value"] = pd.to_numeric(nwp["value"], errors="coerce").where(
        nwp["available_at_issue"] & nwp["is_missing"].eq(0)
    )
    radiation = nwp["variable"].isin(["DSWRF", "DSWRFLX", "DIFSWRF"])
    nwp.loc[radiation & ~nwp["safe_value"].between(0, 2000), "safe_value"] = np.nan

    key = ["prediction_issue_time_kst", "target_time_kst"]
    nwp_long = nwp[nwp["variable"].isin(NWP_VARS)].pivot(
        index=key, columns="variable", values="safe_value"
    ).reset_index()
    nwp_long.columns.name = None
    nwp_long = nwp_long.rename(columns={v: f"forecast_{v}" for v in NWP_VARS})
    requested = (
        nwp.groupby(key, as_index=False)["requested_tm_utc"].first()
    )
    requested["requested_tm_utc"] = pd.to_datetime(
        requested["requested_tm_utc"], errors="raise", utc=True
    ).dt.strftime("%Y%m%d%H%M").astype("int64")
    live = nwp_long.merge(requested, on=key, how="left", validate="one_to_one")

    grid["prediction_issue_time_kst"] = parse_kst(grid["issue_date"].astype(str) + "T10:00:00+09:00")
    grid["run_time_kst_parsed"] = parse_kst(grid["run_time_kst"])
    grid["target_time_kst"] = parse_kst(grid["target_time_kst"])
    grid["first_received_at_kst"] = parse_kst(grid["first_received_at"])
    if grid.duplicated(["issue_date", "variable", "target_time_kst"]).any():
        raise RuntimeError("GRID 발행일·변수·대상시각 중복")
    grid["available_at_issue"] = (
        (grid["run_time_kst_parsed"] <= grid["prediction_issue_time_kst"])
        & (grid["first_received_at_kst"] <= grid["prediction_issue_time_kst"])
    )
    grid["safe_value"] = pd.to_numeric(grid["value"], errors="coerce").where(
        grid["available_at_issue"] & grid["is_missing"].eq(0)
    )
    grid_long = grid[grid["variable"].isin(GRID_VARS)].pivot(
        index=key, columns="variable", values="safe_value"
    ).reset_index()
    grid_long.columns.name = None
    grid_long = grid_long.rename(columns={v: f"forecast_{v}" for v in GRID_VARS})
    live = live.merge(grid_long, on=key, how="left", validate="one_to_one")

    power_cols = [
        "target_time_kst", "valid_5min_slots", "valid_ratio", "plant_ac_power_kw",
        "hourly_energy_kwh", "solar_elevation_deg", "physical_daylight", "quality_status",
    ]
    live = live.merge(power[power_cols], on="target_time_kst", how="left", validate="many_to_one")

    asos["observation_time_kst"] = parse_kst(asos["observation_time"])
    asos["first_received_at_kst"] = parse_kst(asos["first_received_at"])
    issue_asos = build_issue_asos(live[key].drop_duplicates("prediction_issue_time_kst"), asos)
    live = live.merge(issue_asos, on="prediction_issue_time_kst", how="left", validate="many_to_one")
    live = build_reference_asos(live, asos)

    # SQLite의 NULL 때문에 숫자 ASOS 열이 object로 굳지 않도록 명시 변환한다.
    # 시각 열만 제외하고 기존 결합본과 같은 숫자 의미를 유지한다.
    numeric_asos_names = [name for db, name in ASOS_MAP.items() if db != "observation_time"]
    for prefix in ("issue_asos_", "reference_target_asos_"):
        for name in numeric_asos_names:
            col = f"{prefix}{name}"
            if col in live.columns:
                live[col] = pd.to_numeric(live[col], errors="coerce")

    old_max_issue = pd.to_datetime(archive["prediction_issue_time_kst"]).max()
    live = live[live["prediction_issue_time_kst"] > old_max_issue].copy()
    live = live.reindex(columns=archive.columns)
    if live.duplicated(key).any():
        raise RuntimeError("라이브 추가분 발행·대상시각 중복")
    overlap = archive[key].merge(live[key], on=key, how="inner")
    if not overlap.empty:
        raise RuntimeError(f"기존 결합본과 라이브 추가분 중복 {len(overlap)}행")

    # 기존 스키마에서 결측을 허용하는 숫자열은 dtype이 자연스럽게 float로
    # 유지된다. requested_tm_utc는 모든 라이브 행에서 존재해야 한다.
    if live["requested_tm_utc"].isna().any():
        raise RuntimeError("requested_tm_utc 결측")
    live["requested_tm_utc"] = live["requested_tm_utc"].astype(archive["requested_tm_utc"].dtype)
    combined = pd.concat([archive, live], ignore_index=True)
    combined = combined.sort_values(key).reset_index(drop=True)
    if list(combined.columns) != list(archive.columns):
        raise RuntimeError("기존 결합본과 출력 컬럼 스키마 불일치")

    issue_obs_leak = int(
        (live["observation_time_kst"].notna()
         & (live["observation_time_kst"] > live["prediction_issue_time_kst"])).sum()
    )
    invalid_target = int((live["target_time_kst"] <= live["prediction_issue_time_kst"]).sum())
    if issue_obs_leak or invalid_target:
        raise RuntimeError("발행시각 누출 감사 실패")

    weather_cols = [f"forecast_{v}" for v in WEATHER8]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    live.to_parquet(a.output_dir / LIVE_NAME, index=False)
    combined.to_parquet(a.output_dir / OUT_NAME, index=False)
    daily_candidate.to_csv(a.output_dir / DAILY_NAME, index=False, encoding="utf-8-sig")
    summary = {
        "status": "candidate_for_walk_forward_only",
        "base_rows": int(len(archive)),
        "live_added_rows": int(len(live)),
        "combined_rows": int(len(combined)),
        "base_issue_days": int(archive["prediction_issue_time_kst"].nunique()),
        "live_issue_days": int(live["prediction_issue_time_kst"].nunique()),
        "live_issue_range": [str(live["prediction_issue_time_kst"].min()), str(live["prediction_issue_time_kst"].max())],
        "live_power_available_rows": int(live["plant_ac_power_kw"].notna().sum()),
        "live_issue_asos_available_rows": int(live["observation_time_kst"].notna().sum()),
        "live_complete_weather8_rows": int(live[weather_cols].notna().all(axis=1).sum()),
        "live_complete_weather8_issue_days": int(
            live.loc[live[weather_cols].notna().all(axis=1), "prediction_issue_time_kst"].nunique()
        ),
        "daily_candidate_rows": int(len(daily_candidate)),
        "daily_candidate_valid_rows": int(daily_candidate["daily_energy_kwh"].notna().sum()),
        "daily_candidate_latest_valid_day": (
            str(daily_candidate.loc[daily_candidate["daily_energy_kwh"].notna(), "date_kst"].max())
            if daily_candidate["daily_energy_kwh"].notna().any() else None
        ),
        "forecast_missing_by_variable": {c: int(live[c].isna().sum()) for c in weather_cols},
        "nwp_cells_excluded_received_after_issue": int((~nwp["available_at_issue"]).sum()),
        "grid_cells_excluded_received_after_issue": int((~grid["available_at_issue"]).sum()),
        "future_leak_issue_asos_rows": issue_obs_leak,
        "invalid_target_order_rows": invalid_target,
        "reference_target_asos_is_model_input": False,
        "api_calls": 0,
        "existing_archive_modified": False,
        "official_model_or_scheduler_changed": False,
        "output": str(a.output_dir / OUT_NAME),
        "daily_output": str(a.output_dir / DAILY_NAME),
        "consumer_path_note": "medium_term_daily_v1_buan은 JOIN_PARQUET과 DAILY_ENERGY_CSV 두 경로를 모두 후보 경로로 바꿔야 함",
    }
    (a.output_dir / "부안_과거발전_기상결합_라이브연계_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
