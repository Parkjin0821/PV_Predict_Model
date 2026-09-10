"""김제 시간정렬 v1(1단계, 08-31 완료) → 태양고도·야간0·1시간·일간 집계.

부안 템플릿(build_buan_time_aggregates_v1_2026-08-28.py)을 그대로 재사용 -
인버터수(8→10)·plant_id(16783→7018)·좌표만 김제 값으로 교체했다.
API 호출 없음, 원본 정렬 parquet 수정 없음(읽기전용) - 기상(ASOS/NWP/GRID)
백필과 완전히 독립적이라 그것들이 끝나기 전에 먼저 실행할 수 있다.

이 스크립트가 만드는 일간 공식후보 CSV는 결함구간 감사
(audit_gimje_season_testrows_v1_2026-08-31.py, 부안의 audit_buan_
season_testrows_v1_2026-08-28.py와 동일 패턴)의 입력이 된다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PLANT_ID = 7018
LAT = 35.80026670423991
LON = 126.851859588009
TZ_OFFSET_H = 9
EXPECTED_INVERTERS = tuple(range(1, 11))  # 김제 10대(부안은 8대)
DAYLIGHT_DAILY_MIN_RATIO = 0.90
HOURLY_MIN_5MIN_SLOTS = 9

VALID_INVERTER_QUALITY_STATUSES = {
    "observed",
    "physical_zero_night",
    "physical_zero_idle_supported",
    "interpolated_le10min",
}
PLANT_FIVE_QUALITY_STATUSES = (
    "complete_observed",
    "complete_with_short_interpolation",
    "complete_with_night_zero",
    "complete_with_idle_zero",
    "incomplete_invalid_quality",
    "incomplete_no_official_target",
)

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제"
    r"\시간정렬_v1_2026-08-31"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제"
    r"\시간집계_v1_2026-08-31"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="김제 태양고도·야간0·시간/일간 집계")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def solar_elevation_deg(ts: pd.DatetimeIndex) -> np.ndarray:
    """기존 광주/부안과 동일한 NOAA 근사식을 김제 좌표로 재사용."""
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    time_offset = eqtime + 4 * LON - 60 * TZ_OFFSET_H
    tst = hour * 60 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)
    lat_r = np.deg2rad(LAT)
    cos_zenith = (
        np.sin(lat_r) * np.sin(decl)
        + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    )
    return 90.0 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))


def apply_night_zero(aligned: pd.DataFrame) -> pd.DataFrame:
    x = aligned.copy()
    x["grid_time_kst"] = pd.to_datetime(x["grid_time_kst"])
    unique_grid = pd.DatetimeIndex(sorted(x["grid_time_kst"].unique()))
    elev_map = pd.Series(solar_elevation_deg(unique_grid), index=unique_grid)
    x["solar_elevation_deg"] = x["grid_time_kst"].map(elev_map)
    x["physical_night"] = x["solar_elevation_deg"].le(0)

    # 김제 1단계 산출물은 야간에도 실측 0이 상당수 관측돼있다(인버터가
    # 밤에도 0을 정상 보고) - 이미 값이 있는 행은 건드리지 않고, NaN인
    # 행만 물리적 야간일 때 0으로 채운다(부안과 동일 원칙, 임의보간 아님).
    ac_claim = x["physical_night"] & x["ac_power_kw"].isna()
    dc_claim = x["physical_night"] & x["dc_power_kw"].isna()
    x["night_zero_ac"] = ac_claim
    x["night_zero_dc"] = dc_claim
    x.loc[ac_claim, "ac_power_kw"] = 0.0
    x.loc[dc_claim, "dc_power_kw"] = 0.0
    x["night_zero_any"] = ac_claim | dc_claim

    # 정렬 단계 quality_status를 원문 그대로 보존한다. 두 불리언으로
    # 재계산하면 향후 통신결측·모호한 0 상태가 집계에서 유실된다.
    x["quality_status_before_night"] = x["quality_status"].astype("string")
    x["quality_status_after_night"] = x["quality_status_before_night"].copy()
    x.loc[x["night_zero_any"], "quality_status_after_night"] = "physical_zero_night"
    return x


def build_plant_five(x: pd.DataFrame) -> pd.DataFrame:
    ac = x.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    dc = x.pivot(index="grid_time_kst", columns="inverter_number", values="dc_power_kw")
    quality = x.pivot(
        index="grid_time_kst", columns="inverter_number", values="quality_status_after_night"
    )
    for n in EXPECTED_INVERTERS:
        if n not in ac:
            raise RuntimeError(f"인버터 {n}번 누락")
    n_inv = len(EXPECTED_INVERTERS)
    valid_quality = quality.isin(VALID_INVERTER_QUALITY_STATUSES)
    ac_valid = ac.where(valid_quality)
    dc_valid = dc.where(valid_quality)
    out = pd.DataFrame(index=ac.index)
    out["plant_id"] = PLANT_ID
    out["solar_elevation_deg"] = solar_elevation_deg(pd.DatetimeIndex(out.index))
    out["physical_daylight"] = out["solar_elevation_deg"].gt(0)
    out["available_inverter_count"] = ac_valid.notna().sum(axis=1).astype("int8")
    out["observed_inverter_count"] = quality.eq("observed").sum(axis=1).astype("int8")
    out["interpolated_inverter_count"] = quality.eq("interpolated_le10min").sum(axis=1).astype("int8")
    out["night_zero_inverter_count"] = quality.eq("physical_zero_night").sum(axis=1).astype("int8")
    out["idle_zero_inverter_count"] = quality.eq("physical_zero_idle_supported").sum(axis=1).astype("int8")
    out["invalid_quality_inverter_count"] = (~quality.isin(VALID_INVERTER_QUALITY_STATUSES)).sum(axis=1).astype("int8")
    for n in EXPECTED_INVERTERS:
        out[f"inverter_{n}_quality_status"] = quality[n].astype("string")
    out["complete_all_inverters"] = out["available_inverter_count"].eq(n_inv)
    out["plant_ac_power_kw"] = ac_valid.sum(axis=1, min_count=n_inv)
    out["plant_dc_power_kw"] = dc_valid.sum(axis=1, min_count=n_inv)
    out["quality_status"] = np.select(
        [
            out["complete_all_inverters"] & out["invalid_quality_inverter_count"].gt(0),
            out["complete_all_inverters"] & out["night_zero_inverter_count"].gt(0),
            out["complete_all_inverters"] & out["idle_zero_inverter_count"].gt(0),
            out["complete_all_inverters"] & out["interpolated_inverter_count"].gt(0),
            out["complete_all_inverters"],
        ],
        [
            "incomplete_invalid_quality",
            "complete_with_night_zero",
            "complete_with_idle_zero",
            "complete_with_short_interpolation",
            "complete_observed",
        ],
        default="incomplete_no_official_target",
    )
    return out.reset_index()


def build_hourly(plant5: pd.DataFrame) -> pd.DataFrame:
    x = plant5.set_index("grid_time_kst").sort_index()
    g = x.resample("1h")
    out = pd.DataFrame(index=g.size().index)
    valid = x["plant_ac_power_kw"].notna().resample("1h").sum()
    out["valid_5min_slots"] = valid.astype("int8")
    out["valid_ratio"] = valid / 12.0
    out["plant_ac_power_kw"] = g["plant_ac_power_kw"].mean()
    out.loc[valid < HOURLY_MIN_5MIN_SLOTS, "plant_ac_power_kw"] = np.nan
    out["hourly_energy_kwh"] = out["plant_ac_power_kw"]
    out["solar_elevation_deg"] = g["solar_elevation_deg"].mean()
    out["physical_daylight"] = g["physical_daylight"].max().astype(bool)
    for status in PLANT_FIVE_QUALITY_STATUSES:
        out[f"source_{status}_slots"] = x["quality_status"].eq(status).resample("1h").sum().astype("int8")
    out["quality_status"] = np.where(
        valid >= HOURLY_MIN_5MIN_SLOTS, "valid_ge9of12", "invalid_lt9of12"
    )
    return out.reset_index()


def build_daily(plant5: pd.DataFrame) -> pd.DataFrame:
    x = plant5.set_index("grid_time_kst").sort_index()
    x["energy_5min_kwh"] = x["plant_ac_power_kw"] * (5.0 / 60.0)
    day = x.index.floor("D")
    expected_day = x["physical_daylight"].groupby(day).sum()
    valid_day = (x["physical_daylight"] & x["plant_ac_power_kw"].notna()).groupby(day).sum()
    ratio = valid_day.div(expected_day.replace(0, np.nan))
    energy = x["energy_5min_kwh"].groupby(day).sum(min_count=1)
    out = pd.DataFrame({
        "date_kst": expected_day.index,
        "daylight_expected_slots": expected_day.to_numpy(),
        "daylight_valid_slots": valid_day.to_numpy(),
        "daylight_valid_ratio": ratio.to_numpy(),
        "daily_energy_kwh": energy.reindex(expected_day.index).to_numpy(),
    })
    for status in PLANT_FIVE_QUALITY_STATUSES:
        counts = x["quality_status"].eq(status).groupby(day).sum()
        out[f"source_{status}_slots"] = counts.reindex(expected_day.index).to_numpy()
    good = out["daylight_valid_ratio"].ge(DAYLIGHT_DAILY_MIN_RATIO)
    out.loc[~good, "daily_energy_kwh"] = np.nan
    out["quality_status"] = np.where(good, "valid_daylight_ge90pct", "invalid_daylight_lt90pct")
    return out


def main() -> None:
    args = parse_args()
    src = args.input_dir / "김제_인버터별_5분정렬.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_parquet(src)
    required = {
        "grid_time_kst", "inverter_number", "ac_power_kw", "dc_power_kw",
        "was_observed", "was_interpolated", "quality_status",
    }
    missing = sorted(required - set(aligned.columns))
    if missing:
        raise RuntimeError(f"입력 정렬자료 필수열 누락: {missing}")

    inv5 = apply_night_zero(aligned)
    plant5 = build_plant_five(inv5)
    hourly = build_hourly(plant5)
    daily = build_daily(plant5)

    # 공식 총출력은 반드시 10대 합과 같아야 한다(부분합 스케일업 금지 재검증).
    pivot = inv5.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    quality = inv5.pivot(index="grid_time_kst", columns="inverter_number", values="quality_status_after_night")
    expected = pivot.where(quality.isin(VALID_INVERTER_QUALITY_STATUSES)).sum(
        axis=1, min_count=len(EXPECTED_INVERTERS)
    )
    actual = plant5.set_index("grid_time_kst")["plant_ac_power_kw"]
    common = expected.dropna().index.intersection(actual.dropna().index)
    if not np.allclose(expected.loc[common], actual.loc[common], rtol=0, atol=1e-9):
        raise RuntimeError("야간0 적용 후 인버터합계 검증 실패")

    inv5.to_parquet(args.output_dir / "김제_인버터별_5분_야간0포함.parquet", index=False)
    plant5.to_parquet(args.output_dir / "김제_발전소_5분_공식후보.parquet", index=False)
    hourly.to_parquet(args.output_dir / "김제_발전소_1시간_공식후보.parquet", index=False)
    daily.to_csv(args.output_dir / "김제_발전소_일간_공식후보.csv", index=False, encoding="utf-8-sig")

    daylight = plant5["physical_daylight"]
    summary = {
        "status": "time_aggregate_candidate_v1",
        "rules": {
            "night_zero": "only_missing_ac_dc_power_when_solar_elevation_le_0",
            "hourly_gate": f"valid_5min_slots>={HOURLY_MIN_5MIN_SLOTS}_of_12",
            "daily_gate": f"physical_daylight_valid_ratio>={DAYLIGHT_DAILY_MIN_RATIO}",
            "partial_scaling": "forbidden",
            "expected_inverter_count": len(EXPECTED_INVERTERS),
        },
        "five_min": {
            "rows": int(len(plant5)),
            "complete_all_pct": float(100 * plant5["complete_all_inverters"].mean()),
            "daylight_rows": int(daylight.sum()),
            "daylight_complete_pct": float(100 * plant5.loc[daylight, "complete_all_inverters"].mean()),
            "night_rows": int((~daylight).sum()),
            "night_zero_rows": int(plant5["night_zero_inverter_count"].gt(0).sum()),
            "incomplete_daylight_rows": int((daylight & ~plant5["complete_all_inverters"]).sum()),
        },
        "hourly": {
            "rows": int(len(hourly)),
            "valid_rows": int(hourly["plant_ac_power_kw"].notna().sum()),
            "valid_pct": float(100 * hourly["plant_ac_power_kw"].notna().mean()),
        },
        "daily": {
            "rows": int(len(daily)),
            "valid_rows": int(daily["daily_energy_kwh"].notna().sum()),
            "valid_pct": float(100 * daily["daily_energy_kwh"].notna().mean()),
        },
        "not_done": [
            "ASOS/NWP/GRID 발행시각·대상시각 결합(기상 백필 완료 대기중)",
            "결함구간 확정(감사 후보만 나옴, 사람 확인 필요 - 부안과 동일 절차)",
            "모델 학습·검증",
        ],
    }
    (args.output_dir / "김제_시간집계_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
