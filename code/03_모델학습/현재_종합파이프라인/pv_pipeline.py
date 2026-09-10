"""광주 태양광 오프라인 모델의 공통 자료 처리 함수."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
# 08-20 경로 정비: 이 두 상수는 원래 존재하지 않는 폴더
# (`WORKSPACE/gwangju_pv_multihorizon_v3_workspace`,
# `WORKSPACE/결과물/...`)를 가리키고 있어 실행 불가였다. 실제 파일 위치로
# 교정했다(AGENTS.md "2026-08-20 낮" 절 참고). 5분 원자료는
# `gwangju_5min_model_dataset.csv`이며 08-20 기준 여전히 08-19까지의 정의
# (5대 전체 평균 주파수·온도 등)를 담고 있다 — 정상 인버터만 재정의 버전을
# 쓰려면 이 자료 대신 `build_official_hourly_dataset_v1.py`가 만든 시간단위
# 통합 데이터셋을 별도로 참고할 것(이 pv_pipeline.py는 5분 원자료 정제
# 단계만 담당하며 08-20 재정의를 아직 반영하지 않음).
CODEX_RESULTS = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주")
SOURCE_V3 = CODEX_RESULTS / "v3_multihorizon_2026-08-14"
GRID_650 = (
    CODEX_RESULTS
    / "grid_forecast_650d_interim_v2_2026-08-18"
    / "forecast_650d_combined.csv"
)
OUTPUT = ROOT / "outputs"


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def solar_position(index: pd.DatetimeIndex, latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray]:
    """한국 표준시 기준 근사 태양고도와 태양방위각."""
    day = index.dayofyear.to_numpy(float)
    hour = index.hour.to_numpy(float) + index.minute.to_numpy(float) / 60
    gamma = 2 * np.pi / 365 * (day - 1 + (hour - 12) / 24)
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    solar_minutes = (hour * 60 + equation_of_time + 4 * longitude - 60 * 9) % 1440
    hour_angle = np.deg2rad(solar_minutes / 4 - 180)
    latitude_rad = math.radians(latitude)
    cosine_zenith = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    elevation = 90 - np.rad2deg(np.arccos(np.clip(cosine_zenith, -1, 1)))
    azimuth = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(latitude_rad)
        - np.tan(declination) * np.cos(latitude_rad),
    )
    return elevation, (np.rad2deg(azimuth) + 180) % 360


def load_clean_five_minute(config: dict) -> pd.DataFrame:
    """기존 원본 정제 결과를 읽고 모델 공통 5분 기준자료로 고정한다."""
    source = SOURCE_V3 / "03_분석데이터" / "gwangju_5min_model_dataset.csv"
    raw = pd.read_csv(source, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    selected = pd.DataFrame(index=raw.index)
    selected["발전출력_kW"] = pd.to_numeric(raw["plant_output_kw"], errors="coerce")
    selected["입력전력_kW"] = pd.to_numeric(raw["plant_input_power_kw"], errors="coerce")
    selected["입력전류_A"] = pd.to_numeric(raw["plant_input_current_a"], errors="coerce")
    selected["입력전압평균_V"] = pd.to_numeric(raw["mean_input_voltage_v"], errors="coerce")
    selected["주파수평균_Hz"] = pd.to_numeric(raw["mean_frequency_hz"], errors="coerce")
    selected["역률평균"] = pd.to_numeric(raw["mean_power_factor"], errors="coerce")
    selected["인버터평균온도_C"] = pd.to_numeric(raw["mean_inverter_temperature_c"], errors="coerce")
    selected["통신정상비율"] = pd.to_numeric(raw["mean_communication_ok"], errors="coerce")
    selected["가용인버터수"] = pd.to_numeric(raw["inverters_available"], errors="coerce")

    capacity = float(config["site"]["capacity_kw"])
    selected.loc[~selected["발전출력_kW"].between(0, capacity), "발전출력_kW"] = np.nan
    selected.loc[selected["입력전력_kW"] < 0, "입력전력_kW"] = np.nan
    selected.loc[~selected["인버터평균온도_C"].between(-50, 80), "인버터평균온도_C"] = np.nan

    elevation, azimuth = solar_position(
        selected.index,
        float(config["site"]["latitude"]),
        float(config["site"]["longitude"]),
    )
    selected["태양고도_deg"] = elevation
    selected["태양방위각_deg"] = azimuth
    selected["물리적낮"] = (selected["태양고도_deg"] > 0).astype("int8")
    selected["핵심낮시간"] = (selected["태양고도_deg"] > 3).astype("int8")
    selected["발전자료존재"] = selected["발전출력_kW"].notna().astype("int8")
    selected.index.name = "시각"
    return selected


def load_kma_forecast_five_minute() -> pd.DataFrame:
    """전일 05시 발표의 3시간 기상청 예보를 목표일 5분 격자로 확장한다."""
    daily = pd.read_csv(GRID_650, dtype={"발표일": str}, encoding="utf-8-sig")
    daily["발표일"] = daily["발표일"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    daily = daily.drop_duplicates("발표일", keep="last").sort_values("발표일")
    records: list[dict] = []
    for row in daily.to_dict("records"):
        issue_time = pd.to_datetime(row["발표일"], format="%Y%m%d") + pd.Timedelta(hours=5)
        target_date = issue_time.normalize() + pd.Timedelta(days=1)
        for hour in range(0, 24, 3):
            records.append(
                {
                    "예보대상시각": target_date + pd.Timedelta(hours=hour),
                    "예보발표시각": issue_time,
                    "기온예보_C": float(row[f"TMP_{hour:02d}h"]),
                    "하늘상태예보": float(row[f"SKY_{hour:02d}h"]),
                    "습도예보_pct": float(row[f"REH_{hour:02d}h"]),
                }
            )
    three_hour = pd.DataFrame(records).sort_values("예보대상시각")
    pieces = []
    for target_date, part in three_hour.groupby(three_hour["예보대상시각"].dt.normalize()):
        index = pd.date_range(target_date, target_date + pd.Timedelta(hours=23, minutes=55), freq="5min")
        expanded = part.set_index("예보대상시각").reindex(index)
        expanded["기온예보_C"] = expanded["기온예보_C"].interpolate(method="time").ffill().bfill()
        expanded["습도예보_pct"] = expanded["습도예보_pct"].interpolate(method="time").ffill().bfill()
        expanded["하늘상태예보"] = expanded["하늘상태예보"].ffill().bfill()
        expanded["예보발표시각"] = part["예보발표시각"].iloc[0]
        expanded.index.name = "예보대상시각"
        pieces.append(expanded)
    forecast = pd.concat(pieces).sort_index()
    forecast["예보운량_10분율"] = forecast["하늘상태예보"].map({1.0: 0.0, 2.0: 2.5, 3.0: 5.0, 4.0: 10.0})
    return forecast


def aggregate_power(frame: pd.DataFrame, frequency: str, required_fraction: float) -> pd.DataFrame:
    """정제된 5분 값을 평균출력으로 집계하되 충족률 미달 구간은 결측으로 둔다."""
    expected = {"15min": 3, "1h": 12}[frequency]
    count = frame["발전출력_kW"].resample(frequency).count()
    result = frame.resample(frequency).mean(numeric_only=True)
    result["발전자료개수"] = count
    result["발전자료충족률"] = count / expected
    result.loc[result["발전자료충족률"] < required_fraction, "발전출력_kW"] = np.nan
    result["발전자료존재"] = result["발전출력_kW"].notna().astype("int8")
    result.index.name = "시각"
    return result


def build_daily_actual(five_minute: pd.DataFrame) -> pd.DataFrame:
    """5분 평균출력을 kWh로 변환하여 일간 실제 발전량을 만든다."""
    work = five_minute[["발전출력_kW", "물리적낮"]].copy()
    work["5분발전량_kWh"] = work["발전출력_kW"] * (5 / 60)
    groups = work.groupby(work.index.normalize())
    daily = pd.DataFrame(index=groups.size().index)
    daily["일간발전량_kWh"] = groups["5분발전량_kWh"].sum(min_count=1)
    daylight_expected = groups["물리적낮"].sum().replace(0, np.nan)
    daylight_observed = groups.apply(
        lambda x: int(((x["물리적낮"] == 1) & x["발전출력_kW"].notna()).sum()),
        include_groups=False,
    )
    daily["낮시간예상개수"] = daylight_expected
    daily["낮시간실측개수"] = daylight_observed
    daily["낮시간자료충족률"] = daily["낮시간실측개수"] / daily["낮시간예상개수"]
    daily.loc[daily["낮시간자료충족률"] < 0.9, "일간발전량_kWh"] = np.nan
    daily.index.name = "날짜"
    return daily


def build_all() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    five = load_clean_five_minute(config)
    fifteen = aggregate_power(five, "15min", required_fraction=1.0)
    hourly = aggregate_power(five, "1h", required_fraction=0.75)
    daily = build_daily_actual(five)
    forecast = load_kma_forecast_five_minute()

    five.to_parquet(OUTPUT / "정제_5분_기준자료.parquet")
    fifteen.to_parquet(OUTPUT / "집계_15분_자료.parquet")
    hourly.to_parquet(OUTPUT / "집계_1시간_자료.parquet")
    daily.to_parquet(OUTPUT / "집계_일간_실제발전량.parquet")
    forecast.to_parquet(OUTPUT / "기상청_650일_5분확장예보.parquet")

    audit = {
        "5분": {
            "행수": int(len(five)),
            "시작": str(five.index.min()),
            "종료": str(five.index.max()),
            "발전출력_유효행": int(five["발전출력_kW"].notna().sum()),
            "발전출력_결측행": int(five["발전출력_kW"].isna().sum()),
        },
        "15분": {
            "행수": int(len(fifteen)),
            "발전출력_유효행": int(fifteen["발전출력_kW"].notna().sum()),
        },
        "1시간": {
            "행수": int(len(hourly)),
            "발전출력_유효행": int(hourly["발전출력_kW"].notna().sum()),
        },
        "일간": {
            "행수": int(len(daily)),
            "발전량_유효일": int(daily["일간발전량_kWh"].notna().sum()),
        },
        "기상청예보": {
            "발표일수": int(forecast["예보발표시각"].nunique()),
            "행수": int(len(forecast)),
            "대상시작": str(forecast.index.min()),
            "대상종료": str(forecast.index.max()),
            "결측수": int(forecast.isna().sum().sum()),
        },
        "집계원칙": "5분 원본 정제 후 순수 집계; 결측을 0으로 대체하지 않음",
    }
    (OUTPUT / "자료생성_감사.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    print(json.dumps(build_all(), ensure_ascii=False, indent=2))
