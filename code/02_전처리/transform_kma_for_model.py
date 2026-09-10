"""기상청 ASOS 시간·일자료를 모델용 표준자료로 변환하고 발전자료와 병합한다.

주의:
- ASOS는 과거 관측자료다. 미래 목표시각의 입력으로 사용하지 않는다.
- 기존 Open-Meteo 미래예보 열은 출처를 이름에 명시한 채 임시 유지한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "gwangju_pv_multihorizon_v3_workspace" / "outputs" / "gwangju_1hour_model_dataset.csv"
HOURLY_SOURCE = ROOT / "processed" / "기상청_ASOS156_시간환경_20240825_20260804.csv"
DAILY_SOURCE = ROOT / "processed" / "기상청_ASOS156_일환경_20240825_20260804.csv"
HOURLY_OUT = ROOT / "processed" / "기상청_ASOS156_시간환경_모델용.csv"
DAILY_OUT = ROOT / "processed" / "기상청_ASOS156_일환경_모델용.csv"
MERGED_OUT = ROOT / "processed" / "gwangju_1hour_model_dataset_kma_observed.csv"
AUDIT_OUT = ROOT / "processed" / "기상청_ASOS_모델변환_감사.json"


def numeric(frame: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    for name in frame.columns:
        if name not in excluded:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def load_clean_daily() -> pd.DataFrame:
    daily = pd.read_csv(DAILY_SOURCE, low_memory=False)
    daily["날짜"] = pd.to_datetime(daily["날짜"])
    daily = numeric(daily, {"날짜", "지점명", "일기현상"})
    # 기상청 일자료는 무강수·무적설일에 빈칸을 사용한다.
    for name in ["일강수량_mm", "최심신적설_cm", "최심적설_cm", "합계3시간신적설_cm"]:
        if name in daily:
            daily[name] = daily[name].fillna(0)
    daily["합계일사량_kWh_m2"] = daily["합계일사량_MJ_m2"] / 3.6
    return daily.sort_values("날짜").reset_index(drop=True)


def load_clean_hourly(valid_solar_dates: set[pd.Timestamp]) -> pd.DataFrame:
    hourly = pd.read_csv(HOURLY_SOURCE, low_memory=False)
    hourly["시각"] = pd.to_datetime(hourly["시각"])
    hourly = numeric(hourly, {"시각", "지점명", "운형약어"})
    # 강수·적설은 관측 종료시각 직전 구간의 값이다. 빈칸=현상 없음은 일자료와
    # 하루씩 전수 대조해 확인했으므로 0으로 변환한다.
    for name in ["강수량_mm", "적설_cm", "3시간신적설_cm"]:
        hourly[name] = hourly[name].fillna(0)
    # 일사·일조 빈칸은 공식 일합계가 존재하는 날에만 0으로 바꾼다.
    interval_date = (hourly["시각"] - pd.Timedelta(hours=1)).dt.normalize()
    valid = interval_date.isin(valid_solar_dates)
    for name in ["일사량_MJ_m2", "일조시간_hr"]:
        hourly.loc[valid & hourly[name].isna(), name] = 0
    hourly["일사량_W_m2"] = hourly["일사량_MJ_m2"] * (1_000_000 / 3600)
    hourly["전운량_pct"] = hourly["전운량_10분위"] * 10
    hourly["중하층운량_pct"] = hourly["중하층운량_10분위"] * 10
    hourly["강수귀속일"] = interval_date
    return hourly.sort_values("시각").reset_index(drop=True)


def cross_check(hourly: pd.DataFrame, daily: pd.DataFrame) -> dict:
    grouped = hourly.groupby("강수귀속일").agg(
        시간강수합계_mm=("강수량_mm", "sum"),
        시간일사합계_MJ_m2=("일사량_MJ_m2", lambda value: value.sum(min_count=1)),
        시간일조합계_hr=("일조시간_hr", lambda value: value.sum(min_count=1)),
    )
    joined = daily.set_index("날짜").join(grouped, how="left")
    checks = {}
    for name, official, hourly_name in [
        ("강수량", "일강수량_mm", "시간강수합계_mm"),
        ("일사량", "합계일사량_MJ_m2", "시간일사합계_MJ_m2"),
        ("일조시간", "합계일조시간_hr", "시간일조합계_hr"),
    ]:
        part = joined[[official, hourly_name]].dropna()
        difference = part[hourly_name] - part[official]
        checks[name] = {
            "비교일수": len(part),
            "평균절대차이": float(difference.abs().mean()),
            "최대절대차이": float(difference.abs().max()),
        }
    return checks


def merge_model_data(hourly: pd.DataFrame) -> pd.DataFrame:
    model = pd.read_csv(SOURCE, parse_dates=["time"], low_memory=False)
    # 잘못 관측값처럼 이름 붙었던 Open-Meteo historical forecast 열은 제거한다.
    old_observed = [name for name in model.columns if name.startswith("observed_weather_")]
    model = model.drop(columns=old_observed)
    # 미래예보는 아직 Open-Meteo이므로 출처를 열 이름에 명시한다.
    model = model.rename(columns={
        name: f"open_meteo_{name}"
        for name in model.columns
        if name.startswith("forecast_")
    })
    selected = hourly[[
        "시각", "기온_C", "상대습도_pct", "강수량_mm", "전운량_pct",
        "일사량_W_m2", "일조시간_hr", "풍속_m_s", "풍향_deg",
        "현지기압_hPa", "해면기압_hPa", "적설_cm", "지면온도_C",
    ]].copy()
    selected = selected.rename(columns={
        "시각": "time",
        **{name: f"기상청관측_{name}" for name in selected.columns if name != "시각"},
    })
    merged = model.merge(selected, on="time", how="left", validate="one_to_one")
    return merged


def main() -> None:
    daily = load_clean_daily()
    valid_solar_dates = set(daily.loc[daily["합계일사량_MJ_m2"].notna(), "날짜"])
    hourly = load_clean_hourly(valid_solar_dates)
    merged = merge_model_data(hourly)
    checks = cross_check(hourly, daily)
    hourly.to_csv(HOURLY_OUT, index=False, encoding="utf-8-sig")
    daily.to_csv(DAILY_OUT, index=False, encoding="utf-8-sig")
    merged.to_csv(MERGED_OUT, index=False, encoding="utf-8-sig")
    kma_columns = [name for name in merged.columns if name.startswith("기상청관측_")]
    payload = {
        "시간자료행수": len(hourly),
        "일자료행수": len(daily),
        "발전자료병합행수": len(merged),
        "발전자료병합시작": merged["time"].min().isoformat(),
        "발전자료병합종료": merged["time"].max().isoformat(),
        "기상청관측열수": len(kma_columns),
        "기상청관측완전매칭행수": int(merged[kma_columns].notna().all(axis=1).sum()),
        "삭제한오해소지열": "observed_weather_* (실제 출처 Open-Meteo historical forecast)",
        "임시유지미래예보": "open_meteo_forecast_*_previous_day1/2",
        "시간일자료교차검사": checks,
        "누출방지": "모델은 기상청 관측값을 최소 1시간 지연한 과거특성으로만 사용",
    }
    AUDIT_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
