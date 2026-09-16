"""김제·영광 D+1 재검증용 발전량·ASOS·NWP·GRID 라이브 연계 후보."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
OUT_ROOT = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델")
CFG = {
    "김제": {
        "plant_id": 7018, "station": 243, "slug": "gimje",
        "archive": OUT_ROOT / r"김제\과거발전_기상결합_v1_2026-08-31\김제_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet",
        "static_daily": OUT_ROOT / r"김제\시간집계_v1_2026-08-31\김제_발전소_일간_공식후보.csv",
        "block_db": OUT_ROOT / r"김제\blockdata_live_history_gimje_v1_2026-09-01\blockdata_history.sqlite3",
        "kma_db": OUT_ROOT / r"김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",
        "aggregate": ROOT / r"02_전처리\김제\build_gimje_time_aggregates_v1_2026-08-31.py",
    },
    "영광": {
        "plant_id": 7912, "station": 252, "slug": "yeonggwang",
        "archive": OUT_ROOT / r"영광\과거발전_기상결합_v1_2026-09-03\영광_과거발전_ASOS_NWP_GRID_결합_v1_2026-09-03.parquet",
        "static_daily": OUT_ROOT / r"영광\시간집계_v1_2026-09-01\영광_발전소_일간_공식후보.csv",
        "block_db": OUT_ROOT / r"영광\blockdata_live_history_yeonggwang_v1_2026-09-08\blockdata_history.sqlite3",
        "kma_db": OUT_ROOT / r"영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",
        "aggregate": ROOT / r"02_전처리\영광\build_yeonggwang_time_aggregates_v1_2026-09-01.py",
    },
}
NWP_VARS = ("DSWRF", "DSWRFLX", "DIFSWRF", "TCDC", "LCDC", "MCDC", "HCDC")
GRID_VARS = ("REH", "POP", "SKY")
WEATHER8 = ("DSWRF", "TCDC", "LCDC", "MCDC", "HCDC", "REH", "POP", "SKY")
ASOS_MAP = {
    "station": "지점번호", "observation_time": "시각", "temperature_c": "기온_C",
    "rainfall_mm": "강수량_mm", "wind_speed_m_s": "풍속_m_s", "wind_direction_deg": "풍향_deg",
    "humidity_pct": "상대습도_pct", "local_pressure_hpa": "현지기압_hPa",
    "sea_pressure_hpa": "해면기압_hPa", "sunshine_hr": "일조시간_hr",
    "solar_mj_m2": "일사량_MJ_m2", "solar_w_m2": "일사량_W_m2", "snow_cm": "적설_cm",
    "cloud_tenths": "전운량_10분위", "cloud_pct": "전운량_pct", "ground_temperature_c": "지면온도_C",
}


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=CFG)
    return p.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("regional_aggregate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def sql_ro(path: Path, sql: str, params=()):
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try: return pd.read_sql_query(sql, con, params=params)
    finally: con.close()


def parse_kst(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)


def build_live_power(c, agg):
    raw = sql_ro(c["block_db"], """SELECT inverter_number,measurement_time,ac_power,dc_power,quality_status
        FROM inverter_measurements WHERE plant_id=? ORDER BY measurement_time,inverter_number""", (c["plant_id"],))
    if raw.empty: raise RuntimeError("Blockdata 인버터 이력 없음")
    raw["measurement_time_kst"] = parse_kst(raw["measurement_time"])
    raw["grid_time_kst"] = raw["measurement_time_kst"].dt.round("5min")
    raw["offset"] = (raw["measurement_time_kst"] - raw["grid_time_kst"]).abs().dt.total_seconds()
    raw = raw.sort_values(["inverter_number", "grid_time_kst", "offset", "measurement_time_kst"])
    raw = raw.drop_duplicates(["inverter_number", "grid_time_kst"], keep="first")
    start, end = raw["grid_time_kst"].min(), raw["grid_time_kst"].max()
    grid = pd.MultiIndex.from_product(
        [pd.date_range(start, end, freq="5min"), agg.EXPECTED_INVERTERS],
        names=["grid_time_kst", "inverter_number"],
    ).to_frame(index=False)
    x = grid.merge(raw, on=["grid_time_kst", "inverter_number"], how="left", validate="one_to_one")
    x["ac_power_kw"] = pd.to_numeric(x["ac_power"], errors="coerce")
    x["dc_power_kw"] = pd.to_numeric(x["dc_power"], errors="coerce")
    # ★★09-16 수정★★: 기존엔 그 슬롯의 인버터 전수(count==10/8/13)가
    # 보고돼야만 그중 한 대라도 "observed"로 인정했다 - 1대만 빠져도
    # 나머지가 전부 정상 실측이었어도 통째로 지워버려, agg.build_plant_five()
    # 로 넘어오는 시점엔 이미 "전부 있거나 전부 없거나"만 남았다(김제
    # 인버터별 보정 로직 검증 중 발견 - 보정치가 전수게이트 비율과 거의
    # 동일하게 나와서 알아챔). inverter_measurements.quality_status는
    # collect_blockdata_history의 inverter_quality_flags가 인버터 각자
    # 자기 값(ac_power 존재·범위)만 보고 매기는 진짜 개별값이므로, 그룹
    # 게이트 없이 "자기 값이 있는지"만으로 observed를 정한다.
    # 전수완전성(complete_all_inverters)·정식 daily_energy_kwh(min_count=
    # 전수)는 agg.build_plant_five()가 이미 별도로 강제하므로 여기서
    # 미리 지울 필요가 없다 - 영광 재검증 결과 daily_energy_kwh 5일 값
    # 완전 동일(회귀 없음), 김제는 부분관측이 처음으로 인버터별
    # 보정 단계에 도달함.
    has_value = x["ac_power_kw"].notna()
    source_ok = x["quality_status"].isin(["ok", "warning"])
    x["quality_status"] = np.where(has_value & source_ok, "observed", "missing")
    x["was_observed"] = x["quality_status"].eq("observed")
    x["was_interpolated"] = False
    x.loc[~x["was_observed"], ["ac_power_kw", "dc_power_kw"]] = np.nan
    inv5 = agg.apply_night_zero(x)
    plant5 = agg.build_plant_five(inv5)
    return inv5, plant5, agg.build_hourly(plant5), agg.build_daily(plant5)


def issue_asos(issues, asos):
    rows=[]
    for issue in issues:
        eligible=asos[(asos.observation_time_kst<=issue)&(asos.first_received_at_kst<=issue)&(asos.observation_time_kst>=issue-pd.Timedelta("3h"))]
        rec={"prediction_issue_time_kst":issue}
        if not eligible.empty:
            src=eligible.sort_values("observation_time_kst").iloc[-1]
            for db, old in ASOS_MAP.items(): rec[f"issue_asos_{old}"]=src[db]
            rec["observation_time_kst"]=src.observation_time_kst
        rows.append(rec)
    return pd.DataFrame(rows)


def reference_asos(targets, asos):
    src=asos.copy(); src["target_time_kst"]=src.observation_time_kst
    keep=["target_time_kst"]+list(ASOS_MAP); rename={db:f"reference_target_asos_{old}" for db,old in ASOS_MAP.items()}
    return targets.merge(src[keep].rename(columns=rename).drop_duplicates("target_time_kst",keep="last"),on="target_time_kst",how="left",validate="many_to_one")


# ★09-15 추가(Claude, 코덱스 용량저하로 대행 리빌드 중 발견)★: 부안
# 라이브브릿지(build_buan_weather_power_live_bridge_v1_2026-09-14.py의
# correct_daily_denominator)에는 이미 있던 보정이 이 공용(김제·영광)
# 스크립트에는 빠져 있었음 - agg.build_daily()가 "그날 지금까지 들어온
# 슬롯 수"를 분모로 써서 진행 중인 당일이 우연히 전부 유효하면 부분값을
# 100%로 오판정할 수 있는 잠재 결함(09-15 실행 시점엔 두 지역 다 아침
# 결측이 섞여 있어 실제로는 발동 안 했음 - 우연히 피함). 부안와 동일한
# 로직(날짜별 전체 일광 5분 슬롯을 solar_elevation_deg로 재계산)을
# 그대로 재사용해 재구현 없이 이식.
def correct_daily_denominator(daily: pd.DataFrame, agg) -> pd.DataFrame:
    daily = daily.copy()
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
    daily["quality_status"] = np.where(good, "valid_daylight_ge90pct", "invalid_daylight_lt90pct")
    return daily


def main():
    region=cli().region; c=CFG[region]; agg=load_module(c["aggregate"])
    archive=pd.read_parquet(c["archive"]); inv5,plant5,hourly,live_daily=build_live_power(c,agg)
    hourly=hourly.rename(columns={"grid_time_kst":"target_time_kst"})
    nwp=sql_ro(c["kma_db"],"SELECT * FROM nwp_values"); grid=sql_ro(c["kma_db"],"SELECT * FROM grid_forecast")
    asos=sql_ro(c["kma_db"],"SELECT * FROM asos_hourly WHERE station=?",(c["station"],))
    if nwp.empty or grid.empty or asos.empty: raise RuntimeError("라이브 기상 필수 테이블 비어 있음")
    nwp["prediction_issue_time_kst"]=parse_kst(nwp.planned_issue_at_kst); nwp["target_time_kst"]=parse_kst(nwp.target_time_kst); nwp["first_received_at_kst"]=parse_kst(nwp.first_received_at)
    nwp["available"]=nwp.first_received_at_kst<=nwp.prediction_issue_time_kst
    nwp["safe_value"]=pd.to_numeric(nwp.value,errors="coerce").where(nwp.available & nwp.is_missing.eq(0))
    nwp.loc[nwp.variable.isin(["DSWRF","DSWRFLX","DIFSWRF"]) & ~nwp.safe_value.between(0,2000),"safe_value"]=np.nan
    key=["prediction_issue_time_kst","target_time_kst"]
    nl=nwp[nwp.variable.isin(NWP_VARS)].pivot(index=key,columns="variable",values="safe_value").reset_index().rename(columns={v:f"forecast_{v}" for v in NWP_VARS})
    req=nwp.groupby(key,as_index=False).requested_tm_utc.first(); req["requested_tm_utc"]=pd.to_datetime(req.requested_tm_utc,utc=True).dt.strftime("%Y%m%d%H%M").astype("int64")
    live=nl.merge(req,on=key,validate="one_to_one")
    grid["prediction_issue_time_kst"]=parse_kst(grid.issue_date.astype(str)+"T10:00:00+09:00"); grid["run"]=parse_kst(grid.run_time_kst); grid["target_time_kst"]=parse_kst(grid.target_time_kst); grid["first_received_at_kst"]=parse_kst(grid.first_received_at)
    grid["available"]=(grid.run<=grid.prediction_issue_time_kst)&(grid.first_received_at_kst<=grid.prediction_issue_time_kst)
    grid["safe_value"]=pd.to_numeric(grid.value,errors="coerce").where(grid.available & grid.is_missing.eq(0))
    gl=grid[grid.variable.isin(GRID_VARS)].pivot(index=key,columns="variable",values="safe_value").reset_index().rename(columns={v:f"forecast_{v}" for v in GRID_VARS})
    live=live.merge(gl,on=key,how="left",validate="one_to_one")
    power_cols=[x for x in archive.columns if x in hourly.columns and x!="target_time_kst"]
    live=live.merge(hourly[["target_time_kst"]+power_cols],on="target_time_kst",how="left",validate="many_to_one")
    asos["observation_time_kst"]=parse_kst(asos.observation_time); asos["first_received_at_kst"]=parse_kst(asos.first_received_at)
    live=live.merge(issue_asos(live.prediction_issue_time_kst.drop_duplicates(),asos),on="prediction_issue_time_kst",how="left",validate="many_to_one")
    live=reference_asos(live,asos)
    for prefix in ("issue_asos_","reference_target_asos_"):
        for db,name in ASOS_MAP.items():
            col=prefix+name
            if db!="observation_time" and col in live: live[col]=pd.to_numeric(live[col],errors="coerce")
    old_max=pd.to_datetime(archive.prediction_issue_time_kst).max(); live=live[live.prediction_issue_time_kst>old_max].reindex(columns=archive.columns)
    if "is_defect_period" in live: live["is_defect_period"]=False
    live["requested_tm_utc"]=live.requested_tm_utc.astype(archive.requested_tm_utc.dtype)
    combined=pd.concat([archive,live],ignore_index=True).sort_values(key).reset_index(drop=True)
    if list(combined.columns)!=list(archive.columns) or combined.duplicated(key).any(): raise RuntimeError("결합 스키마/키 감사 실패")
    issue_leak=int((live.observation_time_kst.notna()&(pd.to_datetime(live.observation_time_kst)>pd.to_datetime(live.prediction_issue_time_kst))).sum())
    if issue_leak or (live.target_time_kst<=live.prediction_issue_time_kst).any(): raise RuntimeError("발행시각 누출 감사 실패")
    static_daily=pd.read_csv(c["static_daily"]); static_daily["date_kst"]=pd.to_datetime(static_daily.date_kst)
    live_daily["date_kst"]=pd.to_datetime(live_daily.date_kst); live_daily=live_daily[live_daily.date_kst>static_daily.date_kst.max()]
    # ★09-16 수정★: 기존엔 live_daily를 static_daily의 컬럼에 맞춰
    # reindex해서, static에 없는 신규 컬럼(김제 인버터보정 등)이 있으면
    # 조용히 버려졌다. 두 프레임 컬럼의 합집합으로 맞춰 신규 컬럼이
    # 살아남게 한다(static 쪽은 그 컬럼이 NaN으로 채워짐 - 과거 정적
    # 구간엔 보정치가 없다는 뜻이라 정확한 표현).
    union_cols=list(dict.fromkeys(list(static_daily.columns)+list(live_daily.columns)))
    daily=pd.concat([static_daily.reindex(columns=union_cols),live_daily.reindex(columns=union_cols)],ignore_index=True).drop_duplicates("date_kst",keep="first").sort_values("date_kst")
    daily=correct_daily_denominator(daily,agg)
    if hasattr(agg,"finalize_corrected_gate"): daily=agg.finalize_corrected_gate(daily)
    out=OUT_ROOT/region/f"과거발전_기상결합_라이브연계_v1_2026-09-14"; out.mkdir(parents=True,exist_ok=True)
    combined_path=out/f"{region}_과거발전_ASOS_NWP_GRID_결합_라이브연계_v1_2026-09-14.parquet"; daily_path=out/f"{region}_발전소_일간_라이브연계_D1검증용.csv"
    combined.to_parquet(combined_path,index=False); live.to_parquet(out/f"{region}_라이브추가분_ASOS_NWP_GRID_결합_v1_2026-09-14.parquet",index=False); daily.to_csv(daily_path,index=False,encoding="utf-8-sig")
    wc=[f"forecast_{v}" for v in WEATHER8]
    summary={"status":"candidate_for_walk_forward_only","region":region,"base_rows":len(archive),"live_added_rows":len(live),"combined_rows":len(combined),"live_issue_days":live.prediction_issue_time_kst.nunique(),"live_issue_range":[str(live.prediction_issue_time_kst.min()),str(live.prediction_issue_time_kst.max())],"live_power_available_rows":int(live.plant_ac_power_kw.notna().sum()),"live_complete_weather8_rows":int(live[wc].notna().all(axis=1).sum()),"daily_added_rows":len(live_daily),"daily_added_valid_rows":int(live_daily.daily_energy_kwh.notna().sum()),"nwp_cells_excluded_received_after_issue":int((~nwp.available).sum()),"grid_cells_excluded_received_after_issue":int((~grid.available).sum()),"future_leak_issue_asos_rows":issue_leak,"api_calls":0,"official_model_or_scheduler_changed":False,"output":str(combined_path),"daily_output":str(daily_path)}
    (out/f"{region}_과거발전_기상결합_라이브연계_요약.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
