from __future__ import annotations
import importlib.util,json,sqlite3,sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib,numpy as np,pandas as pd

# ★2026-09-16★ 4지역 "+48h pooled 후보"를 라이브로 처음 조립한다.
# 배경: 09-10에 만든 live_feature_assembler_단기_v2_pooled(+24h)는 GRID/
# 정규NWP가 35h까지만 있어 +48h(37~59h)를 못 만든다. +48h는 KIM NC D1D2
# 외에는 그 리드타임 자체가 존재하지 않으므로(GRID collect_kma_asos_grid_
# live_v1의 TARGET_HOURS_KST가 당일 8시각만 수집하는 구조적 한계, AGENTS.md
# 09-16 항목 참고), NC 전용으로 새로 짠다. +24h는 이미 GRID로 4지역 다 되어
# 있으니 이 파일은 딱 +48h만 담당한다(+24h 파일과 역할 분리, 헷갈림 방지).
#
# 4지역 방식 통일 원칙(사용자 지시, 09-16): 광주도 부안/김제/영광과 같은
# NC 기반 pooled 경로에 편입한다. 단, 광주는 blockdata 스키마가 달라
# (plant_input_power_kw/mean_power_factor/mean_input_voltage_v 등, 부안/
# 김제/영광의 단순 plant_ac_power_kw와 다름) 설비·ASOS 읽기는 기존
# live_feature_assembler_단기_v1(오늘 ASOS 동적탐색으로 고친 그 파일)의
# build_live_hourly()를 그대로 재사용하고, 예보만 NC로 갈아끼운다.
#
# 광주 GRID기반 "단기 Shadow"(shadow_predict_단기_v1, UCUBE_ShadowPredict_
# Short_*)는 삭제하지 않고 그대로 둔다 - 광주는 GRID가 정상 작동하는
# 유일한 지역이라 "광주 자체 GRID후보 vs NC pooled후보" 비교용으로
# 계속 쌓아둘 가치가 있음(사용자 확인, 09-16).

ROOT=Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
NCDIR=Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델")
KST=ZoneInfo("Asia/Seoul")
DB=ROOT/"shadow_predictions_short_pooled.sqlite3"; H=range(37,60)
ASOS_LOOKBACK_HOURS=12  # 09-16 광주 ASOS 지연 실측(p95=552분/9시간+) 근거, 단기 라이브 조립기와 동일 기준

REG={
"광주":dict(bundle=ROOT/"광주_준비_2026-09-08"/"outputs"/"단기_재학습_후보_2026-09-09"/"+48h",
            nc=NCDIR/"광주"/"kma_nwp_d1d2_live_v1_2026-09-09"/"kma_nwp_d1d2_live.sqlite3",
            cap=240.58),
"부안":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3",
            kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3",
            bundle=ROOT/"부안_준비_2026-08-28"/"outputs"/"단기_재학습_후보_2026-09-09"/"+48h",
            nc=NCDIR/"부안"/"kma_nwp_d1d2_live_v1_2026-09-08"/"kma_nwp_d1d2_live.sqlite3",
            cap=1000,inv=8,station=243),
"김제":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\blockdata_live_history_gimje_v1_2026-09-01\blockdata_history.sqlite3",
            kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",
            bundle=ROOT/"김제_준비_2026-09-01"/"outputs"/"단기_재학습_후보_2026-09-09"/"+48h",
            nc=NCDIR/"김제"/"kma_nwp_d1d2_live_v1_2026-09-08"/"kma_nwp_d1d2_live.sqlite3",
            cap=1100,inv=10,station=243),
"영광":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\blockdata_live_history_yeonggwang_v1_2026-09-08\blockdata_history.sqlite3",
            kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",
            bundle=ROOT/"영광_준비_2026-09-03"/"outputs"/"단기_재학습_후보_2026-09-09"/"+48h",
            nc=NCDIR/"영광"/"kma_nwp_d1d2_live_v1_2026-09-08"/"kma_nwp_d1d2_live.sqlite3",
            cap=634,inv=13,station=252),
}

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
OLD=load("old_short_plus48h",ROOT/"live_feature_assembler_단기_v1_2026-08-26.py")

def init():
 c=sqlite3.connect(DB);c.execute("""CREATE TABLE IF NOT EXISTS shadow_short_pooled_predictions(id INTEGER PRIMARY KEY,region TEXT,issue_time_kst TEXT,target_time_kst TEXT,lead_h INTEGER,status TEXT,reason TEXT,predicted_kw REAL,missing_features TEXT,model_sha256 TEXT,run_at_kst TEXT,UNIQUE(region,issue_time_kst,target_time_kst))""");c.commit();c.close()
def save(x):
 c=sqlite3.connect(DB);c.execute("INSERT OR REPLACE INTO shadow_short_pooled_predictions(region,issue_time_kst,target_time_kst,lead_h,status,reason,predicted_kw,missing_features,model_sha256,run_at_kst) VALUES(?,?,?,?,?,?,?,?,?,?)",(x["region"],x["issue"],x["target"],x["lead"],x["status"],x.get("reason"),x.get("pred"),json.dumps(x.get("missing",[]),ensure_ascii=False),x.get("sha"),pd.Timestamp.now(tz=KST).isoformat()));c.commit();c.close()

def power(cfg,issue):
 c=sqlite3.connect(cfg["plant"]);s=(issue-timedelta(hours=30)).isoformat();d=pd.read_sql_query("SELECT snapshot_time,plant_ac_power_kw,valid_ac_power_count,expected_inverter_count FROM plant_snapshots WHERE snapshot_time>=? AND snapshot_time<=? ORDER BY snapshot_time",c,params=(s,issue.isoformat()));c.close()
 if d.empty:return pd.Series(dtype=float)
 d["t"]=pd.to_datetime(d.snapshot_time).dt.tz_localize(None);d=d[(d.valid_ac_power_count==cfg["inv"])&(d.expected_inverter_count==cfg["inv"])]
 return d.set_index("t").plant_ac_power_kw.resample("1h").mean()

def asos_latest(cfg,issue):
 # ★09-16 발견·수정★: +24h pooled 스크립트(09-10)의 SELECT를 그대로 베껴
 # 쓰다가 검증 중 발견 - asos_hourly에는 cloud_tenths(0~10)와 cloud_pct
 # (0~100, =tenths*10)가 별도 컬럼으로 존재하는데, SELECT cloud_pct 값을
 # amap의 "전운량_10분위" 자리에 그대로 매핑하고 있었다(부안 실측으로
 # cloud_pct=cloud_tenths*10 확인). +48h 모델은 전운량_10분위·전운량_pct
 # 둘 다 실제 학습 특성이라 10배 스케일이 섞여 들어갈 뻔했다 - 여기서
 # cloud_tenths/cloud_pct를 각각 올바른 이름으로 분리해서 고쳤다.
 # 일사량(solar_mj_m2/solar_w_m2)도 영광 +48h가 실제로 쓰므로 추가.
 c=sqlite3.connect(cfg["kma"]);a=c.execute("SELECT temperature_c,rainfall_mm,wind_speed_m_s,wind_direction_deg,humidity_pct,local_pressure_hpa,sea_pressure_hpa,sunshine_hr,solar_mj_m2,solar_w_m2,cloud_tenths,cloud_pct,ground_temperature_c FROM asos_hourly WHERE station=? AND observation_time<=? ORDER BY observation_time DESC LIMIT 1",(cfg["station"],issue.isoformat())).fetchone();c.close()
 amap=["기온_C","강수량_mm","풍속_m_s","풍향_deg","상대습도_pct","현지기압_hPa","해면기압_hPa","일조시간_hr","일사량_MJ_m2","일사량_W_m2","전운량_10분위","전운량_pct","지면온도_C"];ad={"issue_asos_"+k:v for k,v in zip(amap,a or [])};ad["issue_asos_지점번호"]=cfg["station"]
 if pd.isna(ad.get("issue_asos_강수량_mm")):ad["issue_asos_강수량_mm"]=0.0
 return ad

def nc_forecast(cfg,issue):
 # +48h(37~59h) 리드는 GRID엔 존재하지 않으므로 지역별 KIM NC D1D2 라이브
 # DB(nwp_d1d2_values)에서 직접 읽는다. 발행시각 이후 값 사용 방지를 위해
 # first_received_at<=issue 조건은 그대로 유지(과거 백필행은 target_time_kst
 # 범위 필터에서 자연히 걸러짐).
 win_lo=(issue-timedelta(days=3)).strftime("%Y%m%d");win_hi=(issue+timedelta(days=1)).strftime("%Y%m%d")
 c=sqlite3.connect(cfg["nc"]);g=pd.read_sql_query("SELECT target_time_kst,variable,value,is_missing,first_received_at FROM nwp_d1d2_values WHERE dry_run=0 AND lead_hours BETWEEN 37 AND 59 AND issue_date>=? AND issue_date<=?",c,params=(win_lo,win_hi));c.close()
 if g.empty:return {}
 g.loc[g.is_missing==1,"value"]=np.nan
 g["t"]=pd.to_datetime(g.target_time_kst,utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
 g["frx"]=pd.to_datetime(g.first_received_at,utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
 g=g[(g.t>=issue+timedelta(hours=37))&(g.t<=issue+timedelta(hours=59))&(g.frx<=issue)]
 if g.empty:return {}
 p=g.sort_values("frx").drop_duplicates(["t","variable"],keep="last").pivot(index="t",columns="variable",values="value")
 return p.rename(columns={c2:f"forecast_{c2}" for c2 in p.columns}).to_dict("index")

def _row_for_lead(base,w,issue,lead,features,region,sha,cap,bundle):
 target=issue+timedelta(hours=lead)
 near=min(w,key=lambda t:abs((t-target).total_seconds())) if w else None
 row=dict(base);row.update(w.get(near,{}) if near and abs((near-target).total_seconds())<=5400 else {})
 row["target_hour_sin"]=np.sin(2*np.pi*target.hour/24);row["target_hour_cos"]=np.cos(2*np.pi*target.hour/24)
 row["doy_sin"]=np.sin(2*np.pi*target.dayofyear/365.25);row["doy_cos"]=np.cos(2*np.pi*target.dayofyear/365.25)
 miss=[f for f in features if f not in row or pd.isna(row[f])];x={"region":region,"issue":issue.isoformat(),"target":target.isoformat(),"lead":lead,"sha":sha}
 if miss:x.update(status="대기",reason="필수특성 결측",missing=miss)
 else:x.update(status="성공",pred=float(np.clip(bundle["model"].predict(pd.DataFrame([[row[f] for f in features]],columns=features))[0],0,cap)))
 return x

def regional(region,cfg,issue,bundle,features,sha):
 ps=power(cfg,issue);ad=asos_latest(cfg,issue);base=dict(ad)
 for h in [1,2,3,6,24]:base[f"발전출력_{h}시간전_kW"]=ps.get(issue-timedelta(hours=h),np.nan)
 history=ps.loc[ps.index<issue]
 for h in [6,24]:
  window=history.tail(h)
  base[f"발전출력_{h}시간이동평균_kW"]=window.mean() if len(window)==h else np.nan
  base[f"발전출력_{h}시간이동표준편차_kW"]=window.std() if len(window)==h else np.nan
 w=nc_forecast(cfg,issue)
 for lead in H:save(_row_for_lead(base,w,issue,lead,features,region,sha,cfg["cap"],bundle))

def _gwangju_issue_search(hourly,end_time):
 # live_feature_assembler_단기_v1.assemble_and_predict()의 09-16 동적탐색과
 # 동일 기준(ASOS_LOOKBACK_HOURS=12) - 이 파일도 같은 build_live_hourly()
 # 결과물을 쓰므로 같은 방식으로 issue_time을 찾아야 동일한 안전성을 갖는다.
 core=[c for c in ("기상청관측_기온_C","기상청관측_상대습도_pct","기상청관측_전운량_pct") if c in hourly.columns]
 candidate=end_time.floor("h").tz_localize(None) if getattr(end_time,"tzinfo",None) else end_time.floor("h")
 for _ in range(ASOS_LOOKBACK_HOURS):
  if candidate in hourly.index and core and hourly.loc[candidate,core].notna().all():return candidate
  candidate=candidate-pd.Timedelta(hours=1)
 return None

def gwangju(cfg,now_naive,bundle,features,sha):
 hourly=OLD.build_live_hourly(pd.Timestamp(now_naive).tz_localize(KST),72,0)
 if hourly.empty:
  for lead in H:save({"region":"광주","issue":now_naive.isoformat(),"target":(now_naive+timedelta(hours=lead)).isoformat(),"lead":lead,"sha":sha,"status":"대기","reason":"원자료 부족(hourly 프레임 비어있음)","missing":[]})
  return
 issue=_gwangju_issue_search(hourly,now_naive)
 if issue is None:
  for lead in H:save({"region":"광주","issue":now_naive.isoformat(),"target":(now_naive+timedelta(hours=lead)).isoformat(),"lead":lead,"sha":sha,"status":"대기","reason":f"최근 {ASOS_LOOKBACK_HOURS}시간 내 ASOS 핵심관측 완결 시각 없음","missing":[]})
  return
 row=hourly.loc[issue]
 base={
  "plant_input_power_kw":row.get("plant_input_power_kw",np.nan),
  "mean_power_factor":row.get("mean_power_factor",np.nan),
  "mean_input_voltage_v":row.get("mean_input_voltage_v",np.nan),
  "발전출력_24시간전_kW":hourly["plant_output_kw"].get(issue-timedelta(hours=24),np.nan) if "plant_output_kw" in hourly.columns else np.nan,
  "기상청관측_일조시간_hr":row.get("기상청관측_일조시간_hr",np.nan),
  "기상청관측_상대습도_pct":row.get("기상청관측_상대습도_pct",np.nan),
  "기상청관측_전운량_pct":row.get("기상청관측_전운량_pct",np.nan),
 }
 w=nc_forecast(cfg,issue)
 for lead in H:save(_row_for_lead(base,w,issue,lead,features,"광주",sha,cfg["cap"],bundle))

def main():
 init();now=(pd.Timestamp.now(tz=KST).floor("h")-pd.Timedelta(hours=1)).tz_localize(None)
 for r,cfg in REG.items():
  man=json.loads((cfg["bundle"]/"manifest.json").read_text(encoding="utf-8"));features=man.get("features",man.get("feature_cols"));bundle=joblib.load(cfg["bundle"]/"model.joblib");sha=man["model_sha256"]
  if r=="광주":
   gwangju(cfg,now,bundle,features,sha)
  else:
   regional(r,cfg,now,bundle,features,sha)
 print(DB)
if __name__=="__main__":main()
