from __future__ import annotations
import importlib.util,json,sqlite3,sys
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib,numpy as np,pandas as pd

ROOT=Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인");KST=ZoneInfo("Asia/Seoul")
DB=ROOT/"shadow_predictions_short_pooled.sqlite3"; H=range(14,37)
REG={
"광주":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25\blockdata_history.sqlite3",kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3",bundle=ROOT/"광주_준비_2026-09-08"/"outputs"/"단기_재학습_후보_2026-09-09"/"+24h",cap=240.58,inv=5,station=156),
"부안":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\blockdata_live_v1_2026-08-28\blockdata_history.sqlite3",kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\kma_live_inputs_v1_2026-08-28\kma_live_inputs.sqlite3",bundle=ROOT/"부안_준비_2026-08-28"/"outputs"/"단기_재학습_후보_2026-09-09"/"+24h",cap=1000,inv=8,station=243),
"김제":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\blockdata_live_history_gimje_v1_2026-09-01\blockdata_history.sqlite3",kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\kma_live_inputs_gimje_v1_2026-09-01\kma_live_inputs.sqlite3",bundle=ROOT/"김제_준비_2026-09-01"/"outputs"/"단기_재학습_후보_2026-09-09"/"+24h",cap=1100,inv=10,station=243),
"영광":dict(plant=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\blockdata_live_history_yeonggwang_v1_2026-09-08\blockdata_history.sqlite3",kma=r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\kma_live_inputs_yeonggwang_v1_2026-09-08\kma_live_inputs.sqlite3",bundle=ROOT/"영광_준비_2026-09-03"/"outputs"/"단기_재학습_후보_2026-09-09"/"+24h",cap=634,inv=13,station=252)}

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
OLD=load("old_short",ROOT/"live_feature_assembler_단기_v1_2026-08-26.py")
SOL={"부안":load("bsolar",ROOT/"부안_준비_2026-08-28"/"ultra_short_term_v1_buan_2026-09-08.py"),"김제":load("gsolar",ROOT.parent.parent/"02_전처리"/"김제"/"build_gimje_time_aggregates_v1_2026-08-31.py"),"영광":load("ysolar",ROOT.parent.parent/"02_전처리"/"영광"/"build_yeonggwang_time_aggregates_v1_2026-09-01.py")}

def init():
 c=sqlite3.connect(DB);c.execute("""CREATE TABLE IF NOT EXISTS shadow_short_pooled_predictions(id INTEGER PRIMARY KEY,region TEXT,issue_time_kst TEXT,target_time_kst TEXT,lead_h INTEGER,status TEXT,reason TEXT,predicted_kw REAL,missing_features TEXT,model_sha256 TEXT,run_at_kst TEXT,UNIQUE(region,issue_time_kst,target_time_kst))""");c.commit();c.close()
def save(x):
 c=sqlite3.connect(DB);c.execute("INSERT OR REPLACE INTO shadow_short_pooled_predictions(region,issue_time_kst,target_time_kst,lead_h,status,reason,predicted_kw,missing_features,model_sha256,run_at_kst) VALUES(?,?,?,?,?,?,?,?,?,?)",(x["region"],x["issue"],x["target"],x["lead"],x["status"],x.get("reason"),x.get("pred"),json.dumps(x.get("missing",[]),ensure_ascii=False),x.get("sha"),datetime.now(KST).isoformat()));c.commit();c.close()
def power(cfg,issue):
 c=sqlite3.connect(cfg["plant"]);s=(issue-timedelta(hours=30)).isoformat();d=pd.read_sql_query("SELECT snapshot_time,plant_ac_power_kw,valid_ac_power_count,expected_inverter_count FROM plant_snapshots WHERE snapshot_time>=? AND snapshot_time<=? ORDER BY snapshot_time",c,params=(s,issue.isoformat()));c.close()
 if d.empty:return pd.Series(dtype=float)
 d["t"]=pd.to_datetime(d.snapshot_time).dt.tz_localize(None);d=d[(d.valid_ac_power_count==cfg["inv"])&(d.expected_inverter_count==cfg["inv"])]
 return d.set_index("t").plant_ac_power_kw.resample("1h").mean()
def weather(cfg,issue):
 # ★09-16 발견·수정★: asos_hourly에는 cloud_tenths(0~10, 학습 특성
 # "전운량_10분위"가 실제 기대하는 값)와 cloud_pct(0~100, =tenths*10)가
 # 별도 컬럼인데, 아래 SELECT가 cloud_pct를 뽑아서 amap의 "전운량_10분위"
 # 자리에 그대로 넣고 있었다 - 09-10 배포 이후 지금까지 4지역 +24h pooled
 # 라이브 예측(부안·김제·영광)이 전운량 특성에 10배 스케일 오류를 안고
 # 있었다(부안 실측으로 cloud_pct=cloud_tenths*10 확인, 모델은 전운량_pct를
 # 쓰지 않고 전운량_10분위만 씀). +48h pooled 조립기 신규 제작 중 발견해
 # 즉시 고침. 09-10~09-16 사이 저장된 "성공" 예측 중 이 특성을 실제로 쓴
 # 것은 스케일이 잘못된 입력으로 나온 값이라 신뢰할 수 없음(과거 행을
 # 소급 정정하지는 않음 - shadow_short_pooled_predictions에 그대로 둠).
 c=sqlite3.connect(cfg["kma"]);a=c.execute("SELECT temperature_c,rainfall_mm,wind_speed_m_s,wind_direction_deg,humidity_pct,local_pressure_hpa,sea_pressure_hpa,sunshine_hr,cloud_tenths,ground_temperature_c FROM asos_hourly WHERE station=? AND observation_time<=? ORDER BY observation_time DESC LIMIT 1",(cfg["station"],issue.isoformat())).fetchone()
 g=pd.read_sql_query("SELECT target_time_kst,variable,value,is_missing,first_received_at FROM grid_forecast WHERE target_time_kst>=? AND target_time_kst<=? AND first_received_at<=? AND variable IN ('POP','SKY','REH')",c,params=((issue+timedelta(hours=14)).isoformat(),(issue+timedelta(hours=36)).isoformat(),issue.isoformat()));c.close()
 amap=["기온_C","강수량_mm","풍속_m_s","풍향_deg","상대습도_pct","현지기압_hPa","해면기압_hPa","일조시간_hr","전운량_10분위","지면온도_C"];ad={"issue_asos_"+k:v for k,v in zip(amap,a or [])};ad["issue_asos_지점번호"]=cfg["station"]
 # ASOS 강수량 null은 무강수 시 흔한 원천 표기이므로 기존 전처리와 같이 0으로 해석한다.
 if pd.isna(ad.get("issue_asos_강수량_mm")):ad["issue_asos_강수량_mm"]=0.0
 if g.empty:return ad,{}
 g.loc[g.is_missing==1,"value"]=np.nan;g["t"]=pd.to_datetime(g.target_time_kst).dt.tz_localize(None);p=g.sort_values("first_received_at").drop_duplicates(["t","variable"],keep="last").pivot(index="t",columns="variable",values="value");return ad,p.to_dict("index")
def regional(region,cfg,issue,bundle,features,sha):
 ps=power(cfg,issue);ad,w=weather(cfg,issue);base=dict(ad)
 for h in [1,2,3,6,24]:base[f"발전출력_{h}시간전_kW"]=ps.get(issue-timedelta(hours=h),np.nan)
 history=ps.loc[ps.index<issue]
 for h in [6,24]:
  window=history.tail(h)
  base[f"발전출력_{h}시간이동평균_kW"]=window.mean() if len(window)==h else np.nan
  base[f"발전출력_{h}시간이동표준편차_kW"]=window.std() if len(window)==h else np.nan
 for lead in H:
  target=issue+timedelta(hours=lead); near=min(w,key=lambda t:abs((t-target).total_seconds())) if w else None;row=dict(base);row.update({"forecast_"+k:v for k,v in (w.get(near,{}) if near and abs((near-target).total_seconds())<=5400 else {}).items()});row["target_hour_sin"]=np.sin(2*np.pi*target.hour/24);row["target_hour_cos"]=np.cos(2*np.pi*target.hour/24);row["doy_sin"]=np.sin(2*np.pi*target.dayofyear/365.25);row["doy_cos"]=np.cos(2*np.pi*target.dayofyear/365.25)
  if "solar_elevation_deg" in features:row["solar_elevation_deg"]=float(SOL[region].solar_elevation_deg(pd.DatetimeIndex([target]))[0])
  miss=[f for f in features if f not in row or pd.isna(row[f])];x={"region":region,"issue":issue.isoformat(),"target":target.isoformat(),"lead":lead,"sha":sha}
  if miss:x.update(status="대기",reason="필수특성 결측",missing=miss)
  else:x.update(status="성공",pred=float(np.clip(bundle["model"].predict(pd.DataFrame([[row[f] for f in features]],columns=features))[0],0,cfg["cap"])))
  save(x)
def gwangju(cfg,issue,bundle,features,sha):
 hourly=OLD.build_live_hourly(pd.Timestamp(issue).tz_localize(KST),72,36)
 for lead in H:
  frame=OLD.harness.build_frame(hourly,lead,OLD.harness.FEATURE_SETS["전체후보"]);it=issue
  miss=features if it not in frame.index else [f for f in features if f not in frame or pd.isna(frame.at[it,f])];x={"region":"광주","issue":issue.isoformat(),"target":(issue+timedelta(hours=lead)).isoformat(),"lead":lead,"sha":sha}
  if miss:x.update(status="대기",reason="필수특성 결측",missing=miss)
  else:x.update(status="성공",pred=float(np.clip(bundle["model"].predict(frame.loc[[it],features])[0],0,cfg["cap"])))
  save(x)
def main():
 # 진행 중인 현재 시각대가 아니라 마지막으로 완결된 정시까지만 사용한다.
 init();issue=(pd.Timestamp.now(tz=KST).floor("h")-pd.Timedelta(hours=1)).tz_localize(None)
 for r,cfg in REG.items():
  man=json.loads((cfg["bundle"]/"manifest.json").read_text(encoding="utf-8"));features=man.get("features",man.get("feature_cols"));bundle=joblib.load(cfg["bundle"]/"model.joblib");sha=man["model_sha256"]
  if r=="광주":
   gwangju(cfg,issue,bundle,features,sha)
  else:
   regional(r,cfg,issue,bundle,features,sha)
 print(DB)
if __name__=="__main__":main()
