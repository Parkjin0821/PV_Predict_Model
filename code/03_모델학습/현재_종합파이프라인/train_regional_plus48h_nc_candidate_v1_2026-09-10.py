from __future__ import annotations
import hashlib, importlib.util, json, os, sqlite3, sys
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT=Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
SOURCE=ROOT/"retrain_regional_short_segmented_candidate_v1_2026-09-09.py"
NCDIR=Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델")

def load_source():
 spec=importlib.util.spec_from_file_location("short_retrain",SOURCE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def metrics(y,p):
 e=np.asarray(y)-np.asarray(p);return float(abs(e).mean()),float(np.sqrt((e*e).mean())),len(e)

def nc_frame(region):
 db=NCDIR/region/"kma_nwp_d1d2_live_v1_2026-09-08"/"kma_nwp_d1d2_live.sqlite3"
 con=sqlite3.connect(db);q=pd.read_sql_query("SELECT issue_date,target_time_kst,lead_hours,variable,value,is_missing,dry_run FROM nwp_d1d2_values WHERE dry_run=0 AND lead_hours BETWEEN 37 AND 59",con);con.close()
 q.loc[(q.is_missing==1),"value"]=np.nan;q["target_time_kst"]=pd.to_datetime(q.target_time_kst,utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
 p=q.pivot_table(index=["issue_date","target_time_kst","lead_hours"],columns="variable",values="value",aggfunc="last").reset_index()
 return p.rename(columns={x:f"forecast_{x}" for x in ["DSWRF","TCDC","LCDC","MCDC","HCDC"]})

def run(region,cfg,src,harness):
 base,_,audit=src.build(cfg); base["issue_date"]=base.issue_time_kst.dt.strftime("%Y%m%d")
 issue_cols=[c for c in base.columns if (c.startswith("발전출력_") or c.startswith("issue_asos_")) and pd.api.types.is_numeric_dtype(base[c])]
 issue=base.sort_values("issue_time_kst").drop_duplicates("issue_date")[["issue_date","issue_time_kst",*issue_cols]]
 nc=nc_frame(region).merge(issue,on="issue_date",how="inner")
 hp=pd.read_parquet(cfg["hourly"]);tc=[c for c in ["grid_time_kst","target_time_kst","시각"] if c in hp.columns][0];hp["_t"]=src.naive(hp[tc]).dt.floor("h");actual=hp.sort_values("_t").drop_duplicates("_t",keep="last").set_index("_t")[src.TARGET]
 nc[src.TARGET]=actual.reindex(nc.target_time_kst.dt.floor("h")).to_numpy();t=nc.target_time_kst
 nc["target_hour_sin"]=np.sin(2*np.pi*t.dt.hour/24);nc["target_hour_cos"]=np.cos(2*np.pi*t.dt.hour/24);nc["doy_sin"]=np.sin(2*np.pi*t.dt.dayofyear/365.25);nc["doy_cos"]=np.cos(2*np.pi*t.dt.dayofyear/365.25)
 preferred=[*issue_cols,"forecast_DSWRF","forecast_TCDC","forecast_LCDC","forecast_MCDC","forecast_HCDC","target_hour_sin","target_hour_cos","doy_sin","doy_cos"]
 features=[c for c in preferred if c in nc and nc[c].notna().mean()>=.90]
 req=features+[src.TARGET];nc["issue_day"]=nc.issue_time_kst.dt.normalize();days=pd.DatetimeIndex(sorted(nc.issue_day.unique()));pooled=[]
 for name in ["LightGBM","XGBoost","선형회귀"]:
  ys=[];ps=[]
  for trd,ted in src.folds(days,max(30,len(days)//3)):
   tr=nc[nc.issue_day.isin(trd)].dropna(subset=req);te=nc[nc.issue_day.isin(ted)].dropna(subset=req)
   if len(tr)<100 or len(te)<5:continue
   mdl=__import__("sklearn.linear_model",fromlist=["LinearRegression"]).LinearRegression() if name=="선형회귀" else harness.make_model(name,42);mdl.fit(tr[features],tr[src.TARGET]);pred=np.clip(mdl.predict(te[features]),0,cfg["capacity"]);ys.extend(te[src.TARGET]);ps.extend(pred)
  if ys: pooled.append((name,*metrics(ys,ps)))
 if not pooled:raise RuntimeError(f"{region}: +48h 유효 폴드 없음")
 selected=min(pooled,key=lambda x:x[1])[0];clean=nc.dropna(subset=req);mdl=__import__("sklearn.linear_model",fromlist=["LinearRegression"]).LinearRegression() if selected=="선형회귀" else harness.make_model(selected,42);mdl.fit(clean[features],clean[src.TARGET])
 out=cfg["dir"]/"outputs"/"단기_재학습_후보_2026-09-09"/"+48h";out.mkdir(parents=True,exist_ok=True);path=out/"model.joblib";tmp=out/"model.joblib.tmp";joblib.dump({"model":mdl,"features":features,"region":region,"horizon":"+48h","model_name":selected},tmp);joblib.load(tmp);os.replace(tmp,path)
 man={"status":"공식_후보","region":region,"tier":"단기","horizon":"+48h","horizon_type":"pooled_window","lead_hours_range":"37~59h (실제 NC 격자 38~59h)","definition":"모레 예측 묶음; D+2 KIM NC 5종과 발행시점 이전 발전·ASOS만 사용","source_nc":str(NCDIR/region/"kma_nwp_d1d2_live_v1_2026-09-08"/"kma_nwp_d1d2_live.sqlite3"),"issue_days_available":int(nc.issue_day.nunique()),"training_rows":len(clean),"features":features,"selected_model":selected,"walk_forward":[{"model":a,"MAE_kW":b,"RMSE_kW":c,"test_rows":n} for a,b,c,n in pooled],"leakage_audit":{"max_target_relative_lag_used":0,"all_power_features_anchored_at_or_before_issue":True,"target_weather_lead_min_h":float(nc.lead_hours.min()),"target_weather_lead_max_h":float(nc.lead_hours.max())},"defect_audit":cfg.get("defect_audit","기존 결합자료 라벨 재사용; 별도 감사 미실시"),"promotion_blocker":"영광 결함기간 감사 완료 전 공식 승격 금지" if region=="영광" else None,"운영_연결":"보류; 기존 공식모델·스케줄러 미변경","model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"created_at_kst":datetime.now().astimezone().isoformat()};(out/"manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8");return man

def main():
 src=load_source();h=src.load_harness();print(json.dumps([run(r,src.CFG[r],src,h) for r in ["부안","김제","영광"]],ensure_ascii=False,indent=2))
if __name__=="__main__":main()
