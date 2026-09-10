"""부안·김제 단기(+1h/+24h) 라이브 재현가능 특성 후보 재학습.

기존 공식 번들과 스케줄러는 건드리지 않는다. 아카이브는 하나의 연속
segment로 명시하고, 발행시각 anchor 기반 lag/rolling으로 미래누출을 막는다.
최근 라이브 결측이 90% 이상인 구형 NWP 특성은 후보에서 제외한다.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py"
CFG = {
 "부안": {"dir": ROOT/"부안_준비_2026-08-28", "capacity":1000.0,
  "join":Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\과거발전_기상결합_v1_2026-08-28\부안_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"),
  "summary":ROOT/"부안_준비_2026-08-28"/"outputs/요인재검증_v6_phase2_2026-09-03/phase2_요약.json",
  "hourly":Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28\부안_발전소_1시간_공식후보.parquet")},
 "김제": {"dir": ROOT/"김제_준비_2026-09-01", "capacity":1100.0,
  "join":Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31\김제_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"),
  "summary":ROOT/"김제_준비_2026-09-01"/"outputs/요인재검증_v6_phase2_인근일사량추가_2026-09-07/phase2_요약.json",
  "hourly":Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31\김제_발전소_1시간_공식후보.parquet")},
 "영광": {"dir": ROOT/"영광_준비_2026-09-03", "capacity":634.0,
  "join":Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_v1_2026-09-03\영광_과거발전_ASOS_NWP_GRID_결합_v1_2026-09-03.parquet"),
  "summary":ROOT/"영광_준비_2026-09-03"/"outputs/요인구축_v6_phase2_2026-09-03/phase2_요약.json",
  "hourly":Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01\영광_발전소_1시간_공식후보.parquet"),
  "defect_audit":"미실시"},
}
# 최근 라이브 08-26~09-09 실측에서 90% 이상 결측으로 확인된 구형 NWP.
LIVE_UNREPRODUCIBLE = {"forecast_DSWRF","forecast_TCDC","forecast_LCDC","forecast_MCDC","forecast_HCDC","forecast_DSWRFLX","forecast_DIFSWRF","전주146_인근일사량_W_m2","정읍245_인근일사량_W_m2"}
LAGS=[1,2,3,6,24]; ROLLS=[6,24]; TARGET="plant_ac_power_kw"

def load_harness():
 spec=importlib.util.spec_from_file_location("phase2", PHASE2); mod=importlib.util.module_from_spec(spec); sys.modules["phase2"]=mod; spec.loader.exec_module(mod); return mod._load_harness()
def naive(x):
 x=pd.to_datetime(x,errors="raise"); return x.dt.tz_convert("Asia/Seoul").dt.tz_localize(None) if getattr(x.dt,"tz",None) is not None else x
def build(cfg):
 j=pd.read_parquet(cfg["join"]); j["issue_time_kst"]=naive(j["prediction_issue_time_kst"]); j["target_time_kst"]=naive(j["target_time_kst"])
 j=j[j.target_time_kst>j.issue_time_kst].copy(); j["segment_id"]="archive"; j["boundary_excluded"]=False
 hp=pd.read_parquet(cfg["hourly"]); ht=[c for c in ["grid_time_kst","target_time_kst","시각"] if c in hp.columns][0]; hp["_t"]=naive(hp[ht]).dt.floor("h"); h=hp.sort_values("_t").drop_duplicates("_t",keep="last").set_index("_t")[TARGET]
 for lag in LAGS: j[f"발전출력_{lag}시간전_kW"]=h.reindex(j.issue_time_kst.dt.floor("h")-pd.to_timedelta(lag,"h")).to_numpy()
 hs=h.sort_index(); sh=hs.shift(1)
 for w in ROLLS:
  j[f"발전출력_{w}시간이동평균_kW"]=sh.rolling(w,min_periods=w).mean().reindex(j.issue_time_kst.dt.floor("h")).to_numpy()
  j[f"발전출력_{w}시간이동표준편차_kW"]=sh.rolling(w,min_periods=w).std().reindex(j.issue_time_kst.dt.floor("h")).to_numpy()
 t=j.target_time_kst; j["target_hour_sin"]=np.sin(2*np.pi*(t.dt.hour+t.dt.minute/60)/24); j["target_hour_cos"]=np.cos(2*np.pi*(t.dt.hour+t.dt.minute/60)/24); j["doy_sin"]=np.sin(2*np.pi*t.dt.dayofyear/365.25); j["doy_cos"]=np.cos(2*np.pi*t.dt.dayofyear/365.25)
 old=json.loads(cfg["summary"].read_text(encoding="utf-8")); feats=[x for x in old["최종특성"] if x not in LIVE_UNREPRODUCIBLE and x in j.columns]
 if "issue_asos_강수량_mm" in j.columns: j["issue_asos_강수량_mm"]=pd.to_numeric(j["issue_asos_강수량_mm"],errors="coerce").fillna(0.0)
 # 반드시 lag/rolling + selected weather만 후보에 포함
 missing=[x for x in old["최종특성"] if x not in j.columns and x not in LIVE_UNREPRODUCIBLE]
 if missing: raise RuntimeError(f"{cfg}: missing {missing}")
 return j,feats,{"excluded_live_unreproducible":sorted(set(old["최종특성"])&LIVE_UNREPRODUCIBLE),"max_target_relative_lag_used":0,"segment_count":1}
def folds(days, initial):
 return [(days[:i],days[i:min(i+30,len(days))]) for i in range(initial,len(days),30) if len(days[i:min(i+30,len(days))])]
def run(region,cfg,harness):
 df,feats,audit=build(cfg); req=feats+[TARGET]; results=[]
 for horizon in (1,24):
  sub=df.copy()
  if horizon==24:
   lead=(sub.target_time_kst-sub.issue_time_kst).dt.total_seconds()/3600
   sub=sub[(lead-23.0).abs()<=1.5].copy(); audit_h={**audit,"actual_lead_h_used":23.0,"lead_tolerance_h":1.5}
  else: audit_h={**audit,"actual_lead_h_used":"regional_D+1_all_available_leads"}
  sub["issue_day"]=sub.issue_time_kst.dt.normalize(); days=pd.DatetimeIndex(sorted(sub.issue_day.unique())); pooled=[]
  for name in ["LightGBM","XGBoost","선형회귀"]:
   ys=[]; ps=[]
   for trd,ted in folds(days,max(30,len(days)//3)):
    tr=sub[sub.issue_day.isin(trd)].dropna(subset=req); te=sub[sub.issue_day.isin(ted)].dropna(subset=req)
    if len(tr)<100 or len(te)<5: continue
    m=__import__('sklearn.linear_model',fromlist=['LinearRegression']).LinearRegression() if name=="선형회귀" else harness.make_model(name,42); m.fit(tr[feats],tr[TARGET]); ps.append(np.clip(m.predict(te[feats]),0,cfg["capacity"])); ys.append(te[TARGET].to_numpy())
   if ys:
    y=np.concatenate(ys); p=np.concatenate(ps); pooled.append((name,float(np.abs(p-y).mean()),float(np.sqrt(np.mean((p-y)**2))),len(y)))
  if not pooled:
   print(region,"missing_top",sub[req].isna().mean().sort_values(ascending=False).head(12).to_dict(),"rows",len(sub),"days",len(days)); raise RuntimeError(f"{region}/{horizon}: no valid folds")
  selected=min(pooled,key=lambda x:x[1])[0]; clean=sub.dropna(subset=req); model=harness.make_model(selected,42) if selected!="선형회귀" else __import__('sklearn.linear_model',fromlist=['LinearRegression']).LinearRegression(); model.fit(clean[feats],clean[TARGET])
  out=cfg["dir"] / "outputs" / "단기_재학습_후보_2026-09-09"; d=out/("D+1" if horizon==1 else "+24h"); d.mkdir(parents=True,exist_ok=True); path=d/"model.joblib"; tmp=d/"model.joblib.tmp"; joblib.dump({"model":model,"features":feats,"region":region,"horizon":("D+1" if horizon==1 else "+24h"),"model_name":selected},tmp); _=joblib.load(tmp); os.replace(tmp,path)
  man={"status":"공식_후보","region":region,"tier":"단기","horizon":("D+1" if horizon==1 else "+24h"),"archive_only":True,"archive_period":f"{sub.issue_time_kst.min()}~{sub.issue_time_kst.max()}","seg2_live":"미포함(지역별 라이브 과거 세그먼트 미확보)","defect_audit":cfg.get("defect_audit","기존 결합자료의 결함기간 라벨 재사용; 별도 감사 미실시"),"promotion_blocker":"영광은 결함기간 감사 완료 전 공식 승격 금지" if region=="영광" else None,"features":feats,"excluded_live_unreproducible":audit_h["excluded_live_unreproducible"],"selected_model":selected,"training_rows":len(clean),"walk_forward":[{"model":a,"MAE_kW":b,"RMSE_kW":c,"test_rows":n} for a,b,c,n in pooled],"leakage_audit":audit_h,"운영_연결":"보류; 기존 공식 번들·스케줄러 미변경","model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"created_at_kst":datetime.now().astimezone().isoformat()}; (d/"manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8"); results.append({"horizon":man["horizon"],"training_rows":len(clean),"selected_model":selected,"walk_forward":man["walk_forward"]})
 return {"region":region,"rows":len(df),"features":feats,"results":results}
def main():
 h=load_harness(); print(json.dumps([run(r,c,h) for r,c in CFG.items()],ensure_ascii=False,indent=2))
if __name__=="__main__": main()
