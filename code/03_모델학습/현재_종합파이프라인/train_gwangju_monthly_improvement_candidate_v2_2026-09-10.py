from __future__ import annotations
import importlib.util, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

PIPE=Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
OUT=PIPE/"outputs"/"광주_월간_개선후보_v2_2026-09-10"
DAILY=PIPE/"outputs"/"v7_라이브연계_2026-08-26"/"집계_일간_실제발전량_v5.parquet"

def metrics(y,p):
 y,p=np.asarray(y,float),np.asarray(p,float); e=y-p
 return {"n":len(y),"WAPE_pct":round(abs(e).sum()/abs(y).sum()*100,2),"MAE_kWh":round(abs(e).mean(),1),"RMSE_kWh":round(np.sqrt((e**2).mean()),1)}

def model_factories():
 return {
  "LightGBM_동일조건":lambda:LGBMRegressor(objective="regression_l1",n_estimators=80,learning_rate=.05,num_leaves=7,max_depth=3,min_child_samples=3,reg_lambda=1,verbosity=-1,random_state=20260819,n_jobs=2),
  "GradientBoostingHuber_개선후보":lambda:GradientBoostingRegressor(loss="huber",n_estimators=50,max_depth=2,learning_rate=.04,min_samples_leaf=3,random_state=20260819),
 }

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 d=pd.read_parquet(DAILY); d.index=pd.to_datetime(d.index); ycol="일간발전량_kWh"; acol="가용인버터수_낮시간평균"
 m=d[ycol].resample("MS").agg(total="sum",n="count"); m["days"]=m.index.days_in_month; m["y"]=m.total.where(m.n>=m.days)
 m["avail"]=d[acol].resample("MS").mean()
 idx=pd.date_range(m.index.min(),m.index.max(),freq="MS"); m=m.reindex(idx); f=pd.DataFrame(index=idx)
 for k in [1,2,3]: f[f"lag{k}"]=m.y.shift(k)
 f["mean3"]=m.y.shift(1).rolling(3,min_periods=3).mean();f["month_sin"]=np.sin(2*np.pi*idx.month/12);f["month_cos"]=np.cos(2*np.pi*idx.month/12);f["days"]=idx.days_in_month;f["availability_lag1"]=m.avail.shift(1);f["y"]=m.y
 features=["lag1","lag2","lag3","mean3","month_sin","month_cos","days","availability_lag1"]
 v=f.dropna(subset=features+["y"]); rows=[]; factories=model_factories()
 for i in range(6,len(v)):
  tr,te=v.iloc[:i],v.iloc[i:i+1]; r={"month":str(te.index[0].date()),"actual_kWh":float(te.y.iloc[0]),"전월지속성_kWh":float(te.lag1.iloc[0])}
  for name,mk in factories.items():
   mdl=mk();mdl.fit(tr[features],tr.y);r[name+"_kWh"]=max(0,float(mdl.predict(te[features])[0]))
  rows.append(r)
 o=pd.DataFrame(rows);o.to_csv(OUT/"월간_OOF_상세.csv",index=False,encoding="utf-8-sig")
 result={"전월지속성":metrics(o.actual_kWh,o.전월지속성_kWh)}
 for name in factories: result[name]=metrics(o.actual_kWh,o[name+"_kWh"])
 pd.DataFrame([{"구성":k,**v} for k,v in result.items()]).to_csv(OUT/"월간_성능요약.csv",index=False,encoding="utf-8-sig")
 final=factories["GradientBoostingHuber_개선후보"]();final.fit(v[features],v.y);joblib.dump({"model":final,"features":features},OUT/"model.joblib")
 special={}
 for month in ["2025-09-01","2025-10-01","2026-03-01","2026-05-01"]:
  q=o[o.month==month]
  if len(q): special[month]={"actual_kWh":round(float(q.actual_kWh.iloc[0]),1),"candidate_kWh":round(float(q["GradientBoostingHuber_개선후보_kWh"].iloc[0]),1),"abs_error_kWh":round(abs(float(q.actual_kWh.iloc[0]-q["GradientBoostingHuber_개선후보_kWh"].iloc[0])),1)}
 manifest={"status":"공식_후보_운영연결보류","기존모델_스케줄러_변경":False,"source":str(DAILY),"complete_months":int(m.y.notna().sum()),"latest_complete_month":str(m.y.dropna().index.max().date()),"walk_forward_common_n":len(o),"issue_safe":"모든 lag는 과거월만 사용; 기존의 해당월 평균 가용인버터수는 사용하지 않고 1개월 지연값만 사용","metrics":result,"문제월_감사":special,"판정":"동일 14개월에서 세 지표 모두 기준선 및 LightGBM보다 개선했으나, 최신 완결월이 늘지 않았고 2025-09 부분용량월 오차도 해소되지 않아 잠정 해제 불가. 독립 검증 필요."}
 (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
