"""3지역 초단기·중장기 발행시각 안전 재검증. API 호출 없음."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parent; SEED=42
OUT=ROOT/"outputs"/"3지역_수평별_issue_safe_재검증_v1_2026-09-08"

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m

def score(y,p):
    e=np.asarray(y)-np.asarray(p)
    return {"n":int(len(e)),"MAE_kW":float(np.mean(np.abs(e))),"RMSE_kW":float(np.sqrt(np.mean(e**2)))}

def folds(days,initial,block):
    return [(days[:i],days[i:min(i+block,len(days))]) for i in range(initial,len(days),block)]

def strict_split(d,trdays,tedays,features,target):
    test=d[d.issue_day.isin(tedays)].dropna(subset=features+[target]).copy()
    if test.empty:return d.iloc[:0],test,0
    cutoff=test.issue_time.min()
    candidate=d[d.issue_day.isin(trdays)]
    removed=int((candidate.target_time>=cutoff).sum())
    train=candidate[candidate.target_time<cutoff].dropna(subset=features+[target]).copy()
    return train,test,removed

def tune_chrono(d,trdays,features,target,structures,min_rows=30):
    ds=np.array(sorted(trdays)); n=max(1,int(len(ds)*.2)); fitdays=ds[:-n]; valdays=ds[-n:]
    val=d[d.issue_day.isin(valdays)].dropna(subset=features+[target]);
    if val.empty:return next(iter(structures))
    cutoff=val.issue_time.min(); tr=d[d.issue_day.isin(fitdays)&(d.target_time<cutoff)].dropna(subset=features+[target])
    if len(tr)<min_rows:return next(iter(structures))
    best=(np.inf,next(iter(structures)))
    for name,p in structures.items():
        m=LGBMRegressor(**p,random_state=SEED,n_jobs=-1,verbosity=-1);m.fit(tr[features],tr[target])
        mae=np.mean(np.abs(val[target]-np.clip(m.predict(val[features]),0,None)))
        if mae<best[0]:best=(mae,name)
    return best[1]

def run_ultra_buan():
    m=load('buan_ultra_old',ROOT/'부안_준비_2026-08-28'/'ultra_short_historical_backtest_v1_2026-08-31.py')
    s=m.load_plant_5min_series(); base=m.build_features(s); out={}; removed_total=0
    features=['value','lag_15m','lag_30m','lag_1h','roll_1h_mean','lag_1day_same_time']
    for mins in m.HORIZONS_MIN:
        d=base.copy();d['issue_time']=d.index;d['target_time']=d.index+pd.Timedelta(minutes=mins)
        d['issue_day']=d.issue_time.dt.normalize();d['y']=s.shift(-(mins//5));d['persistence']=d.value
        days=pd.DatetimeIndex(np.sort(d.dropna(subset=features+['y']).issue_day.unique())); ys=[]; preds={"LightGBM":[],"선형회귀":[],"지속성":[]}
        for trd,ted in folds(days,m.INITIAL_TRAIN_DAYS,m.TEST_BLOCK_DAYS):
            tr,te,removed=strict_split(d,trd,ted,features,'y');removed_total+=removed
            if len(tr)<500 or te.empty:continue
            models=[('LightGBM',LGBMRegressor(n_estimators=150,learning_rate=.05,num_leaves=15,max_depth=4,min_child_samples=30,subsample=.9,colsample_bytree=.9,reg_alpha=.1,reg_lambda=1,random_state=SEED,n_jobs=-1,verbosity=-1)),('선형회귀',LinearRegression())]
            ys.append(te.y.to_numpy());preds['지속성'].append(te.persistence.to_numpy())
            for name,model in models:model.fit(tr[features],tr.y);preds[name].append(np.clip(model.predict(te[features]),0,None))
        y=np.concatenate(ys);out[f'+{mins//60}h']={k:score(y,np.concatenate(v)) for k,v in preds.items()}
    return {"status":"issue_safe_revalidated","results":out,"boundary_train_rows_removed":removed_total,"limitations":["가을 표본 없음"]}

def prepare_mod_frame(mod,h):
    base,_=mod.load_base();d,target,features=mod.build_horizon_frame(base,h)
    d=d.copy();d['issue_time']=d['grid_time_kst'];d['target_time']=d.issue_time+pd.Timedelta(hours=h);d['issue_day']=d.issue_time.dt.normalize()
    return d,target,features

def run_ultra_model(region,mod,regime=False):
    results={};removed_total=0
    for h in mod.LEAD_HOURS:
        d,target,features=prepare_mod_frame(mod,h); days=pd.DatetimeIndex(np.sort(d.issue_day.unique())); ys=[];pm=[];pb=[]
        for trd,ted in folds(days,mod.INITIAL_TRAIN_DAYS,mod.TEST_BLOCK_DAYS):
            tr,te,removed=strict_split(d,trd,ted,features,target);removed_total+=removed
            need=['power_lag_0min','kt_now','clearsky_power_target_kw'];te=te.dropna(subset=need)
            if len(tr)<mod.MIN_ROWS_PER_FOLD or te.empty:continue
            chosen=tune_chrono(d,trd,features,target,mod.STRUCTURES if hasattr(mod,'STRUCTURES') else {'raw':dict(n_estimators=200,learning_rate=.05,num_leaves=15,max_depth=5,min_child_samples=20,subsample=.9,colsample_bytree=.9,reg_alpha=.1,reg_lambda=1)})
            params=(mod.STRUCTURES if hasattr(mod,'STRUCTURES') else {'raw':{}})[chosen]
            if regime:
                regs=np.select([tr.kt_now>=.7,tr.kt_now<.3],['맑음','흐림'],default='보통'); testregs=np.select([te.kt_now>=.7,te.kt_now<.3],['맑음','흐림'],default='보통'); pred=np.zeros(len(te))
                for r in ['맑음','보통','흐림']:
                    sub=tr[regs==r];fit=tr if len(sub)<200 else sub; model=LGBMRegressor(**params,random_state=SEED,n_jobs=-1,verbosity=-1);model.fit(fit[features],fit[target]);mask=testregs==r
                    if mask.any():pred[mask]=np.clip(model.predict(te.loc[mask,features]),0,None)
            else:
                model=LGBMRegressor(**params,random_state=SEED,n_jobs=-1,verbosity=-1);model.fit(tr[features],tr[target]);pred=np.clip(model.predict(te[features]),0,None)
            ys.append(te[target].to_numpy());pm.append(pred);pb.append(np.clip(te.kt_now.to_numpy()*te.clearsky_power_target_kw.to_numpy(),0,None))
        y=np.concatenate(ys); results[f'+{h}h']={"모델":score(y,np.concatenate(pm)),"스마트지속성":score(y,np.concatenate(pb))}
    return {"status":"issue_safe_revalidated","results":results,"boundary_train_rows_removed":removed_total,"inner_tuning":"chronological_last_20pct"}

def run_medium(region,mod):
    d,meta=mod.build_daily_dataset();d=d.copy();
    # 발행 당일 전체값(target_day-1)은 발행시각에 미완결. 그 이전 완결일로 교정.
    lookup=d.set_index('target_day')[mod.TARGET]
    d['daily_energy_lag1_kwh']=(d['issue_day']-pd.Timedelta(days=1)).map(lookup)
    days=pd.DatetimeIndex(np.sort(d.issue_day.unique())); fs=mod.CANDIDATE_FEATURES; ys=[];pm=[];pp=[]
    for trd,ted in folds(days,mod.INITIAL_TRAIN_DAYS,mod.TEST_BLOCK_DAYS):
        tr=d[d.issue_day.isin(trd)].dropna(subset=fs+[mod.TARGET]);te=d[d.issue_day.isin(ted)].dropna(subset=fs+[mod.TARGET])
        if len(tr)<mod.MIN_ROWS_PER_FOLD or te.empty:continue
        model=LGBMRegressor(n_estimators=300,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=20,subsample=.9,colsample_bytree=.9,reg_alpha=.1,reg_lambda=1,random_state=SEED,n_jobs=-1,verbosity=-1)
        model.fit(tr[fs],tr[mod.TARGET]);ys.append(te[mod.TARGET].to_numpy());pm.append(np.clip(model.predict(te[fs]),0,None));pp.append(te.daily_energy_lag1_kwh.to_numpy())
    y=np.concatenate(ys); ar=score(y,np.concatenate(pm));br=score(y,np.concatenate(pp));
    a={"n":ar["n"],"MAE_kWh":ar["MAE_kW"],"RMSE_kWh":ar["RMSE_kW"]};b={"n":br["n"],"MAE_kWh":br["MAE_kW"],"RMSE_kWh":br["RMSE_kW"]}
    return {"status":"issue_safe_revalidated","model":a,"persistence_last_complete_day":b,"MAE_improvement_pct":float((1-a['MAE_kWh']/b['MAE_kWh'])*100),"old_issue_day_complete_energy_feature_removed":True}

def main():
    gjv3=load('gj_v3',ROOT/'김제_준비_2026-09-01'/'ultra_short_term_v3_gimje_structuretuning_2026-09-01.py'); gj=gjv3.v2
    ygv3=load('yg_v3',ROOT/'영광_준비_2026-09-03'/'ultra_short_term_v3_yeonggwang_weather_regime_2026-09-07.py'); yg=ygv3.V1
    gjm=load('gj_med',ROOT/'김제_준비_2026-09-01'/'medium_term_daily_v1_gimje_2026-09-01.py')
    ygm=load('yg_med',ROOT/'영광_준비_2026-09-03'/'medium_term_daily_v1_yeonggwang_2026-09-07.py')
    result={"부안":{"초단기":run_ultra_buan(),"단기_D+1":"09-08 issue-safe 완료","중장기":{"status":"not_built_no_existing_model"}},"김제":{"초단기":run_ultra_model('김제',gj),"단기_D+1":"09-08 issue-safe 완료","중장기_일간":run_medium('김제',gjm),"월간연간":"일간예측 롤업이며 독립 예측모델 아님"},"영광":{"초단기":run_ultra_model('영광',yg,True),"단기_D+1":"09-08 issue-safe 완료","중장기_일간":run_medium('영광',ygm),"월간연간":"일간예측 롤업이며 독립 예측모델 아님"}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'결과.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
