"""광주 총출력 이상치 정책 비교: 물리범위·IQR·동료인버터."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA=Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\v3_multihorizon_2026-08-14\03_분석데이터\gwangju_5min_model_dataset.csv")
PEER=Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인\outputs\v6_동료대조_후보_2026-09-01\v6_제외전_플랜트값_150.csv")
OUT=Path(__file__).resolve().parent/"outputs"/"광주_이상치정책_비교_v1_2026-09-09"
CAP=240.0
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False

def hourly(d,y):
    n=y.notna().resample("1h").sum(); h=pd.DataFrame(index=n.index); h["target"]=y.resample("1h").mean().where(n>=9)
    for c in ["observed_weather_temperature_2m","observed_weather_relative_humidity_2m","observed_weather_cloud_cover","observed_weather_shortwave_radiation","solar_elevation_deg"]: h[c]=d[c].resample("1h").mean()
    for lag in [1,2,3,6,24]: h[f"lag_{lag}h"]=h.target.shift(lag)
    h["roll6_mean"]=h.target.shift(1).rolling(6,min_periods=6).mean(); h["roll24_mean"]=h.target.shift(1).rolling(24,min_periods=24).mean()
    h["hour_sin"]=np.sin(2*np.pi*h.index.hour/24); h["hour_cos"]=np.cos(2*np.pi*h.index.hour/24)
    h["doy_sin"]=np.sin(2*np.pi*h.index.dayofyear/365.25); h["doy_cos"]=np.cos(2*np.pi*h.index.dayofyear/365.25)
    return h

def main():
    OUT.mkdir(parents=True,exist_ok=True); d=pd.read_csv(DATA,parse_dates=["time"],low_memory=False).set_index("time").sort_index()
    y=pd.to_numeric(d.plant_output_kw,errors="coerce"); split=d.index.max()-pd.Timedelta(days=90)
    train_day=(d.index<split)&(d.solar_elevation_deg>0)&y.notna(); q1,q3=y[train_day].quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
    peer_times=set(pd.to_datetime(pd.read_csv(PEER,usecols=["grid_time_kst"])["grid_time_kst"]))
    physical=~y.between(0,CAP); global_iqr=(d.solar_elevation_deg>0)&~y.between(lo,hi)
    # 태양고도별 정상 출력 분포 차이를 보존하는 조건부 IQR(학습기간에서만 경계 산출).
    bins=pd.cut(d.solar_elevation_deg,[-90,0,10,20,30,40,50,60,90],include_lowest=True)
    conditional=pd.Series(False,index=d.index)
    for b in bins.dropna().unique():
        tr=train_day&(bins==b); vals=y[tr]
        if len(vals)<100: continue
        a,c=vals.quantile([.25,.75]); l,u=a-1.5*(c-a),c+1.5*(c-a); conditional|=(bins==b)&~y.between(l,u)
    peer=pd.Series(d.index.isin(peer_times),index=d.index)
    masks={"무필터":pd.Series(False,index=d.index),"물리범위_0_240":physical,"전역_IQR_1.5":global_iqr,
           "태양고도조건부_IQR_1.5":conditional,"동료인버터_150행":peer,"물리+동료":physical|peer}
    frames={k:hourly(d,y.mask(m)) for k,m in masks.items()}; features=[c for c in frames["무필터"] if c!="target"]
    common=frames["무필터"].loc[split:].dropna(subset=["target"]).index
    # 정책 비교 채점의 정답은 원시 5대 완전관측률 75% 이상인 시간만 사용.
    raw_complete=d.inverters_raw_observed.eq(d.reported_inverter_count).resample("1h").mean(); common=common.intersection(raw_complete[raw_complete>=.75].index)
    truth=frames["무필터"].loc[common,"target"]
    rows=[]; errors={}
    for name,h in frames.items():
        train=h.loc[h.index<split].dropna(subset=["target"]); test=h.loc[common]
        m=LGBMRegressor(n_estimators=220,learning_rate=.04,num_leaves=31,min_child_samples=30,subsample=.9,colsample_bytree=.9,reg_lambda=.3,random_state=42,n_jobs=4,verbosity=-1)
        m.fit(train[features],train.target); p=np.clip(m.predict(test[features]),0,CAP); e=np.abs(p-truth.to_numpy()); errors[name]=e
        mask=masks[name]; rows.append({"policy":name,"removed_5min_rows":int((mask&y.notna()).sum()),"removed_daylight_rows":int((mask&(d.solar_elevation_deg>0)&y.notna()).sum()),
            "removed_high_solar_gt40deg":int((mask&(d.solar_elevation_deg>40)&y.notna()).sum()),"train_rows":len(train),"test_rows":len(test),"MAE_kW":e.mean(),"RMSE_kW":float(np.sqrt(np.mean((p-truth.to_numpy())**2)))})
    r=pd.DataFrame(rows).sort_values("MAE_kW"); base=errors["물리범위_0_240"]; rng=np.random.default_rng(42)
    for i,row in r.iterrows():
        # 공통 시험행 중 정책별 target 누락 가능성을 피하려고 길이가 다르면 비교 CI는 생략.
        e=errors[row.policy]
        if len(e)==len(base):
            diff=e-base; boot=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(2000)]
            r.loc[i,"delta_MAE_vs_physical_kW"]=diff.mean(); r.loc[i,"delta_CI95_low_kW"],r.loc[i,"delta_CI95_high_kW"]=np.quantile(boot,[.025,.975])
    r.to_csv(OUT/"이상치정책별_제거행_모델성능.csv",index=False,encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(10,5.4)); bars=ax.bar(r.policy,r.MAE_kW,color=["#2563eb" if i==0 else "#94a3b8" for i in range(len(r))]); ax.bar_label(bars,fmt="%.2f",padding=3)
    ax.set(ylabel="MAE (kW)",title="광주 +1h: 이상치 정책별 동일 홀드아웃 성능"); ax.tick_params(axis="x",rotation=18); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(OUT/"01_이상치정책별_MAE.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.4)); z=r.set_index("policy")[["removed_daylight_rows","removed_high_solar_gt40deg"]]; z.plot.bar(ax=ax,color=["#f97316","#dc2626"]); ax.set(ylabel="제거한 5분 행",title="이상치 정책별 제거 규모와 고태양고도 행 포함"); ax.tick_params(axis="x",rotation=18); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(OUT/"02_이상치정책별_제거행.png",dpi=180); plt.close(fig)
    (OUT/"결론.json").write_text(json.dumps({"status":"비교완료_공식변경전","global_iqr_bounds_kw":[float(lo),float(hi)],"best":r.iloc[0].to_dict(),
        "caution":"동료인버터 150행은 기존 v6 감사결과 재사용. IQR 경계는 학습기간에서만 산출. 한 지역 +1h 결과를 타지역·타수평에 일반화 금지."},ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print('IQR',lo,hi); print(r.to_string(index=False))

if __name__=="__main__": main()
