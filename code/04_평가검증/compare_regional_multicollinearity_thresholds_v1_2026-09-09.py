"""부안·김제·영광 VIF·쌍상관 임계값 동일 홀드아웃 비교."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"03_모델학습"/"현재_종합파이프라인"/"retrain_regional_short_segmented_candidate_v1_2026-09-09.py"
OUT=Path(__file__).resolve().parent/"outputs"/"3지역_다중공선성_임계값비교_v1_2026-09-09"
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False

def load():
 s=importlib.util.spec_from_file_location("regional",SRC); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def vif(x):
 z=(x-x.mean())/x.std(ddof=0).replace(0,np.nan); z=z.dropna(axis=1).dropna(); out={}
 for c in z:
  a=z.drop(columns=c).to_numpy(); r2=LinearRegression().fit(a,z[c]).score(a,z[c]) if a.shape[1] else 0; out[c]=np.inf if r2>=1 else 1/(1-r2)
 return pd.Series(out)
def prune(tr,cols,vc,cc):
 if vc is None:return cols,[]
 x=tr[cols].dropna(); vv=vif(x); cm=x.corr(); ty=tr[cols+["plant_ac_power_kw"]].corr()["plant_ac_power_kw"].abs(); severe=set(vv[vv>=vc].index); drop=set()
 for i,a in enumerate(cols):
  for b in cols[i+1:]:
   if a in cm and b in cm and abs(cm.loc[a,b])>=cc:
    w=a if ty.get(a,0)<ty.get(b,0) else b
    if w in severe:drop.add(w)
 return [c for c in cols if c not in drop],sorted(drop)
def main():
 mod=load(); h=mod.load_harness(); OUT.mkdir(parents=True,exist_ok=True); rows=[]
 for region,cfg in mod.CFG.items():
  df,final,_=mod.build(cfg); old=json.loads(cfg["summary"].read_text(encoding="utf-8")); cand=[]
  for c in final+old.get("다중공선성_제거",[]):
   if c in df and c not in mod.LIVE_UNREPRODUCIBLE and c not in cand:cand.append(c)
  df["issue_day"]=df.issue_time_kst.dt.normalize(); days=pd.DatetimeIndex(sorted(df.issue_day.unique())); region_folds=mod.folds(days,max(30,len(days)//3))
  for vc,cc in [(None,None)]+[(v,c) for v in (5.,10.) for c in (.7,.8,.9)]:
   ys=[];ps=[];kept_counts=[];dropped=set();fold_count=0
   for trd,ted in region_folds:
    train=df[df.issue_day.isin(trd)].dropna(subset=cand+[mod.TARGET]); test=df[df.issue_day.isin(ted)].dropna(subset=cand+[mod.TARGET])
    if len(train)<100 or len(test)<5:continue
    keep,drop=prune(train,cand,vc,cc); model=h.make_model("LightGBM",42); model.fit(train[keep],train[mod.TARGET]); ps.append(np.clip(model.predict(test[keep]),0,cfg["capacity"]));ys.append(test[mod.TARGET].to_numpy());kept_counts.append(len(keep));dropped.update(drop);fold_count+=1
   y=np.concatenate(ys);p=np.concatenate(ps);e=p-y
   rows.append({"region":region,"VIF_cutoff":"없음" if vc is None else vc,"pair_corr_cutoff":"없음" if cc is None else cc,"features_kept_median":float(np.median(kept_counts)),"features_dropped_union":len(dropped),"dropped_union":" | ".join(sorted(dropped)),"folds":fold_count,"n_test":len(y),"MAE_kW":np.abs(e).mean(),"RMSE_kW":np.sqrt(np.mean(e**2))})
 r=pd.DataFrame(rows); r.to_csv(OUT/"3지역_VIF_쌍상관_성능비교.csv",index=False,encoding="utf-8-sig")
 fig,axs=plt.subplots(1,3,figsize=(16,5),sharey=False)
 for ax,(region,z) in zip(axs,r.groupby("region")):
  z=z.sort_values("MAE_kW"); labels=[f"{a}/{b}" for a,b in zip(z.VIF_cutoff,z.pair_corr_cutoff)]; bars=ax.bar(labels,z.MAE_kW,color=["#2563eb"]+["#94a3b8"]*(len(z)-1)); ax.set_title(region); ax.tick_params(axis="x",rotation=45); ax.grid(axis="y",alpha=.25); ax.bar_label(bars,fmt="%.1f",fontsize=8)
 axs[0].set_ylabel("MAE (kW)"); fig.suptitle("3지역 다중공선성 임계값 민감도 (VIF/|r|)"); fig.tight_layout(); fig.savefig(OUT/"01_3지역_VIF_쌍상관_MAE.png",dpi=180); plt.close(fig)
 best=r.loc[r.groupby("region").MAE_kW.idxmin()].to_dict("records"); (OUT/"결론.json").write_text(json.dumps({"status":"비교완료_공식변경전","best_by_region":best,"scope":"각 폴드 학습구간 내부 VIF 산출 expanding walk-forward pooled"},ensure_ascii=False,indent=2,default=str),encoding="utf-8"); print(r.sort_values(["region","MAE_kW"]).to_string(index=False))
if __name__=="__main__":main()
