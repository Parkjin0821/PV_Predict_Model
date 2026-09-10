"""부안·김제·영광 인버터 실측 마스킹 복원 비교."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

BASE=Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델")
PATHS={"부안":BASE/r"부안\시간정렬_v1_2026-08-28\부안_인버터별_5분정렬.parquet",
       "김제":BASE/r"김제\시간정렬_v1_2026-08-31\김제_인버터별_5분정렬.parquet",
       "영광":BASE/r"영광\시간정렬_v1_2026-09-01\영광_인버터별_5분정렬.parquet"}
OUT=Path(__file__).resolve().parent/"outputs"/"3지역_결측보간_마스킹복원_v1_2026-09-09"
GAPS=[1,2,3,6]; N=500; plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False

def preds(y,s,n):
    q=np.arange(s,s+n); l,r=s-1,s+n
    side=np.r_[y[s-3:s],y[r:r+3]]
    return {"직전값":np.repeat(y[l],n),"선형":np.interp(q,[l,r],[y[l],y[r]]),"이동평균":np.repeat(np.mean(side),n),
            "스플라인":CubicSpline([s-2,s-1,r,r+1],y[[s-2,s-1,r,r+1]],bc_type="natural")(q)}

def main():
    OUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(42); rows=[]
    for region,path in PATHS.items():
        d=pd.read_parquet(path,columns=["grid_time_kst","inverter_number","ac_power_kw","was_observed"])
        d["grid_time_kst"]=pd.to_datetime(d.grid_time_kst)
        for inv,g in d.groupby("inverter_number"):
            g=g.sort_values("grid_time_kst").reset_index(drop=True); y=g.ac_power_kw.to_numpy(float); obs=g.was_observed.to_numpy(bool); t=g.grid_time_kst
            dt=t.diff().eq(pd.Timedelta(minutes=5)).to_numpy()
            for n in GAPS:
                cand=np.arange(3,len(g)-n-3)
                ok=np.ones(len(cand),dtype=bool); good=obs&np.isfinite(y)
                for off in range(-3,n+3): ok&=good[cand+off]
                for off in range(-2,n+3): ok&=dt[cand+off]
                positive=np.zeros(len(cand),dtype=bool)
                for off in range(n): positive|=y[cand+off]>0
                valid=cand[ok&positive]
                if len(valid)==0: continue
                take=rng.choice(valid,size=min(max(1,N//d.inverter_number.nunique()),len(valid)),replace=False)
                for s in take:
                    truth=y[s:s+n]
                    for method,p in preds(y,int(s),n).items():
                        e=p-truth; rows.append({"region":region,"inverter":int(inv),"gap_minutes":n*5,"method":method,"mae_kw":np.abs(e).mean(),"mse":np.mean(e**2),
                            "p95_base":np.max(np.abs(e)),"violations":int(((p<0)|(p>float(g.loc[s,"ac_power_kw"]+g.loc[s,"ac_power_kw"]+1000))).sum())})
    raw=pd.DataFrame(rows); summary=raw.groupby(["region","gap_minutes","method"],as_index=False).agg(samples=("mae_kw","size"),MAE_kW=("mae_kw","mean"),RMSE_kW=("mse",lambda x:np.sqrt(np.mean(x))),P95_max_error_kW=("p95_base",lambda x:np.quantile(x,.95)))
    raw.to_csv(OUT/"개별결과.csv",index=False,encoding="utf-8-sig"); summary.to_csv(OUT/"3지역_보간법_공백길이별_비교.csv",index=False,encoding="utf-8-sig")
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for ax,(region,z) in zip(axs,summary.groupby("region")):
        for method,g in z.groupby("method"): ax.plot(g.gap_minutes,g.MAE_kW,"o-",label=method)
        ax.set(title=region,xlabel="공백 길이(분)"); ax.grid(alpha=.25)
    axs[0].set_ylabel("복원 MAE (kW/인버터)"); axs[-1].legend(); fig.suptitle("3지역 인버터 실측 마스킹 복원 비교"); fig.tight_layout(); fig.savefig(OUT/"01_3지역_보간법_MAE.png",dpi=180); plt.close(fig)
    print(summary.to_string(index=False))

if __name__=="__main__": main()
