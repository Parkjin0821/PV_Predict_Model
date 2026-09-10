"""Recompute metrics and figures from saved out-of-fold predictions (no training)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
models=['전월지속성','전년동월','계절평균','선형회귀','Ridge','LightGBM','GBHuber']
def metrics(df,col):
    x=df[['actual_kWh',col]].dropna()
    e=x[col]-x.actual_kWh
    return {'n':len(x),'MAE_kWh':float(e.abs().mean()),'RMSE_kWh':float(np.sqrt((e**2).mean())),
            'WAPE_pct':float(e.abs().sum()/x.actual_kWh.abs().sum()*100)}

results={}
for tag,region in [('gwangju','광주'),('yeonggwang','영광')]:
    df=pd.read_csv(ROOT/'results'/f'monthly_{tag}_oof.csv',parse_dates=['month'])
    rows=[]
    for m in models:
        pair=df.dropna(subset=[m+'_kWh','GBHuber_kWh','actual_kWh'])
        for name in dict.fromkeys([m,'GBHuber']):
            rows.append({'comparison':m,'model':name,**metrics(pair,name+'_kWh')})
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'monthly_{tag}_paired_metrics.csv',index=False,encoding='utf-8-sig')
    results[tag]={h:{m:metrics(sub,m+'_kWh') for m in models} for h,sub in [('전체',df),('상반기',df[df.half=='상반기']),('하반기',df[df.half=='하반기'])]}
    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
    for m,label,color in [('actual','실측','#172b4d'),('GBHuber','GB-Huber 후보','#d65f27'),('전년동월','전년동월 기준선','#3a8795'),('Ridge','Ridge 기준선','#9272a5')]:
        col='actual_kWh' if m=='actual' else m+'_kWh'
        ax.plot(df.month,df[col],marker='o',markersize=4,label=label,color=color)
    ax.set(title=f'{region} 월간 예측 — 저장된 walk-forward OOF 재시각화',ylabel='월간 총발전량 (kWh)')
    ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2);ax.tick_params(axis='x',rotation=30)
    fig.savefig(ROOT/'figures'/f'monthly_{tag}_oof.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    for ax,(label,sub) in zip(axes,[('전체',df),('상반기',df[df.half=='상반기']),('하반기',df[df.half=='하반기'])]):
        names=['전월지속성','Ridge','GBHuber']
        vals=[metrics(sub,m+'_kWh')['MAE_kWh'] for m in names]
        ax.bar(names,vals,color=['#74879e','#9272a5','#d65f27'])
        ax.set(title=f'{label} · n={len(sub)}',ylabel='MAE (kWh)');ax.tick_params(axis='x',rotation=20)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:,.0f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'{region} 계절별 성능 — 각 패널 안에서만 동일 표본 비교')
    fig.savefig(ROOT/'figures'/f'monthly_{tag}_season.png',dpi=160);plt.close(fig)
(ROOT/'results'/'monthly_recomputed_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('Monthly OOF metrics recomputed; 4 figures saved. No model retrained.')
