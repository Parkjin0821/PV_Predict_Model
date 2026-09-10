from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_packages"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

FONT = Path("C:/Windows/Fonts/malgun.ttf")
if FONT.exists():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
ORANGE, DARK, GRAY, BLUE = "#F58220", "#222222", "#6B7280", "#2F6FB0"


def save(fig, name):
    fig.savefig(ROOT/"figures"/name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def correlation_heatmap():
    corr = pd.read_csv(ROOT/"outputs"/"correlation_pearson_daylight.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,
                    color="white" if abs(v)>.55 else DARK)
    fig.colorbar(im, ax=ax, fraction=.045, pad=.03, label="Pearson r")
    ax.set_title("광주 15분 데이터 낮 시간 상관분석", fontsize=18, fontweight="bold", pad=16)
    fig.text(.5,.015,"정적 지형·설비값은 단일 발전소 내에서 변하지 않아 상관계수를 계산할 수 없음",
             ha="center",color=GRAY,fontsize=10)
    save(fig,"01_15분_낮시간_상관행렬.png")


def horizon_summary():
    specs=[("5min","5분 갱신 · 1~4시간",60),("15min","15분 갱신 · 1~4시간",60),("1hour","1시간 갱신 · 24~48시간",60)]
    fig, axes=plt.subplots(1,3,figsize=(17,5.5))
    for ax,(name,title,scale) in zip(axes,specs):
        s=pd.read_csv(ROOT/"outputs"/f"{name}_model_scorecard.csv")
        selected=s[s["selected_on_validation"].astype(str).str.lower().eq("true")].sort_values("horizon_minutes")
        x=selected["horizon_minutes"]/scale
        ax.plot(x,selected["daylight_R2"],color=ORANGE,marker="o",lw=2.4,label="R²")
        ax.axhline(.7,color=GRAY,ls="--",lw=1.2,label="목표 0.7")
        ax.set_ylim(min(.65,selected["daylight_R2"].min()-.03),min(1,selected["daylight_R2"].max()+.05))
        ax.set_title(title,fontweight="bold"); ax.set_xlabel("예측수평(시간)"); ax.set_ylabel("낮 시간 R²")
        ax.grid(alpha=.22); ax.legend(loc="best")
    fig.suptitle("시간수평별 광주 PV 예측 성능",fontsize=20,fontweight="bold",y=1.02)
    save(fig,"02_시간수평별_R2.png")


def prediction_examples():
    fig,axes=plt.subplots(3,1,figsize=(15,11),sharex=False)
    for ax,name,horizon,title in [
        (axes[0],"5min",60,"5분 갱신 · 1시간 앞"),
        (axes[1],"15min",240,"15분 갱신 · 4시간 앞"),
        (axes[2],"1hour",1440,"1시간 갱신 · 24시간 앞"),
    ]:
        p=pd.read_csv(ROOT/"outputs"/f"{name}_selected_predictions.csv",parse_dates=["target_time"])
        p=p[p["horizon_minutes"].eq(horizon)].sort_values("target_time")
        cutoff=p["target_time"].max()-pd.Timedelta(days=5)
        p=p[p["target_time"]>=cutoff]
        ax.plot(p["target_time"],p["actual_kw"],color=DARK,lw=2,label="실제")
        ax.plot(p["target_time"],p["predicted_kw"],color=ORANGE,lw=1.7,label="예측")
        ax.set_title(title,fontweight="bold",loc="left"); ax.set_ylabel("AC 출력(kW)")
        ax.grid(alpha=.2); ax.legend(ncol=2,loc="upper right")
    fig.suptitle("시험구간 실제값과 예측값 예시",fontsize=20,fontweight="bold",y=.995)
    fig.tight_layout()
    save(fig,"03_실제값_예측값_비교.png")


def feature_effects():
    bundle=joblib.load(ROOT/"models"/"15min"/"horizon_004.joblib")
    features=bundle["features"]
    linear=bundle["models"]["multiple_regression_ridge"].named_steps["model"]
    coef=pd.Series(linear.coef_,index=features,name="standardized_coefficient").sort_values(key=np.abs,ascending=False)
    light=pd.Series(bundle["models"]["lightgbm"].feature_importances_,index=features,name="lightgbm_importance").sort_values(ascending=False)
    pd.concat([coef,light],axis=1).to_csv(ROOT/"outputs"/"15min_1hour_feature_effects.csv",encoding="utf-8-sig")
    top=coef.head(14).sort_values()
    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    axes[0].barh(range(len(top)),top.values,color=[BLUE if v<0 else ORANGE for v in top.values])
    def label(x):
        return x.replace("target_forecast_","예보_").replace("current_weather_","현재기상_").replace("equipment_current_","설비_")
    axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels([label(x) for x in top.index],fontsize=9)
    axes[0].set_title("Ridge 다중회귀 표준화 계수",fontweight="bold"); axes[0].axvline(0,color=DARK,lw=.8)
    top2=light.head(14).sort_values()
    axes[1].barh(range(len(top2)),top2.values,color=ORANGE)
    axes[1].set_yticks(range(len(top2))); axes[1].set_yticklabels([label(x) for x in top2.index],fontsize=9)
    axes[1].set_title("LightGBM 중요도",fontweight="bold")
    fig.suptitle("15분 갱신 · 1시간 앞 예측 설명요인",fontsize=19,fontweight="bold")
    fig.text(.5,.01,"회귀계수와 변수중요도는 예측 기여도이며 인과효과를 의미하지 않음",ha="center",color=GRAY)
    fig.tight_layout()
    save(fig,"04_다중회귀계수_변수중요도.png")


def daily_performance():
    p=pd.read_csv(ROOT/"outputs"/"daily_selected_predictions.csv",parse_dates=["date"])
    summary=json.loads((ROOT/"outputs"/"model_run_summary.json").read_text(encoding="utf-8"))["daily"]
    fig,ax=plt.subplots(figsize=(14,6))
    ax.plot(p["date"],p["actual_kwh"],color=DARK,lw=2,label="실제")
    ax.plot(p["date"],p["predicted_kwh"],color=ORANGE,lw=2,label="익일 예측")
    ax.set_title(f"일간 누적 발전량 예측 · R² {summary['test_R2']:.3f}",fontsize=19,fontweight="bold")
    ax.set_ylabel("일 발전량(kWh)"); ax.grid(alpha=.22); ax.legend(ncol=2)
    ax.text(.01,.96,f"MAE {summary['test_MAE_kWh']:.1f} kWh  |  RMSE {summary['test_RMSE_kWh']:.1f} kWh  |  MAPE {summary['test_MAPE_pct']:.1f}%",
            transform=ax.transAxes,va="top",fontsize=11,bbox=dict(facecolor="white",edgecolor="#DDDDDD",pad=7))
    save(fig,"05_일간_익일예측_성능.png")


def metric_table():
    rows=[]
    for name,label in [("5min","5분"),("15min","15분"),("1hour","1시간")]:
        s=pd.read_csv(ROOT/"outputs"/f"{name}_model_scorecard.csv")
        s=s[s["selected_on_validation"].astype(str).str.lower().eq("true")]
        for h in ([60,120,180,240] if name!="1hour" else [1440,2160,2880]):
            r=s.iloc[(s["horizon_minutes"]-h).abs().argsort()[:1]].iloc[0]
            rows.append([label,f"{int(r.horizon_minutes/60)}시간",r.model,
                         f"{r.daylight_MAE_kW:.1f}",f"{r.daylight_RMSE_kW:.1f}",f"{r.daylight_R2:.3f}"])
    daily=json.loads((ROOT/"outputs"/"model_run_summary.json").read_text(encoding="utf-8"))["daily"]
    rows.append(["일간","익일",daily["selected_model"],f"{daily['test_MAE_kWh']:.1f} kWh",f"{daily['test_RMSE_kWh']:.1f} kWh",f"{daily['test_R2']:.3f}"])
    fig,ax=plt.subplots(figsize=(13,6)); ax.axis("off")
    table=ax.table(cellText=rows,colLabels=["갱신주기","예측수평","선택모델","MAE","RMSE","R²"],cellLoc="center",loc="center",colWidths=[.12,.12,.25,.14,.14,.1])
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.75)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        if i==0: cell.set_facecolor(ORANGE);cell.set_text_props(color="white",weight="bold")
        elif i%2: cell.set_facecolor("#FFF7F0")
    ax.set_title("광주 PV 예측모델 핵심 성능",fontsize=20,fontweight="bold",pad=15)
    fig.text(.5,.06,"5분·15분·1시간은 낮 시간 AC 출력 기준, 일간은 누적 발전량 기준",ha="center",color=GRAY)
    save(fig,"06_핵심성능_요약표.png")


def main():
    correlation_heatmap(); horizon_summary(); prediction_examples(); feature_effects(); daily_performance(); metric_table()
    print("created 6 figures")


if __name__=="__main__": main()
