"""광주 +1h 모델의 VIF·쌍상관 임계값 민감도 비교.

임계값은 문헌 권고를 확정 규칙으로 간주하지 않고, 동일 시간순 홀드아웃에서
성능·특성수·제거특성을 비교해 프로젝트 채택 근거를 만든다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
RETRAIN = ROOT / "03_모델학습" / "현재_종합파이프라인" / "광주_준비_2026-09-08" / "retrain_단기_v1_segmented_2026-09-09.py"
OUT = Path(__file__).resolve().parent / "outputs" / "광주_다중공선성_임계값비교_v1_2026-09-09"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_retrain():
    spec = importlib.util.spec_from_file_location("gw_retrain", RETRAIN)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def vif(x: pd.DataFrame) -> pd.Series:
    z = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1).dropna()
    result = {}
    for col in z:
        y = z[col].to_numpy(); others = z.drop(columns=col).to_numpy()
        r2 = LinearRegression().fit(others, y).score(others, y) if others.shape[1] else 0
        result[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(result)


def prune(train: pd.DataFrame, features: list[str], vif_cut: float | None, corr_cut: float | None):
    if vif_cut is None or corr_cut is None:
        return features, []
    x = train[features].dropna()
    vv = vif(x); cm = x.corr(); ty = train[features + ["목표_발전출력_kW"]].corr()["목표_발전출력_kW"].abs()
    severe = set(vv[vv >= vif_cut].index); drop = set()
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            if a in cm and b in cm and abs(cm.loc[a, b]) >= corr_cut:
                weaker = a if ty.get(a, 0) < ty.get(b, 0) else b
                if weaker in severe: drop.add(weaker)
    return [c for c in features if c not in drop], sorted(drop)


def main():
    mod = load_retrain(); OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(mod.V4_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    f = mod.build_frame_segment_aware(df, 1)
    features = mod.FINAL12_LIVE + ["발전출력_1시간전_kW", "발전출력_2시간전_kW", "발전출력_3시간전_kW",
        "발전출력_6시간전_kW", "발전출력_24시간전_kW", "발전출력_6시간이동평균_kW", "발전출력_6시간이동표준편차_kW",
        "발전출력_24시간이동평균_kW", "발전출력_24시간이동표준편차_kW", "목표_태양고도_deg", "일주기_sin", "일주기_cos", "연주기_sin", "연주기_cos"]
    u = f[(f.segment_id == "seg1_710d") & (~f.boundary_excluded) & f["목표_낮시간"].astype(bool)].dropna(subset=features + ["목표_발전출력_kW"]).sort_index()
    cut = u.index.max() - pd.Timedelta(days=30); train, test = u[u.index < cut], u[u.index >= cut]
    rows = []
    settings = [(None, None)] + [(v, c) for v in (5., 10.) for c in (.7, .8, .9)]
    for vc, cc in settings:
        keep, dropped = prune(train, features, vc, cc)
        m = LGBMRegressor(n_estimators=220, learning_rate=.04, num_leaves=31, min_child_samples=30,
            subsample=.9, colsample_bytree=.9, reg_lambda=.3, random_state=42, n_jobs=4, verbosity=-1)
        m.fit(train[keep], train["목표_발전출력_kW"]); p = np.clip(m.predict(test[keep]), 0, None)
        e = p - test["목표_발전출력_kW"].to_numpy()
        rows.append({"VIF_cutoff":"없음" if vc is None else vc, "pair_corr_cutoff":"없음" if cc is None else cc,
            "features_kept":len(keep), "features_dropped":len(dropped), "dropped":" | ".join(dropped),
            "n_test":len(test), "MAE_kW":float(np.abs(e).mean()), "RMSE_kW":float(np.sqrt(np.mean(e**2)))})
    result = pd.DataFrame(rows).sort_values("MAE_kW")
    result.to_csv(OUT / "VIF_쌍상관_임계값_성능비교.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 5.4)); labels = [f"VIF {r.VIF_cutoff}\nr {r.pair_corr_cutoff}\n({r.features_kept}개)" for r in result.itertuples()]
    bars = ax.bar(labels, result.MAE_kW, color=["#2563eb" if i == 0 else "#94a3b8" for i in range(len(result))])
    ax.bar_label(bars, fmt="%.2f", padding=3); ax.set(ylabel="MAE (kW)", title="광주 +1h: 다중공선성 임계값 민감도 (동일 시간순 홀드아웃)")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(OUT / "01_VIF_쌍상관_임계값_MAE.png", dpi=180); plt.close(fig)
    best = result.iloc[0].to_dict(); (OUT / "결론.json").write_text(json.dumps({"status":"민감도비교완료_공식변경전", "best":best,
        "interpretation":"국내 문헌의 VIF 기준은 연구별로 다르므로 고정 정답으로 인용하지 않고 동일 홀드아웃 실증으로 선택한다.",
        "source_dataset":str(mod.V4_CSV)}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__": main()
