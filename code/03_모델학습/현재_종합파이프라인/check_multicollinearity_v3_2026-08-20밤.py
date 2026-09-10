# -*- coding: utf-8 -*-
"""5-2번 재실행(08-20 밤, 3차) — v3(라벨규약 수정) 18개 특성의 다중공선성+배포가능성."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

spec = importlib.util.spec_from_file_location(
    "sel_v3", Path(__file__).parent / "select_features_by_correlation_threshold_v3_2026-08-20밤.py"
)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

CORR_DIR = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\correlation_threshold_selection_v3_2026-08-20밤")
OUT_DIR = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\multicollinearity_check_v3_2026-08-20밤")
VIF_SEVERE, PAIR_CORR_HIGH = 10.0, 0.8

SELECTED_18 = [
    "plant_input_power_kw", "plant_input_current_a", "DSWRF", "기상청관측_일사량_W_m2",
    "기상청관측_일조시간_hr", "mean_power_factor", "REH", "mean_input_voltage_v",
    "추정_출력온도", "기상청관측_상대습도_pct", "추정_모듈표면온도", "기상청관측_전운량_pct",
    "mean_inverter_temperature_c", "DSWRFLX_bsrn정제", "LCDC", "TCDC", "POP", "SKY",
]


def compute_vif(x: pd.DataFrame) -> pd.Series:
    x = (x - x.mean()) / x.std(ddof=0)
    vifs = {}
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity_kw = float(df["reported_capacity_kw"].dropna().iloc[0])
    cand = sel.build_candidate_frame(df)
    target = df["plant_output_kw"]
    daylight = (df["solar_elevation_deg"] > 0).astype(float)
    target_corr = pd.read_csv(CORR_DIR / "변수별_상관계수_낮시간.csv").set_index("변수")["상관계수"]

    daylight_cand = cand[daylight > 0]
    x = daylight_cand[SELECTED_18].dropna()
    print(f"VIF 계산용 완전표본: {len(x)}")

    vif = compute_vif(x)
    print("\n[1] VIF"); print(vif.to_string())

    corr_matrix = x.corr(method="pearson")
    high_pairs = []
    cols = corr_matrix.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr_matrix.loc[a, b]
            if abs(r) >= PAIR_CORR_HIGH:
                weaker = a if abs(target_corr.get(a, 0)) < abs(target_corr.get(b, 0)) else b
                high_pairs.append({"변수A": a, "변수B": b, "쌍상관계수": float(r),
                                    "A_타깃상관": float(target_corr.get(a, np.nan)),
                                    "B_타깃상관": float(target_corr.get(b, np.nan)),
                                    "제거후보": weaker})
    pairs_df = pd.DataFrame(high_pairs).sort_values("쌍상관계수", key=abs, ascending=False) if high_pairs else pd.DataFrame()
    print("\n[2] |r|>=0.8 변수쌍"); print(pairs_df.to_string(index=False) if len(pairs_df) else "(없음)")

    drop_candidates = sorted(set(pairs_df["제거후보"])) if len(pairs_df) else []
    severe_vif = set(vif[vif >= VIF_SEVERE].index)
    final_drop = [c for c in drop_candidates if c in severe_vif]
    print(f"\n[3] 최종 가지치기(쌍상관>=0.8 & VIF>=10): {final_drop}")

    pruned = [c for c in SELECTED_18 if c not in final_drop]

    def evaluate(name, cols):
        base = sel.make_base_features(df)
        for col in cols:
            base[col] = cand[col]
        base["목표_발전출력_kW"] = target
        base["목표_낮시간"] = daylight
        features = [c for c in base.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
        f = base.dropna(subset=features + ["목표_발전출력_kW"])
        f = f[f["목표_낮시간"] > 0]
        n = len(f); cut = int(n * 0.85)
        train, test = f.iloc[:cut], f.iloc[cut:]
        model = LGBMRegressor(n_estimators=220, learning_rate=.04, num_leaves=31, min_child_samples=30,
                               subsample=.9, colsample_bytree=.9, reg_lambda=.3, random_state=42, n_jobs=4, verbosity=-1)
        model.fit(train[features], train["목표_발전출력_kW"])
        pred = np.clip(model.predict(test[features]), 0, capacity_kw)
        e = test["목표_발전출력_kW"].to_numpy() - pred
        return dict(구성=name, 특성수=len(cols), 학습=len(train), 시험=len(test),
                    MAE=round(float(np.abs(e).mean()), 3), RMSE=round(float(np.sqrt((e**2).mean())), 3))

    print("\n[4] 원본 vs 가지치기 vs 배포가능 ablation 재검증")
    deploy_safe = [c for c in pruned if c != "mean_inverter_temperature_c"]
    results = [
        evaluate("A_원본18개", SELECTED_18),
        evaluate("B_가지치기", pruned),
        evaluate("C_배포가능(온도제외)", deploy_safe),
    ]
    scorecard = pd.DataFrame(results)
    print(scorecard.to_string(index=False))

    vif.to_csv(OUT_DIR / "VIF.csv", header=["VIF"], encoding="utf-8-sig")
    pairs_df.to_csv(OUT_DIR / "강한상관쌍.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(OUT_DIR / "성능비교.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "요약.json").write_text(json.dumps({
        "VIF": vif.replace(np.inf, None).to_dict(), "강한상관쌍": high_pairs,
        "최종가지치기": final_drop, "가지치기후": pruned, "배포가능_최종": deploy_safe,
        "성능비교": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT_DIR}")
    print(f"\n★최종 배포가능 특성({len(deploy_safe)}개): {deploy_safe}")


if __name__ == "__main__":
    main()
