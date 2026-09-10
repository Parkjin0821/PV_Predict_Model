# -*- coding: utf-8 -*-
"""5번 후속: 타깃 상관계수(|r|>=0.2)로 채택한 외생변수 20개 내부에서
다중공선성(multicollinearity)을 점검한다 — 사용자 지적(08-20): "타깃과의
상관계수만 봤지 피처끼리의 상관은 안 봤다."

방법(표준 통계 관행, 출처 명시)
1. VIF(Variance Inflation Factor): VIF_i = 1/(1-R_i^2), R_i^2는 변수 i를
   나머지 변수 전부로 회귀했을 때의 결정계수. 관행적 기준: VIF<5 양호,
   5<=VIF<10 주의, VIF>=10 심각(O'Brien 2007, "A Caution Regarding Rules
   of Thumb for Variance Inflation Factors", Quality & Quantity 41).
   단, 이 논문 자체가 "VIF만 보고 기계적으로 변수를 빼면 안 된다"는
   경고이므로 여기서도 VIF만으로 자동 제거하지 않고 참고 지표로만 쓴다.
2. 변수쌍 Pearson 상관행렬: |r|>=0.8을 "강한 중복" 관행적 기준으로 병기
   (일반 통계 관행, VIF와 상호보완적으로 같이 보는 것이 표준적).
3. 최종 판단: VIF와 쌍상관 둘 다 높게 나오는 변수쌍에 대해서만, 타깃
   상관계수(지난 단계 산출)가 더 낮은 쪽을 제거 후보로 표시한다. LightGBM
   같은 트리 모델은 다중공선성에 예측정확도 자체는 비교적 안정적이지만
   (분할 기준을 아무 변수로나 대체 가능), 특성중요도 해석과 6번 단계에서
   비교 예정인 Ridge 같은 선형모델에는 영향이 있으므로 미리 짚어둔다.
4. 검증: 원래 20개 특성 세트와, 다중공선성 기준으로 가지치기한 세트로
   동일한 +1시간 ablation을 다시 돌려 MAE·RMSE가 유지/개선되는지 확인한다
   (임계값 튜닝 때와 같은 방식 — 그냥 이론적으로 좋다고 하지 않고 실측 검증).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

DATA_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v1_2026-08-20.csv"
)
CORR_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\correlation_threshold_selection_v1_2026-08-20"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\multicollinearity_check_v1_2026-08-20"
)
SEED = 42

# 08-20 상관분석(|r|>=0.2)에서 채택된 외생변수 20개
SELECTED_20 = [
    "plant_input_power_kw", "plant_input_current_a", "기상청관측_일사량_W_m2",
    "mean_power_factor", "기상청관측_일조시간_hr", "mean_inverter_temperature_c",
    "mean_input_voltage_v", "DSWRFLX_bsrn정제", "DSWRF", "기상청관측_상대습도_pct",
    "기상청관측_전운량_pct", "TCDC", "LCDC", "추정_일조시간_hr", "기상청관측_지면온도_C",
    "MCDC", "HCDC", "추정_출력온도", "추정_모듈표면온도", "DIFSWRF_bsrn정제",
]

VIF_WARN, VIF_SEVERE = 5.0, 10.0
PAIR_CORR_HIGH = 0.8


def compute_vif(x: pd.DataFrame) -> pd.Series:
    x = (x - x.mean()) / x.std(ddof=0)  # 표준화(VIF 값 자체엔 무관하지만 수치안정성)
    vifs = {}
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        reg = LinearRegression().fit(others, y)
        r2 = reg.score(others, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def make_base_features(df: pd.DataFrame) -> pd.DataFrame:
    power = df["plant_output_kw"]
    out = pd.DataFrame(index=df.index)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.shift(1).rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.shift(1).rolling(hours, min_periods=max(3, hours // 2)).std()
    out["목표_태양고도_deg"] = df["solar_elevation_deg"].shift(-1)
    minute = out.index.hour * 60 + out.index.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    return out


def evaluate_feature_set(name: str, cols: list[str], base: pd.DataFrame, df: pd.DataFrame, capacity_kw: float) -> dict:
    frame = base.copy()
    for col in cols:
        frame[f"직전_{col}"] = df[col].shift(1)
    frame["목표_발전출력_kW"] = df["plant_output_kw"].shift(-1)
    frame["목표_낮시간"] = (df["solar_elevation_deg"].shift(-1) > 0).astype(float)
    features = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
    frame = frame.dropna(subset=features + ["목표_발전출력_kW"])
    frame = frame[frame["목표_낮시간"] == 1]
    n = len(frame)
    cut = int(n * 0.85)
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    model = LGBMRegressor(
        n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3,
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    model.fit(train[features], train["목표_발전출력_kW"])
    pred = np.clip(model.predict(test[features]), 0, capacity_kw)
    err = test["목표_발전출력_kW"].to_numpy() - pred
    return {
        "구성": name, "특성수": len(cols), "학습표본수": int(len(train)), "시험표본수": int(len(test)),
        "MAE_kW": float(np.abs(err).mean()), "RMSE_kW": float(np.sqrt((err ** 2).mean())),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity_kw = float(df["reported_capacity_kw"].dropna().iloc[0])
    daylight = df[df["solar_elevation_deg"] > 0]

    target_corr = pd.read_csv(CORR_DIR / "변수별_상관계수_낮시간.csv").set_index("변수")["상관계수"]

    # 20개 변수 전부 값이 있는 공통 표본(중앙값 대체 없이, VIF는 완전표본 기준)
    x = daylight[SELECTED_20].dropna()
    print(f"VIF 계산용 완전표본 수: {len(x)} / {len(daylight)}")

    print("\n[1] VIF 계산")
    vif = compute_vif(x)
    print(vif.to_string())

    print("\n[2] 변수쌍 상관행렬에서 |r|>=0.8인 쌍")
    corr_matrix = x.corr(method="pearson")
    high_pairs = []
    cols = corr_matrix.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr_matrix.loc[a, b]
            if abs(r) >= PAIR_CORR_HIGH:
                weaker = a if abs(target_corr.get(a, 0)) < abs(target_corr.get(b, 0)) else b
                high_pairs.append({
                    "변수A": a, "변수B": b, "쌍상관계수": float(r),
                    "A_타깃상관": float(target_corr.get(a, np.nan)),
                    "B_타깃상관": float(target_corr.get(b, np.nan)),
                    "제거후보(타깃상관 더 약한 쪽)": weaker,
                })
    pairs_df = pd.DataFrame(high_pairs).sort_values("쌍상관계수", key=abs, ascending=False)
    print(pairs_df.to_string(index=False))

    drop_candidates = sorted(set(pairs_df["제거후보(타깃상관 더 약한 쪽)"])) if len(pairs_df) else []
    # VIF>=10이면서 위 제거후보에도 든 변수만 최종 가지치기 대상으로 확정(보수적)
    severe_vif = set(vif[vif >= VIF_SEVERE].index)
    final_drop = [c for c in drop_candidates if c in severe_vif]
    print(f"\n[3] 최종 가지치기 후보(쌍상관>=0.8 & VIF>=10 둘 다 해당): {final_drop}")

    pruned_cols = [c for c in SELECTED_20 if c not in final_drop]

    print("\n[4] 원본 20개 vs 가지치기 세트 ablation 재검증")
    base = make_base_features(df)
    results = [
        evaluate_feature_set("원본_20개(0.2임계값)", SELECTED_20, base, df, capacity_kw),
        evaluate_feature_set(f"가지치기_{len(pruned_cols)}개(다중공선성 제거)", pruned_cols, base, df, capacity_kw),
    ]
    scorecard = pd.DataFrame(results)
    print(scorecard.to_string(index=False))

    vif.to_csv(OUT_DIR / "VIF.csv", header=["VIF"], encoding="utf-8-sig")
    pairs_df.to_csv(OUT_DIR / "강한상관쌍_0.8이상.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(OUT_DIR / "가지치기_전후_성능비교.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "요약.json").write_text(
        json.dumps({
            "VIF": vif.to_dict(), "강한상관쌍": high_pairs,
            "최종가지치기": final_drop, "가지치기후_특성목록": pruned_cols,
            "성능비교": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
