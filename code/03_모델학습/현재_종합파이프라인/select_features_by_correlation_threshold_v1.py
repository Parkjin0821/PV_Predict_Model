# -*- coding: utf-8 -*-
"""5번: 상관분석 임계값을 튜닝 파라미터로 취급해, 여러 |r| 임계값 후보를
실제로 학습·검증해 RMSE가 가장 좋은 임계값을 채택한다.

근거: Kwon et al., "Impact of Correlation-based Feature Selection on
Photovoltaic Power Prediction," IEEE TIMES-iCON 2019 (한국 기상청 API
데이터·PV 발전량 예측, 우리 프로젝트와 조건 거의 동일). 이 논문은
|r| 임계값 0/0.1/0.2/0.3/0.4를 전부 실제로 돌려 RMSE로 비교했고 0.1이
최적이었다(필터 없음보다 33.7% 개선, 0.2 이상은 오히려 악화). 사용자
확정(08-20): "튜닝 파라미터로 진행해서 최적의 RMSE를 통해서 임계값을
설정" — 이 스크립트가 그 방법을 광주 데이터에 그대로 적용한다.

방법
1. 공식 병합 데이터셋(`processed/gwangju_1hour_model_dataset_official_v1_
   2026-08-20.csv`)에서 낮시간(태양고도>0)만 대상으로, plant_output_kw와
   각 "외생 변수"(기상관측·NWP예보·장비텔레메트리·태양방위)의 Pearson
   상관계수를 계산한다. lag/이동통계/태양고도/주기성 같은 구조적 기준
   특성(과거발전량+시간)은 상관분석 대상이 아니라 항상 포함하는 기준선으로
   둔다(이 프로젝트의 기존 ablation 관례 — 환경→시간→설비 순서로 추가와
   동일한 사고방식).
2. 후보 임계값 [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]마다: |r|>=임계값인
   외생변수만 골라 기준선 특성과 합쳐 LightGBM으로 +1시간 뒤 발전량을
   예측하는 모델을 학습하고, 시간순 85:15 분할로 낮시간 시험 MAE·RMSE를
   측정한다(Kwon et al.과 동일하게 RMSE를 기준으로 최적 임계값 선정,
   MAE도 같이 보고 — 이 프로젝트의 "MAPE만 쓰지 않는다" 원칙).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v1_2026-08-20.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\correlation_threshold_selection_v1_2026-08-20"
)
SEED = 42
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

CANDIDATE_COLUMNS = [
    # 기상청 ASOS 실측
    "기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_강수량_mm",
    "기상청관측_전운량_pct", "기상청관측_일사량_W_m2", "기상청관측_일조시간_hr",
    "기상청관측_풍속_m_s", "기상청관측_풍향_deg", "기상청관측_현지기압_hPa",
    "기상청관측_해면기압_hPa", "기상청관측_적설_cm", "기상청관측_지면온도_C",
    # 미래 NWP 계열(3시간 해상도, 자연 결측 포함)
    "DSWRF", "TCDC", "DSWRFLX_bsrn정제", "DIFSWRF_bsrn정제",
    "LCDC", "MCDC", "HCDC", "VEC",
    "추정_모듈표면온도", "추정_출력온도", "추정_일조시간_hr",
    # 장비 텔레메트리(08-20 재정의 반영됨)
    "plant_input_power_kw", "plant_input_current_a", "mean_input_voltage_v",
    "mean_frequency_hz", "mean_power_factor", "mean_inverter_temperature_c",
    "mean_communication_ok", "inverters_available",
    # 태양방위(지형·시간 경계 성격)
    "solar_azimuth_deg",
]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    daylight = df[df["solar_elevation_deg"] > 0]
    rows = []
    for col in CANDIDATE_COLUMNS:
        sub = daylight[[col, "plant_output_kw"]].dropna()
        if len(sub) < 30:
            r, n = np.nan, len(sub)
        else:
            r = float(np.corrcoef(sub[col], sub["plant_output_kw"])[0, 1])
            n = len(sub)
        rows.append({"변수": col, "상관계수": r, "표본수": n, "절대값": abs(r) if pd.notna(r) else np.nan})
    return pd.DataFrame(rows).sort_values("절대값", ascending=False)


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


def evaluate_threshold(
    threshold: float,
    corr: pd.DataFrame,
    base: pd.DataFrame,
    df: pd.DataFrame,
    capacity_kw: float,
) -> dict:
    selected_cols = corr.loc[corr["절대값"] >= threshold, "변수"].tolist()
    frame = base.copy()
    for col in selected_cols:
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
    actual = test["목표_발전출력_kW"].to_numpy()
    err = actual - pred
    return {
        "임계값": threshold,
        "선택된_외생변수_수": len(selected_cols),
        "선택된_외생변수": selected_cols,
        "전체_특성수": len(features),
        "학습표본수": int(len(train)),
        "시험표본수": int(len(test)),
        "MAE_kW": float(np.abs(err).mean()),
        "RMSE_kW": float(np.sqrt((err ** 2).mean())),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity_kw = float(df["reported_capacity_kw"].dropna().iloc[0])

    print("[1/2] 낮시간 기준 Pearson 상관계수 계산 중...")
    corr = compute_correlations(df)
    corr.to_csv(OUT_DIR / "변수별_상관계수_낮시간.csv", index=False, encoding="utf-8-sig")
    print(corr.to_string(index=False))

    base = make_base_features(df)

    print("\n[2/2] 임계값별 ablation 학습 중...")
    results = [evaluate_threshold(t, corr, base, df, capacity_kw) for t in THRESHOLDS]
    scorecard = pd.DataFrame([{k: v for k, v in r.items() if k != "선택된_외생변수"} for r in results])
    print(scorecard.to_string(index=False))

    best = min(results, key=lambda r: r["RMSE_kW"])
    print(f"\n최적 임계값(RMSE 기준): {best['임계값']} (RMSE={best['RMSE_kW']:.3f}kW, MAE={best['MAE_kW']:.3f}kW)")
    print(f"채택된 외생변수({len(best['선택된_외생변수'])}개): {best['선택된_외생변수']}")

    scorecard.to_csv(OUT_DIR / "임계값별_성능표.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "상세결과.json").write_text(
        json.dumps({"results": results, "최적임계값": best["임계값"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
