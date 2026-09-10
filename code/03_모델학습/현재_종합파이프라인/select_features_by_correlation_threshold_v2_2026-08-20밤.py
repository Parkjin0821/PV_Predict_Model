# -*- coding: utf-8 -*-
"""5번 재실행(08-20 밤) — 시간정렬 버그 수정 + 1시간 해상도 NWP 데이터로 재실행.

## 이전 버전(v1) 대비 수정된 것
1. **시간정렬 버그 수정**: v1은 모든 외생변수(관측·NWP 구분 없이)에 일괄
   `shift(1)`을 적용했다(치명적 결함, AGENTS.md "저녁 감사" 절). 이번엔
   변수 성격별로 정렬을 구분한다:
   - **관측·장비텔레메트리**(ASOS, 인버터 등): 발행시각 t에 "방금 관측된"
     값만 안다 → `직전_X = X.shift(0)`(행 인덱스 t의 값 = 그 시각까지 실제로
     들어온 마지막 관측치). *v1은 여기서까지 shift(1)을 더 줘서 t-1 값을
     썼는데, 이번엔 t 시각 자체의 관측값을 쓴다 — t행은 "그 시각 마감까지
     들어온 정보로 t+1시를 예측한다"는 의미로 재정의했다.*
   - **NWP 예보**(DSWRF·TCDC·격자예보 등): 발행시각 t에 이미 **목표시각
     t+1의 예보값**을 안다(수치예보는 하루 전에 발표되므로 1시간 뒤
     예보는 당연히 미리 알려져 있음 — 리드타임이 넉넉해 누출 위험 없음)
     → `목표_X = X.shift(-1)`(행 인덱스 t+1의 값, 즉 목표시각의 예보값).
2. **1시간 해상도 데이터 사용**: `build_official_hourly_dataset_v2_
   2026-08-20밤.py` 산출물 사용. NWP 계열 컬럼의 유효 표본이 33%(3시간에
   1번)에서 ~99.9%(청천모양 보존적 재분배)로 대폭 늘어 학습표본이 훨씬
   커진다.
3. **선택-평가 일관성 확보**: v1은 상관계수표를 동시각 값으로 계산하고
   평가는 shift(1)로 해서 서로 다른 기준이었다(치명적 결함 2). 이번엔
   상관계수도 위 정렬 규칙(관측=t시각, NWP=목표시각)으로 통일해서 계산한다.

## 아직 남은 한계(문서화, 6번에서 처리)
- 상관계수를 여전히 전체 기간(시험구간 포함)으로 계산한다 — "선택 시
  시험구간 정보누출"(치명적 결함2)은 이번에도 완전히 고치지 않았다.
  지금 규모에서 상관계수 기반 1차 필터는 정성적 스크리닝 목적이 크고,
  6번의 정식 모델 선정 단계에서 fold별 재계산으로 다시 검증할 것.
- 시험구간은 여전히 단일 시간순 85:15 분할(치명적 결함3, 계절 편중) —
  6번에서 rolling-origin으로 전환 예정.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v2_2026-08-20밤.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\correlation_threshold_selection_v2_2026-08-20밤"
)
SEED = 42
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

# 관측·장비텔레메트리(발행시각 t까지 알 수 있음 -> shift(0))
OBSERVED_COLUMNS = [
    "기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_강수량_mm",
    "기상청관측_전운량_pct", "기상청관측_일사량_W_m2", "기상청관측_일조시간_hr",
    "기상청관측_풍속_m_s", "기상청관측_풍향_deg", "기상청관측_현지기압_hPa",
    "기상청관측_해면기압_hPa", "기상청관측_적설_cm", "기상청관측_지면온도_C",
    "plant_input_power_kw", "plant_input_current_a", "mean_input_voltage_v",
    "mean_frequency_hz", "mean_power_factor", "mean_inverter_temperature_c",
    "mean_communication_ok", "inverters_available",
]
# NWP 예보(목표시각 t+1에 대해 발행시각 t에 이미 알려져 있음 -> shift(-1))
FORECAST_COLUMNS = [
    "DSWRF", "TCDC", "DSWRFLX_bsrn정제", "DIFSWRF_bsrn정제",
    "LCDC", "MCDC", "HCDC", "VEC", "TMP", "SKY", "REH", "WSD", "POP",
    "추정_모듈표면온도", "추정_출력온도", "추정_일조시간_hr",
]
CANDIDATE_COLUMNS = OBSERVED_COLUMNS + FORECAST_COLUMNS


def build_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """모든 후보 변수를 "목표시각(t+1)에 대해 발행시각(t)에서 안다"는
    기준으로 정렬한 프레임(행 인덱스=발행시각 t)을 만든다."""
    out = pd.DataFrame(index=df.index)
    for c in OBSERVED_COLUMNS:
        out[c] = df[c]  # shift(0): t시각 값
    for c in FORECAST_COLUMNS:
        out[c] = df[c].shift(-1)  # 목표시각(t+1)의 예보값
    return out


def compute_correlations(cand: pd.DataFrame, target_next: pd.Series, daylight_next: pd.Series) -> pd.DataFrame:
    mask = daylight_next > 0
    rows = []
    for col in CANDIDATE_COLUMNS:
        sub = pd.concat([cand.loc[mask, col], target_next[mask]], axis=1).dropna()
        if len(sub) < 30:
            r, n = np.nan, len(sub)
        else:
            r = float(np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1])
            n = len(sub)
        rows.append({"변수": col, "상관계수": r, "표본수": n, "절대값": abs(r) if pd.notna(r) else np.nan})
    return pd.DataFrame(rows).sort_values("절대값", ascending=False)


def make_base_features(df: pd.DataFrame) -> pd.DataFrame:
    power = df["plant_output_kw"]
    out = pd.DataFrame(index=df.index)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours - 1) if hours > 1 else power  # t, t-1, ... (t 시각 자체 포함)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).std()
    out["목표_태양고도_deg"] = df["solar_elevation_deg"].shift(-1)
    minute = out.index.hour * 60 + out.index.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    return out


def evaluate_threshold(threshold: float, corr: pd.DataFrame, base: pd.DataFrame, cand: pd.DataFrame,
                        target_next: pd.Series, daylight_next: pd.Series, capacity_kw: float) -> dict:
    selected_cols = corr.loc[corr["절대값"] >= threshold, "변수"].tolist()
    frame = base.copy()
    for col in selected_cols:
        prefix = "목표_" if col in FORECAST_COLUMNS else "발행시_"
        frame[f"{prefix}{col}"] = cand[col]
    frame["목표_발전출력_kW"] = target_next
    frame["목표_낮시간"] = daylight_next

    features = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
    frame = frame.dropna(subset=features + ["목표_발전출력_kW"])
    frame = frame[frame["목표_낮시간"] > 0]

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
        "임계값": threshold, "선택된_외생변수_수": len(selected_cols), "선택된_외생변수": selected_cols,
        "전체_특성수": len(features), "학습표본수": int(len(train)), "시험표본수": int(len(test)),
        "시험시작": str(test.index.min()), "시험종료": str(test.index.max()),
        "MAE_kW": float(np.abs(err).mean()), "RMSE_kW": float(np.sqrt((err ** 2).mean())),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity_kw = float(df["reported_capacity_kw"].dropna().iloc[0])

    cand = build_candidate_frame(df)
    target_next = df["plant_output_kw"].shift(-1)
    daylight_next = (df["solar_elevation_deg"].shift(-1) > 0).astype(float)

    print("[1/2] 정렬수정 기준 Pearson 상관계수 계산 중...")
    corr = compute_correlations(cand, target_next, daylight_next)
    corr.to_csv(OUT_DIR / "변수별_상관계수_낮시간.csv", index=False, encoding="utf-8-sig")
    print(corr.to_string(index=False))

    base = make_base_features(df)

    print("\n[2/2] 임계값별 ablation 학습 중...")
    results = [evaluate_threshold(t, corr, base, cand, target_next, daylight_next, capacity_kw) for t in THRESHOLDS]
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
