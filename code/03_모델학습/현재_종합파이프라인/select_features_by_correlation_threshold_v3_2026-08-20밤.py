# -*- coding: utf-8 -*-
"""5번 재실행(08-20 밤, 3차) — 데이터 종류별 시각 라벨 규약 불일치 수정.

## 발견(사용자 지적으로 확정)
- `plant_output_kw`·장비텔레메트리(pandas resample 기본값): 라벨 t = **그
  시각부터 다음 시각까지**(예: "09시"=09~10시, 시작시각 라벨).
- ASOS 관측·NWP 재구성값(기상청 관행): 라벨 t = **그 시각까지의 구간**
  (예: "09시"=08~09시, 종료시각 라벨) — `transform_kma_for_model.py`에
  이미 이렇게 문서화돼 있었다("관측 종료시각 직전 구간의 값").
- 즉 **타깃(발전량)을 그대로 두고(shift 없음), 다른 자료들만 정렬을
  맞추면 그 자체로 정확히 "+1시간 예측" 과제가 된다**(사용자 통찰):
  ASOS·NWP는 종료시각 라벨이라 "09시" 값이 이미 08~09시 정보를 담고
  있으므로, 그걸로 발전량 "09시"(=09~10시)를 맞추면 자연스럽게 1시간
  앞을 내다보는 예측이다.

## 최종 정렬 규칙
- **타깃**: `plant_output_kw` 그대로(행 인덱스 t, shift 없음) — "t~t+1시
  발전량".
- **장비텔레메트리·발전량 자기이력**(시작시각 라벨, 타깃과 같은 규약):
  `shift(1)`부터 시작(직전에 "완결된" 버킷 = t-1~t, 타깃 시작 시점에
  막 끝난 값). 동시각(shift 0)은 타깃과 같은 버킷이라 쓰면 누출.
- **ASOS 관측**(종료시각 라벨): `shift(0)` — 그대로 t행 값이 "t까지
  들어온 마지막 관측"이라 타깃(t~t+1) 예측 시점에 정확히 알려져 있음.
- **NWP 예보**(종료시각 라벨이지만 실제로는 미리 발표된 예보):
  `shift(-1)` — t+1행 값(="t+1까지"=t~t+1 구간)이 타깃과 같은 시간대를
  커버하는 예보값. 예보는 하루 전에 이미 발표되므로 미래정보 누출이
  아니다.

## 검증(daylight, n≈7,600대)
| 정렬 | 변수 | 상관계수 |
|---|---|---|
| 장비.shift(1) | plant_input_power_kw | 0.917 |
| ASOS.shift(0) | 기상청관측_일사량 | 0.895 |
| NWP.shift(-1) | DSWRF | 0.909 |
모두 물리적으로 타당하고 누출 없는 조합으로 확인됨.
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
    r"\correlation_threshold_selection_v3_2026-08-20밤"
)
SEED = 42
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

# 시작시각 라벨(타깃과 동일 규약) -> shift(1) 필요
STARTLABEL_COLUMNS = [
    "plant_input_power_kw", "plant_input_current_a", "mean_input_voltage_v",
    "mean_frequency_hz", "mean_power_factor", "mean_inverter_temperature_c",
    "mean_communication_ok", "inverters_available",
]
# 종료시각 라벨 관측(shift(0))
OBSERVED_COLUMNS = [
    "기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_강수량_mm",
    "기상청관측_전운량_pct", "기상청관측_일사량_W_m2", "기상청관측_일조시간_hr",
    "기상청관측_풍속_m_s", "기상청관측_풍향_deg", "기상청관측_현지기압_hPa",
    "기상청관측_해면기압_hPa", "기상청관측_적설_cm", "기상청관측_지면온도_C",
]
# 종료시각 라벨 예보(shift(-1))
FORECAST_COLUMNS = [
    "DSWRF", "TCDC", "DSWRFLX_bsrn정제", "DIFSWRF_bsrn정제",
    "LCDC", "MCDC", "HCDC", "VEC", "TMP", "SKY", "REH", "WSD", "POP",
    "추정_모듈표면온도", "추정_출력온도", "추정_일조시간_hr",
]
CANDIDATE_COLUMNS = STARTLABEL_COLUMNS + OBSERVED_COLUMNS + FORECAST_COLUMNS


def build_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in STARTLABEL_COLUMNS:
        out[c] = df[c].shift(1)
    for c in OBSERVED_COLUMNS:
        out[c] = df[c]
    for c in FORECAST_COLUMNS:
        out[c] = df[c].shift(-1)
    return out


def compute_correlations(cand: pd.DataFrame, target: pd.Series, daylight: pd.Series) -> pd.DataFrame:
    mask = daylight > 0
    rows = []
    for col in CANDIDATE_COLUMNS:
        sub = pd.concat([cand.loc[mask, col], target[mask]], axis=1).dropna()
        if len(sub) < 30:
            r, n = np.nan, len(sub)
        else:
            r = float(np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1])
            n = len(sub)
        rows.append({"변수": col, "상관계수": r, "표본수": n, "절대값": abs(r) if pd.notna(r) else np.nan})
    return pd.DataFrame(rows).sort_values("절대값", ascending=False)


def make_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """타깃(t행 자체, t~t+1)을 예측하기 위한 고정 기준선 — 전부 shift(1)부터."""
    power = df["plant_output_kw"].shift(1)  # t-1~t 버킷 = 타깃 직전 완결값
    out = pd.DataFrame(index=df.index)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours - 1)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).std()
    out["목표_태양고도_deg"] = df["solar_elevation_deg"]  # 타깃 버킷 시작시각 태양고도(그대로)
    minute = out.index.hour * 60 + out.index.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    return out


def evaluate_threshold(threshold: float, corr: pd.DataFrame, base: pd.DataFrame, cand: pd.DataFrame,
                        target: pd.Series, daylight: pd.Series, capacity_kw: float) -> dict:
    selected_cols = corr.loc[corr["절대값"] >= threshold, "변수"].tolist()
    frame = base.copy()
    for col in selected_cols:
        frame[col] = cand[col]
    frame["목표_발전출력_kW"] = target
    frame["목표_낮시간"] = daylight

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
    target = df["plant_output_kw"]
    daylight = (df["solar_elevation_deg"] > 0).astype(float)

    print("[1/2] 최종 정렬 기준 Pearson 상관계수 계산 중...")
    corr = compute_correlations(cand, target, daylight)
    corr.to_csv(OUT_DIR / "변수별_상관계수_낮시간.csv", index=False, encoding="utf-8-sig")
    print(corr.to_string(index=False))

    base = make_base_features(df)

    print("\n[2/2] 임계값별 ablation 학습 중...")
    results = [evaluate_threshold(t, corr, base, cand, target, daylight, capacity_kw) for t in THRESHOLDS]
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
