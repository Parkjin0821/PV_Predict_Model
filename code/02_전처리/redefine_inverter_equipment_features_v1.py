# -*- coding: utf-8 -*-
"""
다음 단계 우선순위 3번: `주파수평균_Hz`(mean_frequency_hz)·`인버터평균온도_C`
(mean_inverter_temperature_c)를 "정상 작동 인버터만의 평균"으로 재정의하고,
기존(5대 전체 평균, 오염됨) 정의 대비 실제 모델 성능에 미치는 영향을
ablation으로 검증한다.

배경(AGENTS.md 2026-08-19 기록, 5개 인버터 63만행 전수 재확인 완료):
- 주파수: 3번·4번 인버터는 발전 중(출력전력>0)일 때 100% 0Hz로 고정되는
  계측/통신 결함이 있다. 정상은 1·2·5번(발전 중 0Hz 비율 0%).
- 인버터온도: 반대로 1·2·5번 인버터는 전 기간 0으로 고정되어 있다.
  정상은 3·4번(발전 중 0도 비율 3번 0.0056%, 4번 0%로 사실상 0).
- 기존 파이프라인은 5대 전체를 그대로 평균해 각 특성에 항상 결함 인버터의
  0값이 섞여 들어간다 → 이 스크립트가 재정의하는 대상.

출처: 광주 인버터 1~5번 실측 로그 xlsx→csv 변환본
`v1_2026-08-14/02_분석데이터/gwangju_inverter_selected_raw.csv`
(5분 간격, 638,075행, 2024-08-25~2026-08-04, inverter 컬럼으로 구분).

방법론(자체 판단, 보고서에 없는 방법):
1. 인버터별로 원시 5분 로그를 1시간 평균으로 직접 리샘플링한다(정상 인버터만
   평균 방식은 기존 파이프라인의 "5분 평균 → 1시간 평균" 2단계 대신
   "원시값 → 1시간 평균" 1단계로 계산한다 — 5분 구간별 실제 표본수가 균일
   하지 않아 완전히 동일한 값은 아니지만, 차이는 무시할 수준이며 목적
   (재정의 전후 비교)에는 영향 없음. 기존 데이터셋의 mean_frequency_hz·
   mean_inverter_temperature_c(5대 전체 평균)를 이 방식으로 재계산해 기존
   파일 값과 근사히 일치하는지 정합성 검사로 확인한다.
2. 재정의: new_mean_frequency_hz = mean(1,2,5번 주파수),
   new_mean_inverter_temperature_c = mean(3,4번 온도).
3. ablation: 발전소 합산 실측 발전량(plant_output_kw, 이미 확보된 시간단위
   KMA 결합 데이터셋 기준)을 목표로, "직전 시각까지 알 수 있는 정보로
   1시간 뒤 발전량을 예측"하는 단순 LightGBM 모델을 3가지 특성 조합으로
   비교한다:
     A. 기준(장비 특성 없음: 발전량 지연값·이동통계·태양고도만)
     B. 기준 + 기존 정의(5대 전체 평균, 오염됨) 주파수·온도
     C. 기준 + 재정의(정상 인버터만 평균) 주파수·온도
   시간순 분할(마지막 15%를 시험구간)로 낮시간(태양고도>0)만 MAE·RMSE 비교.
   본 ablation은 "이 두 특성의 재정의가 실제로 도움/무해/유해 중 무엇인지"만
   확인하는 목적의 경량 실험이며, 프로젝트 공식 다중수평 파이프라인
   (train_hourly_baseline.py 등, 아직 config.json·경로 미정비로 실행 불가)을
   대체하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

RAW_INVERTER_LOG = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\v1_2026-08-14\02_분석데이터"
    r"\gwangju_inverter_selected_raw.csv"
)
HOURLY_KMA_DATASET = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS"
    r"\gwangju_1hour_model_dataset_kma_observed.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\inverter_equipment_feature_redefinition_v1_2026-08-20"
)

NORMAL_FREQ_INVERTERS = [1, 2, 5]
NORMAL_TEMP_INVERTERS = [3, 4]
SEED = 42


def build_hourly_per_inverter() -> pd.DataFrame:
    raw = pd.read_csv(RAW_INVERTER_LOG, parse_dates=["생성일"], low_memory=False)
    raw["주파수"] = pd.to_numeric(raw["주파수"], errors="coerce")
    raw["온도"] = pd.to_numeric(raw["온도"], errors="coerce")
    raw["inverter"] = raw["inverter"].astype(int)

    freq_wide = {}
    temp_wide = {}
    for inv, part in raw.groupby("inverter"):
        part = part.set_index("생성일").sort_index()
        freq_wide[inv] = part["주파수"].resample("1h").mean()
        temp_wide[inv] = part["온도"].resample("1h").mean()
    freq_df = pd.DataFrame(freq_wide).sort_index()
    temp_df = pd.DataFrame(temp_wide).sort_index()

    out = pd.DataFrame(index=freq_df.index)
    out["재계산_기존_주파수평균_Hz_5대"] = freq_df.mean(axis=1)
    out["재계산_기존_인버터평균온도_C_5대"] = temp_df.mean(axis=1)
    out["재정의_주파수평균_Hz_정상만"] = freq_df[NORMAL_FREQ_INVERTERS].mean(axis=1)
    out["재정의_인버터평균온도_C_정상만"] = temp_df[NORMAL_TEMP_INVERTERS].mean(axis=1)
    out.index.name = "time"
    return out


def sanity_check(features: pd.DataFrame, kma: pd.DataFrame) -> dict:
    merged = features.join(
        kma[["mean_frequency_hz", "mean_inverter_temperature_c"]], how="inner"
    )
    freq_diff = (merged["재계산_기존_주파수평균_Hz_5대"] - merged["mean_frequency_hz"]).abs()
    temp_diff = (
        merged["재계산_기존_인버터평균온도_C_5대"] - merged["mean_inverter_temperature_c"]
    ).abs()
    return {
        "비교행수": int(len(merged)),
        "주파수_최대절대오차_Hz": float(freq_diff.max()),
        "주파수_평균절대오차_Hz": float(freq_diff.mean()),
        "온도_최대절대오차_C": float(temp_diff.max()),
        "온도_평균절대오차_C": float(temp_diff.mean()),
    }


def make_lag_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    power = frame["plant_output_kw"]
    out = pd.DataFrame(index=frame.index)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.shift(1).rolling(
            hours, min_periods=max(3, hours // 2)
        ).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.shift(1).rolling(
            hours, min_periods=max(3, hours // 2)
        ).std()
    out["태양고도_deg"] = frame["solar_elevation_deg"].shift(-1)  # 예측대상(t+1) 시각 값
    minute = out.index.hour * 60 + out.index.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    return out


def run_variant(
    name: str,
    base: pd.DataFrame,
    equip: pd.DataFrame | None,
    target_next: pd.Series,
    target_daylight_next: pd.Series,
    capacity_kw: float,
) -> dict:
    frame = base.copy()
    if equip is not None:
        frame = frame.join(equip)
    frame["목표_발전출력_kW"] = target_next
    frame["목표_낮시간"] = target_daylight_next
    features = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
    frame = frame.dropna(subset=features + ["목표_발전출력_kW"])
    frame = frame[frame["목표_낮시간"] == 1]

    n = len(frame)
    cut = int(n * 0.85)
    train, test = frame.iloc[:cut], frame.iloc[cut:]

    model = LGBMRegressor(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.3,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(train[features], train["목표_발전출력_kW"])
    pred = np.clip(model.predict(test[features]), 0, capacity_kw)
    actual = test["목표_발전출력_kW"].to_numpy()
    err = actual - pred
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    importances = dict(zip(features, model.feature_importances_.tolist()))
    return {
        "구성": name,
        "학습표본수": int(len(train)),
        "시험표본수": int(len(test)),
        "시험시작": str(test.index.min()),
        "시험종료": str(test.index.max()),
        "MAE_kW": mae,
        "RMSE_kW": rmse,
        "특성중요도": importances,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 원시 인버터 로그에서 1시간 단위 정상/기존 정의 특성 재계산 중...")
    features = build_hourly_per_inverter()

    kma = pd.read_csv(HOURLY_KMA_DATASET, parse_dates=["time"], low_memory=False).set_index(
        "time"
    ).sort_index()

    print("[2/4] 정합성 검사(재계산 기존정의 vs 기존 데이터셋 값)...")
    check = sanity_check(features, kma)
    print(json.dumps(check, ensure_ascii=False, indent=2))

    print("[3/4] ablation용 결합 데이터 구성...")
    combined = kma[["plant_output_kw", "solar_elevation_deg"]].join(features, how="inner")
    combined["낮시간"] = (combined["solar_elevation_deg"] > 0).astype(int)

    base = make_lag_base_features(combined)
    target_next = combined["plant_output_kw"].shift(-1)
    daylight_next = combined["낮시간"].shift(-1)
    capacity_kw = float(kma["reported_capacity_kw"].iloc[0])

    old_equip = combined[
        ["재계산_기존_주파수평균_Hz_5대", "재계산_기존_인버터평균온도_C_5대"]
    ].rename(
        columns={
            "재계산_기존_주파수평균_Hz_5대": "직전_주파수평균_Hz",
            "재계산_기존_인버터평균온도_C_5대": "직전_인버터평균온도_C",
        }
    )
    new_equip = combined[
        ["재정의_주파수평균_Hz_정상만", "재정의_인버터평균온도_C_정상만"]
    ].rename(
        columns={
            "재정의_주파수평균_Hz_정상만": "직전_주파수평균_Hz",
            "재정의_인버터평균온도_C_정상만": "직전_인버터평균온도_C",
        }
    )

    print("[4/4] ablation 3종 학습·평가 중 (A:기준 / B:기존정의 / C:재정의)...")
    results = [
        run_variant("A_기준(장비특성없음)", base, None, target_next, daylight_next, capacity_kw),
        run_variant("B_기존정의(5대평균,오염)", base, old_equip, target_next, daylight_next, capacity_kw),
        run_variant("C_재정의(정상인버터만)", base, new_equip, target_next, daylight_next, capacity_kw),
    ]

    scorecard = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "특성중요도"} for r in results]
    )
    print(scorecard.to_string(index=False))

    features.to_csv(OUT_DIR / "광주_인버터_주파수온도_재정의_1시간_710일.csv", encoding="utf-8-sig")
    scorecard.to_csv(OUT_DIR / "ablation_성능비교표.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "ablation_상세결과.json").write_text(
        json.dumps({"정합성검사": check, "ablation": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
