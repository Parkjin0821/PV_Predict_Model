# -*- coding: utf-8 -*-
"""일간 총량 직접모델 — v3(고정-tm, 누출없음) 데이터셋 재실행.

## 왜 다시 만드는가
08-20 밤에 만든 `train_daily_official_v1`은 `gwangju_1hour_model_dataset
_official_v2_2026-08-20밤.csv`를 썼는데, 이 v2는 **NWP를 "매 시각
직전 가장 가까운 발표런"으로 수집**해서 익일예측(D 10시 발행 → D+1
전체) 관점에서 미래정보 누출이 있다(AGENTS.md "★★★전체 감사★★★" 절).
일간모델은 NWP를 목표일(D+1) 단위로 집계해서 쓰므로 이 누출을 그대로
물려받는다. 08-21에 NWP를 고정-tm(D 09:00 KST 발표, 리드타임 15~36h)
으로 재수집해 v3 데이터셋을 만들었으므로, **데이터 소스만 v3로 교체**
하고 나머지 로직(집계 방식·특성 구성·모델)은 v1 그대로 재사용한다.

## 참고 — 특성선택 엄밀도는 이번에도 그대로임(범위 축소, v1과 동일)
v1의 자체검증 메모를 그대로 유지: 이 일간모델은 시간단위 모델의
"최종 확정 특성"(상관분석→다중공선성→배포필터)을 거치지 않고 NWP
12종·장비 17종을 넓게 집계해서 쓴다. 일간모델 전용 상관분석·다중공선성
점검은 이번에도 하지 않았다(별도 과제로 명시 이월, 우선순위는 v2→v3
데이터 교체로 누출을 없애는 것이 더 급했음).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "일간_직접모델_v2_2026-08-21"
HOURLY_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v3_fixed_tm_2026-08-21.csv"
)
DAILY_ACTUAL_PARQUET = ROOT / "outputs" / "집계_일간_실제발전량.parquet"
ACTUAL = "실제_일간발전량_kWh"

NWP_AGG = {
    "DSWRF": ["mean", "sum", "max"], "DSWRFLX_bsrn정제": ["mean", "sum"],
    "DIFSWRF_bsrn정제": ["mean", "sum"], "TCDC": ["mean", "max"], "LCDC": ["mean"],
    "MCDC": ["mean"], "HCDC": ["mean"], "REH": ["mean", "min"], "POP": ["mean", "max"],
    "SKY": ["mean"], "TMP": ["mean", "min", "max"], "WSD": ["mean", "max"],
}
# 08-20 밤 배포가능성 결론 반영: mean_inverter_temperature_c 제외
EQUIPMENT_COLS = [
    "plant_input_power_kw", "mean_input_voltage_v", "mean_frequency_hz",
    "mean_power_factor", "mean_communication_ok", "inverters_available",
    "기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_강수량_mm",
    "기상청관측_전운량_pct", "기상청관측_일사량_W_m2", "기상청관측_일조시간_hr",
    "기상청관측_풍속_m_s", "기상청관측_풍향_deg", "기상청관측_현지기압_hPa",
    "기상청관측_해면기압_hPa", "기상청관측_적설_cm", "기상청관측_지면온도_C",
]


def energy_metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    e = y - p
    denom = float(np.abs(y).sum())
    return {
        "표본수": int(len(y)), "평균절대오차_kWh": float(np.abs(e).mean()),
        "평균제곱근오차_kWh": float(np.sqrt(np.mean(e ** 2))),
        "가중절대비율오차_pct": float(np.abs(e).sum() / denom * 100) if denom > 0 else None,
    }


def build_daily_dataset() -> tuple[pd.DataFrame, list[str]]:
    daily_actual = pd.read_parquet(DAILY_ACTUAL_PARQUET).copy()
    daily_actual.index = pd.to_datetime(daily_actual.index)
    data = pd.DataFrame(index=daily_actual.index)
    data[ACTUAL] = daily_actual["일간발전량_kWh"]
    data["7일전_일간발전량_kWh"] = data[ACTUAL].shift(7)
    data["2일전_일간발전량_kWh"] = data[ACTUAL].shift(2)
    data["2일전기준_7일이동평균_kWh"] = data[ACTUAL].shift(2).rolling(7, min_periods=4).mean()
    data["2일전기준_30일이동평균_kWh"] = data[ACTUAL].shift(2).rolling(30, min_periods=15).mean()
    data["2일전기준_30일표준편차_kWh"] = data[ACTUAL].shift(2).rolling(30, min_periods=15).std()
    data["7일전지속성예측_kWh"] = data["7일전_일간발전량_kWh"]

    hourly = pd.read_csv(HOURLY_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    hourly["날짜"] = hourly.index.normalize()

    nwp_daily = hourly.groupby("날짜").agg(NWP_AGG)
    nwp_daily.columns = [f"목표일예보_{name}_{stat}" for name, stat in nwp_daily.columns]
    data = data.join(nwp_daily, how="left")

    equipment_daily = hourly[EQUIPMENT_COLS + ["날짜"]].groupby("날짜").mean(numeric_only=True).shift(2)
    equipment_daily.columns = [f"2일전평균_{name}" for name in equipment_daily.columns]
    data = data.join(equipment_daily, how="left")

    day = data.index.dayofyear
    data["목표일_연주기_sin"] = np.sin(2 * np.pi * day / 365.25)
    data["목표일_연주기_cos"] = np.cos(2 * np.pi * day / 365.25)
    data["목표일_월"] = data.index.month
    data["목표일_요일"] = data.index.dayofweek
    data["해발고도_m"] = float(hourly["site_elevation_dem_m"].dropna().median())
    data["설비용량_kW"] = float(hourly["reported_capacity_kw"].dropna().median())

    features = [c for c in data.columns if c not in {ACTUAL, "7일전지속성예측_kWh"}]
    data = data.dropna(subset=[ACTUAL, "7일전지속성예측_kWh"])
    data.index.name = "예측대상일"
    return data, features


def new_models(seed: int):
    return {
        "LightGBM": lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=500, learning_rate=0.025,
            num_leaves=15, max_depth=6, min_child_samples=14,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbosity=-1, n_jobs=4,
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:absoluteerror", n_estimators=500, learning_rate=0.025,
            max_depth=4, min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, random_state=seed, n_jobs=4,
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    capacity_daily = float(config["site"]["capacity_kw"]) * 24

    data, features = build_daily_dataset()
    print(f"일간 표본수: {len(data)}, 특성수: {len(features)}")

    windows = config["cross_validation_windows"]
    fold_rows = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index <= end)]
        if len(train) < 60 or len(test) < 10:
            print(f"[{w.get('_계절', i)}] 표본 부족으로 건너뜀 (train={len(train)}, test={len(test)})")
            continue
        medians = train[features].median(numeric_only=True)
        row = {"폴드": f"{i}_{w.get('_계절','')}", "학습표본": len(train), "시험표본": len(test)}
        actual = test[ACTUAL].to_numpy()
        pers = test["7일전지속성예측_kWh"].to_numpy()
        row["지속성"] = energy_metrics(actual, pers)
        for name, model in new_models(seed).items():
            model.fit(train[features].fillna(medians), train[ACTUAL])
            pred = np.clip(model.predict(test[features].fillna(medians)), 0, capacity_daily)
            row[name] = energy_metrics(actual, pred)
        fold_rows.append(row)
        for key in ["지속성", "LightGBM", "XGBoost"]:
            m = row[key]
            print(f"  [{row['폴드']}] {key:10s} MAE={m['평균절대오차_kWh']:.1f}kWh RMSE={m['평균제곱근오차_kWh']:.1f}kWh WAPE={m['가중절대비율오차_pct']:.2f}%")

    (OUT / "폴드별_결과.json").write_text(json.dumps(fold_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    for key in ["지속성", "LightGBM", "XGBoost"]:
        maes = [r[key]["평균절대오차_kWh"] for r in fold_rows if key in r]
        rmses = [r[key]["평균제곱근오차_kWh"] for r in fold_rows if key in r]
        wapes = [r[key]["가중절대비율오차_pct"] for r in fold_rows if key in r]
        summary_rows.append({"구성": key, "폴드수": len(maes),
                              "평균MAE_kWh": round(float(np.mean(maes)), 2) if maes else None,
                              "평균RMSE_kWh": round(float(np.mean(rmses)), 2) if rmses else None,
                              "평균WAPE_pct": round(float(np.mean(wapes)), 2) if wapes else None})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "폴드평균_요약.csv", index=False, encoding="utf-8-sig")

    # 마지막 모델(전체 데이터 중 최신 구간까지 학습)을 저장
    medians = data[features].median(numeric_only=True)
    final_models = {}
    for name, model in new_models(seed).items():
        model.fit(data[features].fillna(medians), data[ACTUAL])
        final_models[name] = model
    joblib.dump({"models": final_models, "features": features, "medians": medians.to_dict()},
                OUT / "일간_직접모델.joblib")

    print("\n=== 폴드 평균 요약 ===")
    print(summary.to_string(index=False))
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
