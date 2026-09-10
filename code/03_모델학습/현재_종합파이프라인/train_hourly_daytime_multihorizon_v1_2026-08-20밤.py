# -*- coding: utf-8 -*-
"""익일 낮시간 전체 다중시각 1시간 예측 파이프라인 (08-20 밤 신규).

## 목적
"당일 10시 발행 → 익일 00~23시 각 시각 예측"이 있어야 ①일간모델과의
계층조정 ②Blockdata 시간대별 스키마 대응이 가능하다는 게 6번 후속3의
결론이었다. 이 스크립트가 그 첫 조각이다.

## 설계
- **발행 관례**: 회사 기존 관례(구 `train_daily_and_reconcile.py`)를
  그대로 따라 **매일 10시 발행**으로 고정한다(발행시각 hour==10인 행만
  이슈 시점으로 사용).
- **예측 대상**: 익일 낮시간(06~19시, 전체기간 실측 기준 발전량이
  0이 아닌 시각 분포로 확정 — 아래 실측 확인 참고) 14개 시각. 각 시각은
  `select_features_by_correlation_threshold_v3`가 확립한 수평 개념으로
  horizon=15+H(H=목표시각 0~23) — 10시 발행 기준 다음날 06시는
  horizon=21, 19시는 horizon=34.
- **모델**: 수평(horizon)별로 독립된 LightGBM 1개씩(기존 프로젝트
  관례 — `train_ultra_short_baseline.py`·`train_hourly_official`도
  수평별 독립모델). 특성·정렬은 6번에서 확정한 v3 규칙과 최종 14개를
  그대로 재사용.
- **검증**: config.json의 5계절 rolling-origin 폴드(여름·가을·겨울·봄·
  초여름) 그대로 사용 — "여름 편중" 재발 방지.
- **일간 합계 비교**: 14개 시각 예측을 날짜별로 합산해 "1시간모델
  합계예측_kWh"를 만들고, 06~19시가 실제 일간 총발전량의 몇 %를
  차지하는지 실측으로 확인해 근사 정도를 명시한다(완전히 하루 전체는
  아님 — 새벽·초저녁 자투리 발전 누락).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sel_v3", ROOT / "select_features_by_correlation_threshold_v3_2026-08-20밤.py"
)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

OUT_DIR = ROOT / "outputs" / "익일낮시간_다중시각_v1_2026-08-20밤"

FINAL_14 = [
    "plant_input_power_kw", "DSWRF", "기상청관측_일조시간_hr", "mean_power_factor", "REH",
    "mean_input_voltage_v", "추정_출력온도", "기상청관측_상대습도_pct", "기상청관측_전운량_pct",
    "DSWRFLX_bsrn정제", "LCDC", "TCDC", "POP", "SKY",
]
NATIVE_MISSING_OK = {"DSWRFLX_bsrn정제"}
ISSUE_HOUR = 10
DAYTIME_HOURS = list(range(6, 20))  # 06~19시, 14개


def build_frame_h(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FINAL_14:
        if c in sel.STARTLABEL_COLUMNS:
            out[c] = df[c].shift(1)
        elif c in sel.OBSERVED_COLUMNS:
            out[c] = df[c]
        elif c in sel.FORECAST_COLUMNS:
            out[c] = df[c].shift(-horizon)
        else:
            raise ValueError(c)

    power = df["plant_output_kw"].shift(1)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours - 1)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).std()
    out["목표_태양고도_deg"] = df["solar_elevation_deg"].shift(-(horizon - 1))
    target_time = df.index + pd.to_timedelta(horizon - 1, unit="h")
    minute = target_time.hour * 60 + target_time.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)

    out["목표_발전출력_kW"] = df["plant_output_kw"].shift(-(horizon - 1))
    out["목표대상시각"] = target_time
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    windows = config["cross_validation_windows"]

    df = pd.read_csv(sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()

    print("[사전확인] 06~19시가 일간 총발전량에서 차지하는 비중...")
    daily_total = df["plant_output_kw"].resample("1D").sum(min_count=12)
    daytime_total = df[df.index.hour.isin(DAYTIME_HOURS)]["plant_output_kw"].resample("1D").sum(min_count=8)
    ratio = (daytime_total / daily_total).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  06~19시/일간총량 비율: 평균 {ratio.mean()*100:.2f}%, 중앙값 {ratio.median()*100:.2f}%, 5~95백분위 [{ratio.quantile(.05)*100:.1f}%, {ratio.quantile(.95)*100:.1f}%]")

    all_scores = []
    horizon_test_frames = {}  # 폴드별-시각별 시험 예측(합계 검증용)

    for target_hour in DAYTIME_HOURS:
        horizon = 15 + target_hour
        frame = build_frame_h(df, horizon)
        feature_cols = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표대상시각")]
        required = [c for c in feature_cols if c not in NATIVE_MISSING_OK] + ["목표_발전출력_kW"]

        issue_rows = frame[frame.index.hour == ISSUE_HOUR]

        for i, w in enumerate(windows, start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            train = issue_rows[issue_rows.index < start].dropna(subset=required)
            test = issue_rows[(issue_rows.index >= start) & (issue_rows.index <= end)].dropna(subset=required)
            if len(train) < 60 or len(test) < 10:
                continue
            model = LGBMRegressor(
                n_estimators=180, learning_rate=0.05, num_leaves=25, min_child_samples=15,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
            )
            model.fit(train[feature_cols], train["목표_발전출력_kW"])
            pred = np.clip(model.predict(test[feature_cols]), 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            e = actual - pred
            all_scores.append({
                "목표시각": target_hour, "폴드": f"{i}_{w.get('_계절','')}", "n": len(test),
                "MAE_kW": round(float(np.abs(e).mean()), 3), "RMSE_kW": round(float(np.sqrt((e**2).mean())), 3),
            })
            key = f"{i}_{w.get('_계절','')}"
            fold_pred = pd.DataFrame({
                "목표대상시각": test["목표대상시각"].to_numpy(),
                "예측_kW": pred, "실제_kW": actual,
            })
            horizon_test_frames.setdefault(key, []).append(fold_pred)
        print(f"  목표시각 {target_hour:02d}시(horizon={horizon}) 완료")

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(OUT_DIR / "시각별_폴드별_성능.csv", index=False, encoding="utf-8-sig")

    print("\n=== 목표시각별 평균 성능(5폴드) ===")
    by_hour = scores_df.groupby("목표시각")[["MAE_kW", "RMSE_kW"]].mean().round(3)
    print(by_hour.to_string())

    print("\n=== 낮시간 합계(06~19시) 일간 검증 ===")
    daily_actual_map = daily_total.to_dict()
    summary_rows = []
    for fold_key, parts in horizon_test_frames.items():
        merged = pd.concat(parts, ignore_index=True)
        merged["날짜"] = merged["목표대상시각"].dt.normalize()
        daily_pred = merged.groupby("날짜").agg(합계예측_kW=("예측_kW", "sum"), 시각수=("예측_kW", "size"))
        daily_pred = daily_pred[daily_pred["시각수"] == len(DAYTIME_HOURS)]
        daily_pred["실제_일간총량_kWh"] = daily_pred.index.map(daily_actual_map)
        daily_pred = daily_pred.dropna()
        if len(daily_pred) < 5:
            continue
        # 합계예측(kW를 시간단위로 합산=kWh와 동일 스케일)을 실측 낮시간합계와 비교
        daytime_actual_fold = daytime_total.reindex(daily_pred.index)
        err = daily_pred["합계예측_kW"] - daytime_actual_fold
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        summary_rows.append({"폴드": fold_key, "검증일수": len(daily_pred),
                              "낮시간합계_MAE_kWh": round(mae, 1), "낮시간합계_RMSE_kWh": round(rmse, 1)})
        print(f"  [{fold_key}] 검증일수={len(daily_pred)} 낮시간합계 MAE={mae:.1f}kWh RMSE={rmse:.1f}kWh")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "낮시간합계_일간검증.csv", index=False, encoding="utf-8-sig")
    by_hour.to_csv(OUT_DIR / "시각별_평균성능.csv", encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
