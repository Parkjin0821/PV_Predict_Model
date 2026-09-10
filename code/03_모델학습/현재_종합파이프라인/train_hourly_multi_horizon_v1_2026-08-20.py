# -*- coding: utf-8 -*-
"""6번 후속(08-20 밤) — 남은 항목 처리:
① 저출력 이상구간(2025-08-15~10-20) 처리방식 결정(8번 과제)
② 다른 예측수평(+24h, +48h) 검증

## 예측수평 일반화 방법
`select_features_by_correlation_threshold_v3`가 확립한 정렬 규칙을 그대로
수평 H로 일반화한다:
- 타깃: `plant_output_kw`의 (H-1)번째 미래 행 (`shift(-(H-1))`) — H=1일 때
  타깃 그 자체(shift 0)가 되는 게 v3에서 이미 검증됨.
- NWP 예보(종료라벨): `shift(-H)` — 타깃 버킷과 같은 시간대를 가리키는 라벨.
- 관측·장비텔레메트리: **수평이 늘어나도 "지금 아는 것"은 그대로**이므로
  shift(0)/shift(1) 변경 없음(더 먼 미래를 그 시점 지식만으로 예측하는
  것 — 이게 "예측수평"의 정의 그대로).
- 발전량 자기이력(lag)도 "타깃 시작 시점 기준" 과거로 재계산.

## 이상구간 처리 3안 비교
A. 포함(현행): 그대로 학습에 사용.
B. 제외: 2025-08-15~10-20을 학습에서 완전히 제거.
C. 가중치 하향: 그 구간 표본의 학습 가중치를 0.3으로 낮춤(LightGBM
   sample_weight).
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

OUT_DIR = ROOT / "outputs" / "6번후속_이상구간_예측수평_v1_2026-08-20"

FINAL_14 = [
    "plant_input_power_kw", "DSWRF", "기상청관측_일조시간_hr", "mean_power_factor", "REH",
    "mean_input_voltage_v", "추정_출력온도", "기상청관측_상대습도_pct", "기상청관측_전운량_pct",
    "DSWRFLX_bsrn정제", "LCDC", "TCDC", "POP", "SKY",
]
NATIVE_MISSING_OK = {"DSWRFLX_bsrn정제"}
ANOMALY_START, ANOMALY_END = pd.Timestamp("2025-08-15"), pd.Timestamp("2025-10-20")


def build_candidate_h(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """수평 H에 맞춘 특성·타깃 프레임(행 인덱스=발행시각 t).

    08-20 밤 확정 최종 14개(FINAL_14)만 사용한다 — 전체 36개 후보를 그대로
    쓰면 5번에서 상관분석·다중공선성으로 걸러낸 결론을 무시하게 된다."""
    out = pd.DataFrame(index=df.index)
    for c in FINAL_14:
        if c in sel.STARTLABEL_COLUMNS:
            out[c] = df[c].shift(1)
        elif c in sel.OBSERVED_COLUMNS:
            out[c] = df[c]
        elif c in sel.FORECAST_COLUMNS:
            out[c] = df[c].shift(-horizon)
        else:
            raise ValueError(f"분류되지 않은 특성: {c}")

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
    out["예측수평_시간"] = horizon

    out["목표_발전출력_kW"] = df["plant_output_kw"].shift(-(horizon - 1))
    out["목표_낮시간"] = (df["solar_elevation_deg"].shift(-(horizon - 1)) > 0).astype(float)
    out["_지속성_직전출력_kW"] = df["plant_output_kw"].shift(1)  # 단순지속성(직전 완결값 유지)
    return out


def metrics(y: np.ndarray, p: np.ndarray, capacity_kw: float) -> dict:
    e = y - p
    mae = float(np.abs(e).mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    return {"n": int(len(y)), "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
            "nRMSE_pct": round(rmse / capacity_kw * 100, 3)}


def run_experiment(frame: pd.DataFrame, feature_cols: list[str], windows: list[dict],
                    capacity_kw: float, seed: int, anomaly_mode: str) -> list[dict]:
    required = [c for c in feature_cols if c not in NATIVE_MISSING_OK] + [
        "목표_발전출력_kW", "_지속성_직전출력_kW",
    ]
    daylight = frame[frame["목표_낮시간"] > 0]
    results = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        train = daylight[daylight.index < start].dropna(subset=required).copy()
        test = daylight[(daylight.index >= start) & (daylight.index <= end)].dropna(subset=required)
        if len(train) < 200 or len(test) < 30:
            continue

        weight = np.ones(len(train))
        if anomaly_mode == "제외":
            keep = ~((train.index >= ANOMALY_START) & (train.index <= ANOMALY_END))
            train = train[keep]
            weight = np.ones(len(train))
        elif anomaly_mode == "가중치하향":
            in_anomaly = (train.index >= ANOMALY_START) & (train.index <= ANOMALY_END)
            weight = np.where(in_anomaly, 0.3, 1.0)

        model = LGBMRegressor(
            n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
        )
        model.fit(train[feature_cols], train["목표_발전출력_kW"], sample_weight=weight)
        pred = np.clip(model.predict(test[feature_cols]), 0, capacity_kw)
        pers = np.clip(test["_지속성_직전출력_kW"].to_numpy(), 0, capacity_kw)

        m = metrics(test["목표_발전출력_kW"].to_numpy(), pred, capacity_kw)
        m["단순지속성_RMSE"] = round(float(np.sqrt(np.mean((test["목표_발전출력_kW"].to_numpy() - pers) ** 2))), 3)
        m["폴드"] = f"{i}_{w.get('_계절', '')}"
        results.append(m)
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    windows = config["cross_validation_windows"]

    df = pd.read_csv(sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()

    print("=" * 60 + "\n[실험1] 저출력 이상구간 처리방식 비교 (수평=+1h)\n" + "=" * 60)
    frame_h1 = build_candidate_h(df, 1)
    feature_cols_h1 = [c for c in frame_h1.columns if c not in
                       ("목표_발전출력_kW", "목표_낮시간", "_지속성_직전출력_kW")]
    anomaly_summary = []
    for mode in ["포함(현행)", "제외", "가중치하향"]:
        key = mode.split("(")[0]
        res = run_experiment(frame_h1, feature_cols_h1, windows, capacity_kw, seed, key)
        avg_mae = np.mean([r["MAE_kW"] for r in res])
        avg_rmse = np.mean([r["RMSE_kW"] for r in res])
        avg_nrmse = np.mean([r["nRMSE_pct"] for r in res])
        print(f"  {mode:10s}: 평균 MAE={avg_mae:.3f} RMSE={avg_rmse:.3f} nRMSE={avg_nrmse:.3f}%  (폴드별: {[round(r['RMSE_kW'],2) for r in res]})")
        anomaly_summary.append({"방식": mode, "평균MAE": round(avg_mae, 3), "평균RMSE": round(avg_rmse, 3),
                                 "평균nRMSE": round(avg_nrmse, 3), "폴드별": res})
    pd.DataFrame([{k: v for k, v in r.items() if k != "폴드별"} for r in anomaly_summary]).to_csv(
        OUT_DIR / "이상구간_처리방식_비교.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "이상구간_처리방식_상세.json").write_text(
        json.dumps(anomaly_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60 + "\n[실험2] 예측수평별 성능 (+1h, +24h, +48h)\n" + "=" * 60)
    horizon_summary = []
    for H in [1, 24, 48]:
        frame_h = build_candidate_h(df, H)
        feature_cols_h = [c for c in frame_h.columns if c not in
                           ("목표_발전출력_kW", "목표_낮시간", "_지속성_직전출력_kW")]
        res = run_experiment(frame_h, feature_cols_h, windows, capacity_kw, seed, "포함")
        if not res:
            print(f"  +{H}h: 유효 폴드 없음")
            continue
        avg_mae = np.mean([r["MAE_kW"] for r in res])
        avg_rmse = np.mean([r["RMSE_kW"] for r in res])
        avg_nrmse = np.mean([r["nRMSE_pct"] for r in res])
        avg_pers_rmse = np.mean([r["단순지속성_RMSE"] for r in res])
        skill = round(1 - avg_rmse / avg_pers_rmse, 4) if avg_pers_rmse > 0 else None
        print(f"  +{H:2d}h: 평균 MAE={avg_mae:.3f} RMSE={avg_rmse:.3f} nRMSE={avg_nrmse:.3f}% "
              f"(단순지속성 RMSE={avg_pers_rmse:.3f}, skill={skill}) 폴드수={len(res)}")
        horizon_summary.append({"수평_h": H, "평균MAE": round(avg_mae, 3), "평균RMSE": round(avg_rmse, 3),
                                 "평균nRMSE": round(avg_nrmse, 3), "단순지속성_평균RMSE": round(avg_pers_rmse, 3),
                                 "SkillScore": skill, "폴드수": len(res), "폴드별": res})
    pd.DataFrame([{k: v for k, v in r.items() if k != "폴드별"} for r in horizon_summary]).to_csv(
        OUT_DIR / "예측수평별_성능.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "예측수평별_상세.json").write_text(
        json.dumps(horizon_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
