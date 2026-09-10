# -*- coding: utf-8 -*-
"""일간 계층조정(hierarchical reconciliation) — 공식 KMA 파이프라인 복원.

08-20 밤 계획대로: ①직접 일간모델(`train_daily_official_v1`) +
②익일 낮시간 다중시각 1시간모델 합계(`train_hourly_daytime_multihorizon_v1`)
둘을 시간순 교차검증 OOF로 non-negative 가중결합한다
(`model_common.optimize_nonnegative_weights`, 기존 프로젝트 관행 재사용).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from model_common import optimize_nonnegative_weights, write_json  # noqa: E402

spec_daily = importlib.util.spec_from_file_location("daily_v1", ROOT / "train_daily_official_v1_2026-08-20밤.py")
daily_mod = importlib.util.module_from_spec(spec_daily)
spec_daily.loader.exec_module(daily_mod)

spec_hourly = importlib.util.spec_from_file_location(
    "daytime_v1", ROOT / "train_hourly_daytime_multihorizon_v1_2026-08-20밤.py"
)
hourly_mod = importlib.util.module_from_spec(spec_hourly)
spec_hourly.loader.exec_module(hourly_mod)

OUT_DIR = ROOT / "outputs" / "일간_계층조정_v1_2026-08-20밤"


def daily_oof_predictions(data, features, windows, capacity_daily, seed):
    """폴드별 직접 일간모델 예측(OOF)을 날짜 인덱스로 반환."""
    rows = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index <= end)]
        if len(train) < 60 or len(test) < 10:
            continue
        medians = train[features].median(numeric_only=True)
        model = LGBMRegressor(
            objective="regression_l1", n_estimators=500, learning_rate=0.025,
            num_leaves=15, max_depth=6, min_child_samples=14,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbosity=-1, n_jobs=4,
        )
        model.fit(train[features].fillna(medians), train[daily_mod.ACTUAL])
        pred = np.clip(model.predict(test[features].fillna(medians)), 0, capacity_daily)
        for date, actual, p in zip(test.index, test[daily_mod.ACTUAL], pred):
            rows.append({"날짜": date, "실제_일간총량_kWh": actual, "직접모델_kWh": p, "폴드": f"{i}_{w.get('_계절','')}"})
    return pd.DataFrame(rows)


def hourly_sum_oof_predictions(df, windows, capacity_kw, seed):
    """폴드별 시간모델(06~19시 합) 예측(OOF)을 날짜 인덱스로 반환."""
    horizon_frames = {h: hourly_mod.build_frame_h(df, 15 + h) for h in hourly_mod.DAYTIME_HOURS}
    per_hour_pred = []
    for target_hour, frame in horizon_frames.items():
        feature_cols = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표대상시각")]
        required = [c for c in feature_cols if c not in hourly_mod.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
        issue_rows = frame[frame.index.hour == hourly_mod.ISSUE_HOUR]
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
            for date, p in zip(test["목표대상시각"].dt.normalize(), pred):
                per_hour_pred.append({"날짜": date, "시간예측_kW": p, "폴드": f"{i}_{w.get('_계절','')}"})
    per_hour_df = pd.DataFrame(per_hour_pred)
    daily = per_hour_df.groupby(["날짜", "폴드"], as_index=False).agg(
        시간모델합계_kWh=("시간예측_kW", "sum"), 시각수=("시간예측_kW", "size")
    )
    return daily[daily["시각수"] == len(hourly_mod.DAYTIME_HOURS)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    capacity_kw = float(config["site"]["capacity_kw"])
    capacity_daily = capacity_kw * 24
    windows = config["cross_validation_windows"]

    print("[1/3] 직접 일간모델 OOF 예측...")
    data, features = daily_mod.build_daily_dataset()
    daily_oof = daily_oof_predictions(data, features, windows, capacity_daily, seed)
    print(f"  {len(daily_oof)}건")

    print("[2/3] 시간모델(06~19시 합계) OOF 예측...")
    df = pd.read_csv(hourly_mod.sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    hourly_oof = hourly_sum_oof_predictions(df, windows, capacity_kw, seed)
    print(f"  {len(hourly_oof)}건")

    print("[3/3] 결합 및 가중치 최적화...")
    merged = daily_oof.merge(hourly_oof[["날짜", "시간모델합계_kWh"]], on="날짜", how="inner")
    print(f"  공통 날짜 {len(merged)}건")

    candidates = ["직접모델_kWh", "시간모델합계_kWh"]
    result = optimize_nonnegative_weights(merged["실제_일간총량_kWh"], merged[candidates])
    weights = np.asarray(result["가중치"], dtype=float)
    merged["계층조정_kWh"] = np.clip(merged[candidates].to_numpy() @ weights, 0, capacity_daily)

    print(f"\n가중치: 직접모델={weights[0]:.3f}, 시간모델합계={weights[1]:.3f}")

    def wape(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        e = np.abs(y - p)
        return float(e.sum() / np.abs(y).sum() * 100), float(e.mean()), float(np.sqrt(np.mean((y - p) ** 2)))

    print("\n=== 비교 (전체 OOF, 5폴드 합산) ===")
    rows = []
    for name, col in [("직접모델", "직접모델_kWh"), ("시간모델합계", "시간모델합계_kWh"), ("계층조정", "계층조정_kWh")]:
        w, mae, rmse = wape(merged["실제_일간총량_kWh"], merged[col])
        rows.append({"구성": name, "n": len(merged), "WAPE_pct": round(w, 2), "MAE_kWh": round(mae, 1), "RMSE_kWh": round(rmse, 1)})
        print(f"  {name:10s} WAPE={w:.2f}% MAE={mae:.1f}kWh RMSE={rmse:.1f}kWh")

    pd.DataFrame(rows).to_csv(OUT_DIR / "비교표.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT_DIR / "OOF_상세.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "가중치.json", {**result, "직접모델_가중치": float(weights[0]), "시간모델합계_가중치": float(weights[1])})
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
