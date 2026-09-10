# -*- coding: utf-8 -*-
"""일간 계층조정 — v3, 직접모델 알고리즘 선택 명시(LightGBM vs XGBoost).

## 왜 다시 만드는가 (사용자 지적, 08-21)
v1·v2 둘 다 `daily_oof_predictions`가 **LightGBM만 하드코딩**돼 있었다
— 반면 `train_daily_official_v2_2026-08-21.py`의 독립 폴드평가에서는
XGBoost(WAPE 17.27%)가 LightGBM(18.10%)보다 근소 우위였다. 즉 지금까지
보고한 "계층조정 WAPE 17.28%"는 **직접모델 성분이 실제로는 더 나은
후보(XGBoost)를 시도조차 안 하고 LightGBM으로 고정**된 상태에서 나온
숫자였다 — 월간모델 절엔 "LightGBM 채택"이 명시돼 있는데 일간 계층조정
절엔 이런 표기가 없었던 이유가 바로 이거였다(적어놓지 않은 게 아니라
애초에 비교 자체를 안 했다).

## 이번 수정
`daily_oof_predictions`가 LightGBM·XGBoost 둘 다 폴드별로 OOF를 계산해
**전체 OOF 기준 더 나은 쪽을 채택**하고, 어느 쪽이 선택됐는지 결과에
명시한다(월간모델과 동일한 투명성 원칙). 시간모델합계는 프로젝트 전체
관례대로 LightGBM만 사용(기존 결정 유지, 변경 없음).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from model_common import optimize_nonnegative_weights, write_json  # noqa: E402

spec_daily = importlib.util.spec_from_file_location("daily_v2", ROOT / "train_daily_official_v2_2026-08-21.py")
daily_mod = importlib.util.module_from_spec(spec_daily)
spec_daily.loader.exec_module(daily_mod)

spec_hourly = importlib.util.spec_from_file_location(
    "daytime_v2", ROOT / "train_hourly_daytime_multihorizon_v2_2026-08-21.py"
)
hourly_mod = importlib.util.module_from_spec(spec_hourly)
spec_hourly.loader.exec_module(hourly_mod)

OUT_DIR = ROOT / "outputs" / "일간_계층조정_v3_2026-08-21"

DAILY_MODEL_CANDIDATES = {
    "LightGBM": lambda seed: LGBMRegressor(
        objective="regression_l1", n_estimators=500, learning_rate=0.025,
        num_leaves=15, max_depth=6, min_child_samples=14,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, verbosity=-1, n_jobs=4,
    ),
    "XGBoost": lambda seed: XGBRegressor(
        objective="reg:absoluteerror", n_estimators=500, learning_rate=0.025,
        max_depth=4, min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2.0, random_state=seed, n_jobs=4,
    ),
}


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.abs(y - p)
    return float(e.sum() / np.abs(y).sum() * 100)


def daily_oof_predictions_all_models(data, features, windows, capacity_daily, seed) -> dict[str, pd.DataFrame]:
    """폴드별 직접 일간모델 예측(OOF)을 후보 모델별로 전부 계산."""
    out = {name: [] for name in DAILY_MODEL_CANDIDATES}
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index <= end)]
        if len(train) < 60 or len(test) < 10:
            continue
        medians = train[features].median(numeric_only=True)
        for name, make_model in DAILY_MODEL_CANDIDATES.items():
            model = make_model(seed)
            model.fit(train[features].fillna(medians), train[daily_mod.ACTUAL])
            pred = np.clip(model.predict(test[features].fillna(medians)), 0, capacity_daily)
            for date, actual, p in zip(test.index, test[daily_mod.ACTUAL], pred):
                out[name].append({"날짜": date, "실제_일간총량_kWh": actual, "직접모델_kWh": p,
                                   "폴드": f"{i}_{w.get('_계절','')}"})
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


def hourly_sum_oof_predictions(df, windows, capacity_kw, seed):
    """폴드별 시간모델(06~19시 합) 예측(OOF) — LightGBM만(기존 관례 유지)."""
    harness = hourly_mod.harness
    per_hour_pred = []
    for target_hour in hourly_mod.DAYTIME_HOURS:
        horizon = 15 + target_hour
        frame = hourly_mod.build_frame_h(df, horizon)
        base_cols = [c for c in frame.columns if c not in hourly_mod.CANDIDATE_COLS and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        issue_rows = frame[frame.index.hour == hourly_mod.ISSUE_HOUR]

        for i, w in enumerate(windows, start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            train_all = issue_rows[issue_rows.index < start]
            test_all = issue_rows[(issue_rows.index >= start) & (issue_rows.index <= end)]
            if len(train_all) < 60 or len(test_all) < 10:
                continue

            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(
                tr_for_sel, hourly_mod.CANDIDATE_COLS, threshold=0.3,
                apply_multicollinearity=True, apply_deploy_filter=True,
            )
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 60 or len(test) < 10:
                continue

            model = LGBMRegressor(
                n_estimators=180, learning_rate=0.05, num_leaves=25, min_child_samples=15,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
            )
            model.fit(train[feature_cols], train["목표_발전출력_kW"])
            pred = np.clip(model.predict(test[feature_cols]), 0, capacity_kw)
            target_time = test.index + pd.to_timedelta(horizon - 1, unit="h")
            for date, p in zip(target_time.normalize(), pred):
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

    print("[1/3] 직접 일간모델 OOF 예측 — LightGBM·XGBoost 둘 다 계산해 비교...")
    data, features = daily_mod.build_daily_dataset()
    daily_oof_by_model = daily_oof_predictions_all_models(data, features, windows, capacity_daily, seed)
    for name, oof_df in daily_oof_by_model.items():
        print(f"  {name}: {len(oof_df)}건, WAPE={wape(oof_df['실제_일간총량_kWh'], oof_df['직접모델_kWh']):.2f}%")
    best_daily_name = min(daily_oof_by_model, key=lambda n: wape(
        daily_oof_by_model[n]["실제_일간총량_kWh"], daily_oof_by_model[n]["직접모델_kWh"]))
    print(f"  → 채택: {best_daily_name}(OOF WAPE 최소)")
    daily_oof = daily_oof_by_model[best_daily_name]

    print("\n[2/3] 시간모델(06~19시 합계) OOF 예측(LightGBM, 기존 관례 유지)...")
    df = pd.read_csv(hourly_mod.harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    hourly_oof = hourly_sum_oof_predictions(df, windows, capacity_kw, seed)
    print(f"  {len(hourly_oof)}건")

    print("\n[3/3] 결합 및 가중치 최적화...")
    merged = daily_oof.merge(hourly_oof[["날짜", "시간모델합계_kWh"]], on="날짜", how="inner")
    print(f"  공통 날짜 {len(merged)}건")

    candidates = ["직접모델_kWh", "시간모델합계_kWh"]
    result = optimize_nonnegative_weights(merged["실제_일간총량_kWh"], merged[candidates])
    weights = np.asarray(result["가중치"], dtype=float)
    merged["계층조정_kWh"] = np.clip(merged[candidates].to_numpy() @ weights, 0, capacity_daily)

    print(f"\n직접모델 알고리즘: {best_daily_name} | 가중치: 직접모델={weights[0]:.3f}, 시간모델합계={weights[1]:.3f}")

    print("\n=== 비교 (전체 OOF, 5폴드 합산) ===")
    rows = []
    for name, col in [(f"직접모델({best_daily_name})", "직접모델_kWh"), ("시간모델합계(LightGBM)", "시간모델합계_kWh"),
                       ("계층조정", "계층조정_kWh")]:
        w = wape(merged["실제_일간총량_kWh"], merged[col])
        e = np.abs(merged["실제_일간총량_kWh"] - merged[col])
        mae, rmse = float(e.mean()), float(np.sqrt((e ** 2).mean()))
        rows.append({"구성": name, "n": len(merged), "WAPE_pct": round(w, 2), "MAE_kWh": round(mae, 1), "RMSE_kWh": round(rmse, 1)})
        print(f"  {name:24s} WAPE={w:.2f}% MAE={mae:.1f}kWh RMSE={rmse:.1f}kWh")

    pd.DataFrame(rows).to_csv(OUT_DIR / "비교표.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT_DIR / "OOF_상세.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "가중치.json", {**result, "직접모델_알고리즘": best_daily_name,
                                        "직접모델_가중치": float(weights[0]), "시간모델합계_가중치": float(weights[1])})
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
