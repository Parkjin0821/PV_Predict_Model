# -*- coding: utf-8 -*-
"""익일 낮시간 전체 다중시각 1시간 예측 — v3(고정-tm, 누출없음) 재실행.

## 왜 다시 만드는가
v1(08-20 밤)은 두 가지가 지금 기준으로 무효다:
1. **데이터셋**: `gwangju_1hour_model_dataset_official_v2_2026-08-20밤.csv`
   (NWP가 매 시각 직전 런 기준 — D 10시 발행·D+1 예측 관점에서 미래정보
   누출, AGENTS.md "★★★전체 감사★★★" 절).
2. **특성**: 하드코딩된 `FINAL_14`를 그대로 썼는데, 이 목록 자체가
   v2·1시간 오프셋(+1h) 상관분석 결과라 `DSWRF`·`TCDC`·`LCDC`·
   `DSWRFLX_bsrn정제`(감사에서 확정된 누출 5종 중 4개)가 그대로 들어있고,
   +21~+34h라는 훨씬 긴 수평에 맞는 재선택도 아니었다.

## 이번 버전
- **데이터셋**: `backtest_harness_v1_2026-08-20밤.py`의
  `DATASETS["v3_고정tm"]`(고정-tm 재수집, 누출 없음).
- **특성**: 하드코딩 대신 하네스의 `build_frame(df, horizon, candidate_cols)`
  로 각 목표시각(horizon=15+H, H=06~19)에 맞는 프레임을 만들고,
  `select_features_in_fold`(상관계수 임계값 0.3 + 다중공선성 가지치기
  + Blockdata 배포가능성 필터, 08-21 단기·초단기와 동일 절차)로
  **매 목표시각·매 폴드마다 그 학습구간 안에서** 특성을 다시 고른다
  — 시험구간은 전혀 보지 않는다.
- **나머지(발행 관례 10시, 대상 06~19시 14개, 폴드=5계절)는 v1과 동일.**
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("harness", ROOT / "backtest_harness_v1_2026-08-20밤.py")
harness = importlib.util.module_from_spec(_spec)
sys.modules["harness"] = harness
_spec.loader.exec_module(harness)

OUT_DIR = ROOT / "outputs" / "익일낮시간_다중시각_v2_2026-08-21"

ISSUE_HOUR = 10
DAYTIME_HOURS = list(range(6, 20))  # 06~19시, 14개
CANDIDATE_COLS = harness.sel.CANDIDATE_COLUMNS  # 전체후보(31개) — 목표시각마다 재선택


def build_frame_h(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """하네스의 build_frame을 그대로 쓴다(v1의 로컬 구현과 동일한 정렬규약,
    후보열만 STARTLABEL/OBSERVED/FORECAST 전체로 확장됨)."""
    return harness.build_frame(df, horizon, CANDIDATE_COLS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    windows = config["cross_validation_windows"]

    df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()

    print("[사전확인] 06~19시가 일간 총발전량에서 차지하는 비중...")
    daily_total = df["plant_output_kw"].resample("1D").sum(min_count=12)
    daytime_total = df[df.index.hour.isin(DAYTIME_HOURS)]["plant_output_kw"].resample("1D").sum(min_count=8)
    ratio = (daytime_total / daily_total).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  06~19시/일간총량 비율: 평균 {ratio.mean()*100:.2f}%, 중앙값 {ratio.median()*100:.2f}%, 5~95백분위 [{ratio.quantile(.05)*100:.1f}%, {ratio.quantile(.95)*100:.1f}%]")

    all_scores = []
    feature_log = []  # 목표시각·폴드별로 실제 선택된 특성(감사용)
    horizon_test_frames = {}  # 폴드별-시각별 시험 예측(합계 검증용)

    for target_hour in DAYTIME_HOURS:
        horizon = 15 + target_hour
        frame = build_frame_h(df, horizon)
        base_cols = [c for c in frame.columns if c not in CANDIDATE_COLS and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        issue_rows = frame[frame.index.hour == ISSUE_HOUR]

        for i, w in enumerate(windows, start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            train_all = issue_rows[issue_rows.index < start]
            test_all = issue_rows[(issue_rows.index >= start) & (issue_rows.index <= end)]
            if len(train_all) < 60 or len(test_all) < 10:
                continue

            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(
                tr_for_sel, CANDIDATE_COLS, threshold=0.3,
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
            actual = test["목표_발전출력_kW"].to_numpy()
            e = actual - pred
            all_scores.append({
                "목표시각": target_hour, "폴드": f"{i}_{w.get('_계절','')}", "n": len(test),
                "MAE_kW": round(float(np.abs(e).mean()), 3), "RMSE_kW": round(float(np.sqrt((e**2).mean())), 3),
            })
            feature_log.append({"목표시각": target_hour, "폴드": f"{i}_{w.get('_계절','')}",
                                 "선택특성수": len(chosen), "선택특성": chosen})
            key = f"{i}_{w.get('_계절','')}"
            fold_pred = pd.DataFrame({
                "목표대상시각": test.index + pd.to_timedelta(horizon - 1, unit="h"),
                "예측_kW": pred, "실제_kW": actual,
            })
            horizon_test_frames.setdefault(key, []).append(fold_pred)
        print(f"  목표시각 {target_hour:02d}시(horizon={horizon}) 완료")

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(OUT_DIR / "시각별_폴드별_성능.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_log).to_json(OUT_DIR / "선택특성_로그.json", orient="records", force_ascii=False, indent=2)

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
