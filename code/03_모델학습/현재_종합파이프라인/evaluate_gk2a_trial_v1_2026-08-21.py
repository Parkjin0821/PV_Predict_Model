# -*- coding: utf-8 -*-
"""GK2A 14일 시범자료 교차분석 — VI006 vs IR105 vs 위성없음, 초단기 모델 기준.

## 방법
train_ultra_short_official_v1의 구조를 그대로 쓰되, 시험구간을 위성 시범
기간(2026-04-27~05-10, 14일)으로 고정한 단일 분할로 4가지 특성 구성을
비교한다: ①위성없음(기존 초단기 기준선) ②VI006만 추가 ③IR105만 추가
④둘 다 추가. 학습은 2026-04-27 이전 전체 데이터, 시험은 그 14일뿐.

## 위성 특성의 시간정렬
위성은 NWP처럼 사전예보가 아니라 ASOS와 같은 "관측"이다 — 해당 시각에
찍힌 하늘 상태를 담고 있으므로, 그 시각 이후에만 "안다"고 볼 수 있다.
3시간 간격(8시점/일) 자료이므로, 각 시각의 값을 **그다음 위성 시각
전까지(최대 3시간) 그대로 유지(forward-fill)**해 15분 슬롯에 방송한다
(ASOS를 시간 단위로 방송했던 것과 동일 원리, 간격만 3시간).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ultra", ROOT / "train_ultra_short_official_v1_2026-08-21.py")
ultra = importlib.util.module_from_spec(_spec)
sys.modules["ultra"] = ultra
_spec.loader.exec_module(ultra)

SATELLITE_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_trial_14d_v1_2026-08-21"
    r"\광주_위성픽셀_14일시범_VI006_IR105.csv"
)
OUT_DIR = ROOT / "outputs" / "위성_14일_교차분석_v1_2026-08-21"

TRIAL_START = pd.Timestamp("2026-04-27")
TRIAL_END = pd.Timestamp("2026-05-11")  # 05-10 23:59까지 포함되도록 익일 자정 미만
# ★수정(버그 발견·수정)★: 위성자료가 이 14일 안에만 있어서, 훈련구간을
# "그 이전 전체"로 잡으면 훈련셋의 위성값이 전부 NaN이 되어 상관계수 자체를
# 못 구하고 항상 탈락했다(선택특성수가 위성 유무와 무관하게 똑같았던 원인).
# **14일 안에서 훈련/시험을 나눠야만** 위성 특성이 실제로 학습에 참여한다 —
# 앞 10일 학습, 뒤 4일 시험(소표본 한계는 명시적으로 감안).
TEST_START = pd.Timestamp("2026-05-07")
SAT_COLS = ["VI006_5x5평균", "IR105_5x5평균"]


def load_satellite_broadcast(quarter_index: pd.DatetimeIndex) -> pd.DataFrame:
    sat = pd.read_csv(SATELLITE_CSV, parse_dates=["시각_kst"]).set_index("시각_kst").sort_index()
    sat = sat[SAT_COLS]
    # 3시간 간격 값을 다음 위성 시각 전까지 15분 슬롯에 전진채움(최대 3시간, 그 이상은 결측 유지)
    reindexed = sat.reindex(quarter_index.union(sat.index)).sort_index()
    filled = reindexed.ffill(limit=11)  # 15분 x 11 = 165분 < 180분, 다음 시각 도달 직전까지만
    return filled.reindex(quarter_index)


def build_frame_with_satellite(quarter: pd.DataFrame, hourly_df: pd.DataFrame, horizon_hours: int,
                                sat_broadcast: pd.DataFrame, use_sat_cols: list[str]) -> pd.DataFrame:
    frame = ultra.build_ultra_short_frame(quarter, hourly_df, horizon_hours)
    for c in use_sat_cols:
        frame[c] = sat_broadcast[c]
    return frame


def evaluate(name: str, frame: pd.DataFrame, extra_cols: list[str], capacity_kw: float, seed: int) -> dict:
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS + extra_cols
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]

    # ★수정★: 위성이 있는 14일(TRIAL_START~TRIAL_END) 안에서만 훈련·시험을 나눈다.
    train_all = daylight[(daylight.index >= TRIAL_START) & (daylight.index < TEST_START)]
    test_all = daylight[(daylight.index >= TEST_START) & (daylight.index < TRIAL_END)]

    tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
    chosen = ultra.harness.select_features_in_fold(
        tr_for_sel, candidate_cols, threshold=0.3,
        apply_multicollinearity=True, apply_deploy_filter=True,
    )
    feature_cols = base_cols + chosen
    required = [c for c in feature_cols if c not in ultra.harness.NATIVE_MISSING_OK] + [
        "목표_발전출력_kW", "_지속성_직전출력_kW"
    ]
    train = train_all.dropna(subset=required)
    test = test_all.dropna(subset=required)

    y_train = train["목표_발전출력_kW"].to_numpy()
    y_test = test["목표_발전출력_kW"].to_numpy()
    pers = np.clip(test["_지속성_직전출력_kW"].to_numpy(), 0, capacity_kw)
    ref_rmse = ultra.metrics(y_test, pers, capacity_kw)["RMSE_kW"]

    out = {"구성": name, "학습표본": len(train), "시험표본": len(test),
           "선택특성수": len(chosen), "위성특성_선택여부": [c for c in chosen if c in SAT_COLS],
           "지속성_RMSE": ref_rmse}
    for model_name, cls in [("LightGBM", LGBMRegressor), ("XGBoost", XGBRegressor)]:
        m = ultra.make_model(model_name, seed)
        m.fit(train[feature_cols], y_train)
        p = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
        out[model_name] = ultra.metrics(y_test, p, capacity_kw, ref_rmse)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    quarter = ultra.load_15min_base()
    hourly_df = pd.read_csv(
        ultra.harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False
    ).set_index("time").sort_index()
    sat_broadcast = load_satellite_broadcast(quarter.index)
    trial_mask = (quarter.index >= TRIAL_START) & (quarter.index < TRIAL_END)
    trial_sat = sat_broadcast.loc[trial_mask]
    print(f"위성 방송 결과 유효비율(14일 시범구간 15분 슬롯 {trial_mask.sum()}개 기준): "
          f"VI006 {trial_sat['VI006_5x5평균'].notna().mean()*100:.1f}%, "
          f"IR105 {trial_sat['IR105_5x5평균'].notna().mean()*100:.1f}%")

    variants = {
        "위성없음(기존기준선)": [],
        "VI006만": ["VI006_5x5평균"],
        "IR105만": ["IR105_5x5평균"],
        "VI006+IR105": ["VI006_5x5평균", "IR105_5x5평균"],
    }

    all_rows = []
    for H in [1, 2, 3, 4]:
        print(f"\n=== +{H}h ===")
        for name, extra in variants.items():
            frame = build_frame_with_satellite(quarter, hourly_df, H, sat_broadcast, extra)
            res = evaluate(name, frame, extra, capacity_kw, seed)
            res["수평_h"] = H
            all_rows.append(res)
            sat_pick = res["위성특성_선택여부"] or "(없음)"
            print(f"  [{name:16s}] 특성{res['선택특성수']}개(위성채택:{sat_pick}) "
                  f"LightGBM={res['LightGBM']['RMSE_kW']:.3f} XGBoost={res['XGBoost']['RMSE_kW']:.3f} "
                  f"지속성={res['지속성_RMSE']:.3f}")

    (OUT_DIR / "상세결과.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pd.DataFrame([
        {"수평_h": r["수평_h"], "구성": r["구성"], "선택특성수": r["선택특성수"],
         "위성채택": ",".join(r["위성특성_선택여부"]) or "-",
         "LightGBM_RMSE": r["LightGBM"]["RMSE_kW"], "XGBoost_RMSE": r["XGBoost"]["RMSE_kW"],
         "지속성_RMSE": r["지속성_RMSE"]}
        for r in all_rows
    ])
    summary.to_csv(OUT_DIR / "요약.csv", index=False, encoding="utf-8-sig")
    print("\n\n=== 전체 요약 ===")
    print(summary.to_string(index=False))
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
