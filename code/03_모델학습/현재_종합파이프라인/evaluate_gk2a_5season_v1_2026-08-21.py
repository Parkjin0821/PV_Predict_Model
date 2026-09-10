# -*- coding: utf-8 -*-
"""GK2A 5계절 재검증 교차분석 — VI006 vs IR105 vs 위성없음, 초단기 모델 기준.

## 배경
14일 시범(초여름 폴드 1개, 4일 시험구간)에서 나온 신호(+1h IR105
45%↓, +2~4h VI006 23~28%↓)가 표본이 작아 "확정"이 아니었다. 사용자
결정(2026-08-21)으로 나머지 4개 폴드(여름·가을·겨울·봄)에서 각 10일씩
추가 수집(Codex 실행, 완료)했고, 이 스크립트가 **5계절 전체**에서 같은
패턴이 재현되는지 검증한다.

## 방법 (14일 시범과 동일 원칙, 폴드 단위로 반복)
`evaluate_gk2a_trial_v1_2026-08-21.py`의 핵심 설계를 그대로 재사용한다
— 위성자료가 각 폴드 안(10~14일)에만 있으므로, **그 폴드 안에서만**
훈련/시험을 나눈다(폴드 밖 데이터는 위성값이 통째로 결측이라 상관계수를
못 구해 항상 탈락하는 버그를 이미 겪었다 — 반복 방지). 5개 폴드
전부 동일 원칙(약 70% 훈련 / 약 30% 시험)으로 분할한다.

| 계절 | 전체구간 | 훈련 | 시험 |
|---|---|---|---|
| 초여름(기존 14일) | 04-27~05-10 | 04-27~05-06(10일) | 05-07~05-10(4일) |
| 여름(신규) | 07-17~07-26 | 07-17~07-23(7일) | 07-24~07-26(3일) |
| 가을(신규) | 12-05~12-14 | 12-05~12-11(7일) | 12-12~12-14(3일) |
| 겨울(신규) | 12-16~12-25 | 12-16~12-22(7일) | 12-23~12-25(3일) |
| 봄(신규) | 04-01~04-10 | 04-01~04-07(7일) | 04-08~04-10(3일) |

## 결측 처리 (Codex 수집 보고 반영)
5계절 CSV는 채널당 17/320개(5.3%) 결측이 있고(가을·겨울 UTC 06시에
집중), 두 채널에서 결측 시각이 동일하다. 임의 보간을 하지 않고
기존 파이프라인 그대로 둔다 — 위성 특성이 선택된 경우 `evaluate()`의
`dropna(subset=required)`가 그 특성의 결측 시각을 자동으로 학습/시험
에서 제외하므로, 사실상 "공통 유효시각 기준" 비교가 자연히 이뤄진다
(사용자가 요청한 원칙과 일치, 별도 보간 코드 불필요).

## 위성 시간정렬
위성은 3시간 간격 관측이므로 다음 시각 전까지(최대 165분, 3시간 미만)
전진채움(ffill)해 15분 슬롯에 방송한다(14일 시범과 동일). 5개 폴드가
서로 몇 달씩 떨어져 있어 폴드 경계를 넘는 방송(leak)은 발생하지 않는다
(ffill 한도가 165분뿐이라 자동으로 안전).
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

SAT_CSV_TRIAL14 = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_trial_14d_v1_2026-08-21"
    r"\광주_위성픽셀_14일시범_VI006_IR105.csv"
)
SAT_CSV_5SEASON = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\gk2a_reverify_5season_v1_2026-08-21"
    r"\광주_위성픽셀_5계절재검증_VI006_IR105.csv"
)
OUT_DIR = ROOT / "outputs" / "위성_5계절재검증_v1_2026-08-21"
SAT_COLS = ["VI006_5x5평균", "IR105_5x5평균"]

# (계절, 전체시작, 시험시작, 전체끝(다음날 자정, 배타)) — 위 표와 동일
WINDOWS = [
    {"계절": "초여름(기존14일)", "train_start": "2026-04-27", "test_start": "2026-05-07", "end": "2026-05-11"},
    {"계절": "여름(신규)", "train_start": "2025-07-17", "test_start": "2025-07-24", "end": "2025-07-27"},
    {"계절": "가을(신규)", "train_start": "2025-12-05", "test_start": "2025-12-12", "end": "2025-12-15"},
    {"계절": "겨울(신규)", "train_start": "2025-12-16", "test_start": "2025-12-23", "end": "2025-12-26"},
    {"계절": "봄(신규)", "train_start": "2026-04-01", "test_start": "2026-04-08", "end": "2026-04-11"},
]


def load_satellite_all() -> pd.DataFrame:
    trial14 = pd.read_csv(SAT_CSV_TRIAL14, parse_dates=["시각_kst"])
    trial14["계절"] = "초여름(기존14일)"
    season5 = pd.read_csv(SAT_CSV_5SEASON, parse_dates=["시각_kst"])
    combined = pd.concat([trial14[["계절", "시각_kst"] + SAT_COLS],
                           season5[["계절", "시각_kst"] + SAT_COLS]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["시각_kst"]).set_index("시각_kst").sort_index()
    return combined


def load_satellite_broadcast(quarter_index: pd.DatetimeIndex, sat: pd.DataFrame) -> pd.DataFrame:
    sat = sat[SAT_COLS]
    reindexed = sat.reindex(quarter_index.union(sat.index)).sort_index()
    filled = reindexed.ffill(limit=11)  # 15분 x 11 = 165분 < 180분, 다음 관측 직전까지만
    return filled.reindex(quarter_index)


def build_frame_with_satellite(quarter: pd.DataFrame, hourly_df: pd.DataFrame, horizon_hours: int,
                                sat_broadcast: pd.DataFrame, use_sat_cols: list[str]) -> pd.DataFrame:
    frame = ultra.build_ultra_short_frame(quarter, hourly_df, horizon_hours)
    for c in use_sat_cols:
        frame[c] = sat_broadcast[c]
    return frame


def evaluate_window(name: str, frame: pd.DataFrame, extra_cols: list[str], capacity_kw: float, seed: int,
                     window: dict) -> dict | None:
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS + extra_cols
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]

    train_start = pd.Timestamp(window["train_start"])
    test_start = pd.Timestamp(window["test_start"])
    end = pd.Timestamp(window["end"])

    train_all = daylight[(daylight.index >= train_start) & (daylight.index < test_start)]
    test_all = daylight[(daylight.index >= test_start) & (daylight.index < end)]

    tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
    if len(tr_for_sel) < 10:
        return None
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
    if len(train) < 10 or len(test) < 5:
        return None

    y_train = train["목표_발전출력_kW"].to_numpy()
    y_test = test["목표_발전출력_kW"].to_numpy()
    pers = np.clip(test["_지속성_직전출력_kW"].to_numpy(), 0, capacity_kw)
    ref_rmse = ultra.metrics(y_test, pers, capacity_kw)["RMSE_kW"]

    out = {"계절": window["계절"], "구성": name, "학습표본": len(train), "시험표본": len(test),
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
    sat_all = load_satellite_all()
    sat_broadcast = load_satellite_broadcast(quarter.index, sat_all)

    print("=== 폴드별 위성 방송 유효비율 ===")
    for w in WINDOWS:
        mask = (quarter.index >= pd.Timestamp(w["train_start"])) & (quarter.index < pd.Timestamp(w["end"]))
        sub = sat_broadcast.loc[mask]
        print(f"  {w['계절']}: 15분슬롯{mask.sum()}개 중 "
              f"VI006 {sub['VI006_5x5평균'].notna().mean()*100:.1f}%, "
              f"IR105 {sub['IR105_5x5평균'].notna().mean()*100:.1f}%")

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
            for window in WINDOWS:
                res = evaluate_window(name, frame, extra, capacity_kw, seed, window)
                if res is None:
                    print(f"  [{window['계절']:16s} {name:16s}] 표본부족으로 건너뜀")
                    continue
                res["수평_h"] = H
                all_rows.append(res)
                sat_pick = res["위성특성_선택여부"] or "(없음)"
                print(f"  [{window['계절']:16s} {name:16s}] 특성{res['선택특성수']}개(위성채택:{sat_pick}) "
                      f"LightGBM={res['LightGBM']['RMSE_kW']:.3f} XGBoost={res['XGBoost']['RMSE_kW']:.3f} "
                      f"지속성={res['지속성_RMSE']:.3f}")

    (OUT_DIR / "상세결과.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pd.DataFrame([
        {"수평_h": r["수평_h"], "계절": r["계절"], "구성": r["구성"], "선택특성수": r["선택특성수"],
         "위성채택": ",".join(r["위성특성_선택여부"]) or "-",
         "LightGBM_RMSE": r["LightGBM"]["RMSE_kW"], "XGBoost_RMSE": r["XGBoost"]["RMSE_kW"],
         "지속성_RMSE": r["지속성_RMSE"]}
        for r in all_rows
    ])
    summary.to_csv(OUT_DIR / "요약_폴드별.csv", index=False, encoding="utf-8-sig")

    # 수평x구성별 5계절 평균(계절마다 표본수가 다르므로 단순평균, 계절 수는 함께 표기)
    pooled = summary.groupby(["수평_h", "구성"]).agg(
        폴드수=("계절", "count"),
        LightGBM_RMSE평균=("LightGBM_RMSE", "mean"),
        XGBoost_RMSE평균=("XGBoost_RMSE", "mean"),
        지속성_RMSE평균=("지속성_RMSE", "mean"),
    ).reset_index()
    pooled.to_csv(OUT_DIR / "요약_5계절평균.csv", index=False, encoding="utf-8-sig")

    print("\n\n=== 폴드별 상세 ===")
    print(summary.to_string(index=False))
    print("\n\n=== 수평x구성별 5계절 평균 ===")
    print(pooled.to_string(index=False))
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
