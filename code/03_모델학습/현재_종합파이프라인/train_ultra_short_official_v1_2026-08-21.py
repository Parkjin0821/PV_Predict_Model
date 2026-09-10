# -*- coding: utf-8 -*-
"""초단기(15분 단위, +1~4시간) 모델 — 공식 KMA 파이프라인, 08-21 신규.

## 스펙 재확인 (AGENTS.md "공식 예측 주기" 절)
갱신 주기: 5~15분 롤링 / 데이터·출력 단위: 15분 / **예측수평: +1h·+2h·
+3h·+4h**(15분 후·30분 후를 뜻하지 않음 — "5분~15분"은 예측을 얼마나
자주 갱신하느냐일 뿐, 몇 분 뒤를 맞추느냐가 아니다).

## 시간정렬 설계 (하이브리드 — 이유 포함)
- **장비텔레메트리(발전량 포함, 15분 시작라벨)**: 이건 진짜 15분 해상도로
  갱신되므로 `select_features_by_correlation_threshold_v3`와 동일하게
  shift(1)(직전 완결 15분 구간)을 그대로 15분 단위에 적용.
- **ASOS 관측·NWP 예보(둘 다 시간 단위 원본, 15분 해상도로 갱신되지
  않음)**: 시간단위로 정렬(하네스와 동일 규약: ASOS=shift(0), NWP=
  shift(-H))한 뒤 **그 시간(hour) 안의 4개 15분 슬롯에 그대로 방송
  (broadcast)**한다 — 실제로 그 값이 그 시간 내내 "가장 최근에 알려진
  값"이므로 억지 보간이 아니라 정확한 표현이다.
- **타깃**: 15분 자기 지속 특성과 같은 시작라벨 규약이라, 시간단위처럼
  "(H-1)" 트릭이 성립하지 않는다(15분·시간 해상도가 다른 자료라 자연
  오프셋이 없음). **target = 발전출력_kW.shift(-(H*4))**(H시간 뒤 15분
  버킷, 명시적 정수 오프셋 사용).

## 특성선택·검증
하네스와 동일 원칙 재사용: 폴드(5계절) 학습구간 안에서 상관계수→다중
공선성 가지치기→Blockdata 배포가능성 필터, 시험구간은 전혀 안 봄.
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
_spec = importlib.util.spec_from_file_location(
    "harness", ROOT / "backtest_harness_v1_2026-08-20밤.py"
)
harness = importlib.util.module_from_spec(_spec)
sys.modules["harness"] = harness  # @dataclass 내부에서 sys.modules 조회가 필요함
_spec.loader.exec_module(harness)

QUARTER_15MIN_PARQUET = ROOT / "outputs" / "집계_15분_자료.parquet"
OUT_DIR = ROOT / "outputs" / "초단기_15분_v1_2026-08-21"

# 15분 자료 컬럼명 -> 시간단위 파이프라인 표준명으로 매핑(동일 물리량)
RENAME_15MIN = {
    "입력전력_kW": "plant_input_power_kw",
    "입력전압평균_V": "mean_input_voltage_v",
    "역률평균": "mean_power_factor",
}
# 15분 자료의 주파수·온도는 08-20 밤 재정의 이전 값(정상인버터만 평균 아님)이고,
# 애초에 최종특성에서 배제(주파수=상관 미달, 온도=Blockdata 배포불가)돼
# 있었으므로 이번 초단기 모델에서도 처음부터 후보에서 제외한다.
EQUIP_STARTLABEL = ["plant_input_power_kw", "mean_input_voltage_v", "mean_power_factor"]


def load_15min_base() -> pd.DataFrame:
    df = pd.read_parquet(QUARTER_15MIN_PARQUET)
    df = df.rename(columns=RENAME_15MIN)
    return df


def hourly_features_for_horizon(hourly_df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """ASOS·NWP만 시간단위 규약(harness.sel과 동일)으로 정렬한 프레임.
    (장비 특성은 여기 안 넣는다 — 15분 원자료에서 따로 붙임)"""
    out = pd.DataFrame(index=hourly_df.index)
    sel = harness.sel
    for c in sel.OBSERVED_COLUMNS:
        out[c] = hourly_df[c]
    for c in sel.FORECAST_COLUMNS:
        out[c] = hourly_df[c].shift(-horizon_hours)
    return out


def build_ultra_short_frame(quarter: pd.DataFrame, hourly_df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    hourly_feats = hourly_features_for_horizon(hourly_df, horizon_hours)
    # 시간 단위 특성을 그 시각이 속한 1시간 동안의 4개 15분 슬롯에 방송
    hour_key = quarter.index.floor("1h")
    broadcast = hourly_feats.reindex(hour_key)
    broadcast.index = quarter.index

    frame = pd.DataFrame(index=quarter.index)
    for c in EQUIP_STARTLABEL:
        frame[c] = quarter[c].shift(1)  # 직전 완결 15분 구간(진짜 15분 해상도)
    for c in broadcast.columns:
        frame[c] = broadcast[c]

    power = quarter["발전출력_kW"].shift(1)
    for steps in [1, 2, 4, 8, 16]:  # 15,30,60,120,240분 전
        frame[f"발전출력_{steps*15}분전_kW"] = power.shift(steps - 1)
    for steps in [4, 16]:  # 1시간·4시간 이동통계
        frame[f"발전출력_{steps*15}분이동평균_kW"] = power.rolling(steps, min_periods=max(2, steps // 2)).mean()
        frame[f"발전출력_{steps*15}분이동표준편차_kW"] = power.rolling(steps, min_periods=max(2, steps // 2)).std()

    target_time = quarter.index + pd.Timedelta(hours=horizon_hours)
    frame["목표_태양고도_deg"] = quarter["태양고도_deg"].shift(-horizon_hours * 4)
    minute = target_time.hour * 60 + target_time.minute
    frame["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    frame["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    frame["연주기_sin"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    frame["연주기_cos"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)

    frame["목표_발전출력_kW"] = quarter["발전출력_kW"].shift(-horizon_hours * 4)
    frame["목표_낮시간"] = (frame["목표_태양고도_deg"] > 0).astype(float)
    frame["_지속성_직전출력_kW"] = quarter["발전출력_kW"].shift(1)
    return frame


def metrics(y, p, capacity_kw, ref_rmse=None):
    e = y - p
    mae = float(np.abs(e).mean()); rmse = float(np.sqrt((e ** 2).mean()))
    out = {"n": int(len(y)), "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
           "nMAE_pct": round(mae / capacity_kw * 100, 3), "nRMSE_pct": round(rmse / capacity_kw * 100, 3)}
    if ref_rmse and ref_rmse > 0:
        out["SkillScore"] = round(1 - rmse / ref_rmse, 4)
    return out


def make_model(name, seed):
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
                              subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1)
    return XGBRegressor(n_estimators=220, learning_rate=0.04, max_depth=6, min_child_weight=5,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=0)


def run_horizon(horizon_hours: int, quarter: pd.DataFrame, hourly_df: pd.DataFrame,
                 windows: list[dict], capacity_kw: float, seed: int) -> list[dict]:
    frame = build_ultra_short_frame(quarter, hourly_df, horizon_hours)
    candidate_cols = EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]

    results = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
        if len(train_all) < 500 or len(test_all) < 100:
            continue

        tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
        chosen = harness.select_features_in_fold(
            tr_for_sel, candidate_cols, threshold=0.3,
            apply_multicollinearity=True, apply_deploy_filter=True,
        )
        feature_cols = base_cols + chosen
        required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + [
            "목표_발전출력_kW", "_지속성_직전출력_kW"
        ]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=required)
        if len(train) < 500 or len(test) < 100:
            continue

        y_train = train["목표_발전출력_kW"].to_numpy()
        y_test = test["목표_발전출력_kW"].to_numpy()
        pers = np.clip(test["_지속성_직전출력_kW"].to_numpy(), 0, capacity_kw)
        ref_rmse = metrics(y_test, pers, capacity_kw)["RMSE_kW"]

        row = {"수평_h": horizon_hours, "폴드": f"{i}_{w.get('_계절','')}",
               "학습표본": len(train), "시험표본": len(test), "선택특성수": len(chosen),
               "선택특성": chosen, "지속성_RMSE": ref_rmse}
        for name in ["LightGBM", "XGBoost"]:
            m = make_model(name, seed)
            m.fit(train[feature_cols], y_train)
            p = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
            row[name] = metrics(y_test, p, capacity_kw, ref_rmse)
        results.append(row)
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    windows = config["cross_validation_windows"]

    quarter = load_15min_base()
    hourly_df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    print(f"15분 표본: {len(quarter)}, 시간단위(v3 고정tm) 표본: {len(hourly_df)}")

    all_results = []
    for H in [1, 2, 3, 4]:
        print(f"\n=== +{H}h ===")
        res = run_horizon(H, quarter, hourly_df, windows, capacity_kw, seed)
        all_results.extend(res)
        for r in res:
            print(f"  [{r['폴드']}] 학습{r['학습표본']} 시험{r['시험표본']} 특성{r['선택특성수']}개  "
                  f"지속성RMSE={r['지속성_RMSE']:.2f}  LightGBM={r['LightGBM']['RMSE_kW']:.2f}  "
                  f"XGBoost={r['XGBoost']['RMSE_kW']:.2f}")

    (OUT_DIR / "폴드별_상세.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = []
    for H in [1, 2, 3, 4]:
        rows = [r for r in all_results if r["수평_h"] == H]
        if not rows:
            continue
        for name in ["LightGBM", "XGBoost"]:
            maes = [r[name]["MAE_kW"] for r in rows]
            rmses = [r[name]["RMSE_kW"] for r in rows]
            nrmses = [r[name]["nRMSE_pct"] for r in rows]
            skills = [r[name]["SkillScore"] for r in rows if r[name].get("SkillScore") is not None]
            summary.append({"수평_h": H, "모델": name, "폴드수": len(rows),
                             "평균MAE_kW": round(float(np.mean(maes)), 3),
                             "평균RMSE_kW": round(float(np.mean(rmses)), 3),
                             "평균nRMSE_pct": round(float(np.mean(nrmses)), 3),
                             "평균Skill": round(float(np.mean(skills)), 4) if skills else None})
        pers_rmse = [r["지속성_RMSE"] for r in rows]
        summary.append({"수평_h": H, "모델": "지속성", "폴드수": len(rows),
                         "평균MAE_kW": None, "평균RMSE_kW": round(float(np.mean(pers_rmse)), 3),
                         "평균nRMSE_pct": round(float(np.mean(pers_rmse)) / capacity_kw * 100, 3), "평균Skill": None})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "수평별_요약.csv", index=False, encoding="utf-8-sig")
    print("\n\n=== 수평별 요약 ===")
    print(summary_df.to_string(index=False))
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
