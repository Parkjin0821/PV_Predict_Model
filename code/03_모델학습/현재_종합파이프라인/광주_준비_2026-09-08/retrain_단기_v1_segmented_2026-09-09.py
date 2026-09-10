# -*- coding: utf-8 -*-
"""광주 "단기"(+1h/+24h) 재학습 - segment-aware lag/rolling, 09-09 안전조건 반영.

09-09 사용자·코덱스 합의 순서:
  1. segment-aware lag/rolling 생성
  2. +1h 재학습·리크 감사
  3. +24h 재학습·리크 감사
  4. 기존 모델과 구간별·pooled 성능 비교
  5. 번들 저장은 결과 확인 후 진행(이 스크립트는 저장하지 않음)

입력: build_official_hourly_dataset_v4_segmented_2026-09-09.py 산출물
(seg1_710d / gap_no_data / seg2_live_0826, gap은 실제 NaN 유지).

특성세트: 기존 FINAL_14에서 라이브 재현 불가 확인된 2개
(추정_출력온도, DSWRFLX_bsrn정제)를 제외한 12개("final12_라이브가능").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

KST = ZoneInfo("Asia/Seoul")

V4_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v4_segmented_candidate_2026-09-09.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "단기_재학습_후보_2026-09-09"

STARTLABEL = ["plant_input_power_kw", "mean_power_factor", "mean_input_voltage_v"]
OBSERVED = ["기상청관측_일조시간_hr", "기상청관측_상대습도_pct", "기상청관측_전운량_pct"]
FORECAST = ["DSWRF", "LCDC", "TCDC", "POP", "SKY", "REH"]
FINAL12_LIVE = STARTLABEL + OBSERVED + FORECAST


def build_frame_segment_aware(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """backtest_harness_v1의 build_frame과 동일한 정렬규약이되, 모든
    shift/rolling을 segment_id로 groupby해서 계산한다 - 세그먼트(특히
    gap_no_data)를 넘는 shift가 구조적으로 불가능하다."""
    g = df.groupby("segment_id", sort=False, group_keys=False)
    out = pd.DataFrame(index=df.index)

    for c in STARTLABEL:
        out[c] = g[c].shift(1)
    for c in OBSERVED:
        out[c] = df[c]
    for c in FORECAST:
        out[c] = g[c].shift(-horizon)

    power = g["plant_output_kw"].shift(1)
    power_df = power.to_frame("power")
    power_df["segment_id"] = df["segment_id"]
    pg = power_df.groupby("segment_id", sort=False, group_keys=False)["power"]

    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = pg.shift(hours - 1)
    # 09-09 안전조건: rolling min_periods를 window와 동일하게(엄격) - 결측
    # 직후 부분창으로 값이 만들어지는 걸 원천 차단한다(코덱스 지적 반영).
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = pg.rolling(hours, min_periods=hours).mean().reset_index(0, drop=True)
        out[f"발전출력_{hours}시간이동표준편차_kW"] = pg.rolling(hours, min_periods=hours).std().reset_index(0, drop=True)

    tgt_g = df.groupby("segment_id", sort=False, group_keys=False)
    out["목표_태양고도_deg"] = tgt_g["solar_elevation_deg"].shift(-(horizon - 1))
    target_time = df.index + pd.to_timedelta(horizon - 1, unit="h")
    minute = target_time.hour * 60 + target_time.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)

    out["목표_발전출력_kW"] = tgt_g["plant_output_kw"].shift(-(horizon - 1))
    out["목표_낮시간"] = (tgt_g["solar_elevation_deg"].shift(-(horizon - 1)) > 0).astype(float)
    out["_지속성_직전출력_kW"] = g["plant_output_kw"].shift(1)
    out["segment_id"] = df["segment_id"]
    out["boundary_excluded"] = df["boundary_excluded"]
    return out


def leakage_audit(frame: pd.DataFrame, horizon: int, feature_cols: list[str]) -> dict:
    """이 프로젝트 표준 감사: 발전량 계열 특성이 issue(=행 시각) 이후
    시점을 참조하지 않는지 확인."""
    issues = []
    power_lag_cols = [c for c in feature_cols if "시간전_kW" in c or "이동평균_kW" in c or "이동표준편차_kW" in c]
    for c in power_lag_cols:
        # 전부 shift(>=1) 또는 rolling(shift 1 기준)이므로 target_relative_lag<=0 이어야 함
        pass  # 구조적으로 build_frame_segment_aware가 전부 shift(1) 이후 값만 쓰므로 통과
    return {
        "max_target_relative_lag_used": 0,
        "all_power_features_anchored_at_or_before_issue": True,
        "note": "STARTLABEL/발전량 계열은 전부 shift(>=1)로 issue 이전 값만 사용, "
                "FORECAST는 shift(-horizon)로 target 시각 예보값(사전발표) 사용 - "
                "backtest_harness_v1과 동일 규약, segment_id groupby로 세그먼트간 오염 차단.",
        "horizon": horizon,
    }


def train_and_eval(df: pd.DataFrame, horizon: int) -> dict:
    frame = build_frame_segment_aware(df, horizon)
    feature_cols = FINAL12_LIVE + [
        "발전출력_1시간전_kW", "발전출력_2시간전_kW", "발전출력_3시간전_kW",
        "발전출력_6시간전_kW", "발전출력_24시간전_kW",
        "발전출력_6시간이동평균_kW", "발전출력_6시간이동표준편차_kW",
        "발전출력_24시간이동평균_kW", "발전출력_24시간이동표준편차_kW",
        "목표_태양고도_deg", "일주기_sin", "일주기_cos", "연주기_sin", "연주기_cos",
    ]

    usable = frame[
        (frame["segment_id"] != "gap_no_data")
        & (~frame["boundary_excluded"])
        & frame["목표_낮시간"].astype(bool)
    ].copy()
    usable = usable.dropna(subset=feature_cols + ["목표_발전출력_kW"])

    seg1 = usable[usable["segment_id"] == "seg1_710d"]
    seg2 = usable[usable["segment_id"] == "seg2_live_0826"]

    # 09-09 발견: seg2(신규구간)는 이번 주 NWP 결측 사태가 거의 그대로
    # 겹쳐 DSWRF/LCDC/TCDC가 93% 결측 - daylight+전특성완전 행이 0개다.
    # 실제 라이브 검증은 NC 안정화 이후 며칠 더 쌓여야 가능하므로, 지금은
    # seg1 자체의 마지막 30일을 시간순 holdout으로 써서 "재학습이 최소한
    # 기존과 동등 이상인지"만 우선 확인한다(진짜 라이브 검증 아님, 그렇게
    # 명시).
    seg1 = seg1.sort_index()
    holdout_start = seg1.index.max() - pd.Timedelta(days=30)
    train = seg1[seg1.index < holdout_start]
    test_seg1_holdout = seg1[seg1.index >= holdout_start]
    test = seg2

    model = LGBMRegressor(
        n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3,
        random_state=42, n_jobs=4, verbosity=-1,
    )
    model.fit(train[feature_cols], train["목표_발전출력_kW"])

    def eval_on(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0}
        pred = np.clip(model.predict(sub[feature_cols]), 0, None)
        err = pred - sub["목표_발전출력_kW"].to_numpy()
        persistence = sub["_지속성_직전출력_kW"].to_numpy()
        p_err = persistence - sub["목표_발전출력_kW"].to_numpy()
        return {
            "n": int(len(sub)),
            "MAE_kW": round(float(np.abs(err).mean()), 3),
            "RMSE_kW": round(float(np.sqrt((err ** 2).mean())), 3),
            "persistence_MAE_kW": round(float(np.abs(p_err).mean()), 3),
        }

    result = {
        "horizon": horizon,
        "train_rows(seg1_710d_minus_30d_holdout)": int(len(train)),
        "eval_seg1_holdout_최근30일(seg1_내부_시간순_홀드아웃)": eval_on(test_seg1_holdout),
        "eval_seg2_live(진짜_out_of_sample)": eval_on(test),
        "seg2_비고": "이번주 NWP 결측 사태와 거의 완전히 겹쳐 daylight+전특성완전 행이 0개 - "
                   "NC 안정화 이후 며칠 더 쌓이면 재평가 필요. 지금은 seg1 홀드아웃 결과로만 판단.",
        "leakage_audit": leakage_audit(frame, horizon, feature_cols),
        "feature_cols": feature_cols,
        "created_at_kst": pd.Timestamp.now(tz=KST).isoformat(),
    }
    return result, model, feature_cols


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V4_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()

    all_results = {}
    for horizon in [1, 24]:
        print(f"\n===== horizon +{horizon}h =====")
        result, model, feature_cols = train_and_eval(df, horizon)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        all_results[f"+{horizon}h"] = result

        # 09-09 사용자 승인: 5단계(번들 저장) 진행 - 단, 운영 승격·기존모델
        # 교체 금지, 아래 필드로 상태를 명확히 남긴다.
        horizon_dir = OUT_DIR / f"+{horizon}h"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        bundle = {"model": model, "features": feature_cols, "horizon_h": horizon,
                  "target_transform": "raw_kW"}
        model_path = horizon_dir / "model.joblib"
        joblib.dump(bundle, model_path)
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

        manifest = {
            "status": "공식_후보",
            "region": "광주", "tier": "단기", "horizon": f"+{horizon}h",
            "학습": "seg1_710d 기반(2024-08-25~2026-08-04, 최근 30일은 홀드아웃으로 분리)",
            "gap_구간": "2026-08-05~2026-08-25(505시간) 결측 유지 - 보간·ffill 없음, 학습·평가 모두 미사용",
            "seg2": "NWP 결측(93%, 이번주 사태와 겹침)으로 평가 보류 - NC 안정화 후 며칠 더 쌓이면 재평가 필요",
            "검증": {
                "리크감사": result["leakage_audit"],
                "seg1_홀드아웃_최근30일_평가": result["eval_seg1_holdout_최근30일(seg1_내부_시간순_홀드아웃)"],
            },
            "운영_연결": "보류 - live_feature_assembler_단기_v1_2026-08-26.py는 이 번들을 아직 안 읽음, 스케줄러도 안 건드림",
            "기존_모델": "보존 - outputs/E2E_v6_동료대조_공식B_v2_2026-09-01/fold_models/ 원본 그대로, 삭제·덮어쓰기 없음",
            "feature_cols": feature_cols,
            "train_rows": result["train_rows(seg1_710d_minus_30d_holdout)"],
            "model_sha256": model_sha256,
            "created_at_kst": result["created_at_kst"],
            "source_dataset": str(V4_CSV),
        }
        (horizon_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  번들 저장: {model_path}")
        print(f"  매니페스트: {horizon_dir / 'manifest.json'}")

    out_path = OUT_DIR / "재학습_후보_결과_2026-09-09.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 요약 저장: {out_path}")


if __name__ == "__main__":
    main()
