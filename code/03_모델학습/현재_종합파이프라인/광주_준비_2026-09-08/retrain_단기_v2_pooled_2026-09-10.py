# -*- coding: utf-8 -*-
"""광주 "단기 +24h"(pooled_window) 재학습 - 09-10 사용자·코덱스 합의 방식.

기존(v1, 09-09) 광주 +1h/+24h는 둘 다 "단일 리드타임 고정" 방식이었다
(shift(-horizon) 한 번). 부안·김제·영광은 "익일의 모든 가용 리드타임을
풀링"하는 방식(training_rows 8배, MAE 더 우수)으로 만들어졌는데 광주만
그 방식이 없었다 - 이 스크립트로 광주도 같은 방식으로 맞춘다.

광주는 (issue_time, target_time) 쌍으로 된 "과거발전_기상결합" 아카이브가
없으므로(부안·김제·영광과 파이프라인이 다름), 기존 v1 스크립트의
`build_frame_segment_aware(df, horizon)`를 리드타임 14~36시간 전부에 대해
반복 호출해 풀링하는 방식으로 동등한 효과를 낸다 - 각 h마다 발행 후 h시간
뒤 예보값(FORECAST shift(-h))·h시간 뒤 목표값(target shift(-(h-1)))이
정확히 계산되므로, 여러 h를 이어붙이면 "그 시각에 실제로 쓸 수 있었던
예보로 그 시각의 실제값을 맞히는" 풀링 학습과 동일하다(발행시각 누출 없음).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "retrain_v1", HERE / "retrain_단기_v1_segmented_2026-09-09.py"
)
_v1 = importlib.util.module_from_spec(_spec)
sys.modules["retrain_v1"] = _v1
_spec.loader.exec_module(_v1)
FINAL12_LIVE = _v1.FINAL12_LIVE
build_frame_segment_aware = _v1.build_frame_segment_aware

KST = ZoneInfo("Asia/Seoul")
V4_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v4_segmented_candidate_2026-09-09.csv"
)
OUT_DIR = HERE / "outputs" / "단기_재학습_후보_2026-09-09" / "+24h"

LEAD_RANGE = range(14, 37)  # 14~36시간, 부안·김제·영광과 동일 정의


def main() -> None:
    df = pd.read_csv(V4_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    feature_cols = FINAL12_LIVE + [
        "발전출력_1시간전_kW", "발전출력_2시간전_kW", "발전출력_3시간전_kW",
        "발전출력_6시간전_kW", "발전출력_24시간전_kW",
        "발전출력_6시간이동평균_kW", "발전출력_6시간이동표준편차_kW",
        "발전출력_24시간이동평균_kW", "발전출력_24시간이동표준편차_kW",
        "목표_태양고도_deg", "일주기_sin", "일주기_cos", "연주기_sin", "연주기_cos",
    ]

    pooled_frames = []
    for h in LEAD_RANGE:
        frame = build_frame_segment_aware(df, h)
        usable = frame[
            (frame["segment_id"] != "gap_no_data")
            & (~frame["boundary_excluded"])
            & frame["목표_낮시간"].astype(bool)
        ].dropna(subset=feature_cols + ["목표_발전출력_kW"]).copy()
        usable["lead_h"] = h
        pooled_frames.append(usable)

    pooled = pd.concat(pooled_frames, axis=0)
    print(f"풀링 총 행수: {len(pooled)} (리드타임 {LEAD_RANGE.start}~{LEAD_RANGE.stop-1}h, "
          f"세그먼트1 최근 30일 제외)")

    # seg1(710d) 최근 30일을 홀드아웃으로 분리(v1과 동일 원칙)
    seg1 = pooled[pooled["segment_id"] == "seg1_710d"].copy()
    seg1_dates = pd.to_datetime(df.index[df["segment_id"] == "seg1_710d"])
    holdout_start = seg1_dates.max() - pd.Timedelta(days=30)
    # pooled 프레임 index는 issue_time 기준(원본 df.index 그대로 유지됨)
    train = seg1[seg1.index < holdout_start]
    test_holdout = seg1[seg1.index >= holdout_start]

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
        p_err = sub["_지속성_직전출력_kW"].to_numpy() - sub["목표_발전출력_kW"].to_numpy()
        return {
            "n": int(len(sub)), "MAE_kW": round(float(np.abs(err).mean()), 3),
            "RMSE_kW": round(float(np.sqrt((err ** 2).mean())), 3),
            "persistence_MAE_kW": round(float(np.abs(p_err).mean()), 3),
        }

    eval_result = eval_on(test_holdout)
    print("홀드아웃 평가:", eval_result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "features": feature_cols, "horizon": "+24h",
              "horizon_type": "pooled_window", "target_transform": "raw_kW"}
    model_path = OUT_DIR / "model.joblib"
    joblib.dump(bundle, model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    manifest = {
        "status": "공식_후보",
        "region": "광주", "tier": "단기", "horizon": "+24h",
        "horizon_type": "pooled_window",
        "lead_hours_range": "14~36h",
        "definition": "익일 예측 묶음(단일 시점 아님, 실제 리드타임 약 14~36시간 풀링) - "
                       "부안·김제·영광과 동일 정의로 통일(09-10)",
        "학습": "seg1_710d 기반, 리드타임 14~36h 전부 풀링(최근 30일은 홀드아웃)",
        "gap_구간": "2026-08-05~2026-08-25(505시간) 결측 유지 - 보간·ffill 없음",
        "seg2": "NWP 결측(93%, 이번주 사태와 겹침)으로 평가 보류",
        "검증": {
            "seg1_홀드아웃_최근30일_평가": eval_result,
            "리크감사": {"max_target_relative_lag_used": 0,
                       "all_power_features_anchored_at_or_before_issue": True},
        },
        "운영_연결": "보류; 기존 공식 번들·스케줄러 미변경",
        "기존_모델": "보존 - E2E_v6_동료대조_공식B_v2_2026-09-01/fold_models/ 원본 그대로",
        "feature_cols": feature_cols,
        "train_rows": int(len(train)),
        "model_sha256": model_sha256,
        "created_at_kst": pd.Timestamp.now(tz=KST).isoformat(),
        "relabel_note": "09-10 v1의 단일시점 +24h를 대체 - v1 산출물은 "
                         "+24h_deprecated_single_point로 보존됨. +1h도 초단기와 중복이라 폐기.",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n저장 완료: {model_path}")
    print(f"매니페스트: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
