# -*- coding: utf-8 -*-
"""★golden replay — 단기 +24h, 운영번들 v2 + predict_kw() 전체경로 검증★

Codex 완료기준 체크리스트(08-25) 중 **라이브 API 없이 지금 검증 가능한
항목들**을 이 스크립트로 확인한다:
- 학습 당시와 (재구성한) 실시간 특성 값·열 순서 일치
- 저장 모델과 재적재 모델 예측 차이 0
- 219kW 프로필 적용 및 출력범위 준수
- (참고) 발행시각 이후 자료 미사용은 `harness.build_frame`이 이미
  보장하는 성질을 그대로 재사용하는 것으로 확인(아래 "범위" 참고)

## 방법
`load_short_frame(24)`가 만드는 것과 **재료는 완전히 같지만 조립 순서를
"실시간처럼" 다시 밟는다** — 원본 hourly CSV/parquet을 읽어 `harness.
build_frame()`을 그대로 재사용해 특성 프레임을 새로 만들고, 실제 과거
발행시각 하나를 골라 그 행이 (a) 공식 `load_short_frame(24)`의 같은
행과 특성값이 완전히 같은지, (b) `production_inference_utils.predict_kw()`
로 뽑은 예측이 운영번들 v2로 직접 `.predict()`한 것과 같은지 확인한다.

## 이 스크립트가 증명하는 것 / 증명하지 않는 것
- **증명함**: "원자료(hourly 형태)만 올바르게 조립되면, build_frame()
  재사용 + predict_kw() 경로가 공식 파이프라인·저장모델과 완전히
  일치한다." → 3~4단계(특성조립기·예측실행기)의 **로직**이 옳다는 증거.
- **증명하지 않음**: 그 원자료 자체를 라이브 API로 정확히 수집하는 것
  (1~2단계, Codex 담당) — 이건 여기서 다루지 않는다. 원자료 소스가
  바뀌어도(과거 CSV → 라이브 API) 이 스크립트가 보여주는 조립 로직은
  그대로 재사용 가능하다.

## 산출물
`outputs/golden_replay_단기h24_v1_2026-08-25/판정.json`
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "golden_replay_단기h24_v1_2026-08-25"
HORIZON = 24
BUNDLE_PATH = ROOT / "outputs" / "운영모델_v2_2026-08-25" / "운영모델_단기_h24.joblib"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpc = _load("gr_dpc", "defect_policy_comparison_v1_2026-08-21.py")
harness = dpc.harness
infer = _load("gr_infer", "production_inference_utils_v1_2026-08-25.py")


def build_hourly_from_raw() -> pd.DataFrame:
    """load_short_frame(24)와 완전히 같은 원자료 조립 — 실시간 조립기가
    라이브에서 해야 할 일과 동일한 절차(소스만 과거파일)."""
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(dpc.V5_DIR / "집계_1시간_자료_v5.parquet")
    common = hourly.index.intersection(v5_1h.index)
    hourly = hourly.loc[common].copy()
    hourly["plant_output_kw"] = v5_1h.loc[common, "발전출력_kW"]
    hourly["inverters_available"] = v5_1h.loc[common, "가용인버터수"]
    return hourly


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE_PATH)
    audit = infer.audit_bundle(bundle)
    if not audit["전체통과"]:
        raise RuntimeError(f"번들 자체가 불완전함: {audit}")

    # 1) 재조립 경로: 원자료 → build_frame() 재사용
    hourly = build_hourly_from_raw()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    rebuilt = harness.build_frame(hourly, HORIZON, candidate_cols)
    # load_short_frame()이 build_frame() 뒤에 추가로 붙이는 후처리 —
    # 여기서도 똑같이(재조립 경로가 "원자료→최종 특성"까지 전 과정을
    # 빠짐없이 재현해야 골든리플레이 의미가 있음).
    DIFSWRF_COL = dpc.DIFSWRF_COL
    if DIFSWRF_COL in rebuilt.columns:
        rebuilt[f"{DIFSWRF_COL}_결측여부"] = rebuilt[DIFSWRF_COL].isna().astype(float)
    if bundle.get("클러스터"):
        raise RuntimeError("단기 h24는 클러스터가 없어야 하는데 번들에 있음 — 잘못된 파일")

    # 2) 공식 경로: load_short_frame(24) — 100% 기존 함수, 재구현 아님
    official = dpc.load_short_frame(HORIZON)

    # 발행시각 하나 고르기: 학습기간 내부, daylight, 완전가용, 두 프레임 모두 존재하는 시각
    daylight_official = official[(official["목표_낮시간"] > 0)
                                 & (official["_목표_가용인버터수"] >= 5)]
    common_idx = daylight_official.index.intersection(rebuilt.index)
    train_end = pd.Timestamp(bundle["학습기간_끝"])
    candidates = [t for t in common_idx if t <= train_end]
    if not candidates:
        raise RuntimeError("재현할 발행시각 후보가 없음")
    issue_time = sorted(candidates)[-1]  # 학습기간 안에서 가장 최근 시각

    feat_cols = bundle["features"]
    row_rebuilt = rebuilt.loc[[issue_time], feat_cols]
    row_official = official.loc[[issue_time], feat_cols]

    # (a) 특성값·열순서 일치
    diffs = {}
    all_match = True
    for c in feat_cols:
        a, b = row_rebuilt[c].iloc[0], row_official[c].iloc[0]
        same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(float(a) - float(b)) < 1e-9)
        if not same:
            all_match = False
            diffs[c] = {"재조립": None if pd.isna(a) else float(a), "공식": None if pd.isna(b) else float(b)}

    # (b) predict_kw(재조립 특성) vs 공식 파이프라인 저장모델 직접예측
    # 단기 h24는 타깃유형="raw"라 predict_kw가 태양고도 컬럼을 안 쓰지만,
    # 다른 티어(카파모델) 재사용을 감안해 항상 넣어둔다(rebuilt에 이미 있음).
    live_row = row_rebuilt.copy()
    live_row["목표_태양고도_deg"] = rebuilt.loc[[issue_time], "목표_태양고도_deg"].to_numpy()
    pred_via_replay = infer.predict_kw(bundle, live_row)
    direct = float(np.clip(bundle["model"].predict(row_official[feat_cols])[0], 0, bundle["capacity_kw"]))
    pred_diff = float(abs(pred_via_replay[0] - direct))

    # (c) 219kW 프로필·출력범위
    in_range = bool(0.0 <= pred_via_replay[0] <= bundle["capacity_kw"])

    result = {
        "발행시각": str(issue_time), "티어": "단기", "수평_h": HORIZON,
        "번들audit": audit,
        "특성값_전부일치": all_match, "불일치특성": diffs,
        "재조립경로_예측_kW": round(float(pred_via_replay[0]), 4),
        "공식경로_예측_kW": round(direct, 4),
        "예측차이": pred_diff,
        "219kW_출력범위_준수": in_range,
        "전체판정": bool(all_match and pred_diff < 1e-6 and in_range),
    }
    (OUT / "판정.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n=== golden replay 최종판정: {'통과' if result['전체판정'] else '실패'} ===")


if __name__ == "__main__":
    main()
