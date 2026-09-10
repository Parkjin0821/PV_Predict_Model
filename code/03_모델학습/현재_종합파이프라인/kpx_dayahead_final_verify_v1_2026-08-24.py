# -*- coding: utf-8 -*-
"""KPX day-ahead 최적조합(카파+날씨군집화) 최종 검증 — 재현성 + 시드 민감도.

## 왜 한 번 더 검증하나
이 조합을 공식 반영하면 **협약 연구개발계획서 성과지표(재생발전 예측
오차율, 4차년도 목표 7%)를 산출하는 공식 모델 자체가 바뀐다.** 공인
시험기관 확인 대상이므로 반영 전에 두 가지를 확인한다:
1. **재현성**: 같은 시드로 다시 돌렸을 때 기록된 값(7.360%)이 그대로
   나오는가(결정론적 재현 확인).
2. **시드 민감도**: 시드를 바꿔도 개선이 유지되는가. 이게 중요한 이유는
   `choose_structure_kfold`(구조선택)·`add_weather_clusters`(KMeans)·
   모델 학습이 전부 시드에 의존하기 때문이다. **한 시드에서만 나오는
   우연이면 공식 반영하면 안 된다.**

## 판정(사전동결)
- 재현성: 기존 기록값과 0.001%p 이내 일치
- 시드 민감도: **시험한 모든 시드에서 개선율 ≥1% AND 계절악화 <5%**
  (즉 원래 채택규칙을 시드마다 전부 통과해야 함). 하나라도 실패하면
  공식 반영 보류.

## 산출물 (`outputs/KPX_dayahead_최종검증_2026-08-24/`)
`시드민감도.csv`, `판정.json`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "KPX_dayahead_최종검증_2026-08-24"
CAPACITY_KW = 219.0
RECORDED_NMAE = 7.360          # 앞선 검증에서 기록된 카파+군집 NMAE
RECORDED_TOLERANCE = 0.001
EXTRA_SEEDS = [1, 2026, 7]     # 공식시드(42) 외 추가 시드


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dr = _load("fv_dr", "kpx_dayahead_candidate_reverify_v1_2026-08-24.py")
mc = _load("fv_mc", "kpx_dayahead_more_candidates_v1_2026-08-24.py")
improvement, harness, N_INVERTERS, WINDOWS = dr.improvement, dr.harness, dr.N_INVERTERS, dr.WINDOWS
clearsky = mc.clearsky
nmae = dr.nmae


def run_one_seed(seed: int, frame: pd.DataFrame, candidate_cols, base_cols) -> pd.DataFrame:
    """배포판(raw+구조선택) vs 후보(카파+날씨군집화)를 한 시드로 비교."""
    daylight = frame[frame["목표_낮시간"] > 0]
    rows = []
    for fold, s, e in WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        all_target = daylight[(daylight.index >= start) & (daylight.index < end)]
        full_mask_all = daylight["_목표_가용인버터수"] >= N_INVERTERS
        train_all = daylight[(daylight.index < start) & full_mask_all]
        test_all = all_target[full_mask_all.loc[all_target.index]] if len(all_target) else all_target
        if len(train_all) < 200 or len(test_all) < 30:
            continue
        chosen = harness.select_features_in_fold(
            train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = ([c for c in features if c not in harness.NATIVE_MISSING_OK]
                    + ["목표_발전출력_kW", "_청천_kW", "_카파"])
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=[c for c in features if c not in harness.NATIVE_MISSING_OK]
                               + ["목표_발전출력_kW", "_청천_kW"])
        if len(train) < 200 or len(test) < 30:
            continue

        # 배포판: raw + 구조선택
        params0, _ = improvement.choose_structure_kfold(
            "단기", train, features, "목표_발전출력_kW", "raw", CAPACITY_KW, seed)
        base_model = improvement.make_model("단기", seed, params0)
        base_model.fit(train[features], train["목표_발전출력_kW"])
        p_base = np.clip(base_model.predict(test[features]), 0, CAPACITY_KW)

        # 후보: 카파 타깃 + 구조선택 + 날씨군집화 특성
        params_k, _ = improvement.choose_structure_kfold(
            "단기", train, features, "_카파", "kappa", CAPACITY_KW, seed)
        tr_cl, te_cl, feat_cl, _ = improvement.add_weather_clusters(train, test, features, seed)
        model_kc = improvement.make_model("단기", seed, params_k)
        model_kc.fit(tr_cl[feat_cl], tr_cl["_카파"])
        p_cand = np.clip(model_kc.predict(te_cl[feat_cl]) * test["_청천_kW"].to_numpy(), 0, CAPACITY_KW)

        for yy, pb, pc in zip(test["목표_발전출력_kW"].to_numpy(), p_base, p_cand):
            rows.append({"폴드": fold, "실제": yy, "배포판": pb, "후보예측": pc})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    official_seed = int(config["random_seed"])

    frame, candidate_cols = dr.build_frame_with_obs(CAPACITY_KW)
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간") and not c.startswith("__target_obs__")]
    frame["_청천_kW"] = np.clip(
        CAPACITY_KW * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]

    print("=== [1/2] 재현성 검증(공식시드 %d) ===" % official_seed)
    rows = run_one_seed(official_seed, frame, candidate_cols, base_cols)
    v = dr.verdict_from_rows(rows)
    diff = abs(v["NMAE_후보"] - RECORDED_NMAE)
    reproduced = diff <= RECORDED_TOLERANCE
    print(f"  기록값 {RECORDED_NMAE}% / 이번값 {v['NMAE_후보']}% / 차이 {diff:.4f}%p → "
          f"{'재현 성공' if reproduced else '★재현 실패★'}")
    print(f"  {v}")

    print("\n=== [2/2] 시드 민감도 검증 ===")
    results = [{"시드": official_seed, "구분": "공식시드", **v}]
    for seed in EXTRA_SEEDS:
        r = run_one_seed(seed, frame, candidate_cols, base_cols)
        vv = dr.verdict_from_rows(r)
        results.append({"시드": seed, "구분": "추가시드", **vv})
        print(f"  시드 {seed}: NMAE {vv['NMAE_배포판']}% → {vv['NMAE_후보']}% "
              f"(개선 {vv['개선율_pct']}%, 최대계절악화 {vv['최대계절악화_pct']}%) "
              f"채택={vv['채택']}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "시드민감도.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 시드별 결과 ===")
    print(df[["시드", "구분", "NMAE_배포판", "NMAE_후보", "개선율_pct", "최대계절악화_pct", "채택"]].to_string(index=False))

    all_adopt = bool(df["채택"].all())
    min_gain = float(df["개선율_pct"].min())
    max_worse = float(df["최대계절악화_pct"].max())
    verdict = reproduced and all_adopt

    print(f"\n재현성: {'통과' if reproduced else '실패'}")
    print(f"시드 민감도: 전 시드 채택={all_adopt} (최소 개선율 {min_gain:.2f}%, 최대 계절악화 {max_worse:.2f}%)")
    print(f"\n★최종 판정: {'공식 반영 가능' if verdict else '공식 반영 보류'}★")

    report = {
        "대상": "KPX day-ahead 최적조합(청천지수 카파 변환 + 날씨군집화)",
        "재현성": {"기록값_pct": RECORDED_NMAE, "재실행값_pct": v["NMAE_후보"],
                 "차이_pp": round(diff, 4), "통과": reproduced},
        "시드민감도": {"시험시드": [official_seed] + EXTRA_SEEDS,
                   "전시드_채택": all_adopt, "최소개선율_pct": round(min_gain, 2),
                   "최대계절악화_pct": round(max_worse, 2)},
        "판정규칙": "재현성 0.001%p 이내 AND 모든 시드에서 개선율≥1%·계절악화<5%",
        "최종판정": "공식 반영 가능" if verdict else "공식 반영 보류",
    }
    (OUT / "판정.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
