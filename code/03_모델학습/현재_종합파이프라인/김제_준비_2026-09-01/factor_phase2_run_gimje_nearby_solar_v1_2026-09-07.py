# -*- coding: utf-8 -*-
"""김제 요인재검증 2단계 재실행 - 전주146·정읍245 인근 실측일사량을 후보에
추가해 MAE가 실제로 개선되는지 검증(09-07).

## 배경
09-03 공식 phase2(XGBoost MAE 41.92kW)는 김제 자체 ASOS(부안243 재사용)에
일사계가 없어 일사량 계열 입력이 전혀 없는 채로 나온 결과였다. 09-01에
가장 가까운 실측 일사량 보유 지점(전주146·정읍245)과의 상관을 탐색했으나
그때는 인근관측소 백필이 46%/45%뿐이라 표본이 1,133행에서 정체돼 있었다.
09-07 백필이 710/710일(100%) 완료된 뒤 재검증(`check_correlation_
nearby_solar_gimje_v1_2026-09-01.py`)한 결과, 21개 폴드 전부에서 상관
0.80~0.84로 안정적이고 기존 대안(solar_elevation 0.69, DSWRF 0.52)보다
뚜렷이 강함을 확인했다 - 그 스크립트 자체가 명시한 다음 단계("모델
MAE 비교로 넘어갈 것")를 지금 수행한다.

## 재사용한 것(재구현 안 함)
- `factor_phase2_run_gimje_2026-09-03.py`의 phase1 로드·frame 생성·
  fold 생성 로직 그대로.
- `factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py`의
  `run_phase2()`(VIF+쌍상관 가지치기, LightGBM/XGBoost/선형회귀 3모델
  비교) 그대로 - 새 후보 2개도 기존 후보들과 동일한 기준으로 검토되게
  함(특혜 없음).
- `check_correlation_nearby_solar_gimje_v1_2026-09-01.py`의 인근일사량
  로드 방식(같은 시각 관측 join, 시차 문제 없음) 그대로.

## 산출물 분리 이유
09-03 공식 phase2_요약.json은 그대로 두고, 이 결과는 별도 폴더
(`요인재검증_v6_phase2_인근일사량추가_2026-09-07`)에 저장한다 - 개선이
확인되면 그때 공식 채택 여부를 사용자와 논의(임의로 공식본을 덮어쓰지
않음).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
PHASE1_SCRIPT = HERE / "factor_reverify_v6_hourly_gimje_2026-09-03.py"
PHASE1_SUMMARY = HERE / "outputs" / "요인재검증_v6_hourly_2026-09-03" / "요약.json"
PHASE2_MODULE = PIPELINE_ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py"
OUT_DIR = HERE / "outputs" / "요인재검증_v6_phase2_인근일사량추가_2026-09-07"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30

NEARBY = {
    "전주146_인근일사량_W_m2": Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
        r"\전주146\기상청_ASOS146_시간환경_20240825_20260804.csv"
    ),
    "정읍245_인근일사량_W_m2": Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
        r"\정읍245\기상청_ASOS245_시간환경_20240825_20260804.csv"
    ),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_nearby_solar(path: Path, colname: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["시각"]).dt.tz_localize(None)
    out = pd.DataFrame({"target_time_kst": ts, colname: df["일사량_W_m2"].to_numpy()})
    return out.drop_duplicates("target_time_kst")


def main() -> None:
    if not PHASE1_SUMMARY.is_file():
        raise FileNotFoundError(f"1단계 산출물 없음(먼저 실행 필요): {PHASE1_SUMMARY}")
    phase1 = json.loads(PHASE1_SUMMARY.read_text(encoding="utf-8"))
    baseline_cols = phase1["선택_baseline"]
    candidate_cols = list(phase1["선택_후보"])
    print(f"[김제] 1단계 채택값 로드: baseline {len(baseline_cols)}개, 기존후보 {len(candidate_cols)}개")

    phase1_mod = _load_module("gimje_phase1_20260907", PHASE1_SCRIPT)
    phase2_mod = _load_module("phase2_common_gimje_20260907", PHASE2_MODULE)
    v1 = _load_module("gimje_v1_for_folds_20260907", phase1_mod.V1_SCRIPT)

    frame, _ = phase1_mod.build_hourly_frame()
    before_rows = len(frame)

    new_cols = []
    for colname, path in NEARBY.items():
        if not path.is_file():
            print(f"[대기] {colname} 파일 없음 - 건너뜀: {path}")
            continue
        solar = load_nearby_solar(path, colname)
        frame = frame.merge(solar, on="target_time_kst", how="left")
        new_cols.append(colname)
        print(f"[김제] {colname} 병합 완료(결측 {frame[colname].isna().sum()}/{len(frame)})")

    if not new_cols:
        raise SystemExit("인근 일사량 컬럼을 하나도 못 붙였습니다 - 파일 경로 확인 필요.")
    assert len(frame) == before_rows, "병합 후 행수가 바뀜 - many-to-one 위반 의심(중복 시각)"

    candidate_cols_augmented = candidate_cols + new_cols

    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    folds = v1.expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    def pooled_score(y_true, pred):
        err = y_true - pred
        return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
                "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}

    result = phase2_mod.run_phase2(
        frame=frame, folds=folds, target_col=TARGET,
        baseline_cols=baseline_cols, candidate_cols=candidate_cols_augmented,
        native_missing_ok=phase1_mod.NATIVE_MISSING_OK, pooled_score_fn=pooled_score,
        seed=SEED, out_dir=OUT_DIR, region_label="김제(인근일사량 추가)",
    )

    # 09-03 공식 phase2와 나란히 비교할 수 있게 참고용으로 남김.
    official_path = HERE / "outputs" / "요인재검증_v6_phase2_2026-09-03" / "phase2_요약.json"
    if official_path.is_file():
        official = json.loads(official_path.read_text(encoding="utf-8"))
        official_best = min(
            (m for m in official["모델비교"] if m["MAE_kW"] is not None),
            key=lambda m: m["MAE_kW"],
        )
        new_best = min(
            (m for m in result["모델비교"] if m["MAE_kW"] is not None),
            key=lambda m: m["MAE_kW"],
        )
        print(f"\n=== 비교: 09-03 공식(인근일사량 없음) vs 09-07(인근일사량 추가) ===")
        print(f"09-03 공식: {official_best['모델']} MAE {official_best['MAE_kW']}kW")
        print(f"09-07 신규: {new_best['모델']} MAE {new_best['MAE_kW']}kW")
        delta = official_best["MAE_kW"] - new_best["MAE_kW"]
        pct = delta / official_best["MAE_kW"] * 100
        print(f"차이: {delta:+.2f}kW ({pct:+.1f}%, 양수면 개선)")


if __name__ == "__main__":
    main()
