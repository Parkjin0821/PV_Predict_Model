# -*- coding: utf-8 -*-
"""김제 요인재검증 2단계(다중공선성+모델선정) 러너 - 자정 예약용.
부안 러너(factor_phase2_run_buan_2026-09-03.py)와 동일 원리, 경로만 교체.
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
OUT_DIR = HERE / "outputs" / "요인재검증_v6_phase2_2026-09-03"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30

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


def main() -> None:
    if not PHASE1_SUMMARY.is_file():
        raise FileNotFoundError(f"1단계 산출물 없음(먼저 실행 필요): {PHASE1_SUMMARY}")
    phase1 = json.loads(PHASE1_SUMMARY.read_text(encoding="utf-8"))
    baseline_cols = phase1["선택_baseline"]
    candidate_cols = phase1["선택_후보"]
    print(f"[김제] 1단계 채택값 로드: baseline {len(baseline_cols)}개, 후보 {len(candidate_cols)}개")

    phase1_mod = _load_module("gimje_phase1_20260903", PHASE1_SCRIPT)
    phase2_mod = _load_module("phase2_common_gimje_20260903", PHASE2_MODULE)
    v1 = _load_module("gimje_v1_for_folds", phase1_mod.V1_SCRIPT)

    frame, _ = phase1_mod.build_hourly_frame()
    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    folds = v1.expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    def pooled_score(y_true, pred):
        err = y_true - pred
        return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
                "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}

    phase2_mod.run_phase2(
        frame=frame, folds=folds, target_col=TARGET,
        baseline_cols=baseline_cols, candidate_cols=candidate_cols,
        native_missing_ok=phase1_mod.NATIVE_MISSING_OK, pooled_score_fn=pooled_score,
        seed=SEED, out_dir=OUT_DIR, region_label="김제",
    )


if __name__ == "__main__":
    main()
