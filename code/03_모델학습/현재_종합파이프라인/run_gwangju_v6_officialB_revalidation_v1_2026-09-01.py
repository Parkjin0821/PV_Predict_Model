# -*- coding: utf-8 -*-
"""Run the historical official-B v1/v2 E2E route against isolated v6 data.

This wrapper changes only data/output directories in the existing validated
modules.  Production bundles and all v5 output directories are read-only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
ROOT = PROJECT / "03_모델학습" / "현재_종합파이프라인"
V6_DATA = ROOT / "outputs" / "v6_동료대조_후보_2026-09-01"
V6_V1 = ROOT / "outputs" / "E2E_v6_동료대조_공식B_v1_2026-09-01"
V6_V2 = ROOT / "outputs" / "E2E_v6_동료대조_공식B_v2_2026-09-01"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_empty_or_absent(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to mix with existing candidate output: {path}")


def main() -> None:
    required = [
        V6_DATA / "집계_15분_자료_v5.parquet",
        V6_DATA / "집계_1시간_자료_v5.parquet",
        V6_DATA / "집계_일간_실제발전량_v5.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"v6 candidate inputs missing: {missing}")
    require_empty_or_absent(V6_V1)
    require_empty_or_absent(V6_V2)

    v1 = load("gwangju_v6_e2e_v1", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
    v1.V5_DIR = V6_DATA
    v1.dpc.V5_DIR = V6_DATA
    v1.OUT = V6_V1
    v1.MODEL_DIR = V6_V1 / "fold_models"
    print("=== v6 official-B v1 full 8-horizon rerun ===")
    v1.main()

    v2 = load("gwangju_v6_e2e_v2", "e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py")
    v2.SRC_V1 = V6_V1
    v2.OUT = V6_V2
    v2.MODEL_DIR = V6_V2 / "fold_models"
    v2.e2e.V5_DIR = V6_DATA
    v2.e2e.dpc.V5_DIR = V6_DATA
    v2.e2e.OUT = V6_V1
    v2.e2e.MODEL_DIR = V6_V1 / "fold_models"
    print("=== v6 official-B v2 (+2h/+4h selected structure) rerun ===")
    v2.main()


if __name__ == "__main__":
    main()
