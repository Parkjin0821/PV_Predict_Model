"""Collect the missing 60 Gwangju grid-forecast issue days without altering originals.

This wrapper reuses the authenticated collector already maintained under the
UCUBE project, but redirects recovery and merged outputs into a new versioned
delivery directory.  The existing 650-day combined CSV is treated as the
immutable base dataset.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE_COLLECTOR = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\grid_forecast_resume_20260606.py"
)
BASE_650_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\기상청_과거단기예보\기상청_과거단기예보_기온하늘습도_650일.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\grid_forecast_710d_collection_v3_2026-08-19"
)


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "ucube_grid_forecast_resume_20260606", SOURCE_COLLECTOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"수집 스크립트를 불러올 수 없습니다: {SOURCE_COLLECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not SOURCE_COLLECTOR.is_file():
        raise FileNotFoundError(SOURCE_COLLECTOR)
    if not BASE_650_CSV.is_file():
        raise FileNotFoundError(BASE_650_CSV)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collector = load_collector()

    # The original collector expects two historical fragments.  Point both
    # immutable base inputs at the verified 650-day combined CSV; its final
    # merge de-duplicates by issue date before appending the 60 recovery days.
    collector.BASE_CSV_1 = BASE_650_CSV
    collector.BASE_CSV_2 = BASE_650_CSV
    collector.RECOVERY_CSV = OUTPUT_DIR / "광주_격자예보_3시간단위_재수집_20260606_20260804.csv"
    collector.FINAL_CSV = OUTPUT_DIR / "광주_격자예보_3시간단위_TMP_SKY_REH_710일.csv"
    collector.main()


if __name__ == "__main__":
    main()
