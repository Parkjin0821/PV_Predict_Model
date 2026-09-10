"""영광 격자예보(REH·POP·SKY) 백필을 기존 결합자료에 추가 결합.

김제 join_gimje_grid_forecast_v1_2026-08-31.py와 완전히 동일한 방법론 -
경로만 영광으로 교체. 사용자 09-03 지시로 코드만 작성(실행은 사용자가
직접). 발행시각 정합성 근거(NWP 10:00 KST vs GRID 05:00 KST, 같은
발표일 결합이므로 미래누출 아님)는 부안·김제에서 이미 검증된 원칙을
그대로 재사용 - 영광도 같은 두 API를 동일 패턴으로 쓰므로 그대로
적용된다.

★선행조건★: combine_yeonggwang_history_power_asos_nwp_v1_2026-09-03.py와
영광 GRID 710일 백필(이미 완료, 09-03 12:39) 둘 다 완료돼있어야 한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COMBINED = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_v1_2026-09-03"
    r"\영광_과거발전_ASOS_NWP_결합.parquet"
)
GRID_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31\GRID"
    r"\영광_격자예보_REH_POP_SKY_710일.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_v1_2026-09-03"
)
OUT_PARQUET = OUT_DIR / "영광_과거발전_ASOS_NWP_GRID_결합_v1_2026-09-03.parquet"

FCST_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)
VARIABLES = ("REH", "POP", "SKY")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--combined", type=Path, default=COMBINED)
    p.add_argument("--grid-csv", type=Path, default=GRID_CSV)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    a = cli()
    if not a.combined.is_file():
        raise FileNotFoundError(f"기존 결합자료 없음(combine_yeonggwang_history_power_asos_nwp_v1_2026-09-03.py 먼저 실행): {a.combined}")
    if not a.grid_csv.is_file():
        raise FileNotFoundError(f"격자예보 백필 CSV 없음: {a.grid_csv}")

    base = pd.read_parquet(a.combined)
    for col in ("prediction_issue_time_kst", "target_time_kst"):
        if col not in base.columns:
            raise RuntimeError(f"기존 결합자료에 {col} 열이 없음")
    if base.duplicated(["prediction_issue_time_kst", "target_time_kst"]).any():
        raise RuntimeError("기존 결합자료에 발행/대상시각 중복 존재 - 결합 전 원인 확인 필요")

    grid = pd.read_csv(a.grid_csv, dtype={"발표일": str})
    required_cols = [f"{v}_{h:02d}h" for h in FCST_HOURS for v in VARIABLES]
    missing_cols = [c for c in required_cols if c not in grid.columns]
    if missing_cols:
        raise RuntimeError(f"격자예보 CSV에 필요한 열이 없음: {missing_cols}")
    if grid["발표일"].duplicated().any():
        raise RuntimeError("격자예보 CSV에 발표일 중복 존재")
    if grid[required_cols].isna().any().any():
        raise RuntimeError("격자예보 CSV에 결측 셀 존재 - backfill 검증을 먼저 통과해야 함")
    expected_days = pd.date_range("2024-08-25", "2026-08-04", freq="D")
    if len(grid) != len(expected_days):
        raise RuntimeError(
            f"GRID 백필이 아직 안 끝났습니다({len(grid)}/{len(expected_days)}일) - "
            "710일 전부 완료한 뒤 다시 실행할 것(조용히 부분데이터로 결합하지 않음)."
        )

    grid["issue_date"] = pd.to_datetime(grid["발표일"], format="%Y%m%d", errors="raise")

    records = []
    for _, row in grid.iterrows():
        target_day = row["issue_date"] + pd.Timedelta(days=1)
        for h in FCST_HOURS:
            rec = {
                "issue_date": row["issue_date"],
                "target_time_kst": target_day + pd.Timedelta(hours=h),
            }
            for v in VARIABLES:
                rec[f"forecast_{v}"] = pd.to_numeric(row[f"{v}_{h:02d}h"], errors="raise")
            records.append(rec)
    grid_long = pd.DataFrame(records)
    if grid_long.duplicated(["issue_date", "target_time_kst"]).any():
        raise RuntimeError("격자예보 변환 결과에 발표일·대상시각 중복 발생")

    base = base.copy()
    base["issue_date"] = base["prediction_issue_time_kst"].dt.normalize()

    for col in (f"forecast_{v}" for v in VARIABLES):
        if col in base.columns:
            raise RuntimeError(f"기존 결합자료에 이미 {col} 열이 있음 - 재실행/중복 결합 의심")

    joined = base.merge(
        grid_long, on=["issue_date", "target_time_kst"], how="left", validate="one_to_one"
    )

    unmatched = int(joined["forecast_REH"].isna().sum())
    matched_issue_days = int(joined.loc[joined["forecast_REH"].notna(), "issue_date"].nunique())
    base_issue_days = int(base["issue_date"].nunique())

    a.output_dir.mkdir(parents=True, exist_ok=True)
    joined = joined.drop(columns=["issue_date"])
    joined.to_parquet(a.output_dir / OUT_PARQUET.name, index=False)

    summary = {
        "status": "historical_archive_join_candidate",
        "note": "NWP와 동일한 상속 한계(발행시각 10:00 KST 기준 05:00 격자예보는 항상 선행이라 "
                "미래누출 아님) - 격자예보 자체의 '당시 실제수신시각'도 NWP와 마찬가지로 검증 불가.",
        "rows": len(joined),
        "base_rows": len(base),
        "base_issue_days": base_issue_days,
        "grid_issue_days": int(grid_long["issue_date"].nunique()),
        "matched_issue_days": matched_issue_days,
        "unmatched_rows": unmatched,
        "defect_period_rows": int(joined["is_defect_period"].sum()) if "is_defect_period" in joined.columns else None,
        "forecast_missing_by_variable": {
            f"forecast_{v}": int(joined[f"forecast_{v}"].isna().sum()) for v in VARIABLES
        },
        "output": str(a.output_dir / OUT_PARQUET.name),
    }
    (a.output_dir / "영광_GRID결합_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if unmatched:
        raise RuntimeError(
            f"결합 후 {unmatched}행이 격자예보와 매칭 안 됨 - 발표일 범위 불일치 등 원인 확인 필요"
            "(임의로 무시하지 않음, 위 요약 JSON 참고)"
        )


if __name__ == "__main__":
    main()
