"""김제 격자예보(REH·POP·SKY) 백필을 기존 결합자료에 추가 결합.

부안 join_buan_grid_forecast_v1_2026-08-31.py와 완전히 동일한 방법론 -
경로만 김제로 교체.

★선행조건★: combine_gimje_history_power_asos_nwp_v1_2026-08-31.py와
backfill_gimje_grid_forecast_v1_2026-08-31.py(GRID 710일 백필) 둘 다
완료돼있어야 한다. 없으면 즉시 FileNotFoundError로 멈춘다(임의로
건너뛰거나 결측 채움 없음).

## 발행시각 정합성(부안에서 실측 확인한 원칙 그대로 적용)
NWP(익일예보) 공식 발행시각은 10:00 KST, 격자예보(동네예보)는 항상
05:00 KST(tmfc=발표일+"05")에 발표된 값을 조회한다 - 05:00이 10:00보다
이르므로 같은 발표일의 격자예보값을 NWP 발행시각(10:00) 시점의 특성으로
붙여도 미래누출이 아니다. 정확한 발행시각(clock) 일치가 아니라
발표일(issue_date) + 대상시각(target_time_kst) 일치로 결합한다(두 API의
발행시각 관례가 다르기 때문). 김제도 같은 두 API(NWP 점조회·격자예보)를
동일 패턴으로 쓰므로 이 근거가 그대로 적용된다 - 김제 CSV로 재검증할 것.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COMBINED = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31"
    r"\김제_과거발전_ASOS_NWP_결합.parquet"
)
GRID_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31\GRID"
    r"\김제_격자예보_REH_POP_SKY_710일.csv"
)
OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31"
)
OUT_PARQUET = OUT_DIR / "김제_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"

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
        raise FileNotFoundError(f"기존 결합자료 없음(combine_gimje_history_power_asos_nwp_v1_2026-08-31.py 먼저 실행): {a.combined}")
    if not a.grid_csv.is_file():
        raise FileNotFoundError(
            f"격자예보 백필 CSV 없음 - backfill_gimje_grid_forecast_v1_2026-08-31.py"
            f"를 먼저 완료해야 함: {a.grid_csv}"
        )

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
            "backfill_gimje_grid_forecast_v1_2026-08-31.py를 710일 전부 완료한 뒤 "
            "다시 실행할 것(조용히 부분데이터로 결합하지 않음)."
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
    (a.output_dir / "김제_GRID결합_요약.json").write_text(
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
