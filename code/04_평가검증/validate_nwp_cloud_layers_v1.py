# -*- coding: utf-8 -*-
"""수치예보모델 층별 운량(LCDC 하층·MCDC 중층·HCDC 상층)을 ASOS 실측
전운량·중하층운량과 비교해 MAE·상관계수를 산출한다.

`validate_nwp_solar_cloud_v1.py`(DSWRF/TCDC 검증)의 후속으로, 08-19에 완료된
2차 배치(--vars LCDC MCDC HCDC)는 그동안 결측·값범위만 확인했고 실측 대비
검증은 아직 하지 않았다 — 이 스크립트가 그 공백을 채운다.

ASOS는 층별(저/중/고) 운량을 따로 주지 않고 "전운량"과 "중하층운량"만
제공하므로, 층별 예보값을 실측과 1:1로 정확히 대응시킬 수는 없다. 대신:
1. 각 층(LCDC/MCDC/HCDC) 개별 값과 ASOS 전운량_frac의 상관계수만 참고용으로 본다.
2. 기상학의 표준 무작위 중첩 가정(random overlap assumption,
   전운량 = 1-(1-c1)(1-c2)(1-c3))으로 LCDC+MCDC+HCDC를 합성해 ASOS
   전운량_frac과, LCDC+MCDC 합성을 ASOS 중하층운량_frac과 비교한다.
   이 합성식은 이 프로젝트의 자체 판단(표준 기상학 방법론 차용)이며 보고서
   원문 방법론이 아님을 명시한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

NWP_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_solar_cloud_710d_v1_2026-08-19"
)
LAYER_CSV = NWP_DIR / "광주_수치예보_LCDC_MCDC_HCDC_710일.csv"

ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\기상청_ASOS156_시간환경_모델용.csv"
)

OUT_DIR = NWP_DIR / "검증"
FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def load_layers_long() -> pd.DataFrame:
    df = pd.read_csv(LAYER_CSV, dtype={"발표일": str})
    records = []
    for row in df.to_dict("records"):
        issue_day = datetime.strptime(row["발표일"], "%Y%m%d")
        target_day = issue_day + timedelta(days=1)
        for hour in FCST_HOURS:
            records.append(
                {
                    "timestamp": target_day + timedelta(hours=hour),
                    "LCDC_fcst": row.get(f"LCDC_{hour:02d}h"),
                    "MCDC_fcst": row.get(f"MCDC_{hour:02d}h"),
                    "HCDC_fcst": row.get(f"HCDC_{hour:02d}h"),
                }
            )
    long_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    long_df["합성_전운량_랜덤중첩"] = 1 - (
        (1 - long_df["LCDC_fcst"]) * (1 - long_df["MCDC_fcst"]) * (1 - long_df["HCDC_fcst"])
    )
    long_df["합성_중하층운량_랜덤중첩"] = 1 - (
        (1 - long_df["LCDC_fcst"]) * (1 - long_df["MCDC_fcst"])
    )
    return long_df


def load_asos_hourly() -> pd.DataFrame:
    df = pd.read_csv(ASOS_CSV)
    df["timestamp"] = pd.to_datetime(df["시각"])
    keep = df[["timestamp", "전운량_pct", "중하층운량_pct"]].copy()
    keep["전운량_frac"] = keep["전운량_pct"] / 100.0
    keep["중하층운량_frac"] = keep["중하층운량_pct"] / 100.0
    return keep


def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"n": 0, "mae": None, "corr": None}
    diff = (y_true[mask] - y_pred[mask]).abs()
    return {
        "n": int(mask.sum()),
        "mae": float(diff.mean()),
        "corr": float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1]),
    }


def main() -> int:
    if not LAYER_CSV.exists():
        print(f"아직 수집 결과가 없습니다: {LAYER_CSV}")
        return 1

    layers = load_layers_long()
    asos = load_asos_hourly()
    merged = layers.merge(asos, on="timestamp", how="inner")

    results = {
        "LCDC vs 전운량_frac(참고용)": summarize(merged["전운량_frac"], merged["LCDC_fcst"]),
        "MCDC vs 전운량_frac(참고용)": summarize(merged["전운량_frac"], merged["MCDC_fcst"]),
        "HCDC vs 전운량_frac(참고용)": summarize(merged["전운량_frac"], merged["HCDC_fcst"]),
        "합성_전운량_랜덤중첩 vs ASOS 전운량_frac": summarize(
            merged["전운량_frac"], merged["합성_전운량_랜덤중첩"]
        ),
        "합성_중하층운량_랜덤중첩 vs ASOS 중하층운량_frac": summarize(
            merged["중하층운량_frac"], merged["합성_중하층운량_랜덤중첩"]
        ),
    }

    print(f"매칭된 시각 수(교집합): {len(merged)}")
    for name, r in results.items():
        print(f"[{name}] n={r['n']}, MAE={r['mae']}, corr={r['corr']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "nwp_cloud_layers_vs_asos_matched.csv", index=False, encoding="utf-8-sig")
    lines = [f"매칭된 시각 수: {len(merged)}"]
    lines += [f"{name}: n={r['n']}, MAE={r['mae']}, corr={r['corr']}" for name, r in results.items()]
    (OUT_DIR / "nwp_cloud_layers_vs_asos_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\n결과 저장: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
