# -*- coding: utf-8 -*-
"""광주 D+1 4계절 climatology 기준선을 폴드 학습행만으로 재계산한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
OOF = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25" / "최종_직접모델_OOF.csv"
RAW = ROOT / "outputs" / "v7_라이브연계_2026-08-26" / "집계_일간_실제발전량_v5.parquet"
OUT = ROOT / "outputs" / "광주_일간D1_4계절_폴드내부climatology_재계산_v1_2026-09-14"
WINDOWS = {
    "1_여름": "2025-06-01", "2_가을_결함종료후": "2025-11-17",
    "3_겨울": "2025-12-15", "4_봄": "2026-02-15", "5_초여름": "2026-04-15",
}


def season4(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.month.map({1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
        6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을", 12: "겨울"})


def metric(g: pd.DataFrame, pred: str) -> dict:
    e = g["실제_kWh"] - g[pred]
    return {"n": int(len(g)), "독립일수": int(g["날짜"].nunique()),
            "MAE_kWh": float(e.abs().mean()), "RMSE_kWh": float(np.sqrt(np.mean(e**2))),
            "WAPE_pct": float(100 * e.abs().sum() / g["실제_kWh"].abs().sum())}


def main() -> None:
    oof = pd.read_csv(OOF, encoding="utf-8-sig", parse_dates=["날짜"])
    raw = pd.read_parquet(RAW)
    raw.index = pd.to_datetime(raw.index)
    actual_col = "일간발전량_kWh"
    if actual_col not in raw.columns:
        raise RuntimeError(f"원자료에 {actual_col} 없음")
    hist = raw[actual_col].dropna().rename("학습실제_kWh").to_frame()
    oof["계절4"] = season4(oof["날짜"])
    oof["폴드시험시작"] = oof["폴드"].map(lambda x: pd.Timestamp(WINDOWS[x]))
    clim = []
    for _, row in oof.iterrows():
        train = hist[hist.index < row["폴드시험시작"]].copy()
        train["계절4"] = season4(pd.Series(train.index, index=train.index))
        same = train.loc[train["계절4"] == row["계절4"], "학습실제_kWh"]
        if same.empty:
            same = train["학습실제_kWh"]
        clim.append(float(same.mean()))
    oof["climatology_폴드내부_kWh"] = clim

    rows = []
    for s in ["봄", "여름", "가을", "겨울"]:
        g = oof[oof["계절4"] == s]
        if g.empty:
            continue
        m = metric(g, "예측_kWh")
        c = metric(g, "climatology_폴드내부_kWh")
        rows.append({"계절": s, **{f"모델_{k}": v for k, v in m.items()},
                     **{f"climatology_{k}": v for k, v in c.items()},
                     "MAE_개선율_pct": 100 * (c["MAE_kWh"] - m["MAE_kWh"]) / c["MAE_kWh"],
                     "RMSE_개선율_pct": 100 * (c["RMSE_kWh"] - m["RMSE_kWh"]) / c["RMSE_kWh"],
                     "판정": "표본충분" if m["독립일수"] >= 30 else "표본부족_판정보류"})
    overall = metric(oof, "예측_kWh")
    overall_c = metric(oof, "climatology_폴드내부_kWh")
    manifest = {"OOF행수": len(oof), "모델": overall, "climatology_폴드내부": overall_c,
                "leakage_audit": "통과: 각 예측행은 해당 폴드 시험시작일 이전 원자료만 기준선에 사용",
                "defect_audit": "기존 공식B OOF 및 결함구간 제외 정책 승계",
                "status": "재계산_후보_운영연결보류"}
    OUT.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUT / "동일행_OOF_폴드내부climatology.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "4계절_모델대_폴드내부climatology.csv", index=False, encoding="utf-8-sig")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
