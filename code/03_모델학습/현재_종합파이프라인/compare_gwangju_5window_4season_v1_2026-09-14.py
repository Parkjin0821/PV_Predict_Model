# -*- coding: utf-8 -*-
"""광주 일간 D+1 기존 5개 시간순 시험창을 4계절 기준으로 재집계한다.

모델을 다시 학습하지 않는다. 기존 최종 OOF의 날짜·실측·예측을 그대로
사용해 계절 라벨만 기상학적 4계절(봄 3~5, 여름 6~8, 가을 9~11,
겨울 12~2)로 부여한다. 따라서 표본 차이 없이 분류 체계만 비교한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25" / "최종_직접모델_OOF.csv"
OUT = ROOT / "outputs" / "광주_일간D1_5창대4계절_동일행비교_v1_2026-09-14"


def season4(month: int) -> str:
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    if month in (9, 10, 11):
        return "가을"
    return "겨울"


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby(group_col, sort=False):
        err = g["실제_kWh"] - g["예측_kWh"]
        rows.append({
            group_col: label,
            "n": int(len(g)),
            "시작일": g["날짜"].min().date().isoformat(),
            "종료일": g["날짜"].max().date().isoformat(),
            "MAE_kWh": float(err.abs().mean()),
            "RMSE_kWh": float(np.sqrt(np.mean(err ** 2))),
            "WAPE_pct": float(100 * err.abs().sum() / g["실제_kWh"].abs().sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig")
    df["날짜"] = pd.to_datetime(df["날짜"])
    if df["날짜"].duplicated().any():
        raise RuntimeError("OOF 날짜 중복 발견")
    if df[["실제_kWh", "예측_kWh"]].isna().any().any():
        raise RuntimeError("OOF 실측/예측 결측 발견")

    df["계절4"] = df["날짜"].dt.month.map(season4)
    old = summarize(df, "폴드")
    new = summarize(df, "계절4")
    cross = pd.crosstab(df["폴드"], df["계절4"], margins=True)

    err = df["실제_kWh"] - df["예측_kWh"]
    overall = {
        "동일시험행_n": int(len(df)),
        "날짜중복_n": int(df["날짜"].duplicated().sum()),
        "기존시간순시험창수": int(df["폴드"].nunique()),
        "새계절범주수": int(df["계절4"].nunique()),
        "MAE_kWh": float(err.abs().mean()),
        "RMSE_kWh": float(np.sqrt(np.mean(err ** 2))),
        "WAPE_pct": float(100 * err.abs().sum() / df["실제_kWh"].abs().sum()),
        "leakage_audit": "통과: 기존 OOF 고정, 재학습/재예측 없음",
        "defect_audit": "기존 공식B 결함구간 제외 OOF 정책 승계",
        "status": "분류표준_후보_운영연결보류",
        "판정": (
            "4계절 분류 채택 가능. 단, 시간순 검증창은 5개 유지하고 "
            "2025·2026 여름을 같은 계절 범주로 집계한다."
        ),
        "분기와의관계": "기상학적 계절과 달력분기(Q1~Q4)는 별도 기준",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "동일행_OOF_4계절라벨.csv", index=False, encoding="utf-8-sig")
    old.to_csv(OUT / "기존5개시험창_성능.csv", index=False, encoding="utf-8-sig")
    new.to_csv(OUT / "4계절_성능.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "기존창_4계절_교차표.csv", encoding="utf-8-sig")
    (OUT / "manifest.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n[4계절 성능]")
    print(new.to_string(index=False))
    print("\n[기존창 x 4계절]")
    print(cross.to_string())


if __name__ == "__main__":
    main()
