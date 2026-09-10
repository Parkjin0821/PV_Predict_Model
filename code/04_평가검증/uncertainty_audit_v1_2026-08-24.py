"""광주 PV의 NWP 운영 정합성과 예측구간을 독립 감사한다.

주의: 예측구간 입력은 기존 v3 E2E OOF다. v5 공식 재학습 전까지
방법 검증용 예비 결과이며 공식 모델 성능을 갱신하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
NWP_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\kma_nwp_dayahead_fixed_tm_710d_v2_2026-08-20"
    r"\광주_익일예보_고정tm_DSWRF_DSWRFLX_DIFSWRF_TCDC_LCDC_MCDC_HCDC_710일.csv"
)
OOF_DIR = (
    PROJECT / "03_모델학습" / "현재_종합파이프라인" / "outputs"
    / "E2E_리플레이백테스트_v1_2026-08-21"
)
OOF_AC = OOF_DIR / "행단위_ac_power_예측정답.csv"
OOF_DAILY = OOF_DIR / "행단위_daily_final_예측정답.csv"


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal finite-sample 'higher' quantile."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("nan")
    level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(scores, level, interpolation="higher"))


def audit_nwp() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(NWP_CSV, dtype={"발표일": str, "발행시각_kst": str,
                                     "요청_tm_utc": str, "요청_tm_kst": str})
    issue_day = pd.to_datetime(df["발표일"], format="%Y%m%d", errors="coerce")
    issue_at = pd.to_datetime(df["발행시각_kst"], format="%Y%m%d%H%M", errors="coerce")
    run_utc = pd.to_datetime(df["요청_tm_utc"], format="%Y%m%d%H%M", errors="coerce")
    run_kst = pd.to_datetime(df["요청_tm_kst"], format="%Y%m%d%H%M", errors="coerce")
    expected_days = pd.date_range(issue_day.min(), issue_day.max(), freq="D")
    vars_ = ["DSWRF", "DSWRFLX", "DIFSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]
    value_cols = [f"{v}_{h:02d}h" for v in vars_ for h in range(0, 24, 3)]

    rows = [
        ("행수_710", len(df) == 710, len(df), "정확히 710 발표일"),
        ("발표일_연속", len(expected_days) == len(df) and issue_day.nunique() == len(df),
         issue_day.nunique(), f"{issue_day.min():%Y-%m-%d}~{issue_day.max():%Y-%m-%d}"),
        ("발행시각_D10KST", bool((issue_at == issue_day + pd.Timedelta(hours=10)).all()),
         int((issue_at == issue_day + pd.Timedelta(hours=10)).sum()), "D 10:00 KST"),
        ("요청런_D00UTC", bool((run_utc == issue_day).all()),
         int((run_utc == issue_day).sum()), "D 00:00 UTC"),
        ("요청런_D09KST", bool((run_kst == issue_day + pd.Timedelta(hours=9)).all()),
         int((run_kst == issue_day + pd.Timedelta(hours=9)).sum()), "D 09:00 KST"),
        ("런_발행_선후관계", bool((run_kst < issue_at).all()),
         float(((issue_at - run_kst).dt.total_seconds() / 3600).median()), "중앙 1시간 선행"),
        ("수집완료", bool(df["수집완료"].fillna(False).astype(bool).all()),
         int(df["수집완료"].fillna(False).astype(bool).sum()), "API 응답 TMFC 일치 검사 후 저장"),
        ("실제_게시완료시각_증빙", False, np.nan,
         "파일에 게시/입수시각 로그가 없어 D09 런이 D10 전에 실제 배포됐는지는 미입증"),
    ]
    summary = pd.DataFrame(rows, columns=["항목", "통과", "측정값", "설명"])
    missing = pd.DataFrame({
        "변수": vars_,
        "전체셀": [len(df) * 8] * len(vars_),
        "결측셀": [int(df[[f"{v}_{h:02d}h" for h in range(0, 24, 3)]].isna().sum().sum()) for v in vars_],
    })
    missing["결측률_pct"] = missing["결측셀"] / missing["전체셀"] * 100
    return summary, missing


def load_oof() -> pd.DataFrame:
    ac = pd.read_csv(OOF_AC, parse_dates=["발행시각", "대상시각"])
    ac = ac.rename(columns={"실제_kW": "실제", "예측_kW": "예측", "대상시각": "대상"})
    ac["단위"] = "kW"
    daily = pd.read_csv(OOF_DAILY, parse_dates=["발행시각", "대상일"])
    daily = daily.rename(columns={"실제_kWh": "실제", "예측_kWh": "예측", "대상일": "대상"})
    daily["단위"] = "kWh"
    daily["수평_h"] = "D+1"
    keep = ["티어", "수평_h", "공식구성", "폴드", "발행시각", "대상", "실제", "예측", "단위"]
    return pd.concat([ac[keep], daily[keep]], ignore_index=True)


def sequential_conformal(oof: pd.DataFrame, capacities: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = []
    summaries = []
    for (tier, horizon), g in oof.groupby(["티어", "수평_h"], sort=False):
        fold_order = (g.groupby("폴드")["발행시각"].min().sort_values().index.tolist())
        prior = pd.DataFrame()
        for fold_idx, fold in enumerate(fold_order):
            test = g[g["폴드"] == fold].copy()
            if fold_idx == 0:
                prior = pd.concat([prior, test], ignore_index=True)
                continue
            scores = np.abs(prior["실제"].to_numpy() - prior["예측"].to_numpy())
            cap = capacities["daily"] if tier == "일간" else capacities["hourly"]
            for nominal, alpha in [(80, 0.20), (90, 0.10)]:
                q = finite_sample_quantile(scores, alpha)
                lo = np.clip(test["예측"].to_numpy() - q, 0, cap)
                hi = np.clip(test["예측"].to_numpy() + q, 0, cap)
                y = test["실제"].to_numpy()
                covered = (y >= lo) & (y <= hi)
                part = test.copy()
                part["명목포함률_pct"] = nominal
                part["보정표본수"] = len(scores)
                part["보정절대잔차_q"] = q
                part["하한"] = lo
                part["상한"] = hi
                part["포함"] = covered
                detailed.append(part)
                summaries.append({
                    "티어": tier, "수평_h": horizon, "폴드": fold,
                    "명목포함률_pct": nominal, "실제포함률_pct": covered.mean() * 100,
                    "포함률차이_pp": covered.mean() * 100 - nominal,
                    "평균구간폭": float(np.mean(hi - lo)), "보정표본수": len(scores),
                    "보정절대잔차_q": q, "시험표본수": len(test),
                })
            prior = pd.concat([prior, test], ignore_index=True)
    return pd.concat(detailed, ignore_index=True), pd.DataFrame(summaries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    nwp_summary, nwp_missing = audit_nwp()
    oof = load_oof()
    capacities = {"hourly": 219.0, "daily": 219.0 * 24}
    interval_rows, interval_summary = sequential_conformal(oof, capacities)

    nwp_summary.to_csv(args.output / "NWP_운영정합성_감사.csv", index=False, encoding="utf-8-sig")
    nwp_missing.to_csv(args.output / "NWP_변수별_결측.csv", index=False, encoding="utf-8-sig")
    interval_summary.to_csv(args.output / "예측구간_폴드별_예비검증.csv", index=False, encoding="utf-8-sig")
    interval_rows.to_csv(
        args.output / "예측구간_행단위_예비.csv.gz", index=False,
        encoding="utf-8-sig", compression="gzip",
    )

    agg = (interval_rows.groupby(["티어", "수평_h", "명목포함률_pct"], as_index=False)
           .agg(실제포함률_pct=("포함", lambda x: float(x.mean() * 100)),
                평균구간폭=("하한", lambda x: 0.0), 시험표본수=("포함", "size")))
    widths = (interval_rows.assign(구간폭=interval_rows["상한"] - interval_rows["하한"])
              .groupby(["티어", "수평_h", "명목포함률_pct"])["구간폭"].mean())
    agg["평균구간폭"] = [widths.loc[tuple(x)] for x in agg[["티어", "수평_h", "명목포함률_pct"]].to_numpy()]
    agg.to_csv(args.output / "예측구간_수평별_예비요약.csv", index=False, encoding="utf-8-sig")

    report = {
        "판정": "예비 감사만 완료 — 공식 성능 갱신 금지",
        "NWP_시각구조": "D09 KST 런 < D10 KST 발행 구조는 전 행 통과",
        "NWP_미해결": "실제 게시완료시각 로그가 없어 1시간 내 운영 입수 가능성은 미입증",
        "예측구간_주의": "v3 OOF 사용. 인버터5 결함을 고친 v5 공식 OOF 생성 후 반드시 재산출",
        "용량프로필": "219kW",
    }
    (args.output / "판정.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(nwp_summary.to_string(index=False))
    print("\n예측구간 수평별 예비요약")
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
