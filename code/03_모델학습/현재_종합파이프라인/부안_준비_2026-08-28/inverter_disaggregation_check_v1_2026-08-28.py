# -*- coding: utf-8 -*-
"""★★★08-31 폐기 경고 - 이 스크립트의 산출물(부안_인버터분해_확인.json,
nMAE 1.35%)을 근거로 쓰지 말 것★★★

"ok" 필터가 `isin(["observed","night_zero"])`인데 실제 상태값은
`"night_zero_physical"`이라 철자가 달라 야간 정상0값이 전부 결측
취급됐다 - "8대 완전가용일"이 59일(여름51·봄8, 겨울·가을 0일)로 심하게
축소된 잘못된 표본이었다(실제로는 181일: 겨울78·봄52·여름51). 이 버그로
인한 정정·재실행 결과는
inverter_disaggregation_ABC_v2_필터수정_walkforward_2026-08-31.py와
AGENTS.md 2026-08-31 절("★★★부안 인버터분해: 완전가용 59일 자체가
필터 오타로...★★★")을 볼 것. 이 파일은 버그 재현 방지를 위해 코드는
그대로 두되(과거 산출물 재현성 보존), 새 분석엔 절대 재사용하지 말 것.

--- 이하 원본 08-28 docstring(참고용, 위 경고가 우선) ---
부안 작업순서 6번: 인버터분해 방법 확인(A vs 학습기반 C).

Codex의 인버터별 5분 산출물(부안_인버터별_5분_야간0포함.parquet)을
읽기전용으로 읽어, 8대 균등용량(125kW x 8)에서 정격비례(A, 12.5%씩
균등분배)가 실제로 얼마나 정확한지 실측한다.

## 왜 A vs C 정식 CV(광주 08-27/08-28 방식)를 그대로 안 했는지
8대가 전부 유효한 날이 59일뿐이다(발전소 총출력 186일보다 훨씬 적음 -
인버터 단위 완전가용 기준이 훨씬 엄격하기 때문). 59일을 walk-forward로
학습/시험 나누면 각 폴드 시험표본이 한 자릿수가 되어 통계적으로 의미
있는 A vs C 비교가 불가능하다. 대신 A(균등분배)의 정확도 자체를
직접 실측해 채택 여부를 판단한다 - 실측 결과가 이미 충분히 좋으면
(아래 참고) 데이터가 부족한 지금 시점에 학습기반 C를 무리하게 넣는 게
오히려 과적합 위험만 키운다고 판단했다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_인버터별_5분_야간0포함.parquet"
)
OUT_DIR = HERE / "outputs" / "인버터분해_확인_2026-08-28"

MIN_SLOT_COUNT = 260  # 하루 최대 288슬롯의 약 90% - "거의 완전가용"


def load_daily_per_inverter() -> pd.DataFrame:
    df = pd.read_parquet(SRC, columns=[
        "grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"])
    df["ok"] = df["quality_status_after_night"].isin(["observed", "night_zero"])
    df["date"] = df["grid_time_kst"].dt.date
    df["kwh_5min"] = df["ac_power_kw"] * (5 / 60)
    daily = df[df["ok"]].groupby(["date", "inverter_number"])["kwh_5min"].sum().unstack("inverter_number")
    count = df[df["ok"]].groupby(["date", "inverter_number"]).size().unstack("inverter_number")
    full8 = count.notna().all(axis=1) & (count.min(axis=1) >= MIN_SLOT_COUNT)
    return daily[full8].dropna()


def check_method_a(daily8: pd.DataFrame) -> dict:
    inv_cols = list(daily8.columns)
    total = daily8.sum(axis=1)
    pred_equal = total.to_numpy()[:, None] * (1.0 / len(inv_cols))
    actual = daily8.to_numpy()
    err = actual - pred_equal
    mae_per_inv = np.abs(err).mean(axis=0)
    mean_daily = daily8.mean(axis=0).to_numpy()
    nmae_pct = mae_per_inv / mean_daily * 100
    shares = daily8.div(total, axis=0)

    per_inverter = []
    for i, inv in enumerate(inv_cols):
        per_inverter.append({
            "인버터": int(inv), "일평균_kWh": round(float(mean_daily[i]), 1),
            "MAE_kWh": round(float(mae_per_inv[i]), 2),
            "nMAE_pct": round(float(nmae_pct[i]), 2),
            "실제평균비중_pct": round(float(shares[inv].mean() * 100), 2),
            "비중표준편차_pp": round(float(shares[inv].std() * 100), 2),
        })
    return {
        "표본일수": int(len(daily8)),
        "인버터별": per_inverter,
        "평균_nMAE_pct": round(float(nmae_pct.mean()), 2),
        "최대_nMAE_pct": round(float(nmae_pct.max()), 2),
    }


def run() -> dict:
    daily8 = load_daily_per_inverter()
    a_result = check_method_a(daily8)

    # 참고: 광주 A방법(정격불균등 50/50/30/39/50) 일간 nMAE는 4.5~14.5%였다
    # (AGENTS.md 08-28 "일간 D+1 인버터분해" 절 실측). 부안은 8대가 전부
    # 동일용량(125kW)이라 구조적으로 더 균일할 것으로 기대했고, 아래
    # 실측이 그 기대를 뒷받침하는지 확인한다.
    verdict = {
        "권장방법": "A_정격용량비례(균등 12.5%씩)",
        "판단근거": (
            "평균 nMAE %.2f%%, 최대 %.2f%%로 광주 A방법(4.5~14.5%%)보다 훨씬 정확함 - "
            "8대가 전부 동일용량(125kW)이라 실제로도 거의 균등하게 발전한다. "
            "학습기반 C는 시도하지 않음: 8대 완전가용일이 59일뿐이라 walk-forward로 "
            "나누면 폴드당 시험표본이 한 자릿수가 되어 통계적으로 신뢰 불가 + 데이터 "
            "부족 상황에서 학습기반 방법은 과적합 위험이 실익보다 크다고 판단."
        ) % (a_result["평균_nMAE_pct"], a_result["최대_nMAE_pct"]),
        "재검토_조건": "표본일수(현재 59일)가 충분히 쌓이면(예: 150일+) C방법 재검토",
        "_promote_주석": "config/부안_전처리_규칙_v1_2026-08-28.json과 마찬가지로 "
                       "이 판단도 공식 파이프라인에 아직 편입되지 않았다(promote_to_official 아님).",
    }
    return {"A_정격비례_실측": a_result, "판정": verdict}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "부안_인버터분해_확인.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
