# -*- coding: utf-8 -*-
"""★★★08-31 폐기 경고 - 이 스크립트의 산출물(부안_인버터분해_ABC_결과.json,
B 0.63%/C 0.65%/A 1.35%)을 근거로 쓰지 말 것★★★

"ok" 필터가 `isin(["observed","night_zero"])`인데 실제 상태값은
`"night_zero_physical"`이라 야간 정상0값이 전부 결측 취급됐다 -
"8대 완전가용일"이 59일(여름51·봄8, 겨울·가을 0일)로 축소된 잘못된
표본이었다(실제로는 181일). 게다가 이 스크립트는 LOO 교차검증이라
대상일 이후 날짜도 학습에 포함돼 미래누출 문제도 있었다(Codex 지적).
필터 수정 + 시간순 walk-forward로 재실행한
inverter_disaggregation_ABC_v2_필터수정_walkforward_2026-08-31.py와
AGENTS.md 2026-08-31 절을 볼 것. 이 파일은 재현성 보존을 위해 코드는
그대로 두되, 새 분석엔 절대 재사용하지 말 것.

--- 이하 원본 08-31 오전 docstring(참고용, 위 경고가 우선) ---
부안 인버터분해 A·B·C 완전판 실측(08-31, 사용자 지시: "A/B/C 모두
진행해야 하는 부분이니 셋 다 해보자").

## 08-28 결정과 이번 작업의 관계 (되돌리는 게 아니라 채워넣는 것)
08-28엔 A(정격비례)만 직접 실측하고 "8대 완전가용일이 59일뿐이라
walk-forward 블록 CV로 나누면 폴드당 시험표본이 한 자릿수가 돼
C를 통계적으로 신뢰 못 한다"는 이유로 B·C를 시도하지 않았다(A가
이미 nMAE 1.35%로 충분히 좋기도 했음). 이번엔 **블록 walk-forward
대신 leave-one-out(LOO) 교차검증**을 쓴다 - 이건 08-28의 판단을
뒤집는 게 아니라, "적은 표본에 적합한 다른 검증기법"을 쓰는 것이다.
LOO는 표본이 작을 때 특히 권장되는 표준 기법이고(태양광발전 예측에
LOO를 실제로 쓴 선행연구 다수 확인 - 예: Almeida et al. 2015/2017,
Solar Energy, 실제 5개 PV 발전소 자료에 LOO-CV 적용, 각각 285회·111회
인용), 59개 시험폴드(=59일 각각 1회씩 held-out)를 전부 모아 집계하면
통계적으로 의미 있는 표본크기(N=59)가 된다 - 블록 walk-forward의
"폴드당 시험표본 한 자릿수" 문제를 구조적으로 피한다.

## 그래도 남아있는 한계(숨기지 않음)
- 가을 데이터가 전무하다(부안 전체 공통 한계) - day-of-year 특성이
  가을 구간엔 전혀 학습되지 않은 채로 평가된다.
- 59일은 여전히 적은 표본이라, 이번 결과도 "재검토 조건 150일+"을
  완전히 대체하는 최종판정이 아니라 참고용 중간점검으로 취급한다.
- Codex의 부안_인버터별_5분_야간0포함.parquet을 읽기전용으로만 쓴다.
  API 호출·원본 수정 없음.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
SRC = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_인버터별_5분_야간0포함.parquet"
)
OUT_DIR = HERE / "outputs" / "인버터분해_ABC_2026-08-31"
SEED = 42
MIN_SLOT_COUNT = 260  # 하루 최대 288슬롯의 약 90% - "거의 완전가용"
SUM_TOL_KWH = 1e-6  # 방법별 예측이 그날 실제 총합과 정확히 일치해야 함(분해이지 별도예측 아님)


def _season(month: int) -> str:
    """audit_buan_season_testrows_v1_2026-08-28.py와 동일 정의(일관성 유지)."""
    if month in (12, 1, 2):
        return "겨울"
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    return "가을"


def load_daily_per_inverter() -> pd.DataFrame:
    """inverter_disaggregation_check_v1_2026-08-28.py와 동일 로직 재사용."""
    df = pd.read_parquet(SRC, columns=[
        "grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"])
    df["ok"] = df["quality_status_after_night"].isin(["observed", "night_zero"])
    df["date"] = df["grid_time_kst"].dt.date
    df["kwh_5min"] = df["ac_power_kw"] * (5 / 60)
    daily = df[df["ok"]].groupby(["date", "inverter_number"])["kwh_5min"].sum().unstack("inverter_number")
    count = df[df["ok"]].groupby(["date", "inverter_number"]).size().unstack("inverter_number")
    full8 = count.notna().all(axis=1) & (count.min(axis=1) >= MIN_SLOT_COUNT)
    return daily[full8].dropna()


def _score(daily8: pd.DataFrame, pred: np.ndarray, method_name: str) -> dict:
    inv_cols = list(daily8.columns)
    actual = daily8.to_numpy()
    total = actual.sum(axis=1)

    # 분해 원칙 검증: 방법이 뭐든 인버터별 예측의 합은 그날 실제 총합과 같아야 한다.
    pred_total = pred.sum(axis=1)
    max_sum_err = float(np.max(np.abs(pred_total - total)))
    if max_sum_err > SUM_TOL_KWH:
        raise AssertionError(f"{method_name}: 합계보존 위반(최대오차 {max_sum_err} kWh)")

    err = actual - pred
    mae_per_inv = np.abs(err).mean(axis=0)
    mean_daily = daily8.mean(axis=0).to_numpy()
    nmae_pct = mae_per_inv / mean_daily * 100
    shares_actual = daily8.div(daily8.sum(axis=1), axis=0)

    per_inverter = []
    for i, inv in enumerate(inv_cols):
        per_inverter.append({
            "인버터": int(inv), "일평균_kWh": round(float(mean_daily[i]), 1),
            "MAE_kWh": round(float(mae_per_inv[i]), 2),
            "nMAE_pct": round(float(nmae_pct[i]), 2),
            "실제평균비중_pct": round(float(shares_actual[inv].mean() * 100), 2),
        })
    return {
        "방법": method_name,
        "합계보존_최대오차_kWh": round(max_sum_err, 8),
        "인버터별": per_inverter,
        "평균_nMAE_pct": round(float(nmae_pct.mean()), 2),
        "최대_nMAE_pct": round(float(nmae_pct.max()), 2),
    }


def method_a(daily8: pd.DataFrame) -> dict:
    """정격용량비례 - 8대 전부 125kW 동일이라 균등 12.5%. 과거이력 불필요(CV 불필요)."""
    total = daily8.sum(axis=1).to_numpy()
    # 날짜별 총발전량의 1/8을 인버터 8개 열에 각각 배치한다. 이전 코드는
    # (날짜수, 1) 배열만 만들어 pred.sum(axis=1)이 총합의 1/8이 됐다.
    pred = np.repeat(
        total[:, None] * (1.0 / daily8.shape[1]),
        daily8.shape[1],
        axis=1,
    )
    return _score(daily8, pred, "A_정격용량비례(균등12.5%)")


def method_b_loo(daily8: pd.DataFrame) -> dict:
    """계절중앙값 - 대상일을 제외한 같은 계절의 다른 날들로 중앙값비중을 구해 예측(LOO)."""
    seasons = pd.Series([_season(d.month) for d in daily8.index], index=daily8.index)
    shares_all = daily8.div(daily8.sum(axis=1), axis=0)
    total = daily8.sum(axis=1)

    pred = np.zeros_like(daily8.to_numpy(), dtype=float)
    skipped = []
    season_min_others = 3  # 같은 계절 최소 3일(자기 제외)은 있어야 중앙값을 신뢰
    for idx, day in enumerate(daily8.index):
        same_season_mask = (seasons == seasons.loc[day]) & (daily8.index != day)
        others = shares_all[same_season_mask]
        if len(others) < season_min_others:
            skipped.append({"날짜": str(day), "계절": seasons.loc[day], "같은계절_다른날수": int(len(others))})
            # 표본부족: 전체(계절무관) 중앙값으로 대체하되 반드시 기록해 숨기지 않는다.
            others = shares_all[shares_all.index != day]
        median_share = others.median()
        norm_share = median_share / median_share.sum()
        pred[idx, :] = total.loc[day] * norm_share.to_numpy()

    result = _score(daily8, pred, "B_계절중앙값(LOO)")
    result["표본부족으로_전체중앙값대체된_날"] = skipped
    result["_주의"] = ("계절별 다른날수 3일 미만이면 계절중앙값 대신 전체(계절무관) 중앙값을 "
                     "썼다 - 임의로 숨기지 않고 목록을 남김.")
    return result


def method_c_loo(daily8: pd.DataFrame) -> dict:
    """LightGBM 학습기반 비중분해(LOO) - 특성: day-of-year 순환인코딩 + 그날 총발전량(로그)."""
    inv_cols = list(daily8.columns)
    shares_all = daily8.div(daily8.sum(axis=1), axis=0)
    total = daily8.sum(axis=1)
    doy = pd.Series([d.timetuple().tm_yday for d in daily8.index], index=daily8.index)
    feat = pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "log_total_kwh": np.log1p(total),
    }, index=daily8.index)

    pred = np.zeros_like(daily8.to_numpy(), dtype=float)
    for idx, day in enumerate(daily8.index):
        train_mask = daily8.index != day
        raw_shares = np.zeros(len(inv_cols))
        for j, inv in enumerate(inv_cols):
            model = LGBMRegressor(n_estimators=40, learning_rate=0.1, num_leaves=4,
                                  max_depth=2, min_child_samples=10, subsample=0.9,
                                  reg_alpha=0.5, reg_lambda=2.0, random_state=SEED,
                                  n_jobs=-1, verbosity=-1)
            model.fit(feat[train_mask], shares_all.loc[train_mask, inv])
            raw_shares[j] = max(float(model.predict(feat.loc[[day]])[0]), 0.0)
        if raw_shares.sum() <= 0:
            raw_shares = np.full(len(inv_cols), 1.0 / len(inv_cols))
        norm_shares = raw_shares / raw_shares.sum()
        pred[idx, :] = total.loc[day] * norm_shares

    result = _score(daily8, pred, "C_LightGBM비중(LOO)")
    result["_주의"] = "일별 58행 학습 x 인버터 8개 x 날짜 59개 = 472개 소형모델. 표본이 " \
                     "작아 08-28 재검토조건(150일+)을 대체하는 결과 아님 - 참고용 중간점검."
    return result


def run() -> dict:
    daily8 = load_daily_per_inverter()
    season_coverage = pd.Series([_season(d.month) for d in daily8.index]).value_counts().to_dict()

    a = method_a(daily8)
    b = method_b_loo(daily8)
    c = method_c_loo(daily8)

    ranking = sorted([a, b, c], key=lambda r: r["평균_nMAE_pct"])
    summary = {
        "표본일수": int(len(daily8)),
        "계절별_표본일수": season_coverage,
        "가을_표본": season_coverage.get("가을", 0),
        "검증방식": "leave-one-out(LOO) - A는 과거이력 불필요(결정론적)라 CV 불필요, B·C는 "
                 "대상일을 제외한 나머지 표본으로 학습 후 그날을 복원한다. 자기행 누출은 없지만 "
                 "대상일 이후 자료도 학습에 포함되므로 시계열 운영 관점의 미래누출 없는 검증은 아니다. "
                 "학술근거: Almeida et al. 2015/2017(Solar Energy)이 실제 PV 발전소 자료에 "
                 "LOO-CV를 적용한 선행사례.",
        "순위_평균nMAE기준": [{"방법": r["방법"], "평균_nMAE_pct": r["평균_nMAE_pct"],
                          "최대_nMAE_pct": r["최대_nMAE_pct"]} for r in ranking],
        "1위": ranking[0]["방법"],
        "_판정주의": ("이 결과는 08-28에 세운 'C 재검토조건(150일+)'을 충족한 최종판정이 "
                    "아니라, 08-31에 LOO 기법으로 A/B/C를 회고적으로 비교한 중간점검이다. "
                    "B·C는 미래 날짜가 학습에 들어가므로 운영 채택 근거로 사용할 수 없고, "
                    "향후 시간순 rolling-origin 검증을 별도로 통과해야 한다. "
                    "가을 표본이 0일이라 day-of-year 기반 특성(B·C 둘 다)이 가을엔 전혀 검증 "
                    "안 됐다는 한계를 그대로 남긴다. promote_to_official 대상 아님."),
    }
    return {"A": a, "B": b, "C": c, "요약": summary}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT_DIR / "부안_인버터분해_ABC_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
