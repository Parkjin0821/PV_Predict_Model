# -*- coding: utf-8 -*-
"""김제 총출력모델 상관분석·다중공선성(09-01) - 부안 08-31 Codex 독립감사
4건(leakage·하루밀림·과장·지표집계오류) 중 leakage 교훈을 처음부터 반영.

## 부안과 다른 점(재발방지)
1. **leakage 없음**: 부안은 처음에 843행 전체(시험기간 포함)로 상관·VIF를
   계산해놓고 그 결론을 walk-forward 전체에 적용했다가 Codex 감사에서
   걸렸다(check_multicollinearity_buan_v1_2026-08-31.py). 김제는 처음부터
   각 폴드의 "그 폴드 학습구간 데이터만"으로 상관·VIF를 다시 계산해서
   결론이 폴드마다 안정적인지 확인한다(전체표본 결과는 참고용 탐색치로만
   별도 표시, 특성선택 근거로 안 씀).
2. **결함구간 제외**: `is_defect_period` 컬럼(target_time_kst 기준,
   08-31 확정)으로 미리 걸러낸다 - 부안은 결함구간을 안 걸렀었다.
3. 지표 집계는 이번 스크립트 범위 밖(모델 성능 비교는 다음 단계)이지만,
   폴드 요약은 부안 v5와 동일하게 "폴드별" + "전체(모든 폴드 통합)" 둘 다
   보여줘서 단순평균 함정을 피한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
DATA = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31"
    r"\김제_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"
)
OUT_DIR = HERE / "outputs" / "김제_상관분석_다중공선성_2026-09-01"

TARGET = "plant_ac_power_kw"
VIF_SEVERE, PAIR_CORR_HIGH = 10.0, 0.8
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 10

WEATHER8 = ["forecast_DSWRF", "forecast_TCDC", "forecast_LCDC", "forecast_MCDC", "forecast_HCDC",
            "forecast_REH", "forecast_POP", "forecast_SKY"]
ASOS4 = ["issue_asos_기온_C", "issue_asos_풍속_m_s", "issue_asos_상대습도_pct", "issue_asos_전운량_pct"]
CANDIDATE_FEATURES = WEATHER8 + ASOS4 + ["solar_elevation_deg"]


def load_daylight_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["prediction_issue_time_kst"] = pd.to_datetime(df["prediction_issue_time_kst"])
    df["target_time_kst"] = pd.to_datetime(df["target_time_kst"])
    df["issue_day"] = df["prediction_issue_time_kst"].dt.normalize()

    before = len(df)
    df = df.loc[~df["is_defect_period"]].copy()
    removed_defect_rows = before - len(df)

    df = df[df["quality_status"] == "valid_ge9of12"]
    df["physical_daylight"] = df["physical_daylight"].astype(bool)
    df = df[df["physical_daylight"]]
    return df, removed_defect_rows


def expanding_folds_full_coverage(issue_days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    """부안 v5와 동일 - 마지막 폴드가 남은 날을 전부 흡수해 평가에서 빠지는 발행일이 없게 한다."""
    n = len(issue_days)
    folds = []
    end = initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((issue_days[:end], issue_days[end:test_end]))
        end = test_end
    return folds


def compute_vif(x: pd.DataFrame) -> pd.Series:
    x = (x - x.mean()) / x.std(ddof=0)
    vifs = {}
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def target_correlation(x: pd.DataFrame, y: pd.Series) -> dict:
    return {c: float(x[c].corr(y)) for c in x.columns}


def high_pair_correlations(x: pd.DataFrame, target_corr: dict) -> list[dict]:
    corr_matrix = x.corr(method="pearson")
    cols = corr_matrix.columns.tolist()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr_matrix.loc[a, b]
            if abs(r) >= PAIR_CORR_HIGH:
                weaker = a if abs(target_corr[a]) < abs(target_corr[b]) else b
                pairs.append({"변수A": a, "변수B": b, "쌍상관계수": round(float(r), 3),
                              "A_타깃상관": round(target_corr[a], 3), "B_타깃상관": round(target_corr[b], 3),
                              "제거후보": weaker})
    return pairs


def exploratory_full_sample(df: pd.DataFrame) -> dict:
    """★참고용 탐색치★ - 전체표본(폴드 무관) 계산. 특성선택 최종근거로 쓰지 않는다."""
    x = df[CANDIDATE_FEATURES].dropna()
    y = df.loc[x.index, TARGET]
    target_corr = target_correlation(x, y)
    vif = compute_vif(x)
    pairs = high_pair_correlations(x, target_corr)
    return {
        "_주의": "leakage 있음(전체표본에 시험구간 포함) - 탐색용일 뿐 특성선택 근거 아님. "
                 "최종 판단은 '폴드내부_재검증' 항목을 볼 것.",
        "표본행수": len(x),
        "타깃상관계수(내림차순)": dict(sorted(
            {k: round(v, 3) for k, v in target_corr.items()}.items(), key=lambda kv: -abs(kv[1]))),
        "VIF": {k: (None if np.isinf(v) else round(float(v), 2)) for k, v in vif.items()},
        "쌍상관_0.8이상": pairs,
    }


def fold_internal_check(df: pd.DataFrame, folds: list[tuple]) -> list[dict]:
    """★leakage 방지 핵심★ - 각 폴드의 학습구간 데이터만으로 상관·VIF를 매번 새로 계산."""
    rows = []
    for i, (train_days, _test_days) in enumerate(folds, start=1):
        train = df[df["issue_day"].isin(train_days)]
        x = train[CANDIDATE_FEATURES].dropna()
        row = {"폴드": i, "학습발행일수": len(train_days), "학습행수(낮시간완전표본)": len(x)}
        if len(x) < MIN_ROWS_PER_FOLD:
            row["skip_reason"] = f"표본 {len(x)}행 < 최소 {MIN_ROWS_PER_FOLD}행"
            rows.append(row)
            continue
        y = train.loc[x.index, TARGET]
        target_corr = target_correlation(x, y)
        row["타깃상관"] = {k: round(v, 3) for k, v in target_corr.items()}
        vif = compute_vif(x)
        row["VIF_최대"] = round(float(vif.replace(np.inf, np.nan).max()), 2)
        row["VIF_10이상_변수"] = sorted(vif[vif >= VIF_SEVERE].index.tolist())
        rows.append(row)
    return rows


def summarize_fold_stability(fold_rows: list[dict]) -> dict:
    """폴드별 타깃상관을 변수별로 모아 min/max/mean, 부호일관성을 본다 -
    "이 변수는 거의 모든 폴드에서 상관이 약하다"는 결론이 안정적인지 확인."""
    per_feature: dict[str, list[float]] = {c: [] for c in CANDIDATE_FEATURES}
    for row in fold_rows:
        corr = row.get("타깃상관")
        if not corr:
            continue
        for c in CANDIDATE_FEATURES:
            if c in corr:
                per_feature[c].append(corr[c])

    summary = {}
    weak_stable_candidates = []
    for c, vals in per_feature.items():
        if not vals:
            continue
        arr = np.array(vals)
        summary[c] = {
            "폴드수": len(arr),
            "평균": round(float(arr.mean()), 3),
            "최소": round(float(arr.min()), 3),
            "최대": round(float(arr.max()), 3),
            "부호일관": bool((arr >= 0).all() or (arr <= 0).all()),
            "모든폴드_절대값0.15미만": bool((np.abs(arr) < 0.15).all()),
        }
        if summary[c]["모든폴드_절대값0.15미만"]:
            weak_stable_candidates.append(c)
    return {"변수별_폴드간_상관요약": summary, "제거후보(전_폴드에서_|r|<0.15)": weak_stable_candidates}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, removed_defect_rows = load_daylight_dataset()
    issue_days = pd.DatetimeIndex(np.sort(df["issue_day"].unique()))
    folds = expanding_folds_full_coverage(issue_days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    exploratory = exploratory_full_sample(df)
    fold_rows = fold_internal_check(df, folds)
    stability = summarize_fold_stability(fold_rows)

    result = {
        "데이터출처": str(DATA),
        "결함구간_제외행수(is_defect_period)": removed_defect_rows,
        "낮시간_완전표본_전체행수": exploratory["표본행수"],
        "issue_days_total": int(len(issue_days)),
        "폴드구성": [{"폴드": i + 1, "학습일수": len(tr), "시험일수": len(te)}
                  for i, (tr, te) in enumerate(folds)],
        "탐색용_전체표본_상관VIF(leakage있음_참고만)": exploratory,
        "폴드내부_재검증(leakage없음_특성선택_실제근거)": fold_rows,
        "폴드간_안정성_요약": stability,
        "_다음단계": "여기서 나온 '제거후보(전_폴드에서_|r|<0.15)'를 확정 후보로 삼아 "
                   "rolling-origin 교차검증(모델 성능 비교)으로 넘어갈 것. "
                   "VIF>=10 & 쌍상관>=0.8 동시충족 변수는 폴드별 VIF_10이상_변수 목록도 함께 볼 것.",
    }

    (OUT_DIR / "김제_상관분석_다중공선성_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
