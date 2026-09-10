# -*- coding: utf-8 -*-
"""김제 - 전주(146)/정읍(245) 실측 일사량과 김제 실제발전량 상관분석(09-01).

배경: 김제가 재사용 중인 ASOS243(부안)에는 일사량계가 없어
`issue_asos_일사량_W_m2`·`reference_target_asos_일사량_W_m2`가 전부 결측이다
(check_correlation_multicollinearity_gimje_v1_2026-09-01.py에서 이미 확인).
가장 가까운 실측 일사량 보유 지점(전주 27.4km, 정읍 26.4km - 09-01 KMA
API 3개 시각 실측조회 + 공식 2017 등급표 대조로 확인)의 관측값을 김제
실제발전량과 직접 상관분석해서, 대체 입력으로 쓸 가치가 있는지 먼저
탐색한다. **이 스크립트는 탐색용** - 기존 check_correlation_
multicollinearity_gimje 스크립트의 leakage 방지 원칙(폴드 학습구간
내부에서만 재계산)을 그대로 재사용해서 "폴드마다 안정적인지"까지 함께
확인한다.

join 방식: 전주/정읍 ASOS 관측시각을 김제 결합테이블의 target_time_kst와
그대로 맞춘다(같은 시각의 실측 일사량 vs 같은 시각의 실제발전량 - 예보
join이 아니라 관측 동시각 join이므로 시차 문제 없음).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
COMBINED = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31"
    r"\김제_과거발전_ASOS_NWP_GRID_결합_v1_2026-08-31.parquet"
)
NEARBY = {
    "전주146_27.4km": Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
        r"\전주146\기상청_ASOS146_시간환경_20240825_20260804.csv"
    ),
    "정읍245_26.4km": Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\인근관측소백필_v1_2026-09-01"
        r"\정읍245\기상청_ASOS245_시간환경_20240825_20260804.csv"
    ),
}
OUT_DIR = HERE / "outputs" / "김제_인근일사량_상관분석_2026-09-01"

TARGET = "plant_ac_power_kw"
REFERENCE_FEATURES = ["forecast_DSWRF", "solar_elevation_deg", "forecast_TCDC"]
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 10


def load_nearby_solar(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["시각"]).dt.tz_localize(None)
    out = pd.DataFrame({"target_time_kst": ts, tag: df["일사량_W_m2"].to_numpy()})
    return out.drop_duplicates("target_time_kst")


def load_base() -> pd.DataFrame:
    df = pd.read_parquet(COMBINED)
    df["target_time_kst"] = pd.to_datetime(df["target_time_kst"])
    before = len(df)
    df = df.loc[~df["is_defect_period"]].copy()
    removed_defect = before - len(df)
    df = df[df["quality_status"] == "valid_ge9of12"]
    df["physical_daylight"] = df["physical_daylight"].astype(bool)
    df = df[df["physical_daylight"]]
    return df, removed_defect


def expanding_folds_full_coverage(issue_days, initial, block):
    n = len(issue_days)
    folds = []
    end = initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((issue_days[:end], issue_days[end:test_end]))
        end = test_end
    return folds


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, removed_defect = load_base()

    tags = []
    for tag, path in NEARBY.items():
        if not path.is_file():
            print(f"[대기] {tag} 파일이 아직 없습니다 - 백필 진행 중일 수 있음: {path}")
            continue
        solar = load_nearby_solar(path, tag)
        base = base.merge(solar, on="target_time_kst", how="left")
        tags.append(tag)

    if not tags:
        raise SystemExit("인근 지점 데이터가 하나도 준비되지 않았습니다.")

    base["issue_day"] = base["target_time_kst"].dt.normalize()
    features = REFERENCE_FEATURES + tags
    x_full = base[features].dropna()
    y_full = base.loc[x_full.index, TARGET]

    exploratory = {
        "_주의": "leakage 있음(전체표본에 시험구간 포함) - 탐색용, 특성선택 최종근거 아님. "
                 "'폴드내부_재검증'을 볼 것.",
        "표본행수": int(len(x_full)),
        "타깃상관계수(내림차순)": dict(sorted(
            {c: round(float(x_full[c].corr(y_full)), 3) for c in features}.items(),
            key=lambda kv: -abs(kv[1]))),
    }

    issue_days = pd.DatetimeIndex(np.sort(base["issue_day"].unique()))
    folds = expanding_folds_full_coverage(issue_days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    fold_rows = []
    for i, (train_days, _test_days) in enumerate(folds, start=1):
        train = base[base["issue_day"].isin(train_days)]
        x = train[features].dropna()
        row = {"폴드": i, "학습발행일수": len(train_days), "학습행수": len(x)}
        if len(x) < MIN_ROWS_PER_FOLD:
            row["skip_reason"] = f"표본 {len(x)}행 < 최소 {MIN_ROWS_PER_FOLD}행"
            fold_rows.append(row)
            continue
        y = train.loc[x.index, TARGET]
        row["타깃상관"] = {c: round(float(x[c].corr(y)), 3) for c in features}
        fold_rows.append(row)

    per_feature = {c: [] for c in features}
    for row in fold_rows:
        corr = row.get("타깃상관")
        if not corr:
            continue
        for c in features:
            if c in corr:
                per_feature[c].append(corr[c])
    stability = {}
    for c, vals in per_feature.items():
        if not vals:
            continue
        arr = np.array(vals)
        stability[c] = {
            "폴드수": len(arr), "평균": round(float(arr.mean()), 3),
            "최소": round(float(arr.min()), 3), "최대": round(float(arr.max()), 3),
            "부호일관": bool((arr >= 0).all() or (arr <= 0).all()),
        }

    result = {
        "데이터출처_기본": str(COMBINED),
        "데이터출처_인근지점": {k: str(v) for k, v in NEARBY.items()},
        "결함구간_제외행수": removed_defect,
        "포함된_인근지점": tags,
        "탐색용_전체표본(leakage있음_참고만)": exploratory,
        "폴드내부_재검증(leakage없음)": fold_rows,
        "폴드간_안정성_요약": stability,
        "_다음단계": "여기서 상관이 기존 부안기반 대안(solar_elevation·DSWRF)보다 "
                   "뚜렷이 강하고 폴드간 안정적이면, 결합테이블에 정식 특성으로 "
                   "추가해 모델 성능(MAE) 비교로 넘어갈 것.",
    }

    (OUT_DIR / "김제_인근일사량_상관분석_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
