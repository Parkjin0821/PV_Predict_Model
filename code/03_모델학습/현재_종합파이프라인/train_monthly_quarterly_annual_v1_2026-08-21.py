# -*- coding: utf-8 -*-
"""중장기(월간·분기·연간) 누적 발전량 예측 — 신규 착수(08-21).

## 배경
공식 스펙(AGENTS.md "공식 예측 주기" 절)의 중장기 운영은 "일간·월간·
연간 집계, 예측수평 분기·연간"이다. 지금까지 **일간 총량만 만들어졌고
월간·분기·연간은 프로젝트 시작부터 한 번도 착수되지 않았다**
(AGENTS.md 전체에 "중장기(일간/분기/연간) 파이프라인 미착수" 메모가
반복 등장). 사용자 지시로 이번에 착수.

## ★데이터 제약(착수 전 반드시 확인, 반복 참고할 것)★
실측 발전량은 2024-08-25~2026-08-04, 약 24개월치뿐이다. 게다가:
- 2024-08(시작월)·2026-08(종료월)은 원래 부분월(각 7일·4일)이라 제외.
- **2025-08~11이 알려진 저출력 이상구간**(AGENTS.md, 원인 미확보)과
  맞물려 일별 자료 자체가 크게 빠져있다(08월 11/31일·09월 0/30일·10월
  0/31일·11월 13/30일만 존재) — "저출력"이 아니라 사실상 **결측**이다.
  이 구간을 포함한 월은 "완전월"에서 제외한다(임의 보간 금지 원칙 준수).
- 이 결측 때문에 **2025년 전체가 완전한 1개년이 아니다** — 즉 지금
  자료로는 **연간 예측의 실측 검증표본이 0개**다. 완전 분기도 5개뿐.

## 설계 원칙 (데이터 제약에 맞춰 축소, 명시)
1. **월간**: 완전월(그 달의 실제 일수만큼 일별 자료가 다 있는 달)만
   타깃으로 쓴다. 자기이력(lag1·lag2·lag3·lag12)·이동평균·계절성만으로
   예측한다 — 월 단위로 몇 주~몇 달 앞선 기상예보 자체가 없으므로
   (KMA API허브 NWP는 최대 +36h) 기후값·과거이력 기반 예측이 된다.
   Expanding-window walk-forward로 검증한다.
2. **분기**: 월간모델의 OOF 예측을 3개월 합산하는 **상향식(bottom-up)
   집계**로만 만든다 — 완전분기가 5개뿐이라 분기 전용 ML모델을 학습·
   검증하는 건 통계적으로 무의미하다(자체 판단).
3. **연간**: 검증 가능한 실측 완전 1개년이 없으므로 **정식 모델을
   만들지 않는다.** 참고용으로 "최근 12개 유효월 합계"만 제시하고
   "검증 불가"를 명시한다 — 없는 실측을 있는 것처럼 임의 추정해
   학습데이터로 쓰지 않는다는 프로젝트 원칙을 따른다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "월간_분기_연간_v1_2026-08-21"
DAILY_ACTUAL_PARQUET = ROOT / "outputs" / "집계_일간_실제발전량.parquet"

MIN_TRAIN_MONTHS = 6  # walk-forward 최소 학습표본(월)


def energy_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return {"표본수": 0, "MAE_kWh": None, "RMSE_kWh": None, "WAPE_pct": None}
    e = y - p
    denom = float(np.abs(y).sum())
    return {
        "표본수": int(len(y)),
        "MAE_kWh": float(np.abs(e).mean()),
        "RMSE_kWh": float(np.sqrt(np.mean(e ** 2))),
        "WAPE_pct": float(np.abs(e).sum() / denom * 100) if denom > 0 else None,
    }


def build_monthly_series() -> pd.DataFrame:
    """일간 실측을 월간 집계하고, '완전월' 여부를 실제 그 달 일수와 대조해 판정."""
    daily = pd.read_parquet(DAILY_ACTUAL_PARQUET).copy()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    monthly = daily["일간발전량_kWh"].resample("MS").agg(합계_kWh="sum", 유효일수="count")
    monthly["그달_전체일수"] = monthly.index.days_in_month
    # 시작·종료 부분월(자료수집 경계)까지 자동으로 걸러지도록, 1일 이상 결측이면 불완전월로 판정
    monthly["완전월"] = monthly["유효일수"] >= monthly["그달_전체일수"]
    return monthly


def build_feature_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """완전한 달력월 인덱스(결측월 포함, 빈 자리는 NaN)로 재색인 후 특성 생성."""
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    m = monthly.reindex(full_index)
    target = m["합계_kWh"].where(m["완전월"].fillna(False))  # 불완전월은 타깃에서 NaN 처리

    feat = pd.DataFrame(index=full_index)
    feat["1개월전"] = target.shift(1)
    feat["2개월전"] = target.shift(2)
    feat["3개월전"] = target.shift(3)
    feat["12개월전(전년동월)"] = target.shift(12)
    feat["직전3개월평균"] = target.shift(1).rolling(3, min_periods=3).mean()
    feat["직전6개월평균"] = target.shift(1).rolling(6, min_periods=6).mean()
    feat["월"] = full_index.month
    feat["월_sin"] = np.sin(2 * np.pi * full_index.month / 12)
    feat["월_cos"] = np.cos(2 * np.pi * full_index.month / 12)
    feat["그달_일수"] = full_index.days_in_month
    feat["목표_kWh"] = target
    return feat


def walk_forward(feat: pd.DataFrame, feature_cols: list[str], require_cols: list[str], seed: int) -> pd.DataFrame:
    """expanding-window: 그 시점까지의 완전월만으로 학습, 다음 완전월 하나를 예측.

    ★설계 메모★: 2025-08~11 결측구간 때문에 12개월전(전년동월)·직전6개월평균
    같은 장거리 lag까지 전부 non-null을 요구하면 표본이 25개월 중 1개로
    붕괴한다(실측 확인됨). LightGBM이 결측을 원생 지원하므로, **"1개월전"
    (직전월 지속성, 가장 핵심)만 필수로 요구**하고 나머지 lag·이동평균은
    NaN이어도 학습·예측에 그대로 흘려보낸다(트리모델이 분기 시 결측을
    한쪽으로 보내는 방식으로 처리, fillna 안 함 — 08-20 밤 다중공선성
    가지치기 때와 동일하게 "네이티브 결측 허용" 원칙 재사용)."""
    valid = feat.dropna(subset=require_cols + ["목표_kWh"])
    rows = []
    for i in range(MIN_TRAIN_MONTHS, len(valid)):
        train = valid.iloc[:i]
        test = valid.iloc[i:i + 1]
        model = LGBMRegressor(
            objective="regression_l1", n_estimators=80, learning_rate=0.05,
            num_leaves=7, max_depth=3, min_child_samples=3,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbosity=-1, n_jobs=2,
        )
        model.fit(train[feature_cols], train["목표_kWh"])
        pred_ml = float(np.clip(model.predict(test[feature_cols])[0], 0, None))
        actual = float(test["목표_kWh"].iloc[0])
        rows.append({
            "월": test.index[0], "실제_kWh": actual, "LightGBM_kWh": pred_ml,
            "전월지속성_kWh": float(test["1개월전"].iloc[0]),
            "전년동월_kWh": float(test["12개월전(전년동월)"].iloc[0]) if pd.notna(test["12개월전(전년동월)"].iloc[0]) else None,
            "직전3개월평균_kWh": float(test["직전3개월평균"].iloc[0]),
        })
    return pd.DataFrame(rows)


def summarize(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in [("LightGBM", "LightGBM_kWh"), ("전월지속성", "전월지속성_kWh"),
                       ("직전3개월평균", "직전3개월평균_kWh")]:
        m = energy_metrics(oof["실제_kWh"], oof[col])
        rows.append({"구성": name, **m})
    yoy = oof.dropna(subset=["전년동월_kWh"])
    if len(yoy) > 0:
        m = energy_metrics(yoy["실제_kWh"], yoy["전년동월_kWh"])
        rows.append({"구성": "전년동월(표본적음)", **m})
    return pd.DataFrame(rows)


def quarterly_bottom_up(oof: pd.DataFrame) -> pd.DataFrame:
    """월간 OOF 예측을 분기(3개월) 단위로 합산 — 그 분기 3개월이 전부 OOF에
    있는 경우만 채택(부분분기는 만들지 않음)."""
    df = oof.copy()
    df["분기"] = pd.PeriodIndex(df["월"], freq="Q")
    # ★버그수정(08-21)★: pandas groupby.sum()은 기본적으로 NaN을 조용히 0으로
    # 취급한다 — "직전3개월평균"처럼 초기 몇 개월이 결측(장거리 lag 계산불가)인
    # 열을 그냥 sum()하면 그 결측월이 0으로 들어가 분기합계가 과소추정된다.
    # min_count=3(=그 분기 3개월 전부 non-NaN이어야 함)으로 명시해 방지한다.
    sum3 = lambda s: s.sum(min_count=3)  # noqa: E731
    grouped = df.groupby("분기").agg(
        월수=("실제_kWh", "size"),
        실제_kWh=("실제_kWh", sum3),
        LightGBM_kWh=("LightGBM_kWh", sum3),
        전월지속성_kWh=("전월지속성_kWh", sum3),
        직전3개월평균_kWh=("직전3개월평균_kWh", sum3),
    )
    complete = grouped[grouped["월수"] == 3].copy()
    yoy = df.dropna(subset=["전년동월_kWh"]).groupby("분기")["전년동월_kWh"].agg(["sum", "size"])
    complete["전년동기_kWh"] = yoy.loc[yoy["size"] == 3, "sum"] if len(yoy) else np.nan
    return complete


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = int(config["random_seed"])

    monthly = build_monthly_series()
    monthly.to_csv(OUT / "월간_원자료_완전월판정.csv", encoding="utf-8-sig")
    n_complete = int(monthly["완전월"].sum())
    print(f"전체 {len(monthly)}개월 중 완전월 {n_complete}개")
    print(monthly[["합계_kWh", "유효일수", "그달_전체일수", "완전월"]].to_string())

    feat = build_feature_frame(monthly)
    feature_cols = ["1개월전", "2개월전", "3개월전", "12개월전(전년동월)",
                     "직전3개월평균", "직전6개월평균", "월_sin", "월_cos", "그달_일수"]
    require_cols = ["1개월전"]  # 나머지 lag는 결측 허용(LightGBM 네이티브 처리)

    print(f"\n[월간] walk-forward 가능 표본: {len(feat.dropna(subset=require_cols + ['목표_kWh']))}개 "
          f"(최소학습 {MIN_TRAIN_MONTHS}개월 제외 후 실제 시험표본은 더 적음)")

    oof = walk_forward(feat, feature_cols, require_cols, seed)
    oof.to_csv(OUT / "월간_OOF_상세.csv", index=False, encoding="utf-8-sig")
    monthly_summary = summarize(oof)
    monthly_summary.to_csv(OUT / "월간_성능요약.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 월간 성능 (시험표본 {len(oof)}개월, expanding walk-forward) ===")
    print(monthly_summary.to_string(index=False))

    print("\n[분기] 월간 OOF를 3개월 합산(bottom-up) — 완전분기만 채택")
    q = quarterly_bottom_up(oof)
    q.to_csv(OUT / "분기_bottomup_상세.csv", encoding="utf-8-sig")
    print(f"완전분기 표본수: {len(q)}")
    if len(q) > 0:
        print(q.to_string())
        q_summary = []
        for name, col in [("LightGBM(월간합산)", "LightGBM_kWh"), ("전월지속성(월간합산)", "전월지속성_kWh"),
                           ("직전3개월평균(월간합산)", "직전3개월평균_kWh")]:
            m = energy_metrics(q["실제_kWh"], q[col])
            q_summary.append({"구성": name, **m})
        if "전년동기_kWh" in q.columns and q["전년동기_kWh"].notna().any():
            yq = q.dropna(subset=["전년동기_kWh"])
            m = energy_metrics(yq["실제_kWh"], yq["전년동기_kWh"])
            q_summary.append({"구성": "전년동기(표본극소)", **m})
        q_summary_df = pd.DataFrame(q_summary)
        q_summary_df.to_csv(OUT / "분기_성능요약.csv", index=False, encoding="utf-8-sig")
        print("\n=== 분기 성능(bottom-up, 완전분기 {}개뿐 — 참고용) ===".format(len(q)))
        print(q_summary_df.to_string(index=False))
    else:
        print("완전분기 0개 — 분기 성능 산출 불가")

    print("\n[연간] ★검증 불가★ — 데이터 기간(25개월, 4개월 결측 포함) 안에 "
          "완전한 실측 1개년이 존재하지 않는다.")
    # "최근 12개 유효월 합계" 같은 값은 중간에 결측월을 건너뛴 불연속 구간이라
    # 진짜 연간치의 대용으로 쓰면 과소추정을 숨기는 결과가 된다(오해 소지 있음
    # — 08-21 자체 검토로 폐기). 대신 "연속으로 결측 없이 이어진 가장 긴 구간"
    # 만 정직하게 보고한다.
    complete_flags = monthly["완전월"].reindex(pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")).fillna(False)
    run_id = (complete_flags != complete_flags.shift()).cumsum()
    runs = pd.DataFrame({
        "길이": complete_flags.groupby(run_id).size(),
        "완전월여부": complete_flags.groupby(run_id).first(),
        "시작": complete_flags.groupby(run_id).apply(lambda s: s.index.min()),
        "끝": complete_flags.groupby(run_id).apply(lambda s: s.index.max()),
    })
    best_run = runs[runs["완전월여부"]].sort_values("길이", ascending=False).iloc[0]
    note = (
        f"참고: 결측 없이 연속으로 이어진 가장 긴 완전월 구간은 "
        f"{best_run['시작'].strftime('%Y-%m')}~{best_run['끝'].strftime('%Y-%m')} "
        f"({int(best_run['길이'])}개월, 12개월 미만)뿐이다. 어떤 방식으로도 지금 자료만으로는 "
        "신뢰 가능한 연간 총량 참고치조차 만들 수 없다(불연속 12개월을 억지로 합치면 "
        "빠진 달만큼 과소추정된 값을 마치 연간치처럼 오인시킬 위험이 있어 산출하지 않음)."
    )
    print(f"  {note}")
    (OUT / "연간_참고사항.txt").write_text(
        "연간 예측은 정식 모델을 만들지 않았다(사유: 완전한 실측 1개년 검증표본 0개, "
        f"연속 완전월 최장구간 {int(best_run['길이'])}개월뿐).\n"
        + note + "\n"
        "데이터가 3년 이상으로 늘어나 최소 1~2개 완전년도 홀드아웃이 가능해지면 "
        "재착수할 것.\n",
        encoding="utf-8",
    )

    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
