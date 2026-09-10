# -*- coding: utf-8 -*-
"""광주·영광 월간 총발전량(kWh) 예측 - 공식기준선 vs 실험후보 동일조건 비교(09-10, Claude).

사용자 지시(09-10): GB-Huber를 국내 논문에 억지로 끼워맞추지 말고,
1) 공식 기준선은 국내에서 설명하기 쉬운 전년동월·계절평균·선형/Ridge로 두고
2) GB-Huber·LightGBM은 "실험 후보"로 두고
3) 동일한 월별 walk-forward 조건에서 전부 비교해서
4) GB-Huber가 여러 지표·계절에서 반복적으로 우세할 때만 승격 후보로 올린다.

기존 광주 v2(train_gwangju_monthly_improvement_candidate_v2_2026-09-10.py)와
영광 v1(train_yeonggwang_monthly_candidate_v1_2026-09-10.py)의 특성(lag1-3,
mean3, month_sin/cos, days, availability_lag1/daylight_ratio_lag1)은 그대로
유지하고, 다음을 추가한다:
  - 전년동월(lag12): 공식기준선 후보
  - 계절평균(seasonal climatology): 해당 예측월 시점까지 관측된 "같은 달"
    실적만의 평균 - 미래정보 누출 없음(walk-forward 내부에서 매 시점 재계산)
  - 선형회귀/Ridge: lag1-3, mean3, month_sin/cos, availability_lag1(또는
    daylight_ratio_lag1) 특성으로 학습하는 단순 선형모델(국내 논문에서
    가장 흔히 설명 가능한 회귀 기준선)

판정 규칙: GB-Huber가 MAE/RMSE/WAPE 세 지표 모두에서 모든 공식기준선보다
개선되고, 계절(상반기/하반기) 분할에서도 어느 한쪽에서만 좋은 게 아니라
양쪽 다 개선일 때만 "승격 검토 후보"로 표시한다. 하나라도 어긋나면
"공식_후보_운영연결보류" 상태 유지.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from lightgbm import LGBMRegressor

PIPE = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
OUT = PIPE / "outputs" / "월간_기준선비교_v1_2026-09-10"


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {
        "n": len(y),
        "WAPE_pct": round(abs(e).sum() / abs(y).sum() * 100, 2),
        "MAE_kWh": round(abs(e).mean(), 1),
        "RMSE_kWh": round(np.sqrt((e ** 2).mean()), 1),
    }


def load_gwangju():
    daily_path = PIPE / "outputs" / "v7_라이브연계_2026-08-26" / "집계_일간_실제발전량_v5.parquet"
    d = pd.read_parquet(daily_path)
    d.index = pd.to_datetime(d.index)
    ycol, acol = "일간발전량_kWh", "가용인버터수_낮시간평균"
    m = d[ycol].resample("MS").agg(total="sum", n="count")
    m["days"] = m.index.days_in_month
    m["y"] = m.total.where(m.n >= m.days)
    m["extra"] = d[acol].resample("MS").mean()  # availability_lag1 대체용 원천
    return m, "availability_lag1"


def load_yeonggwang():
    daily_csv = Path(
        r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
        r"\영광_발전소_일간_공식후보.csv"
    )
    daily = pd.read_csv(daily_csv, parse_dates=["date_kst"]).set_index("date_kst")
    daily.loc[daily["quality_status"] != "valid_daylight_ge90pct", "daily_energy_kwh"] = np.nan
    ycol, ratio_col = "daily_energy_kwh", "daylight_valid_ratio"
    m = daily[ycol].resample("MS").agg(total="sum", n="count")
    m["days"] = m.index.days_in_month
    m["y"] = m.total.where(m.n >= m.days)
    m["extra"] = daily[ratio_col].resample("MS").mean()
    return m, "daylight_ratio_lag1"


def build_features(m: pd.DataFrame, extra_name: str) -> tuple[pd.DataFrame, list[str]]:
    idx = pd.date_range(m.index.min(), m.index.max(), freq="MS")
    m = m.reindex(idx)
    f = pd.DataFrame(index=idx)
    for k in [1, 2, 3, 12]:
        f[f"lag{k}"] = m.y.shift(k)
    f["mean3"] = m.y.shift(1).rolling(3, min_periods=3).mean()
    f["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["days"] = idx.days_in_month
    f[extra_name] = m.extra.shift(1)
    f["y"] = m.y
    f["_month_of_year"] = idx.month
    features_core = ["lag1", "lag2", "lag3", "mean3", "month_sin", "month_cos", "days", extra_name]
    return f, features_core


def seasonal_climatology(f: pd.DataFrame, up_to_idx: int) -> pd.Series:
    """워크포워드 각 시점에서, 그 시점까지 관측된 "같은 달" 실적만으로 평균낸
    계절평균 예측값을 반환한다(미래월 정보 사용 없음)."""
    hist = f.iloc[:up_to_idx]
    by_month = hist.groupby("_month_of_year")["y"].mean()
    return by_month


def run_region(name: str, m: pd.DataFrame, extra_name: str) -> dict:
    f, features_core = build_features(m, extra_name)
    features_linear = features_core  # lag1-3, mean3, month_sin/cos, days, extra
    # lag12(전년동월)는 원본 데이터가 2024-08부터라 최근에야 확보되는
    # 특성이다 - 이걸 전체 모집단 게이트에 넣으면 표본이 11개월로 줄어
    # walk-forward 자체가 불가능해진다(09-10 실측 확인). 그래서 나머지
    # 5개 기준선/후보는 기존과 동일하게 전체 모집단(n=14)으로 비교하고,
    # 전년동월만 "쓸 수 있는 달"에 한해 별도 n으로 따로 보고한다 -
    # 데이터 부족을 감추지 않고 그대로 드러낸다.
    v = f.dropna(subset=features_core + ["y"])
    print(f"[{name}] lag12 포함 특성결측없는 학습가능 월: {len(v)}, "
          f"기간: {v.index.min().date()}~{v.index.max().date()}")

    if len(v) < 12:
        print(f"[{name}] 표본 부족(12개월 미만) - 중단")
        return {}

    initial = max(6, len(v) // 3)
    rows = []
    for i in range(initial, len(v)):
        tr, te = v.iloc[:i], v.iloc[i:i + 1]
        te_month = int(te["_month_of_year"].iloc[0])
        r = {
            "month": str(te.index[0].date()),
            "half": "상반기" if te_month <= 6 else "하반기",
            "actual_kWh": float(te.y.iloc[0]),
            "전월지속성_kWh": float(te.lag1.iloc[0]),
            "전년동월_kWh": float(te.lag12.iloc[0]),
        }
        # 계절평균(해당 시점까지 실적만으로, 같은 달) - te가 v의 i번째이므로
        # f 전체에서 해당 날짜 이전(=tr과 동일 구간)까지만 사용
        f_idx_pos = f.index.get_loc(te.index[0])
        clim = seasonal_climatology(f, f_idx_pos)
        clim_val = clim.get(te_month, np.nan)
        # .get()은 키가 "없을" 때만 기본값을 쓴다 - 키는 있는데 값이 NaN인
        # 경우(해당 달의 유일한 과거표본이 결측월이었던 경우, 예: 광주
        # 2024-08은 불완전월이라 y=NaN)까지 걸러야 한다(09-10 실측으로 발견).
        r["계절평균_kWh"] = float(clim_val) if pd.notna(clim_val) else float(tr.y.mean())

        lr = LinearRegression().fit(tr[features_linear], tr.y)
        r["선형회귀_kWh"] = max(0.0, float(lr.predict(te[features_linear])[0]))

        rg = Ridge(alpha=1.0).fit(tr[features_linear], tr.y)
        r["Ridge_kWh"] = max(0.0, float(rg.predict(te[features_linear])[0]))

        lgbm = LGBMRegressor(
            objective="regression_l1", n_estimators=80, learning_rate=.05,
            num_leaves=7, max_depth=3, min_child_samples=3, reg_lambda=1,
            verbosity=-1, random_state=20260819, n_jobs=2,
        ).fit(tr[features_core], tr.y)
        r["LightGBM_kWh"] = max(0.0, float(lgbm.predict(te[features_core])[0]))

        gbh = GradientBoostingRegressor(
            loss="huber", n_estimators=50, max_depth=2, learning_rate=.04,
            min_samples_leaf=3, random_state=20260819,
        ).fit(tr[features_core], tr.y)
        r["GBHuber_kWh"] = max(0.0, float(gbh.predict(te[features_core])[0]))

        rows.append(r)

    o = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    o.to_csv(OUT / f"{name}_월간_비교_OOF_상세.csv", index=False, encoding="utf-8-sig")

    def metrics_nan_safe(actual, pred):
        # 전년동월(lag12)은 데이터 히스토리가 짧아(2024-08~) 초반 테스트월에서
        # 결측일 수 있다 - 그 행만 빼고 계산하고, 실제 쓰인 n을 그대로 보고한다
        # (결측을 감추지 않고 표본이 줄었다는 사실 자체를 드러낸다).
        a, p = np.asarray(actual, float), np.asarray(pred, float)
        mask = ~np.isnan(p)
        if mask.sum() == 0:
            return {"n": 0, "WAPE_pct": None, "MAE_kWh": None, "RMSE_kWh": None}
        return metrics(a[mask], p[mask])

    model_cols = ["전월지속성", "전년동월", "계절평균", "선형회귀", "Ridge", "LightGBM", "GBHuber"]
    # 참고용 - 각 방법의 "자기 표본"에서의 단독 성능(표본수가 방법마다 다를 수 있음)
    overall = {c: metrics_nan_safe(o.actual_kWh, o[f"{c}_kWh"]) for c in model_cols}
    by_half = {
        half: {c: metrics_nan_safe(sub.actual_kWh, sub[f"{c}_kWh"]) for c in model_cols}
        for half, sub in o.groupby("half")
    }

    def paired_metrics(sub: pd.DataFrame, cand: str, base: str):
        # 승격판정은 "둘 다 유효한 행"만으로 짝지어 비교한다 - 전년동월처럼
        # 표본이 적은 기준선과 비교할 때, GBHuber를 표본이 더 많은 전체
        # 모집단(n=14)에서 계산한 값과 비교하면 서로 다른 달을 비교하는
        # 것이라 불공정하다(09-10 검토 중 발견 - 영광에서 전년동월 표본만
        # 유난히 쉬운 달들로 치우쳤을 가능성을 배제 못 함).
        p1 = sub[f"{cand}_kWh"].to_numpy(dtype=float)
        p2 = sub[f"{base}_kWh"].to_numpy(dtype=float)
        a = sub["actual_kWh"].to_numpy(dtype=float)
        mask = ~np.isnan(p1) & ~np.isnan(p2)
        if mask.sum() == 0:
            return None
        return {"n": int(mask.sum()), cand: metrics(a[mask], p1[mask]), base: metrics(a[mask], p2[mask])}

    MIN_N_FOR_BASELINE = 8  # 짝지은 표본이 이보다 적으면 "반드시 이겨야 할 상대"에서 제외(참고용으로만 표시)
    baselines_all = ["전월지속성", "전년동월", "계절평균", "선형회귀", "Ridge"]
    paired_overall = {b: paired_metrics(o, "GBHuber", b) for b in baselines_all}
    baselines = [b for b in baselines_all if paired_overall[b] and paired_overall[b]["n"] >= MIN_N_FOR_BASELINE]
    excluded_thin = [b for b in baselines_all if b not in baselines]
    gbh_wins_all_metrics_overall = all(
        paired_overall[b]["GBHuber"][met] < paired_overall[b][b][met]
        for b in baselines for met in ["MAE_kWh", "RMSE_kWh", "WAPE_pct"]
    ) if baselines else False

    paired_by_half = {
        half: {b: paired_metrics(sub, "GBHuber", b) for b in baselines_all}
        for half, sub in o.groupby("half")
    }
    gbh_wins_both_halves = all(
        paired_by_half.get(half, {}).get(b) is not None
        and paired_by_half[half][b]["n"] >= max(3, MIN_N_FOR_BASELINE // 2)
        and paired_by_half[half][b]["GBHuber"]["MAE_kWh"] < paired_by_half[half][b][b]["MAE_kWh"]
        for half in paired_by_half for b in baselines
    ) if len(paired_by_half) >= 2 and baselines else False

    verdict = (
        "승격_검토_후보"
        if (gbh_wins_all_metrics_overall and gbh_wins_both_halves)
        else "공식_후보_운영연결보류(기준선 미돌파)"
    )

    result = {
        "region": name,
        "walk_forward_n": len(o),
        "overall_단독표본": overall,
        "by_half_단독표본": by_half,
        "paired_vs_GBHuber_동일표본": paired_overall,
        "paired_by_half_vs_GBHuber_동일표본": paired_by_half,
        "baselines_used_for_verdict": baselines,
        "baselines_excluded_thin_sample": excluded_thin,
        "gbh_wins_all_metrics_overall": gbh_wins_all_metrics_overall,
        "gbh_wins_both_halves": gbh_wins_both_halves,
        "판정": verdict,
        "판정근거": (
            f"GB-Huber가 {'·'.join(baselines) if baselines else '(비교가능 기준선 없음)'} "
            f"공식기준선 전부를 MAE·RMSE·WAPE 3개 지표 및 상/하반기 양쪽에서 이겨야 승격검토. "
            + (f"전년동월 등 {'·'.join(excluded_thin)}은 표본이 {MIN_N_FOR_BASELINE}개월 미만이라 "
               f"참고용으로만 표시하고 승격판정 기준에서는 제외함. " if excluded_thin else "")
            + "국내 문헌에 GB-Huber 자체 사례는 없음 - 국내 시계열검증 원칙(순차검증, "
            "결측월 제외)은 따르되 강건손실(Huber)은 프로젝트 자체 비교실험으로 "
            "도입했다는 것을 명시."
        ),
    }
    (OUT / f"{name}_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{name}] 판정: {verdict}")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    m_gj, extra_gj = load_gwangju()
    m_yg, extra_yg = load_yeonggwang()
    r_gj = run_region("광주", m_gj, extra_gj)
    r_yg = run_region("영광", m_yg, extra_yg)
    print("\n=== 요약 ===")
    for r in (r_gj, r_yg):
        if r:
            print(r["region"], "->", r["판정"])


if __name__ == "__main__":
    main()
