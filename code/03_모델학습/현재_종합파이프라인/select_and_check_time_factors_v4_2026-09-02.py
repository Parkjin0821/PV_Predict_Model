# -*- coding: utf-8 -*-
"""5번+5-2번 재실행(★4차, 09-02 밤★) — 시간요인을 "무조건 포함 기준선"에서
빼고 환경·설비요인과 동등한 후보로 상관분석·다중공선성 검증에 포함한다.

## 왜 다시 하는가(사용자 지적, 08-18 확정 스펙 미이행 발견)
08-18에 사용자가 확정한 "공식 4대 예측요인" 문서(AGENTS.md)에는 시간요인
기본항목이 "일·월·시간·연"(+파생: 요일·계절·태양고도·태양방위·예측수평)
으로 명시돼 있었다. 그런데 실제 v1~v3 스크립트의 `make_base_features()`는
과거발전량 lag/rolling과 함께 태양고도·일주기_sin/cos·연주기_sin/cos만
"고정 기준선"(상관분석·다중공선성 검증 대상에서 제외)으로 박아넣었고,
**월·일·시간·연 원본값은 아예 만든 적도 없다** - 08-18 확정 스펙이
제대로 구현 안 된 채로 환경요인만 별도 상관분석이 진행됐던 것.

## 이번 수정
- **진짜 고정 기준선은 과거발전량(lag/rolling)만** - 08-18 문서의
  "과거발전량+시간 기준모델에 환경→지형→설비 단계적 추가" 표현에서
  "과거발전량"과 "시간"을 하나로 묶어 자동으로 기준선 취급한 게 08-20의
  실수였다고 판단(문서상 시간요인은 4대 요인 중 하나로 명시돼 있었지
  기준선 취급을 명시한 적 없음).
- **시간요인 12개를 신규 후보로 추가**: 연·월·일·시간(원본) + 요일·계절
  (파생) + 태양고도·태양방위(파생, `solar_elevation_deg`/`solar_azimuth_deg`
  - 08-19 데이터셋에 이미 존재하나 후보 목록에 누락돼 있었음) + 일주기/
  연주기 sin·cos(파생 주기성, 기존엔 기준선에 있었으나 이번엔 후보로 강등).
  예측수평은 이 스크립트가 +1시간 단일수평만 다루므로 행별로 변하지
  않아 상관분석 후보에서 제외(다수평 비교 시 별도 다룰 것).
- 기존 환경·설비 후보(`select_features_by_correlation_threshold_v3`의
  STARTLABEL+OBSERVED+FORECAST 36개)는 재구현 없이 그대로 재사용.
- **선형(Pearson) 상관계수의 한계 명시**: 시간(0~23시)·월(1~12) 같은
  순환형 변수는 실제 관계가 U자형/비선형이라 Pearson |r|이 과소평가할
  수 있다(예: 새벽 0시와 밤 23시는 값은 멀지만 실제로는 인접). 그래서
  상관계수 임계값 필터만 보지 않고, **①선택 여부와 무관하게 전부 넣은
  ablation, ②최종 가지치기 결과** 두 가지를 실제 재학습으로 같이
  비교해 선형 필터가 비선형 중요변수를 놓쳤는지 확인한다.

## 사용법
```
python select_and_check_time_factors_v4_2026-09-02.py
```
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sel_v3", ROOT / "select_features_by_correlation_threshold_v3_2026-08-20밤.py")
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

OUT_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\시간요인포함_재검증_v4_2026-09-02")
SEED = 42
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
VIF_SEVERE, PAIR_CORR_HIGH = 10.0, 0.8

# 08-18 확정 스펙 그대로: 기본항목(연·월·일·시간) + 파생(요일·계절·
# 태양고도·태양방위·주기성). 예측수평은 단일수평 분석이라 후보에서 제외.
TIME_FACTOR_COLUMNS = [
    "연도", "월", "일", "시간",              # 08-18 "기본항목"
    "요일", "계절",                          # 08-18 "파생 활용항목" 일부
    "태양고도_deg", "태양방위_deg",           # 08-18 "파생 활용항목" 일부
    "일주기_sin", "일주기_cos", "연주기_sin", "연주기_cos",  # 파생 주기성
]

# 08-18 문서상 "과거발전량+시간" 중 진짜 기준선은 과거발전량뿐이라고
# 판단(위 docstring 참고) - 시간요인은 전부 위 TIME_FACTOR_COLUMNS로
# 후보에 넣고, 여기 남는 건 자기회귀(lag/rolling)뿐이다.
def make_true_baseline(df: pd.DataFrame) -> pd.DataFrame:
    power = df["plant_output_kw"].shift(1)
    out = pd.DataFrame(index=df.index)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours - 1)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).std()
    return out


def month_to_season(month: int) -> int:
    # 기상청 관행 4계절 구분(12~2 겨울, 3~5 봄, 6~8 여름, 9~11 가을)
    if month in (12, 1, 2):
        return 1  # 겨울
    if month in (3, 4, 5):
        return 2  # 봄
    if month in (6, 7, 8):
        return 3  # 여름
    return 4  # 가을


def build_time_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)
    out["연도"] = idx.year
    out["월"] = idx.month
    out["일"] = idx.day
    out["시간"] = idx.hour
    out["요일"] = idx.dayofweek
    out["계절"] = [month_to_season(m) for m in idx.month]
    # 08-19 공식 데이터셋에 이미 계산돼 있던 값 재사용(재구현 아님) -
    # pv_pipeline.solar_position()과 같은 공식으로 만들어진 컬럼.
    out["태양고도_deg"] = df["solar_elevation_deg"]
    out["태양방위_deg"] = df["solar_azimuth_deg"]
    minute = idx.hour * 60 + idx.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out


def compute_correlations(cand: pd.DataFrame, all_cols: list[str], target: pd.Series,
                          daylight: pd.Series) -> pd.DataFrame:
    mask = daylight > 0
    rows = []
    for col in all_cols:
        sub = pd.concat([cand.loc[mask, col], target[mask]], axis=1).dropna()
        if len(sub) < 30:
            r, n = np.nan, len(sub)
        else:
            r = float(np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1])
            n = len(sub)
        rows.append({"변수": col, "분류": "시간요인" if col in TIME_FACTOR_COLUMNS else "환경·설비요인",
                     "상관계수": r, "표본수": n, "절대값": abs(r) if pd.notna(r) else np.nan})
    return pd.DataFrame(rows).sort_values("절대값", ascending=False)


def evaluate_threshold(threshold: float, corr: pd.DataFrame, baseline: pd.DataFrame,
                        cand: pd.DataFrame, target: pd.Series, daylight: pd.Series,
                        capacity_kw: float) -> dict:
    selected_cols = corr.loc[corr["절대값"] >= threshold, "변수"].tolist()
    frame = baseline.copy()
    for col in selected_cols:
        frame[col] = cand[col]
    frame["목표_발전출력_kW"] = target
    frame["목표_낮시간"] = daylight

    features = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
    frame = frame.dropna(subset=features + ["목표_발전출력_kW"])
    frame = frame[frame["목표_낮시간"] > 0]

    n = len(frame)
    cut = int(n * 0.85)
    train, test = frame.iloc[:cut], frame.iloc[cut:]

    model = LGBMRegressor(
        n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3,
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    model.fit(train[features], train["목표_발전출력_kW"])
    pred = np.clip(model.predict(test[features]), 0, capacity_kw)
    err = test["목표_발전출력_kW"].to_numpy() - pred
    time_selected = [c for c in selected_cols if c in TIME_FACTOR_COLUMNS]
    return {
        "임계값": threshold, "선택된_후보수": len(selected_cols),
        "선택된_시간요인": time_selected, "선택된_시간요인_수": len(time_selected),
        "선택된_후보": selected_cols,
        "전체_특성수": len(features), "학습표본수": int(len(train)), "시험표본수": int(len(test)),
        "MAE_kW": float(np.abs(err).mean()), "RMSE_kW": float(np.sqrt((err ** 2).mean())),
    }


def compute_vif(x: pd.DataFrame) -> pd.Series:
    x = (x - x.mean()) / x.std(ddof=0)
    vifs = {}
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def retrain_eval(name: str, baseline: pd.DataFrame, cols: list[str], cand: pd.DataFrame,
                  target: pd.Series, daylight: pd.Series, capacity_kw: float) -> dict:
    frame = baseline.copy()
    for c in cols:
        frame[c] = cand[c]
    frame["목표_발전출력_kW"] = target
    frame["목표_낮시간"] = daylight
    features = [c for c in frame.columns if c not in ("목표_발전출력_kW", "목표_낮시간")]
    f = frame.dropna(subset=features + ["목표_발전출력_kW"])
    f = f[f["목표_낮시간"] > 0]
    n = len(f)
    cut = int(n * 0.85)
    train, test = f.iloc[:cut], f.iloc[cut:]
    model = LGBMRegressor(n_estimators=220, learning_rate=.04, num_leaves=31, min_child_samples=30,
                           subsample=.9, colsample_bytree=.9, reg_lambda=.3, random_state=SEED,
                           n_jobs=4, verbosity=-1)
    model.fit(train[features], train["목표_발전출력_kW"])
    pred = np.clip(model.predict(test[features]), 0, capacity_kw)
    e = test["목표_발전출력_kW"].to_numpy() - pred
    return dict(구성=name, 특성수=len(features), 학습=len(train), 시험=len(test),
                MAE_kW=round(float(np.abs(e).mean()), 3), RMSE_kW=round(float(np.sqrt((e ** 2).mean())), 3))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    capacity_kw = float(df["reported_capacity_kw"].dropna().iloc[0])

    env_equip_cand = sel.build_candidate_frame(df)   # 재구현 없음(기존 v3 그대로)
    time_cand = build_time_factor_frame(df)
    cand = pd.concat([env_equip_cand, time_cand], axis=1)
    all_cols = sel.CANDIDATE_COLUMNS + TIME_FACTOR_COLUMNS

    target = df["plant_output_kw"]
    daylight = (df["solar_elevation_deg"] > 0).astype(float)
    baseline = make_true_baseline(df)   # 과거발전량만(시간요인은 전부 후보로 강등)

    print(f"[0] 후보 총 {len(all_cols)}개 (환경·설비 {len(sel.CANDIDATE_COLUMNS)} + 시간요인 {len(TIME_FACTOR_COLUMNS)})")
    print(f"    진짜 고정 기준선(과거발전량만): {list(baseline.columns)}")

    print("\n[1/4] Pearson 상관계수 계산(환경·설비 + 시간요인 동일 취급)...")
    corr = compute_correlations(cand, all_cols, target, daylight)
    corr.to_csv(OUT_DIR / "변수별_상관계수_전체.csv", index=False, encoding="utf-8-sig")
    print(corr.to_string(index=False))

    print("\n[2/4] 임계값별 ablation 학습 중(시간요인 포함 후보군)...")
    results = [evaluate_threshold(t, corr, baseline, cand, target, daylight, capacity_kw) for t in THRESHOLDS]
    scorecard = pd.DataFrame([{k: v for k, v in r.items() if k not in ("선택된_후보", "선택된_시간요인")} for r in results])
    print(scorecard.to_string(index=False))
    best = min(results, key=lambda r: r["RMSE_kW"])
    print(f"\n최적 임계값(RMSE 기준): {best['임계값']} (RMSE={best['RMSE_kW']:.3f}, MAE={best['MAE_kW']:.3f})")
    print(f"채택된 시간요인({best['선택된_시간요인_수']}개): {best['선택된_시간요인']}")
    print(f"채택된 전체 후보({best['선택된_후보수']}개): {best['선택된_후보']}")

    print("\n[3/4] 최적 임계값 선택셋 다중공선성(VIF+쌍상관) 점검...")
    selected = best["선택된_후보"]
    daylight_cand = cand[daylight > 0]
    x = daylight_cand[selected].dropna()
    print(f"VIF 계산용 완전표본: {len(x)}")
    vif = compute_vif(x)
    print(vif.to_string())

    target_corr = corr.set_index("변수")["상관계수"]
    corr_matrix = x.corr(method="pearson")
    high_pairs = []
    cols = corr_matrix.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr_matrix.loc[a, b]
            if abs(r) >= PAIR_CORR_HIGH:
                weaker = a if abs(target_corr.get(a, 0)) < abs(target_corr.get(b, 0)) else b
                high_pairs.append({"변수A": a, "변수B": b, "쌍상관계수": float(r), "제거후보": weaker})
    pairs_df = pd.DataFrame(high_pairs).sort_values("쌍상관계수", key=abs, ascending=False) if high_pairs else pd.DataFrame()
    print("|r|>=0.8 변수쌍:"); print(pairs_df.to_string(index=False) if len(pairs_df) else "(없음)")

    drop_candidates = sorted(set(pairs_df["제거후보"])) if len(pairs_df) else []
    severe_vif = set(vif[vif >= VIF_SEVERE].index)
    final_drop = [c for c in drop_candidates if c in severe_vif]
    pruned = [c for c in selected if c not in final_drop]
    print(f"\n최종 가지치기(쌍상관>=0.8 & VIF>=10 둘 다): {final_drop}")
    print(f"가지치기 후 최종({len(pruned)}개): {pruned}")

    print("\n[4/4] 실제 재학습 비교(구 벤치마크 vs 이번 결과)...")
    OLD_14 = [
        "plant_input_power_kw", "기상청관측_일사량_W_m2", "mean_power_factor",
        "기상청관측_일조시간_hr", "DSWRFLX_bsrn정제", "mean_input_voltage_v", "REH",
        "기상청관측_전운량_pct", "추정_출력온도", "POP", "TCDC", "LCDC",
        "기상청관측_상대습도_pct", "SKY",
    ]
    comparisons = [
        retrain_eval("A_구벤치마크(환경설비14개만, 시간요인=구기준선)",
                      sel.make_base_features(df), OLD_14, cand, target, daylight, capacity_kw),
        retrain_eval("B_이번_선택셋(가지치기 전, 시간요인 포함 후보군)",
                      baseline, selected, cand, target, daylight, capacity_kw),
        retrain_eval("C_이번_최종(가지치기 후)",
                      baseline, pruned, cand, target, daylight, capacity_kw),
        retrain_eval("D_전체후보_전부포함(필터없음, 상한선 참고용)",
                      baseline, all_cols, cand, target, daylight, capacity_kw),
    ]
    comp_df = pd.DataFrame(comparisons)
    print(comp_df.to_string(index=False))

    corr.to_csv(OUT_DIR / "변수별_상관계수_전체.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(OUT_DIR / "임계값별_성능표.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(OUT_DIR / "VIF.csv", header=["VIF"], encoding="utf-8-sig")
    pairs_df.to_csv(OUT_DIR / "강한상관쌍.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(OUT_DIR / "최종성능비교.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "요약.json").write_text(json.dumps({
        "최적임계값": best["임계값"], "선택된_후보": selected, "선택된_시간요인": best["선택된_시간요인"],
        "가지치기": final_drop, "최종": pruned, "성능비교": comparisons,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
