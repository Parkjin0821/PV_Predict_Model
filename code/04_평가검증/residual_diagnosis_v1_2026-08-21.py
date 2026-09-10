# -*- coding: utf-8 -*-
"""② 현재 확정모델 잔차 진단 — 로드맵 08-22 2단계.

## 목적
①(일간 특성거버넌스)에서 "가을 폴드만 두 모델 다 악화"라는 이상신호가
나왔다. 하이퍼파라미터 튜닝(④)에 들어가기 전에, **확정된 3개 티어
(초단기 LightGBM, 단기 XGBoost, 일간 계층조정)가 계절·시간대·구름
전이 구간 중 정확히 어디서 약한지** 먼저 알아야 어떤 개선이 실제로
필요한지 판단할 수 있다(사용자 확정 순서).

## 방법
1. **단기(XGBoost)·초단기(LightGBM)**: 공식 성능표를 만들 때 쓴 것과
   **완전히 동일한 설정**(v3 고정-tm, 전체후보, 폴드 내 재선택+다중공선성
   +배포필터)으로 다시 학습하되, 이번엔 **행 단위 예측값을 저장**한다
   (기존 하네스/ultra 스크립트는 폴드 집계지표만 남기고 행 단위는
   버렸음 — 진단을 위해 이번에만 남긴다, 채점 로직 자체는 안 건드림).
2. **일간(계층조정)**: 이미 저장된 `일간_계층조정_v2_2026-08-21/
   OOF_상세.csv`를 그대로 재사용한다(재학습 불필요, 이미 행 단위 있음).
3. 구름전이 프록시: 기상청 관측 전운량(전운량_pct)의 **시간당 변화량
   절대값**을 3분위(안정/보통/전이)로 나눠, 각 예측 시점(발행시각 또는
   목표시각 직전)이 어느 구간에 속하는지 매긴다. "전이" 구간은 맑음↔
   흐림이 빠르게 바뀌는 시점이라 예측이 어려울 것으로 예상되는 구간.
4. 계절(폴드)·시간대(목표시각의 시)·구름전이구간별로 MAE·RMSE를
   집계해 어디가 약한지 표로 낸다.

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\04_평가검증"
python residual_diagnosis_v1_2026-08-21.py
```
초단기·단기 재학습이 포함돼 있어 수 분 정도 걸린다(위성 API 호출은
전혀 없음 — 로컬 재학습만).

## 산출물
`outputs/잔차진단_v1_2026-08-21/` 아래:
- `단기_행단위.csv`, `초단기_행단위.csv` (개별 예측·실제·오차·계절·시간대·구름전이구간)
- `단기_계절별.csv`, `단기_시간대별.csv`, `단기_구름전이별.csv` (+ 초단기·일간 동일 패턴)
- `요약.txt` (어느 구간이 가장 약한지 사람이 읽을 요약)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 08-21 수정: 콘솔 codepage(cp949 등)가 한글/em-dash를 못 받아 출력이
# 깨지거나(Windows 콘솔) 크래시하는(파일 리다이렉트) 문제가 있었다.
# 실행 환경과 무관하게 항상 UTF-8로 표준출력하도록 고정한다.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1] / "03_모델학습" / "현재_종합파이프라인"

_spec_h = importlib.util.spec_from_file_location("harness", ROOT / "backtest_harness_v1_2026-08-20밤.py")
harness = importlib.util.module_from_spec(_spec_h)
sys.modules["harness"] = harness
_spec_h.loader.exec_module(harness)

_spec_u = importlib.util.spec_from_file_location("ultra", ROOT / "train_ultra_short_official_v1_2026-08-21.py")
ultra = importlib.util.module_from_spec(_spec_u)
sys.modules["ultra"] = ultra
_spec_u.loader.exec_module(ultra)

OUT = Path(__file__).resolve().parent / "outputs" / "잔차진단_v1_2026-08-21"
DAILY_OOF_CSV = ROOT / "outputs" / "일간_계층조정_v2_2026-08-21" / "OOF_상세.csv"

SHORT_HORIZONS = [1, 24, 48]
ULTRA_HORIZONS = [1, 2, 3, 4]


def build_cloud_transition(df: pd.DataFrame) -> pd.Series:
    """시간당 전운량 변화량 절대값을 3분위(안정/보통/전이)로 나눈다."""
    change = df["기상청관측_전운량_pct"].diff().abs()
    q1, q2 = change.quantile([1 / 3, 2 / 3])
    bucket = pd.cut(change, bins=[-0.01, q1, q2, 100], labels=["안정", "보통", "전이"])
    return bucket


def short_term_rows(df: pd.DataFrame, cloud_bucket: pd.Series, config, capacity_kw: float, seed: int) -> pd.DataFrame:
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    rows = []
    for H in SHORT_HORIZONS:
        frame = harness.build_frame(df, H, candidate_cols)
        daylight = frame[frame["목표_낮시간"] > 0]
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        for i, w in enumerate(config["cross_validation_windows"], start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            fold_name = f"{i}_{w.get('_계절','')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            if len(train_all) < 200 or len(test_all) < 30:
                continue
            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(
                tr_for_sel, candidate_cols, threshold=0.3, apply_multicollinearity=True, apply_deploy_filter=True,
            )
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue

            model = harness.make_model("XGBoost", seed)  # 단기 공식 채택 모델
            model.fit(train[feature_cols], train["목표_발전출력_kW"])
            pred = np.clip(model.predict(test[feature_cols]), 0, capacity_kw)
            target_time = test.index + pd.to_timedelta(H - 1, unit="h")

            fold_df = pd.DataFrame({
                "발행시각": test.index, "목표시각": target_time,
                "실제_kW": test["목표_발전출력_kW"].to_numpy(), "예측_kW": pred,
                "수평_h": H, "폴드": fold_name,
            })
            fold_df["오차"] = fold_df["실제_kW"] - fold_df["예측_kW"]
            fold_df["구름전이구간"] = cloud_bucket.reindex(fold_df["발행시각"]).to_numpy()
            fold_df["시간대"] = fold_df["목표시각"].dt.hour
            rows.append(fold_df)
            print(f"  [단기 +{H}h {fold_name}] n={len(fold_df)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ultra_short_rows(quarter: pd.DataFrame, hourly_df: pd.DataFrame, cloud_bucket: pd.Series,
                      config, capacity_kw: float, seed: int) -> pd.DataFrame:
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS
    rows = []
    for H in ULTRA_HORIZONS:
        frame = ultra.build_ultra_short_frame(quarter, hourly_df, H)
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        daylight = frame[frame["목표_낮시간"] > 0]
        for i, w in enumerate(config["cross_validation_windows"], start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            fold_name = f"{i}_{w.get('_계절','')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            if len(train_all) < 500 or len(test_all) < 100:
                continue
            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(
                tr_for_sel, candidate_cols, threshold=0.3, apply_multicollinearity=True, apply_deploy_filter=True,
            )
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + [
                "목표_발전출력_kW", "_지속성_직전출력_kW"
            ]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 500 or len(test) < 100:
                continue

            model = ultra.make_model("LightGBM", seed)  # 초단기 공식 채택 모델
            model.fit(train[feature_cols], train["목표_발전출력_kW"])
            pred = np.clip(model.predict(test[feature_cols]), 0, capacity_kw)
            target_time = test.index + pd.to_timedelta((H * 4 - 1) * 15, unit="min")

            fold_df = pd.DataFrame({
                "발행시각": test.index, "목표시각": target_time,
                "실제_kW": test["목표_발전출력_kW"].to_numpy(), "예측_kW": pred,
                "수평_h": H, "폴드": fold_name,
            })
            fold_df["오차"] = fold_df["실제_kW"] - fold_df["예측_kW"]
            issue_hour = fold_df["발행시각"].dt.floor("h")
            fold_df["구름전이구간"] = cloud_bucket.reindex(issue_hour).to_numpy()
            fold_df["시간대"] = fold_df["목표시각"].dt.hour
            rows.append(fold_df)
            print(f"  [초단기 +{H}h {fold_name}] n={len(fold_df)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(df: pd.DataFrame, group_col: str, label: str, unit: str = "kW") -> pd.DataFrame:
    """08-21 수정: 티어마다 단위가 다르다(초단기·단기=kW, 일간=kWh) —
    열 이름에 단위를 그대로 반영해 뒤섞이지 않게 한다. 축 이름(어떤
    기준으로 묶었는지)도 같이 남겨 요약.txt에서 축별로 분리할 수 있게
    한다."""
    g = df.groupby(group_col)["오차"].agg(
        표본수="size", **{f"MAE_{unit}": lambda s: s.abs().mean(), f"RMSE_{unit}": lambda s: np.sqrt((s ** 2).mean())},
    ).reset_index().rename(columns={group_col: "구간값"})
    g.insert(0, "축", group_col)
    g.insert(0, "티어", label)
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    cloud_bucket = build_cloud_transition(df)

    print("=== 단기(XGBoost) 재학습 + 행 단위 저장 ===")
    short_df = short_term_rows(df, cloud_bucket, config, capacity_kw, seed)
    short_df.to_csv(OUT / "단기_행단위.csv", index=False, encoding="utf-8-sig")

    print("\n=== 초단기(LightGBM) 재학습 + 행 단위 저장 ===")
    quarter = ultra.load_15min_base()
    ultra_df = ultra_short_rows(quarter, df, cloud_bucket, config, capacity_kw, seed)
    ultra_df.to_csv(OUT / "초단기_행단위.csv", index=False, encoding="utf-8-sig")

    print("\n=== 일간(계층조정) 기존 OOF 재사용 ===")
    daily_oof = pd.read_csv(DAILY_OOF_CSV, parse_dates=["날짜"], encoding="utf-8-sig")
    daily_oof["오차"] = daily_oof["실제_일간총량_kWh"] - daily_oof["계층조정_kWh"]
    daily_oof["월"] = daily_oof["날짜"].dt.month
    daily_cloud_daily_mean = df["기상청관측_전운량_pct"].resample("D").mean()
    daily_change = daily_cloud_daily_mean.diff().abs()
    q1, q2 = daily_change.quantile([1 / 3, 2 / 3])
    daily_oof["구름전이구간"] = pd.cut(
        daily_change.reindex(daily_oof["날짜"]).to_numpy(), bins=[-0.01, q1, q2, 100], labels=["안정", "보통", "전이"]
    )
    daily_oof.to_csv(OUT / "일간_행단위.csv", index=False, encoding="utf-8-sig")

    tables = []
    if len(short_df):
        tables += [summarize(short_df, "폴드", "단기", "kW"), summarize(short_df, "시간대", "단기", "kW"),
                   summarize(short_df, "구름전이구간", "단기", "kW"), summarize(short_df, "수평_h", "단기", "kW")]
    if len(ultra_df):
        tables += [summarize(ultra_df, "폴드", "초단기", "kW"), summarize(ultra_df, "시간대", "초단기", "kW"),
                   summarize(ultra_df, "구름전이구간", "초단기", "kW"), summarize(ultra_df, "수평_h", "초단기", "kW")]
    tables += [
        summarize(daily_oof, "폴드", "일간", "kWh"), summarize(daily_oof, "월", "일간", "kWh"),
        summarize(daily_oof, "구름전이구간", "일간", "kWh"),
    ]
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT / "잔차요약_전체.csv", index=False, encoding="utf-8-sig")

    # ★08-21 수정★: 전에는 티어 안에서 축(계절/시간대/구름전이/수평)을
    # 전부 섞어 MAE 기준 top5를 뽑았는데, 시간대(hour)처럼 원래 절대오차
    # 스케일이 큰 축이 항상 상위를 차지해 계절·구름전이 결과가 묻혔다.
    # 축별로 완전히 분리해서 각자 전체를 보여준다.
    lines = ["=== 잔차 진단 요약(티어×축별 분리) ===\n"]
    for tier in combined["티어"].unique():
        lines.append(f"[{tier}]")
        for axis in combined.loc[combined["티어"] == tier, "축"].unique():
            sub = combined[(combined["티어"] == tier) & (combined["축"] == axis)]
            mae_col = [c for c in sub.columns if c.startswith("MAE_")][0]
            lines.append(f"  -- 축: {axis} --")
            lines.append(sub.drop(columns=["티어", "축"]).sort_values(mae_col, ascending=False).to_string(index=False))
        lines.append("")
    summary_text = "\n".join(lines)
    (OUT / "요약.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
