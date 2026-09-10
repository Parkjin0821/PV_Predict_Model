# -*- coding: utf-8 -*-
"""③ raw kW vs 청천지수 정규화 비교 — 로드맵 08-21 3단계, 초단기 우선.

## 왜 초단기부터, 그리고 왜 이 비교인가 (②잔차진단 직결)
②잔차진단에서 **초단기(15분,+1~4h)만 구름전이 구간에서 크게 취약**
(전이 14.91kW vs 안정 11.80kW, +26%)하다는 게 확인됐다. 단기·일간은
거의 영향이 없었다. 청천지수 정규화가 물리적으로 가장 도움될 상황이
정확히 "구름이 빠르게 바뀌는 순간"이므로, **초단기·구름전이 구간에
집중해서** raw kW 타깃 방식과 청천지수(κ=실제/이론청천) 정규화 타깃
방식을 직접 비교한다.

## 청천지수 계산 (미래 ASOS 관측 사용 금지 — 순수 천문학 공식만)
- **Haurwitz 모델**(1945, 대기측정 불필요·태양고도만으로 계산되는
  표준 청천일사량 공식, 널리 인용됨):
  `GHI_clear = 1098 × sin(θ) × exp(-0.059 / sin(θ))` (θ=태양고도, W/m²)
- **청천 발전량 프록시**: `청천_kW = 설비용량_kW × GHI_clear / 1000`
  (KS 표준시험조건 1000W/m²에서 정격용량이 나온다는 통상 가정 — 패널
  변환효율 곡선까지는 모델링 안 함, 근사).
- **청천지수**: `κ = 목표_발전출력_kW / 청천_kW`, 계산 자체가 **태양
  기하학(위치+시각)만으로 발행시점에 이미 다 계산 가능** — 미래 관측을
  전혀 안 쓰므로 정보누출 없음.
- 저고도(태양고도<5°)에서 분모가 0에 가까워 κ가 불안정해지므로, 이
  구간은 두 방식 공정 비교에서 제외한다(원래도 발전량 자체가 미미).

## 학습·비교 방법
- 기존 공식 초단기 파이프라인(`train_ultra_short_official_v1`)과
  **완전히 동일한 특성 후보·폴드 내 재선택(상관계수+다중공선성+배포
  필터)**을 그대로 재사용 — 타깃만 raw kW ↔ κ로 바꾼다.
- κ 모델은 예측된 κ에 청천_kW를 다시 곱해 kW로 역정규화한 뒤 raw kW
  모델과 **완전히 동일한 시험행에서** 비교한다.
- 전체 성능뿐 아니라 ②에서 쓴 것과 동일한 구름전이구간(안정/보통/전이,
  발행시각 기준 전운량 시간당 변화량 3분위)별로 나눠 **"전이 구간에서
  특히 도움되는가"**를 직접 확인한다 — 이게 이번 비교의 핵심 질문.

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python ultra_short_clearsky_v1_2026-08-21.py
```
API 호출 없음(로컬 재학습만, v3 데이터 재사용). 초단기 재학습 4개
수평×5폴드×2방식이라 수 분 정도 걸린다.

## 산출물
`outputs/초단기_청천지수비교_v1_2026-08-21/`
- `행단위_비교.csv` (발행시각·수평·폴드·구름전이구간·실제·raw예측·κ예측)
- `요약_전체.csv`, `요약_구름전이별.csv`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

_spec_h = importlib.util.spec_from_file_location("harness", ROOT / "backtest_harness_v1_2026-08-20밤.py")
harness = importlib.util.module_from_spec(_spec_h)
sys.modules["harness"] = harness
_spec_h.loader.exec_module(harness)

_spec_u = importlib.util.spec_from_file_location("ultra", ROOT / "train_ultra_short_official_v1_2026-08-21.py")
ultra = importlib.util.module_from_spec(_spec_u)
sys.modules["ultra"] = ultra
_spec_u.loader.exec_module(ultra)

OUT = ROOT / "outputs" / "초단기_청천지수비교_v1_2026-08-21"
HORIZONS = [1, 2, 3, 4]
MIN_ELEVATION_DEG = 5.0  # 이 미만은 청천지수 분모 불안정 — 공정비교에서 제외


def clear_sky_ghi(elevation_deg: np.ndarray) -> np.ndarray:
    """Haurwitz(1945) 청천일사량 모델. 태양고도만으로 계산, 관측 불필요."""
    theta = np.deg2rad(np.clip(elevation_deg, 0.01, 90))
    sin_t = np.sin(theta)
    ghi = 1098.0 * sin_t * np.exp(-0.059 / sin_t)
    return np.clip(ghi, 0, None)


def build_cloud_transition(df: pd.DataFrame) -> pd.Series:
    """②잔차진단과 동일 정의(시간당 전운량 변화량 절대값 3분위)."""
    change = df["기상청관측_전운량_pct"].diff().abs()
    q1, q2 = change.quantile([1 / 3, 2 / 3])
    return pd.cut(change, bins=[-0.01, q1, q2, 100], labels=["안정", "보통", "전이"])


def run_horizon(H: int, quarter: pd.DataFrame, hourly_df: pd.DataFrame, cloud_bucket: pd.Series,
                 windows: list[dict], capacity_kw: float, seed: int) -> list[dict]:
    frame = ultra.build_ultra_short_frame(quarter, hourly_df, H)
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    clear_sky_kw = capacity_kw * clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0
    frame["_청천_kW"] = np.clip(clear_sky_kw, 1e-3, None)  # 0나눗셈 방지
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]

    daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= MIN_ELEVATION_DEG)]

    rows = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절','')}"
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
        if len(train_all) < 500 or len(test_all) < 100:
            print(f"  [{fold_name}] 표본 부족으로 건너뜀")
            continue

        tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
        chosen = harness.select_features_in_fold(
            tr_for_sel, candidate_cols, threshold=0.3, apply_multicollinearity=True, apply_deploy_filter=True,
        )
        feature_cols = base_cols + chosen
        required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + [
            "목표_발전출력_kW", "_카파", "_청천_kW"
        ]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=required)
        if len(train) < 500 or len(test) < 100:
            print(f"  [{fold_name}] 표본 부족으로 건너뜀")
            continue

        y_test_kw = test["목표_발전출력_kW"].to_numpy()
        clear_kw_test = test["_청천_kW"].to_numpy()

        model_raw = ultra.make_model("LightGBM", seed)
        model_raw.fit(train[feature_cols], train["목표_발전출력_kW"])
        pred_raw = np.clip(model_raw.predict(test[feature_cols]), 0, capacity_kw)

        model_kappa = ultra.make_model("LightGBM", seed)
        model_kappa.fit(train[feature_cols], train["_카파"])
        pred_kappa_raw = model_kappa.predict(test[feature_cols])
        pred_kappa_kw = np.clip(pred_kappa_raw * clear_kw_test, 0, capacity_kw)

        fold_df = pd.DataFrame({
            "발행시각": test.index, "수평_h": H, "폴드": fold_name,
            "실제_kW": y_test_kw, "raw예측_kW": pred_raw, "카파예측_kW": pred_kappa_kw,
        })
        fold_df["오차_raw"] = fold_df["실제_kW"] - fold_df["raw예측_kW"]
        fold_df["오차_카파"] = fold_df["실제_kW"] - fold_df["카파예측_kW"]
        issue_hour = fold_df["발행시각"].dt.floor("h")
        fold_df["구름전이구간"] = cloud_bucket.reindex(issue_hour).to_numpy()
        rows.append(fold_df)

        mae_raw = fold_df["오차_raw"].abs().mean()
        mae_kappa = fold_df["오차_카파"].abs().mean()
        print(f"  [{fold_name}] +{H}h n={len(fold_df)} MAE raw={mae_raw:.2f} 카파={mae_kappa:.2f} "
              f"({'카파 개선' if mae_kappa < mae_raw else 'raw 우세'})")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    hourly_df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    cloud_bucket = build_cloud_transition(hourly_df)
    quarter = ultra.load_15min_base()

    all_rows = []
    for H in HORIZONS:
        print(f"\n=== +{H}h ===")
        all_rows += run_horizon(H, quarter, hourly_df, cloud_bucket, config["cross_validation_windows"], capacity_kw, seed)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(OUT / "행단위_비교.csv", index=False, encoding="utf-8-sig")

    def summary_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        g = df.groupby(group_cols).agg(
            표본수=("오차_raw", "size"),
            MAE_raw=("오차_raw", lambda s: s.abs().mean()),
            MAE_카파=("오차_카파", lambda s: s.abs().mean()),
            RMSE_raw=("오차_raw", lambda s: np.sqrt((s ** 2).mean())),
            RMSE_카파=("오차_카파", lambda s: np.sqrt((s ** 2).mean())),
        ).reset_index()
        g["MAE_개선율_pct"] = (1 - g["MAE_카파"] / g["MAE_raw"]) * 100
        return g

    if len(combined):
        overall = summary_table(combined, ["수평_h"])
        overall.to_csv(OUT / "요약_전체.csv", index=False, encoding="utf-8-sig")
        by_cloud = summary_table(combined, ["수평_h", "구름전이구간"])
        by_cloud.to_csv(OUT / "요약_구름전이별.csv", index=False, encoding="utf-8-sig")

        print("\n=== 수평별 전체 요약 ===")
        print(overall.to_string(index=False))
        print("\n=== 수평×구름전이구간별 요약(핵심 — 전이구간에서 개선율 확인) ===")
        print(by_cloud.to_string(index=False))

    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
