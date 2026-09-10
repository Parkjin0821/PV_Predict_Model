# -*- coding: utf-8 -*-
"""일간 D+1 인버터 분해 CV v2 - NWP 기상특성 추가(C 후보 재도전).

## 배경
08-28 v1(`inverter_disaggregation_daily_cv_v1_2026-08-28.py`)은 발행시각이
없어 계절성(day-of-year)과 1일전/7일전 비중 lag만으로 C_LightGBM비중을
학습했다. `config/inverter_disaggregation.json`에 그 결과를 채택하며
"기상특성 없는 1차 버전 한계 - 추후 기상특성 추가 시 C가 재도전할 여지는
있다"고 직접 남겨뒀다. 사용자 09-03 지시("일간 D+1 기상특성 추가해서
재검증해줘")로 그 재도전을 이번에 실행한다.

## v1과 다른 점(이것뿐)
- `_build_features`에 공식 일간모델(`e2e_retrain_v5_공식B_v1_2026-08-24.py`
  `build_daily_dataset_v5()`)이 이미 만들어 쓰는 "목표일예보_*" NWP
  일단위 집계특성(DSWRF·TCDC·LCDC·MCDC·HCDC·REH·TMP·WSD·POP 등의
  sum/mean/min/max, DIFSWRF 결측여부)을 그대로 join한다 - 새로 계산하지
  않고 공식 함수 재사용.
- `_feature_columns()`가 그 목표일예보_* 컬럼들을 추가로 포함.
- 그 외(폴드, PREREGISTERED_RULES, A/B 후보, 채점·판정 로직, reconciliation
  허용오차)는 v1과 완전히 동일 - 결과를 보기 전에 정해진 기준을 그대로
  재사용한다(사후 기준변경 금지 원칙 유지).

## 결측 처리
NWP 특성은 결측이 있을 수 있다(DIFSWRF·DSWRFLX는 2026-06-01부터 상류
결측 - AGENTS.md 08-26 트랙 참고). LightGBM native missing을 그대로
쓴다(중앙값 대체 없음, 공식 일간모델과 동일 정책).

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python inverter_disaggregation_daily_cv_v2_기상특성추가_2026-09-03.py
```

## 산출물 (`outputs/인버터_분해교차검증_일간_v2_기상특성추가_2026-09-03/`)
v1과 동일한 5개 파일 + `v1_대비_비교.csv`(같은 폴드에서 C의 MAE/RMSE가
기상특성 추가로 실제 개선됐는지 나란히 비교).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "인버터_분해교차검증_일간_v2_기상특성추가_2026-09-03"
V1_DECISION_PATH = ROOT / "outputs" / "인버터_분해교차검증_일간_v1_2026-08-28" / "사전등록규칙_판정표.csv"
DAILY_OOF_PATH = (ROOT / "outputs" / "E2E_v5_공식B_v4_최종통합_2026-08-25" /
                  "행단위_daily_예측정답.csv")
AUDIT_SCRIPT = ROOT / "audit_inverter_history_v1_2026-08-27.py"
HOURLY_CV_SCRIPT = ROOT / "inverter_disaggregation_cv_v1_2026-08-27.py"
DAILY_MODEL_SCRIPT = ROOT / "e2e_retrain_v5_공식B_v1_2026-08-24.py"
CAPACITY_KW = 240.58

RECONCILE_TOL_KWH = 0.5
SEED = 42

PREREGISTERED_RULES = {
    "판정단위": "일간(D+1) 단독",
    "기준선": "A_정격용량비례",
    "규칙1": "A 대비 전체 MAE와 RMSE가 모두 1% 이상 개선",
    "규칙2": "어느 공식 계절폴드에서도 RMSE가 A보다 5% 이상 악화되지 않음",
    "규칙3": "각 인버터 nMAE 악화가 A 대비 1.0%p 이내",
    "규칙4": "공식 OOF 동일 시험행 100% 사용 및 5대 예측합계 오차 0.5kWh 이하",
    "동률": "RMSE가 사실상 같으면 A→B→C 순으로 단순한 방법 우선",
    "주의": "결과를 보기 전에 코드에 고정한 규칙이며 실행 후 변경하지 않음",
    "v1과_차이": "v1(08-28)과 채점·채택 기준은 완전히 동일. C 후보에만 "
               "공식 일간모델의 목표일예보_* NWP 일단위 집계특성을 추가.",
}


def _load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_daily_oof() -> pd.DataFrame:
    if not DAILY_OOF_PATH.is_file():
        raise FileNotFoundError(DAILY_OOF_PATH)
    oof = pd.read_csv(DAILY_OOF_PATH, encoding="utf-8-sig", parse_dates=["대상일"])
    required = {"대상일", "폴드", "실제_kWh", "예측_kWh"}
    absent = sorted(required - set(oof.columns))
    if absent:
        raise ValueError(f"일간 공식 OOF 필수컬럼 누락: {absent}")
    oof = oof.rename(columns={"실제_kWh": "실제_kW", "예측_kWh": "예측_kW"})
    oof["티어"] = "일간"
    oof["수평_h"] = 1
    oof["대상시각"] = oof["대상일"]
    return oof.sort_values("대상일").reset_index(drop=True)


def _build_daily_actual(audit_mod) -> pd.DataFrame:
    per_inv = audit_mod.load_all_inverters_5min()
    agg1h = audit_mod.aggregate_inverter_power(per_inv, "1h")
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    full5 = agg1h[agg1h["가용인버터수"] == 5]
    hours_per_day = full5.resample("D").size()
    daily = full5[inv_cols].resample("D").sum(min_count=1)
    daily = daily[hours_per_day >= 20].dropna(how="any")
    daily["일간실제합계_kW"] = daily[inv_cols].sum(axis=1)
    return daily


def _attach_daily_inverter_actual(oof: pd.DataFrame, daily_actual: pd.DataFrame):
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    joined = oof.join(daily_actual[inv_cols + ["일간실제합계_kW"]], on="대상일")
    missing = joined[inv_cols].isna().any(axis=1)
    if missing.any():
        n_missing = int(missing.sum())
        print(f"경고: 일간 OOF {len(joined)}건 중 {n_missing}건은 인버터 5대 "
             "완전가용일이 아니라 이 CV 시험셋에서 제외한다(가짜값 생성 금지).")
        joined = joined.loc[~missing].copy()
    diff = (joined["일간실제합계_kW"] - joined["실제_kW"]).abs()
    bad = diff > RECONCILE_TOL_KWH
    if bad.any():
        sample = joined.loc[bad, ["대상일", "폴드", "실제_kW", "일간실제합계_kW"]].head(10)
        raise RuntimeError(
            "인버터 합계와 공식 일간 실제값 불일치(%.2fkWh 초과):\n%s"
            % (RECONCILE_TOL_KWH, sample.to_string(index=False)))
    return joined.reset_index(drop=True), float(diff.max())


def _recent_share_lag_daily(target_dates: pd.Series, daily_actual: pd.DataFrame,
                            lag_days: int) -> pd.DataFrame:
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    shares = daily_actual[inv_cols].div(
        daily_actual[inv_cols].sum(axis=1, min_count=5), axis=0).dropna()
    query = pd.DatetimeIndex(target_dates) - pd.Timedelta(days=lag_days)
    selected = shares.reindex(query, method="ffill", tolerance=pd.Timedelta(days=3))
    selected.index = target_dates.index
    return selected


def _load_weather_features() -> tuple[pd.DataFrame, list[str]]:
    """공식 일간모델(e2e_retrain_v5_공식B_v1)이 이미 만드는 목표일예보_*
    NWP 일단위 집계특성을 그대로 가져온다 - 재계산하지 않는다."""
    e2e = _load_module("daily_disagg_e2e_20260903", DAILY_MODEL_SCRIPT)
    data, all_features = e2e.build_daily_dataset_v5(CAPACITY_KW)
    weather_cols = [c for c in all_features
                    if c.startswith("목표일예보_") or c == "목표일_DIFSWRF_유효개수"]
    weather = data[weather_cols].copy()
    weather.index.name = "대상일"
    return weather, weather_cols


def _build_features(frame: pd.DataFrame, daily_actual: pd.DataFrame,
                     weather: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["대상_doy_sin"] = np.sin(2 * np.pi * frame["대상일"].dt.dayofyear / 365.25)
    frame["대상_doy_cos"] = np.cos(2 * np.pi * frame["대상일"].dt.dayofyear / 365.25)
    frame["대상_month"] = frame["대상일"].dt.month
    for label, lag in (("1일전", 1), ("7일전", 7)):
        recent = _recent_share_lag_daily(frame["대상일"], daily_actual, lag)
        for i in range(1, 6):
            frame[f"{label}_인버터{i}_비중"] = recent[f"인버터{i}_kW"].to_numpy()
    w = weather.reindex(pd.DatetimeIndex(frame["대상일"]))
    w.index = frame.index
    frame = pd.concat([frame, w], axis=1)
    return frame


def _feature_columns(weather_cols: list[str]) -> list[str]:
    fixed = ["대상_doy_sin", "대상_doy_cos"]
    recent = [f"{label}_인버터{i}_비중" for label in ("1일전", "7일전") for i in range(1, 6)]
    return fixed + recent + weather_cols


def _predict_a(cv, n: int) -> np.ndarray:
    return np.tile(cv.CAPACITY_SHARE, (n, 1))


def _predict_b(cv, train: pd.DataFrame, test_months: np.ndarray) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    season = cv._season(train["대상일"].dt.month)
    work = shares.copy()
    work["계절"] = season
    by_season = work.groupby("계절")[inv_cols].median()
    global_share = shares.median().fillna(pd.Series(cv.CAPACITY_SHARE, index=inv_cols))
    rows = []
    for month in test_months:
        s = cv._season(pd.Index([month]))[0]
        rows.append((by_season.loc[s] if s in by_season.index else global_share).to_numpy(float))
    return cv._normalize_shares(np.asarray(rows))


def _predict_c(cv, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    train = train.dropna(subset=inv_cols).copy()
    if len(train) < 60:
        raise RuntimeError(f"C 후보 학습행 부족(일간은 하루 1행이라 절대량이 적음): {len(train)}")
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    valid = np.isfinite(shares).all(axis=1)
    train, shares = train.loc[valid], shares.loc[valid]
    predictions = []
    for i, col in enumerate(inv_cols):
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7,
                              max_depth=4, min_child_samples=20, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED + i, n_jobs=-1, verbosity=-1)
        model.fit(train[features], shares[col])
        predictions.append(model.predict(test[features]))
    return cv._normalize_shares(np.column_stack(predictions))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "사전등록_채택규칙.json").write_text(
        json.dumps(PREREGISTERED_RULES, ensure_ascii=False, indent=2), encoding="utf-8")

    cv = _load_module("inverter_disagg_cv_daily_v2_20260903", HOURLY_CV_SCRIPT)
    audit_mod = _load_module("inverter_audit_daily_v2_20260903", AUDIT_SCRIPT)

    oof = _load_daily_oof()
    daily_actual = _build_daily_actual(audit_mod)
    oof, max_reconcile = _attach_daily_inverter_actual(oof, daily_actual)
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    weather, weather_cols = _load_weather_features()
    features = _feature_columns(weather_cols)
    print(f"NWP 일단위 기상특성 {len(weather_cols)}개 추가(목표일예보_* 등) - "
          f"공식 일간모델(e2e_retrain_v5_공식B_v1)에서 그대로 재사용")

    history_frame = _build_features(
        daily_actual.reset_index(names="대상일")[["대상일"] + inv_cols], daily_actual, weather)
    test_frame_all = _build_features(
        oof[["대상일", "폴드", "티어", "수평_h", "실제_kW", "예측_kW"] + inv_cols].copy(),
        daily_actual, weather)

    output_parts, audit_rows = [], []
    for fold, start_text, end_text in cv.OFFICIAL_WINDOWS:
        test_frame = test_frame_all[test_frame_all["폴드"] == fold].copy()
        if test_frame.empty:
            raise RuntimeError(f"공식 시험행 없음: 일간 {fold}")
        cutoff = test_frame["대상일"].min()
        train = history_frame[history_frame["대상일"] < cutoff].copy()
        train = train[train[inv_cols].notna().all(axis=1)]
        if len(train) < 60:
            raise RuntimeError(f"학습행 부족: 일간 {fold} n={len(train)}")

        test_months = test_frame["대상일"].dt.month.to_numpy()
        share_predictions = {
            "A_정격용량비례": _predict_a(cv, len(test_frame)),
            "B_계절중앙값": _predict_b(cv, train, test_months),
            "C_LightGBM비중_기상특성추가": _predict_c(cv, train, test_frame, features),
        }
        actual = test_frame[inv_cols].to_numpy(float)
        total_pred = test_frame["예측_kW"].to_numpy(float)
        for method, shares in share_predictions.items():
            assert len(shares) == len(test_frame), (
                "%s: 비중 예측 길이(%d)가 시험행수(%d)와 다름"
                % (method, len(shares), len(test_frame)))
            pred = shares * total_pred[:, None]
            block = test_frame[["폴드", "티어", "수평_h", "대상일", "실제_kW", "예측_kW"]].copy()
            block = block.rename(columns={"대상일": "대상시각"})
            block["방법"] = method
            for i in range(1, 6):
                block[f"인버터{i}_실제_kW"] = actual[:, i - 1]
                block[f"인버터{i}_예측_kW"] = pred[:, i - 1]
                block[f"인버터{i}_예측비중"] = shares[:, i - 1]
            block["인버터예측합계_kW"] = pred.sum(axis=1)
            block["합계차이_kW"] = block["인버터예측합계_kW"] - block["예측_kW"]
            output_parts.append(block)
        audit_rows.append({"티어": "일간", "폴드": fold, "학습행수": len(train),
                           "시험행수": len(test_frame), "C특성수": len(features),
                           "학습최대대상일": train["대상일"].max(), "시험최소대상일": cutoff,
                           "시간누출없음": bool(train["대상일"].max() < cutoff)})

    predictions = pd.concat(output_parts, ignore_index=True)
    max_diff = float(predictions["합계차이_kW"].abs().max())
    if max_diff > RECONCILE_TOL_KWH:
        raise AssertionError(f"5대 예측합계 reconciliation 실패: {max_diff}")

    # cv._adjudicate는 방법이름을 METHOD_ORDER 딕셔너리에서 찾으므로
    # 이번 실행의 "C_LightGBM비중_기상특성추가" 이름을 임시로 등록.
    cv.METHOD_ORDER = dict(cv.METHOD_ORDER)
    cv.METHOD_ORDER["C_LightGBM비중_기상특성추가"] = cv.METHOD_ORDER.get("C_LightGBM비중", 2)

    detail, folds = cv._score(predictions)
    decision = cv._adjudicate(detail, folds, predictions)
    predictions.to_csv(OUT / "행단위_인버터별_예측정답.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "인버터별_전체성능.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "계절폴드별_성능.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "사전등록규칙_판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(OUT / "누출_동일행_감사.csv", index=False, encoding="utf-8-sig")

    if V1_DECISION_PATH.is_file():
        v1 = pd.read_csv(V1_DECISION_PATH)
        v1_c = v1[v1["방법"] == "C_LightGBM비중"][
            ["수평_h", "MAE개선율_pct", "RMSE개선율_pct", "폴드최대RMSE악화_pct",
             "인버터최대nMAE악화_pp", "채택기준통과"]
        ].rename(columns=lambda c: f"v1_{c}" if c != "수평_h" else c)
        v2_c = decision[decision["방법"] == "C_LightGBM비중_기상특성추가"][
            ["수평_h", "MAE개선율_pct", "RMSE개선율_pct", "폴드최대RMSE악화_pct",
             "인버터최대nMAE악화_pp", "채택기준통과"]
        ].rename(columns=lambda c: f"v2_{c}" if c != "수평_h" else c)
        compare = v1_c.merge(v2_c, on="수평_h", how="outer")
        compare.to_csv(OUT / "v1_대비_비교.csv", index=False, encoding="utf-8-sig")
        print("\n=== v1(기상특성 없음) vs v2(기상특성 추가) ===")
        print(compare.to_string(index=False))

    print("\n=== 일간 D+1 인버터 분해 v2 사전등록 판정 ===")
    print(decision.to_string(index=False))
    print(f"\n인버터합계-공식실제 reconciliation 최대차이(입력단계): {max_reconcile:.4f} kWh")
    print(f"5대 예측합계 최대차이(출력단계): {max_diff:.3e} kWh")


if __name__ == "__main__":
    main()
