# -*- coding: utf-8 -*-
"""일간 D+1 총출력(공식 OOF)을 인버터 1~5 예측으로 후처리 분해 - CV.

08-27 `inverter_disaggregation_cv_v1_2026-08-27.py`(초단기/단기)와 완전히
같은 원리·같은 사전등록기준을 일간(D+1) 티어에 적용한다. 사용자 지적
("이미 인버터별로 나누는 거 검증했었는데 3/4h·24/48h가 안 되는 게 이해가
안 된다")에 대한 정정: **그 수평들은 이미 다 됨(A방법으로 정상 동작)** -
안 된 건 일간 D+1 하나뿐이었고, 이 스크립트가 그 마지막 구멍을 메운다.

## 초단기/단기 CV와 다른 점(재사용 불가능했던 이유)
- 일간 공식 OOF(`E2E_v5_공식B_v4_최종통합_2026-08-25/행단위_daily_예측정답.csv`)
  에는 `발행시각`이 없다(하루 1행, `대상일`만 있음) - 초단기/단기 CV의
  `_model_frame`/`_recent_share_at`은 시간단위 lag(0h/24h)를 전제해서
  그대로 못 쓴다. 대신 일단위 lag(1일전/7일전 비중)로 새로 만들었다.
- NWP 기상특성은 이번 1차 버전에 넣지 않았다(일단위로 안정적으로 재사용
  가능한 기존 집계 함수가 없었음) - 계절성(day-of-year sin/cos)과 최근
  비중 lag만 사용한다. 이후 필요하면 확장 가능(아래 "미래 확장" 참고).
- 그 외(CAPACITY, OFFICIAL_WINDOWS, 사전등록기준, _normalize_shares,
  _score/_adjudicate 로직)는 전부 CV 스크립트에서 import로 재사용했다 -
  재구현하지 않았다.

## 절대 하지 않는 것
- 결과를 본 뒤 채택기준을 바꾸지 않는다(PREREGISTERED_RULES를 실행 맨
  처음에 파일로 먼저 저장).
- 공식 운영모델·API·운영 DB는 건드리지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "인버터_분해교차검증_일간_v1_2026-08-28"
DAILY_OOF_PATH = (ROOT / "outputs" / "E2E_v5_공식B_v4_최종통합_2026-08-25" /
                  "행단위_daily_예측정답.csv")
AUDIT_SCRIPT = ROOT / "audit_inverter_history_v1_2026-08-27.py"
HOURLY_CV_SCRIPT = ROOT / "inverter_disaggregation_cv_v1_2026-08-27.py"

RECONCILE_TOL_KWH = 0.5  # 일단위 누적이라 시간단위 TOL_KW(0.011)보다 여유를 둠
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
    "_08-27CV와_차이": "일간은 발행시각이 없어 시간단위 lag 대신 일단위(1일전/"
                     "7일전) 비중 lag만 사용, NWP 기상특성은 1차 버전에 미포함",
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
    oof["수평_h"] = 1  # D+1을 그대로 "수평 1"로 취급(일간 자체가 유일 수평)
    oof["대상시각"] = oof["대상일"]
    return oof.sort_values("대상일").reset_index(drop=True)


def _build_daily_actual(audit_mod) -> pd.DataFrame:
    """인버터 5대가 하루 종일(가용인버터수==5인 모든 1시간 슬롯) 살아있는
    날짜만 일단위로 합산한다 - 부분가용일을 섞으면 비중이 왜곡된다."""
    per_inv = audit_mod.load_all_inverters_5min()
    agg1h = audit_mod.aggregate_inverter_power(per_inv, "1h")
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    full5 = agg1h[agg1h["가용인버터수"] == 5]
    hours_per_day = full5.resample("D").size()
    daily = full5[inv_cols].resample("D").sum(min_count=1)
    # 하루 24시간 전부(또는 최소 20시간 이상) 가용한 날만 신뢰 - 부분일 제외
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


def _build_features(frame: pd.DataFrame, daily_actual: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["대상_doy_sin"] = np.sin(2 * np.pi * frame["대상일"].dt.dayofyear / 365.25)
    frame["대상_doy_cos"] = np.cos(2 * np.pi * frame["대상일"].dt.dayofyear / 365.25)
    frame["대상_month"] = frame["대상일"].dt.month
    for label, lag in (("1일전", 1), ("7일전", 7)):
        recent = _recent_share_lag_daily(frame["대상일"], daily_actual, lag)
        for i in range(1, 6):
            frame[f"{label}_인버터{i}_비중"] = recent[f"인버터{i}_kW"].to_numpy()
    return frame


def _feature_columns() -> list[str]:
    fixed = ["대상_doy_sin", "대상_doy_cos"]
    recent = [f"{label}_인버터{i}_비중" for label in ("1일전", "7일전") for i in range(1, 6)]
    return fixed + recent


def _predict_a(cv, n: int) -> np.ndarray:
    return np.tile(cv.CAPACITY_SHARE, (n, 1))


def _predict_b(cv, train: pd.DataFrame, test_months: np.ndarray) -> np.ndarray:
    """일간엔 '시'가 없어 계절만으로 중앙값 비중을 쓴다(B의 일간판).
    train: 학습행(대상_month_test 컬럼 없어도 됨). test_months: 시험행 각각의
    대상월(정수 배열) - 이 길이만큼 예측행이 나온다."""
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

    cv = _load_module("inverter_disagg_cv_daily_20260828", HOURLY_CV_SCRIPT)
    audit_mod = _load_module("inverter_audit_daily_20260828", AUDIT_SCRIPT)

    oof = _load_daily_oof()
    daily_actual = _build_daily_actual(audit_mod)
    oof, max_reconcile = _attach_daily_inverter_actual(oof, daily_actual)
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    features = _feature_columns()

    # ★버그수정(첫 실행에서 발견)★: 학습 풀은 공식 OOF 321행(2025-06-01
    # 이후, fold 시험구간만 존재)이 아니라 인버터 원본 이력 전체(2024-08-25
    # 부터)여야 한다 - OOF만 쓰면 각 fold의 첫 학습 컷오프 이전에 학습행이
    # 0건이 되는 사고가 난다(실측: 1_여름 폴드에서 n=0으로 즉시 실패).
    # 학습 풀(history_frame)은 daily_actual 전체를 그대로 특성화하고,
    # 시험 풀(test_frame_all)은 공식 OOF(실제_kW/예측_kW 보유)만 별도로
    # 특성화한다 - 둘 다 같은 _build_features()를 써서 특성 정의가
    # 어긋나지 않게 한다.
    history_frame = _build_features(
        daily_actual.reset_index(names="대상일")[["대상일"] + inv_cols], daily_actual)
    test_frame_all = _build_features(
        oof[["대상일", "폴드", "티어", "수평_h", "실제_kW", "예측_kW"] + inv_cols].copy(),
        daily_actual)

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
            "C_LightGBM비중": _predict_c(cv, train, test_frame, features),
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
            # 발행시각 없음(일간 OOF 자체가 하루 1행) - 있지도 않은 값을
            # 지어내지 않는다. 필요하면 대상시각으로 직접 조회할 것.
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
    detail, folds = cv._score(predictions)
    decision = cv._adjudicate(detail, folds, predictions)
    predictions.to_csv(OUT / "행단위_인버터별_예측정답.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "인버터별_전체성능.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "계절폴드별_성능.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "사전등록규칙_판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(OUT / "누출_동일행_감사.csv", index=False, encoding="utf-8-sig")
    print("\n=== 일간 D+1 인버터 분해 사전등록 판정 ===")
    print(decision.to_string(index=False))
    print(f"\n인버터합계-공식실제 reconciliation 최대차이(입력단계): {max_reconcile:.4f} kWh")
    print(f"5대 예측합계 최대차이(출력단계): {max_diff:.3e} kWh")


if __name__ == "__main__":
    main()
