# -*- coding: utf-8 -*-
"""공식 발전소 총출력 OOF를 인버터 1~5 예측으로 후처리 분해한다.

A 정격용량 비례, B 학습구간 계절×시간대 발전비중 중앙값, C 시간·기상·
최근 인버터 비중 LightGBM을 공식 5계절 rolling-origin의 동일 시험행에서
비교한다. 모든 후보는 5대 합계가 기존 공식 총예측과 정확히 일치하도록
비음수 단체(simplex) 정규화한다. API·운영 DB·기존 모델은 수정하지 않는다.
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
OUT = ROOT / "outputs" / "인버터_분해교차검증_v1_2026-08-27"
OOF_PATH = (ROOT / "outputs" / "E2E_v5_공식B_v4_최종통합_2026-08-25" /
            "행단위_ac_power_예측정답.csv")
AUDIT_SCRIPT = ROOT / "audit_inverter_history_v1_2026-08-27.py"
DPC_SCRIPT = ROOT / "defect_policy_comparison_v1_2026-08-21.py"
CAPACITY = np.array([50.0, 50.0, 30.0, 39.0, 50.0], dtype=float)
CAPACITY_SHARE = CAPACITY / CAPACITY.sum()
DEFECT_START = pd.Timestamp("2025-08-15")
DEFECT_END_EXCLUSIVE = pd.Timestamp("2025-11-17")
TOL_KW = 0.011
SEED = 42

OFFICIAL_WINDOWS = [
    ("1_여름", "2025-06-01", "2025-08-14"),
    ("2_가을_결함종료후", "2025-11-17", "2025-12-14"),
    ("3_겨울", "2025-12-15", "2026-02-14"),
    ("4_봄", "2026-02-15", "2026-04-14"),
    ("5_초여름", "2026-04-15", "2026-08-04"),
]

WEATHER_FEATURES = [
    "DSWRF", "TCDC", "LCDC", "HCDC", "TMP", "SKY", "REH", "WSD", "POP",
    "목표_태양고도_deg", "일주기_sin", "일주기_cos", "연주기_sin", "연주기_cos",
]
METHOD_ORDER = {"A_정격용량비례": 0, "B_계절시간중앙값": 1,
                "C_LightGBM비중": 2}
PREREGISTERED_RULES = {
    "판정단위": "티어×수평별",
    "기준선": "A_정격용량비례",
    "규칙1": "A 대비 전체 MAE와 RMSE가 모두 1% 이상 개선",
    "규칙2": "어느 공식 계절폴드에서도 RMSE가 A보다 5% 이상 악화되지 않음",
    "규칙3": "각 인버터 nMAE 악화가 A 대비 1.0%p 이내",
    "규칙4": "공식 OOF 동일 시험행 100% 사용 및 5대 예측합계 오차 1e-9kW 이하",
    "동률": "RMSE가 사실상 같으면 A→B→C 순으로 단순한 방법 우선",
    "주의": "결과를 보기 전에 코드에 고정한 규칙이며 실행 후 변경하지 않음",
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


def _season(month: pd.Series | pd.Index) -> np.ndarray:
    values = np.asarray(month)
    return np.select([np.isin(values, [3, 4, 5]), np.isin(values, [6, 7, 8]),
                      np.isin(values, [9, 10, 11])],
                     ["봄", "여름", "가을"], default="겨울")


def _normalize_shares(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    sums = raw.sum(axis=1, keepdims=True)
    fallback = np.tile(CAPACITY_SHARE, (len(raw), 1))
    return np.divide(raw, sums, out=fallback, where=sums > 1e-12)


def _load_official_oof() -> pd.DataFrame:
    if not OOF_PATH.is_file():
        raise FileNotFoundError(OOF_PATH)
    oof = pd.read_csv(OOF_PATH, encoding="utf-8-sig",
                      parse_dates=["발행시각", "대상시각"])
    required = {"티어", "수평_h", "폴드", "발행시각", "대상시각", "실제_kW", "예측_kW"}
    absent = sorted(required - set(oof.columns))
    if absent:
        raise ValueError(f"공식 OOF 필수컬럼 누락: {absent}")
    expected_folds = {x[0] for x in OFFICIAL_WINDOWS}
    unknown = sorted(set(oof["폴드"].dropna()) - expected_folds)
    if unknown:
        raise ValueError(f"공식 5폴드 밖의 폴드 발견: {unknown}")
    return oof.sort_values(["티어", "수평_h", "발행시각"]).reset_index(drop=True)


def _attach_inverter_actual(oof: pd.DataFrame, audit_mod):
    per_inv = audit_mod.load_all_inverters_5min()
    agg15 = audit_mod.aggregate_inverter_power(per_inv, "15min")
    agg1h = audit_mod.aggregate_inverter_power(per_inv, "1h")
    parts = []
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    for tier, group in oof.groupby("티어", sort=False):
        source = agg15 if tier == "초단기" else agg1h
        values = source.reindex(pd.DatetimeIndex(group["대상시각"]))[
            inv_cols + ["가용인버터수"]]
        values.index = group.index
        joined = group.join(values)
        parts.append(joined)
    out = pd.concat(parts).sort_index()
    missing = out[inv_cols].isna().any(axis=1) | (out["가용인버터수"] < 5)
    if missing.any():
        sample = out.loc[missing, ["티어", "수평_h", "폴드", "대상시각"]].head(10)
        raise RuntimeError("정책B 공식 OOF에 인버터 5대 완전가용이 아닌 시험행이 있음:\n" +
                           sample.to_string(index=False))
    out["인버터실제합계_kW"] = out[inv_cols].sum(axis=1)
    out["공식실제_차이_kW"] = out["인버터실제합계_kW"] - out["실제_kW"]
    bad = out["공식실제_차이_kW"].abs() > TOL_KW
    if bad.any():
        sample = out.loc[bad, ["티어", "수평_h", "폴드", "대상시각", "실제_kW",
                               "인버터실제합계_kW", "공식실제_차이_kW"]].head(10)
        raise RuntimeError("인버터 합계와 공식 실제값 불일치(0.011kW 초과):\n" +
                           sample.to_string(index=False))
    return out, per_inv, agg15, agg1h


def _recent_share_at(issue: pd.DatetimeIndex, agg: pd.DataFrame, lag: pd.Timedelta) -> pd.DataFrame:
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    shares = agg[inv_cols].div(agg[inv_cols].sum(axis=1, min_count=5), axis=0).dropna()
    query = issue - lag
    # 오직 발행시각 이전 값만 사용. 6시간 넘게 오래된 값은 결측으로 둔다.
    selected = shares.reindex(query, method="ffill", tolerance=pd.Timedelta("6h"))
    selected.index = issue
    return selected


def _model_frame(dpc, tier: str, horizon: int, agg: pd.DataFrame) -> pd.DataFrame:
    frame = dpc.load_ultra_frame(horizon) if tier == "초단기" else dpc.load_short_frame(horizon)
    frame = frame.copy().sort_index()
    offset = pd.Timedelta(hours=horizon if tier == "초단기" else horizon - 1)
    frame["대상시각"] = frame.index + offset
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    actual = agg.reindex(pd.DatetimeIndex(frame["대상시각"]))[inv_cols]
    actual.index = frame.index
    frame[inv_cols] = actual
    frame["대상_month"] = frame["대상시각"].dt.month
    frame["대상_hour"] = frame["대상시각"].dt.hour
    frame["대상_hour_sin"] = np.sin(2 * np.pi * frame["대상_hour"] / 24)
    frame["대상_hour_cos"] = np.cos(2 * np.pi * frame["대상_hour"] / 24)
    frame["대상_doy_sin"] = np.sin(2 * np.pi * frame["대상시각"].dt.dayofyear / 365.25)
    frame["대상_doy_cos"] = np.cos(2 * np.pi * frame["대상시각"].dt.dayofyear / 365.25)
    for label, lag in (("최근", pd.Timedelta(0)), ("24시간전", pd.Timedelta("24h"))):
        recent = _recent_share_at(frame.index, agg, lag)
        for i in range(1, 6):
            frame[f"{label}_인버터{i}_비중"] = recent[f"인버터{i}_kW"].to_numpy()
    return frame


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    fixed = ["대상_hour_sin", "대상_hour_cos", "대상_doy_sin", "대상_doy_cos"]
    recent = [f"{label}_인버터{i}_비중" for label in ("최근", "24시간전")
              for i in range(1, 6)]
    weather = [c for c in WEATHER_FEATURES if c in frame.columns]
    cols = fixed + weather + recent
    if not weather:
        raise RuntimeError("C 후보에 사용할 기상특성을 공식 프레임에서 찾지 못함")
    return cols


def _predict_b(train: pd.DataFrame, test_times: pd.Series) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    meta = pd.DataFrame({"계절": _season(train["대상시각"].dt.month),
                         "시": train["대상시각"].dt.hour}, index=train.index)
    work = pd.concat([meta, shares], axis=1)
    by_key = work.groupby(["계절", "시"])[inv_cols].median()
    by_hour = work.groupby("시")[inv_cols].median()
    global_share = shares.median().fillna(pd.Series(CAPACITY_SHARE, index=inv_cols))
    rows = []
    for ts in test_times:
        key = (_season(pd.Index([ts.month]))[0], ts.hour)
        if key in by_key.index:
            rows.append(by_key.loc[key].to_numpy(float))
        elif ts.hour in by_hour.index:
            rows.append(by_hour.loc[ts.hour].to_numpy(float))
        else:
            rows.append(global_share.to_numpy(float))
    return _normalize_shares(np.asarray(rows))


def _predict_c(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    train = train.dropna(subset=inv_cols).copy()
    if len(train) < 300:
        raise RuntimeError(f"C 후보 학습행 부족: {len(train)}")
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    valid = np.isfinite(shares).all(axis=1)
    train, shares = train.loc[valid], shares.loc[valid]
    predictions = []
    for i, col in enumerate(inv_cols):
        model = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              max_depth=5, min_child_samples=40, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED + i, n_jobs=-1, verbosity=-1)
        model.fit(train[features], shares[col])
        predictions.append(model.predict(test[features]))
    return _normalize_shares(np.column_stack(predictions))


def _metrics(actual: np.ndarray, pred: np.ndarray, capacity: float) -> dict:
    mask = np.isfinite(actual) & np.isfinite(pred)
    y, p = actual[mask], pred[mask]
    if not len(y):
        return {"n": 0, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan,
                "R2": np.nan, "nMAE_pct": np.nan}
    mse = mean_squared_error(y, p)
    return {"n": int(len(y)), "MAE": float(mean_absolute_error(y, p)),
            "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else np.nan,
            "nMAE_pct": float(mean_absolute_error(y, p) / capacity * 100)}


def _score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail, folds = [], []
    for keys, group in predictions.groupby(["티어", "수평_h", "방법"], sort=False):
        tier, horizon, method = keys
        for inv in range(1, 6):
            m = _metrics(group[f"인버터{inv}_실제_kW"].to_numpy(),
                         group[f"인버터{inv}_예측_kW"].to_numpy(), CAPACITY[inv - 1])
            detail.append({"티어": tier, "수평_h": horizon, "방법": method,
                           "인버터": inv, "용량_kW": CAPACITY[inv - 1], **m})
        all_actual = group[[f"인버터{i}_실제_kW" for i in range(1, 6)]].to_numpy().ravel()
        all_pred = group[[f"인버터{i}_예측_kW" for i in range(1, 6)]].to_numpy().ravel()
        m = _metrics(all_actual, all_pred, CAPACITY.mean())
        detail.append({"티어": tier, "수평_h": horizon, "방법": method,
                       "인버터": "전체_동일가중", "용량_kW": CAPACITY.mean(), **m})
        for fold, sub in group.groupby("폴드"):
            a = sub[[f"인버터{i}_실제_kW" for i in range(1, 6)]].to_numpy().ravel()
            p = sub[[f"인버터{i}_예측_kW" for i in range(1, 6)]].to_numpy().ravel()
            fm = _metrics(a, p, CAPACITY.mean())
            folds.append({"티어": tier, "수평_h": horizon, "방법": method,
                          "폴드": fold, **fm})
    return pd.DataFrame(detail), pd.DataFrame(folds)


def _adjudicate(detail: pd.DataFrame, folds: pd.DataFrame,
                predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tier, horizon), group in detail[detail["인버터"] == "전체_동일가중"].groupby(
            ["티어", "수평_h"]):
        base = group[group["방법"] == "A_정격용량비례"].iloc[0]
        candidates = []
        for _, row in group.iterrows():
            method = row["방법"]
            fold_base = folds[(folds["티어"] == tier) & (folds["수평_h"] == horizon) &
                              (folds["방법"] == "A_정격용량비례")].set_index("폴드")
            fold_now = folds[(folds["티어"] == tier) & (folds["수평_h"] == horizon) &
                             (folds["방법"] == method)].set_index("폴드")
            fold_worst = float(((fold_now["RMSE"] / fold_base["RMSE"] - 1) * 100).max())
            inv_base = detail[(detail["티어"] == tier) & (detail["수평_h"] == horizon) &
                              (detail["방법"] == "A_정격용량비례") &
                              (detail["인버터"] != "전체_동일가중")].set_index("인버터")
            inv_now = detail[(detail["티어"] == tier) & (detail["수평_h"] == horizon) &
                             (detail["방법"] == method) &
                             (detail["인버터"] != "전체_동일가중")].set_index("인버터")
            inv_worst_pp = float((inv_now["nMAE_pct"] - inv_base["nMAE_pct"]).max())
            pred_sub = predictions[(predictions["티어"] == tier) &
                                   (predictions["수평_h"] == horizon) &
                                   (predictions["방법"] == method)]
            reconcile = float(pred_sub["합계차이_kW"].abs().max())
            mae_imp = float((1 - row["MAE"] / base["MAE"]) * 100)
            rmse_imp = float((1 - row["RMSE"] / base["RMSE"]) * 100)
            passed = (method == "A_정격용량비례" or
                      (mae_imp >= 1.0 and rmse_imp >= 1.0 and fold_worst < 5.0 and
                       inv_worst_pp <= 1.0 and reconcile <= 1e-9))
            item = {"티어": tier, "수평_h": horizon, "방법": method,
                    "MAE개선율_pct": mae_imp, "RMSE개선율_pct": rmse_imp,
                    "폴드최대RMSE악화_pct": fold_worst,
                    "인버터최대nMAE악화_pp": inv_worst_pp,
                    "합계최대차이_kW": reconcile, "채택기준통과": bool(passed)}
            rows.append(item)
            if passed:
                candidates.append((float(row["RMSE"]), METHOD_ORDER[method], method))
        chosen = sorted(candidates, key=lambda x: (round(x[0], 9), x[1]))[0][2]
        for item in rows:
            if item["티어"] == tier and item["수평_h"] == horizon:
                item["최종선택"] = item["방법"] == chosen
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "사전등록_채택규칙.json").write_text(
        json.dumps(PREREGISTERED_RULES, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_mod = _load_module("inverter_audit_20260827", AUDIT_SCRIPT)
    dpc = _load_module("inverter_disagg_dpc_20260827", DPC_SCRIPT)
    oof = _load_official_oof()
    oof, _, agg15, agg1h = _attach_inverter_actual(oof, audit_mod)
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    output_parts, audit_rows = [], []

    for (tier, horizon), test_all in oof.groupby(["티어", "수평_h"], sort=False):
        horizon = int(horizon)
        agg = agg15 if tier == "초단기" else agg1h
        frame = _model_frame(dpc, tier, horizon, agg)
        features = _feature_columns(frame)
        for fold, start_text, end_text in OFFICIAL_WINDOWS:
            test_oof = test_all[test_all["폴드"] == fold].copy()
            if test_oof.empty:
                raise RuntimeError(f"공식 시험행 없음: {tier} +{horizon}h {fold}")
            test_issue = pd.DatetimeIndex(test_oof["발행시각"])
            test = frame.reindex(test_issue).copy()
            if test["대상시각"].isna().any():
                raise RuntimeError(f"공식 프레임 정렬 실패: {tier} +{horizon}h {fold}")
            # 시험 대상시각보다 과거인 행만 학습. 정책B 결함구간은 타깃에서 제외.
            cutoff = test_oof["대상시각"].min()
            train = frame[frame["대상시각"] < cutoff].copy()
            defect = (train["대상시각"] >= DEFECT_START) & (train["대상시각"] < DEFECT_END_EXCLUSIVE)
            train = train[~defect & train[inv_cols].notna().all(axis=1)]
            if len(train) < 300:
                raise RuntimeError(f"학습행 부족: {tier} +{horizon}h {fold} n={len(train)}")

            share_predictions = {
                "A_정격용량비례": np.tile(CAPACITY_SHARE, (len(test_oof), 1)),
                "B_계절시간중앙값": _predict_b(train, test["대상시각"]),
                "C_LightGBM비중": _predict_c(train, test, features),
            }
            actual = test_oof[inv_cols].to_numpy(float)
            total_pred = test_oof["예측_kW"].to_numpy(float)
            for method, shares in share_predictions.items():
                pred = shares * total_pred[:, None]
                block = test_oof[["plant_id", "plant_name", "티어", "수평_h", "폴드",
                                  "발행시각", "대상시각", "실제_kW", "예측_kW"]].copy()
                block["방법"] = method
                for i in range(1, 6):
                    block[f"인버터{i}_실제_kW"] = actual[:, i - 1]
                    block[f"인버터{i}_예측_kW"] = pred[:, i - 1]
                    block[f"인버터{i}_예측비중"] = shares[:, i - 1]
                block["인버터예측합계_kW"] = pred.sum(axis=1)
                block["합계차이_kW"] = block["인버터예측합계_kW"] - block["예측_kW"]
                output_parts.append(block)
            audit_rows.append({"티어": tier, "수평_h": horizon, "폴드": fold,
                               "학습행수": len(train), "시험행수": len(test_oof),
                               "C특성수": len(features), "C특성": "|".join(features),
                               "학습최대대상시각": train["대상시각"].max(),
                               "시험최소대상시각": cutoff,
                               "시간누출없음": bool(train["대상시각"].max() < cutoff),
                               "정책B결함학습행수": int(defect.loc[train.index.intersection(defect.index)].sum())
                               if len(train) else 0})

    predictions = pd.concat(output_parts, ignore_index=True)
    max_diff = float(predictions["합계차이_kW"].abs().max())
    if max_diff > 1e-9:
        raise AssertionError(f"5대 예측합계 reconciliation 실패: {max_diff}")
    detail, folds = _score(predictions)
    decision = _adjudicate(detail, folds, predictions)
    predictions.to_csv(OUT / "행단위_인버터별_예측정답.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "인버터별_전체성능.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "계절폴드별_성능.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "사전등록규칙_판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(OUT / "누출_동일행_감사.csv", index=False, encoding="utf-8-sig")
    print("\n=== 인버터 분해 사전등록 판정 ===")
    print(decision.to_string(index=False))
    print(f"\n5대 합계 최대차이: {max_diff:.3e} kW")


if __name__ == "__main__":
    main()
