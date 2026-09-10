# -*- coding: utf-8 -*-
"""부안 인버터 분해 CV(방법론①, 09-07) - 광주 08-27/김제·영광 09-07이
쓰는 "사전등록기준" 방법론으로 통일(4개 지역 방법론 통일, 사용자 확정
"1,2 비교해서 전체적으로 더 잘 나오는 걸로" → 방법론①로 최종 확정).

## 배경
부안은 08-28에 "인버터 8대 전부가용일이 59일뿐"이라는 이유로 정식 A vs
C CV를 건너뛰고 A(균등비례) 실측확인만 했다가(`inverter_disaggregation_
check_v1_2026-08-28.py`, 이후 필터버그로 폐기), 08-31에 필터를 고쳐
181일로 늘려 방법론②(실측 일간총량 walk-forward, `inverter_
disaggregation_ABC_v2~v4_2026-08-31.py`)로 공식채택(B_계절중앙값)까지
갔다. **방법론①(예측총량 기반 사전등록CV)은 부안에서 한 번도 정식
실행된 적이 없다** - 이번에 처음 한다.

## 재사용한 것(재구현 안 함)
- 총출력 모델: `backtest_buan_v1_2026-09-07.py`가 쓰는 phase2
  공식채택모델(LightGBM, 특성 그대로) - `factor_reverify_v6_hourly_
  buan_2026-09-03.py`의 build_hourly_frame(v5)·v5.expanding_folds_
  full_coverage, `factor_phase2_multicollinearity_modelselect_v1_
  2026-09-03.py`의 harness.make_model.
- 인버터 분해 CV 골격: 김제·영광 09-07 스크립트와 100% 동일 로직
  (_score/_adjudicate/_predict_a/b/c, 사전등록규칙 문구·임계값) -
  인버터수·용량표·총출력모델 로더만 부안 것으로 교체.

## 부안 고유 사항
- 인버터 8기, **전부 125kW 균등**(김제 10기 110kW 균등과 같은 유형,
  영광 13기 12x50+1x34 이종용량과 다름) - A는 균등 1/8.
- INITIAL_TRAIN_DAYS=60/TEST_BLOCK_DAYS=20 (부안 총출력모델의 기존
  관례 그대로, 김제·영광의 90/30과 다름 - 지역별 폴드 규약을 억지로
  통일하지 않음, 각 지역 총출력모델의 기존 규약을 그대로 따름).
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

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
OUT = HERE / "outputs" / "인버터_분해교차검증_v1_buan_2026-09-07"
PHASE1_SCRIPT = HERE / "factor_reverify_v6_hourly_buan_2026-09-03.py"
PHASE2_MODULE = PIPELINE_ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py"
OFFICIAL_PHASE2 = HERE / "outputs" / "요인재검증_v6_phase2_2026-09-03" / "phase2_요약.json"
INVERTER_PARQUET = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\부안\시간집계_v1_2026-08-28"
    r"\부안_인버터별_5분_야간0포함.parquet"
)
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20

N_INVERTERS = 8
CAPACITY = np.array([125.0] * N_INVERTERS, dtype=float)  # 8대 전부 균등(08-28 실측 확인)
CAPACITY_SHARE = CAPACITY / CAPACITY.sum()
SEED = 42
TOL_KW = 1.0

PREREGISTERED_RULES = {
    "판정단위": "부안_총출력(단일 D+1 hourly 트랙, phase2 공식모델)",
    "기준선": "A_균등비례",
    "규칙1": "A 대비 전체 MAE와 RMSE가 모두 1% 이상 개선",
    "규칙2": "어느 폴드에서도 RMSE가 A보다 5% 이상 악화되지 않음",
    "규칙3": "각 인버터 nMAE 악화가 A 대비 1.0%p 이내",
    "규칙4": "동일 시험행 100% 사용 및 8대 예측합계 오차 1.0kW 이하",
    "동률": "RMSE가 사실상 같으면 A→B→C 순으로 단순한 방법 우선",
    "주의": "결과를 보기 전에 코드에 고정한 규칙이며 실행 후 변경하지 않음",
    "타지역_대비_차이": "인버터수 8(전부 125kW 균등, 김제 10x110kW 균등과 같은 유형) - "
                      "A는 균등1/8. 총출력모델 폴드는 부안 기존 관례(60일학습+20일블록).",
    "부안_고유_경위": "08-28엔 8대 전부가용일이 59일뿐이라 이 CV를 건너뛰고 A만 확인했었음 - "
                    "09-01 필터버그 수정으로 181일 확보된 지금(09-07)이 사실상 첫 정식 실행.",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_inverter_wide() -> pd.DataFrame:
    cols = ["grid_time_kst", "inverter_number", "ac_power_kw", "quality_status_after_night"]
    raw = pd.read_parquet(INVERTER_PARQUET, columns=cols)
    valid_status = {"observed", "physical_zero_night", "night_zero_physical"}
    raw["유효"] = raw["quality_status_after_night"].isin(valid_status)
    raw.loc[~raw["유효"], "ac_power_kw"] = np.nan

    wide = raw.pivot(index="grid_time_kst", columns="inverter_number", values="ac_power_kw")
    wide.columns = [f"인버터{i}_kW" for i in wide.columns]
    valid_wide = raw.pivot(index="grid_time_kst", columns="inverter_number", values="유효")
    valid_wide.columns = [f"인버터{i}_kW" for i in valid_wide.columns]
    wide = wide.sort_index()
    valid_wide = valid_wide.sort_index()

    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    count_valid = valid_wide[inv_cols].resample("1h").sum()
    mean_power = wide[inv_cols].resample("1h").mean()
    for c in inv_cols:
        mean_power.loc[count_valid[c] < 9, c] = np.nan
    mean_power["가용인버터수"] = count_valid[inv_cols].ge(9).sum(axis=1)
    mean_power["발전소합계_kW"] = mean_power[inv_cols].sum(axis=1, min_count=N_INVERTERS)
    return mean_power


def _season(month) -> np.ndarray:
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


def _recent_share_at(issue: pd.DatetimeIndex, agg: pd.DataFrame, lag: pd.Timedelta) -> pd.DataFrame:
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    shares = agg[inv_cols].div(agg[inv_cols].sum(axis=1, min_count=N_INVERTERS), axis=0).dropna()
    query = issue - lag
    selected = shares.reindex(query, method="ffill", tolerance=pd.Timedelta("6h"))
    selected.index = issue
    return selected


def _predict_a(n: int) -> np.ndarray:
    return np.tile(CAPACITY_SHARE, (n, 1))


def _predict_b(train: pd.DataFrame, test_times: pd.Series) -> np.ndarray:
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
    shares = train[inv_cols].div(train[inv_cols].sum(axis=1), axis=0)
    meta = pd.DataFrame({"계절": _season(train["target_time_kst"].dt.month),
                         "시": train["target_time_kst"].dt.hour}, index=train.index)
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
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]
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
    for method, group in predictions.groupby("방법", sort=False):
        for inv in range(1, N_INVERTERS + 1):
            m = _metrics(group[f"인버터{inv}_실제_kW"].to_numpy(),
                         group[f"인버터{inv}_예측_kW"].to_numpy(), CAPACITY[inv - 1])
            detail.append({"방법": method, "인버터": inv, "용량_kW": CAPACITY[inv - 1], **m})
        all_actual = group[[f"인버터{i}_실제_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
        all_pred = group[[f"인버터{i}_예측_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
        m = _metrics(all_actual, all_pred, CAPACITY.mean())
        detail.append({"방법": method, "인버터": "전체_동일가중", "용량_kW": CAPACITY.mean(), **m})
        for fold, sub in group.groupby("폴드"):
            a = sub[[f"인버터{i}_실제_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
            p = sub[[f"인버터{i}_예측_kW" for i in range(1, N_INVERTERS + 1)]].to_numpy().ravel()
            fm = _metrics(a, p, CAPACITY.mean())
            folds.append({"방법": method, "폴드": fold, **fm})
    return pd.DataFrame(detail), pd.DataFrame(folds)


def _adjudicate(detail: pd.DataFrame, folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    method_order = {"A_균등비례": 0, "B_계절시간중앙값": 1, "C_LightGBM비중": 2}
    rows = []
    group = detail[detail["인버터"] == "전체_동일가중"]
    base = group[group["방법"] == "A_균등비례"].iloc[0]
    candidates = []
    for _, row in group.iterrows():
        method = row["방법"]
        fold_base = folds[folds["방법"] == "A_균등비례"].set_index("폴드")
        fold_now = folds[folds["방법"] == method].set_index("폴드")
        fold_worst = float(((fold_now["RMSE"] / fold_base["RMSE"] - 1) * 100).max())
        inv_base = detail[(detail["방법"] == "A_균등비례") &
                          (detail["인버터"] != "전체_동일가중")].set_index("인버터")
        inv_now = detail[(detail["방법"] == method) &
                         (detail["인버터"] != "전체_동일가중")].set_index("인버터")
        inv_worst_pp = float((inv_now["nMAE_pct"] - inv_base["nMAE_pct"]).max())
        pred_sub = predictions[predictions["방법"] == method]
        reconcile = float(pred_sub["합계차이_kW"].abs().max())
        mae_imp = float((1 - row["MAE"] / base["MAE"]) * 100)
        rmse_imp = float((1 - row["RMSE"] / base["RMSE"]) * 100)
        passed = (method == "A_균등비례" or
                  (mae_imp >= 1.0 and rmse_imp >= 1.0 and fold_worst < 5.0 and
                   inv_worst_pp <= 1.0 and reconcile <= TOL_KW))
        item = {"방법": method, "MAE개선율_pct": mae_imp, "RMSE개선율_pct": rmse_imp,
                "폴드최대RMSE악화_pct": fold_worst, "인버터최대nMAE악화_pp": inv_worst_pp,
                "합계최대차이_kW": reconcile, "채택기준통과": bool(passed)}
        rows.append(item)
        if passed:
            candidates.append((float(row["RMSE"]), method_order[method], method))
    chosen = sorted(candidates, key=lambda x: (round(x[0], 9), x[1]))[0][2]
    for item in rows:
        item["최종선택"] = item["방법"] == chosen
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "사전등록_채택규칙.json").write_text(
        json.dumps(PREREGISTERED_RULES, ensure_ascii=False, indent=2), encoding="utf-8")

    official = json.loads(OFFICIAL_PHASE2.read_text(encoding="utf-8"))
    features = official["최종특성"]
    model_name = official["선정모델(MAE기준)"]
    print(f"[부안] 총출력 공식모델 재사용: {model_name}, 특성 {len(features)}개")

    phase1_mod = _load_module("buan_phase1_invcv", PHASE1_SCRIPT)
    phase2_mod = _load_module("phase2_common_buan_invcv", PHASE2_MODULE)
    harness = phase2_mod._load_harness()
    v5 = phase1_mod._load_module("buan_v5_for_folds_invcv", phase1_mod.V5_SCRIPT)

    frame = phase1_mod.build_hourly_frame(v5)
    if isinstance(frame, tuple):
        frame = frame[0]
    TARGET = "plant_ac_power_kw"
    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    folds = v5.expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
    required = [c for c in features if c not in getattr(phase1_mod, "NATIVE_MISSING_OK", set())]

    inv_wide = _load_inverter_wide()
    inv_cols = [f"인버터{i}_kW" for i in range(1, N_INVERTERS + 1)]

    output_parts, audit_rows = [], []
    for fold_i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤"
        train_total = frame[frame["issue_day"].isin(train_days)].dropna(subset=required + [TARGET])
        test_total = frame[frame["issue_day"].isin(test_days)].dropna(subset=required + [TARGET])
        if len(train_total) < 300 or len(test_total) < 30:
            print(f"[폴드{fold_i}] 총출력 표본 부족(train={len(train_total)}, "
                  f"test={len(test_total)}) - 생략")
            continue

        model_total = harness.make_model(model_name, SEED)
        model_total.fit(train_total[features], train_total[TARGET])
        total_pred_test = np.clip(model_total.predict(test_total[features]), 0, CAPACITY.sum())

        target_times = pd.DatetimeIndex(test_total["target_time_kst"])
        inv_actual = inv_wide.reindex(target_times)[inv_cols + ["가용인버터수"]]
        full_ok = (inv_actual["가용인버터수"] == N_INVERTERS).to_numpy()
        if full_ok.sum() < 30:
            print(f"[폴드{fold_i}] 8기 전부가용 시험행 부족({full_ok.sum()}) - 생략")
            continue

        test_full = test_total.loc[full_ok].copy()
        actual = inv_actual.loc[full_ok, inv_cols].to_numpy(float)
        total_pred_full = total_pred_test[full_ok]

        c_frame = test_full[["target_time_kst"] + features].copy()
        c_frame["대상_hour_sin"] = np.sin(2 * np.pi * c_frame["target_time_kst"].dt.hour / 24)
        c_frame["대상_hour_cos"] = np.cos(2 * np.pi * c_frame["target_time_kst"].dt.hour / 24)
        recent = _recent_share_at(pd.DatetimeIndex(c_frame["target_time_kst"]), inv_wide, pd.Timedelta(0))
        recent24 = _recent_share_at(pd.DatetimeIndex(c_frame["target_time_kst"]), inv_wide, pd.Timedelta("24h"))
        for i in range(1, N_INVERTERS + 1):
            c_frame[f"최근_인버터{i}_비중"] = recent[f"인버터{i}_kW"].to_numpy()
            c_frame[f"24시간전_인버터{i}_비중"] = recent24[f"인버터{i}_kW"].to_numpy()

        train_target_times = pd.DatetimeIndex(train_total["target_time_kst"])
        train_inv = inv_wide.reindex(train_target_times)[inv_cols + ["가용인버터수"]]
        train_ok = (train_inv["가용인버터수"] == N_INVERTERS).to_numpy()
        c_train_frame = train_total.loc[train_ok, ["target_time_kst"] + features].copy()
        c_train_frame["대상_hour_sin"] = np.sin(2 * np.pi * c_train_frame["target_time_kst"].dt.hour / 24)
        c_train_frame["대상_hour_cos"] = np.cos(2 * np.pi * c_train_frame["target_time_kst"].dt.hour / 24)
        recent_tr = _recent_share_at(pd.DatetimeIndex(c_train_frame["target_time_kst"]), inv_wide, pd.Timedelta(0))
        recent24_tr = _recent_share_at(pd.DatetimeIndex(c_train_frame["target_time_kst"]), inv_wide, pd.Timedelta("24h"))
        for i in range(1, N_INVERTERS + 1):
            c_train_frame[f"최근_인버터{i}_비중"] = recent_tr[f"인버터{i}_kW"].to_numpy()
            c_train_frame[f"24시간전_인버터{i}_비중"] = recent24_tr[f"인버터{i}_kW"].to_numpy()

        c_train_frame[inv_cols] = train_inv.loc[train_ok, inv_cols].to_numpy(float)
        b_train = c_train_frame.copy()

        share_predictions = {
            "A_균등비례": _predict_a(len(test_full)),
            "B_계절시간중앙값": _predict_b(b_train, c_frame["target_time_kst"]),
            "C_LightGBM비중": _predict_c(c_train_frame, c_frame,
                                       list(features) + ["대상_hour_sin", "대상_hour_cos"] +
                                       [f"최근_인버터{i}_비중" for i in range(1, N_INVERTERS + 1)] +
                                       [f"24시간전_인버터{i}_비중" for i in range(1, N_INVERTERS + 1)]),
        }
        for method, shares in share_predictions.items():
            pred = shares * total_pred_full[:, None]
            block = pd.DataFrame({
                "폴드": fold_i, "대상시각": test_full["target_time_kst"].to_numpy(),
                "발전소실제_kW": inv_actual.loc[full_ok, inv_cols].sum(axis=1).to_numpy(),
                "발전소예측_kW": total_pred_full,
            })
            block["방법"] = method
            for i in range(1, N_INVERTERS + 1):
                block[f"인버터{i}_실제_kW"] = actual[:, i - 1]
                block[f"인버터{i}_예측_kW"] = pred[:, i - 1]
            block["인버터예측합계_kW"] = pred.sum(axis=1)
            block["합계차이_kW"] = block["인버터예측합계_kW"] - block["발전소예측_kW"]
            output_parts.append(block)
        audit_rows.append({"폴드": fold_i, "학습행수": len(train_total),
                           "시험행수(8기전부가용)": int(full_ok.sum()),
                           "총출력_학습최대대상시각": train_total["target_time_kst"].max(),
                           "총출력_시험최소대상시각": test_total["target_time_kst"].min(),
                           "시간누출없음": bool(train_total["target_time_kst"].max() <
                                             test_total["target_time_kst"].min())})
        print(f"[폴드{fold_i}] 완료 - 시험행 {int(full_ok.sum()):,}건")

    if not output_parts:
        raise RuntimeError("모든 폴드가 표본 부족으로 생략됨 - 결과 없음")

    predictions = pd.concat(output_parts, ignore_index=True)
    max_diff = float(predictions["합계차이_kW"].abs().max())
    if max_diff > TOL_KW:
        raise AssertionError(f"8대 예측합계 reconciliation 실패: {max_diff}")
    detail, folds_score = _score(predictions)
    decision = _adjudicate(detail, folds_score, predictions)
    predictions.to_csv(OUT / "행단위_인버터별_예측정답.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "인버터별_전체성능.csv", index=False, encoding="utf-8-sig")
    folds_score.to_csv(OUT / "폴드별_성능.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "사전등록규칙_판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(OUT / "누출_동일행_감사.csv", index=False, encoding="utf-8-sig")
    print("\n=== 부안 인버터 분해 사전등록 판정(방법론①) ===")
    print(decision.to_string(index=False))
    print(f"\n8대 합계 최대차이: {max_diff:.3e} kW")


if __name__ == "__main__":
    main()
