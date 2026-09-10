"""부안·영광 D+1 인버터분해를 발행시각 기준으로 재검증한다.

API 호출 없음. 총출력 특성과 C 방식의 최근 인버터 비중은 모두
prediction_issue_time_kst 이하 관측만 사용한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAFE_SCRIPT = ROOT / "rebuild_regional_dayahead_issue_safe_v1_2026-09-08.py"
SEED = 42

CFG = {
    "부안": {
        "dir": ROOT / "부안_준비_2026-08-28",
        "old": "inverter_disaggregation_cv_v1_buan_2026-09-07.py",
        "initial": 60, "block": 20, "method_a": "A_균등비례",
    },
    "영광": {
        "dir": ROOT / "영광_준비_2026-09-03",
        "old": "inverter_disaggregation_cv_v1_yeonggwang_2026-09-07.py",
        "initial": 90, "block": 30, "method_a": "A_정격용량비례",
    },
    "김제": {
        "dir": ROOT / "김제_준비_2026-09-01",
        "old": "inverter_disaggregation_cv_v1_gimje_2026-09-03.py",
        "initial": 90, "block": 30, "method_a": "A_균등비례",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def add_issue_safe_share_features(
    frame: pd.DataFrame, inv_wide: pd.DataFrame, n: int
) -> pd.DataFrame:
    """발행시각 현재/24시간 전의 마지막 관측 비중만 붙인다."""
    out = frame.copy()
    issue = pd.DatetimeIndex(out["issue_time_kst"])
    for label, lag in (("발행시각_최근", pd.Timedelta(0)),
                       ("발행시각_24시간전", pd.Timedelta("24h"))):
        shares = inv_wide[[f"인버터{i}_kW" for i in range(1, n + 1)]].copy()
        shares = shares.div(shares.sum(axis=1, min_count=n), axis=0).dropna()
        query = issue - lag
        selected = shares.reindex(query, method="ffill", tolerance=pd.Timedelta("6h"))
        selected.index = out.index
        for i in range(1, n + 1):
            out[f"{label}_인버터{i}_비중"] = selected[f"인버터{i}_kW"].to_numpy()
    return out


def run_region(region: str, cfg: dict, safe, harness) -> dict:
    old = load_module(f"old_inv_{region}", cfg["dir"] / cfg["old"])
    safe_cfg = safe.CFG[region]
    frame, features, leakage_audit = safe.build_safe_frame(safe_cfg)
    days = pd.DatetimeIndex(np.sort(frame["issue_day"].dropna().unique()))
    fold_list = safe.folds(days, cfg["initial"], cfg["block"])
    required = features + [safe.TARGET]

    inv_wide = old._load_inverter_wide()
    n = old.N_INVERTERS
    inv_cols = [f"인버터{i}_kW" for i in range(1, n + 1)]
    parts, audit_rows = [], []

    for fold_i, (train_days, test_days) in enumerate(fold_list, start=1):
        if train_days.max() >= test_days.min():
            raise AssertionError("발행일 폴드 누출")
        train_total = frame[frame["issue_day"].isin(train_days)].dropna(subset=required)
        test_total = frame[frame["issue_day"].isin(test_days)].dropna(subset=required)
        if len(train_total) < 300 or len(test_total) < 30:
            continue

        total_model = harness.make_model("XGBoost", SEED)
        total_model.fit(train_total[features], train_total[safe.TARGET])
        total_pred = np.clip(total_model.predict(test_total[features]), 0, old.CAPACITY.sum())

        test_inv = inv_wide.reindex(pd.DatetimeIndex(test_total["target_time_kst"]))
        full_ok = (test_inv["가용인버터수"] == n).to_numpy()
        if full_ok.sum() < 30:
            continue
        test_full = test_total.loc[full_ok].copy()
        actual = test_inv.loc[full_ok, inv_cols].to_numpy(float)
        total_pred = total_pred[full_ok]

        train_inv = inv_wide.reindex(pd.DatetimeIndex(train_total["target_time_kst"]))
        train_ok = (train_inv["가용인버터수"] == n).to_numpy()
        c_train = train_total.loc[train_ok, ["issue_time_kst", "target_time_kst"] + features].copy()
        c_test = test_full[["issue_time_kst", "target_time_kst"] + features].copy()
        for x in (c_train, c_test):
            x["대상_hour_sin"] = np.sin(2 * np.pi * x["target_time_kst"].dt.hour / 24)
            x["대상_hour_cos"] = np.cos(2 * np.pi * x["target_time_kst"].dt.hour / 24)
        c_train = add_issue_safe_share_features(c_train, inv_wide, n)
        c_test = add_issue_safe_share_features(c_test, inv_wide, n)
        c_train[inv_cols] = train_inv.loc[train_ok, inv_cols].to_numpy(float)

        share_features = (features + ["대상_hour_sin", "대상_hour_cos"] +
                          [f"발행시각_최근_인버터{i}_비중" for i in range(1, n + 1)] +
                          [f"발행시각_24시간전_인버터{i}_비중" for i in range(1, n + 1)])
        c_required = share_features + inv_cols
        c_train_clean = c_train.dropna(subset=c_required)
        c_test_clean_mask = c_test[share_features].notna().all(axis=1).to_numpy()
        if len(c_train_clean) < 300 or c_test_clean_mask.sum() < 30:
            continue

        # 세 방법 모두 정확히 같은 시험행을 사용한다.
        test_full = test_full.loc[c_test_clean_mask]
        actual = actual[c_test_clean_mask]
        total_pred = total_pred[c_test_clean_mask]
        c_test = c_test.loc[c_test_clean_mask]

        shares_by_method = {
            cfg["method_a"]: old._predict_a(len(test_full)),
            "B_계절시간중앙값": old._predict_b(c_train_clean, c_test["target_time_kst"]),
            "C_LightGBM비중": old._predict_c(c_train_clean, c_test, share_features),
        }
        for method, shares in shares_by_method.items():
            pred = shares * total_pred[:, None]
            block = pd.DataFrame({
                "폴드": fold_i,
                "발행시각": test_full["issue_time_kst"].to_numpy(),
                "대상시각": test_full["target_time_kst"].to_numpy(),
                "발전소실제_kW": actual.sum(axis=1),
                "발전소예측_kW": total_pred,
                "방법": method,
            })
            for i in range(1, n + 1):
                block[f"인버터{i}_실제_kW"] = actual[:, i - 1]
                block[f"인버터{i}_예측_kW"] = pred[:, i - 1]
            block["인버터예측합계_kW"] = pred.sum(axis=1)
            block["합계차이_kW"] = block["인버터예측합계_kW"] - block["발전소예측_kW"]
            parts.append(block)
        audit_rows.append({
            "폴드": fold_i, "학습행": len(train_total), "동일시험행": len(test_full),
            "학습최대발행일": train_days.max(), "시험최소발행일": test_days.min(),
            "총출력_issue_safe": True, "C비중_target_time_관측사용": False,
            "C비중_최대가용시각": "issue_time_kst",
        })

    if not parts:
        raise RuntimeError(f"{region}: 유효 폴드 없음")
    predictions = pd.concat(parts, ignore_index=True)
    max_diff = float(predictions["합계차이_kW"].abs().max())
    if max_diff > old.TOL_KW:
        raise AssertionError(f"{region}: 합계보존 위반 {max_diff}")
    detail, fold_scores = old._score(predictions)
    decision = old._adjudicate(detail, fold_scores, predictions)

    slug = {"부안": "buan", "김제": "gimje", "영광": "yeonggwang"}[region]
    out = cfg["dir"] / "outputs" / f"인버터_분해교차검증_issue_safe_v1_{slug}_2026-09-08"
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "행단위_인버터별_예측정답.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out / "인버터별_전체성능.csv", index=False, encoding="utf-8-sig")
    fold_scores.to_csv(out / "폴드별_성능.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(out / "사전등록규칙_판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(out / "누출_동일행_감사.csv", index=False, encoding="utf-8-sig")
    selected = decision.loc[decision["최종선택"], "방법"].iloc[0]
    overall = detail[(detail["방법"] == selected) & (detail["인버터"] == "전체_동일가중")].iloc[0]
    summary = {
        "status": "issue_safe_revalidated", "region": region,
        "total_model": "XGBoost", "selected_disaggregation": selected,
        "same_test_rows": int(len(predictions) / 3),
        "selected_nMAE_pct": float(overall["nMAE_pct"]),
        "selected_MAE_kW_per_inverter_equal_weight": float(overall["MAE"]),
        "selected_RMSE_kW_per_inverter_equal_weight": float(overall["RMSE"]),
        "max_reconciliation_error_kw": max_diff,
        "leakage_audit": {**leakage_audit,
            "c_share_features_anchored_at_or_before_issue": True,
            "target_time_observed_share_features_used": False},
    }
    (out / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    safe = load_module("safe_d1_for_inv", SAFE_SCRIPT)
    phase2 = safe.load_module("safe_phase2_for_inv", safe.PHASE2)
    harness = phase2._load_harness()
    print(json.dumps([run_region(r, c, safe, harness) for r, c in CFG.items()], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
