# -*- coding: utf-8 -*-
"""⑤ 모든 후보 통합 최종 재검증 — 로드맵 08-21 5단계.

## ★후보·판정기준은 실행 전 사전동결(AGENTS.md 참고, 바꾸지 말 것)★
결과를 보고 후보를 추가하거나 기준을 바꾸지 않는다. 아래 목록·규칙이
전부다.

### 후보(동결)
- **초단기(LightGBM)**: +1h=raw만 / +2h={raw,청천지수,청천지수+튜닝} /
  +3h={raw,청천지수} / +4h={raw,청천지수,청천지수+튜닝}
- **단기(XGBoost)**: +1h={기본,튜닝} / +24h={기본만} / +48h={기본,튜닝}
- **일간**: {전체특성,거버넌스특성} × {기본파라미터,튜닝파라미터} = 4후보,
  **전부 계층조정까지 재산출**(직접모델 단독 비교 금지)

### 채택 기준(동결, 순서대로 적용)
1. 모든 후보를 정확히 같은 시험행에서 비교(표본수 다르면 비교 안 함)
2. MAE·RMSE 둘 다 개선 → 채택 후보
3. 한쪽만 개선 → 기존유지
4. 개선율 1% 미만 → 기존유지(운영복잡성)
5. 특정 계절 MAE 5%+ 악화 → 보류(계절불안정)
6. 동률이면 구조 단순한 쪽

### ★인식론적 한계(보고서에도 그대로 쓸 것)★
이건 "후보와 판정기준을 사전에 고정한 동일 5계절 rolling-origin 통합
재검증"이다. 완전히 안 본 독립시험이 아니다(①~④가 전부 같은 5계절
폴드를 보며 만든 후보이므로). 지금 데이터에서 가능한 가장 공정한
최종 비교이지만, 새 데이터가 들어오면 그걸로 다시 외부검증해야 한다.

## 재사용 모듈
청천지수 계산(`ultra_short_clearsky_v1`), 특성거버넌스
(`daily_feature_governance_v1`), 하이퍼파라미터 탐색
(`hyperparameter_tuning_v1`)의 함수를 그대로 import해서 쓴다 — 로직
중복 구현 안 함, 이미 검증된 함수 재사용.
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


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("harness", "backtest_harness_v1_2026-08-20밤.py")
ultra = _load("ultra", "train_ultra_short_official_v1_2026-08-21.py")
clearsky = _load("clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
daily_mod = _load("daily_v2", "train_daily_official_v2_2026-08-21.py")
daily_gov = _load("daily_gov", "daily_feature_governance_v1_2026-08-21.py")
tuning = _load("tuning", "hyperparameter_tuning_v1_2026-08-21.py")

sys.path.insert(0, str(ROOT))
from model_common import optimize_nonnegative_weights  # noqa: E402

OUT = ROOT / "outputs" / "최종통합재검증_v1_2026-08-21"
DAILY_OOF_REF = ROOT / "outputs" / "일간_계층조정_v2_2026-08-21" / "OOF_상세.csv"

# 후보명 복잡도(규칙6 동률 tie-break용, 클수록 복잡)
COMPLEXITY = {
    "raw": 0, "기본": 0, "전체특성+기본파라미터": 0,
    "청천지수": 1, "튜닝": 1, "거버넌스특성+기본파라미터": 1, "전체특성+튜닝파라미터": 1,
    "청천지수+튜닝": 2, "거버넌스특성+튜닝파라미터": 2,
}


def energy_metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    e = y - p
    denom = float(np.abs(y).sum())
    return {"n": int(len(y)), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "WAPE_pct": float(np.abs(e).sum() / denom * 100) if denom > 0 else None}


def decide(baseline: dict, candidate: dict, baseline_season: dict, cand_season: dict, label: str) -> dict:
    """동결된 채택기준 1~5를 순서대로 적용. 6(동률 tie-break)은 그룹 단위로 main()에서 처리."""
    if baseline["n"] != candidate["n"]:
        return {"후보": label, "판정": "비교불가(표본수 다름)", "MAE": candidate["MAE"], "RMSE": candidate["RMSE"]}
    mae_impr = (baseline["MAE"] - candidate["MAE"]) / baseline["MAE"] * 100
    rmse_impr = (baseline["RMSE"] - candidate["RMSE"]) / baseline["RMSE"] * 100
    row = {"후보": label, "MAE": candidate["MAE"], "RMSE": candidate["RMSE"],
           "MAE개선율_pct": round(mae_impr, 2), "RMSE개선율_pct": round(rmse_impr, 2)}
    if not (mae_impr > 0 and rmse_impr > 0):
        row["판정"] = "기존유지(한쪽만 개선 또는 악화)"
        return row
    # 규칙4 해석(명시): "개선율 1% 미만"을 MAE·RMSE 중 하나라도 1%를
    # 못 넘으면 걸리는 것으로 엄격하게 적용한다(더 관대한 해석도 가능하나,
    # "운영복잡성을 늘리지 않는다"는 규칙4 취지상 보수적으로 판단).
    if mae_impr < 1.0 or rmse_impr < 1.0:
        row["판정"] = "기존유지(개선율<1%)"
        return row
    worst = 0.0
    for season, b_mae in baseline_season.items():
        if season in cand_season and b_mae > 0:
            degrade = (cand_season[season] - b_mae) / b_mae * 100
            worst = max(worst, degrade)
    row["최대계절악화율_pct"] = round(worst, 2)
    row["판정"] = "보류(계절불안정)" if worst >= 5.0 else "채택후보"
    return row


def season_mae_map(df: pd.DataFrame, fold_col: str, err_col: str) -> dict:
    return df.groupby(fold_col)[err_col].apply(lambda s: s.abs().mean()).to_dict()


# ── 초단기 ─────────────────────────────────────────────────────────
ULTRA_TARGETS = {
    1: ["raw"],
    2: ["raw", "청천지수", "청천지수+튜닝"],
    3: ["raw", "청천지수"],
    4: ["raw", "청천지수", "청천지수+튜닝"],
}


def run_ultra(config, capacity_kw, seed) -> pd.DataFrame:
    quarter = ultra.load_15min_base()
    hourly_df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS
    windows = config["cross_validation_windows"]

    results = []
    for H, labels in ULTRA_TARGETS.items():
        frame = ultra.build_ultra_short_frame(quarter, hourly_df, H)
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        clear_kw = capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0
        frame["_청천_kW"] = np.clip(clear_kw, 1e-3, None)
        frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]

        need_kappa = any("청천지수" in l for l in labels)
        elev_thresh = clearsky.MIN_ELEVATION_DEG if need_kappa else 0.0
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= elev_thresh)]

        # 튜닝이 필요한 경우, 이 수평의 카파 타깃으로 폴드 전체에서 단일 배포파라미터를 뽑는다
        need_tuning = any("튜닝" in l for l in labels)
        tuned_params = None

        fold_rows = {label: [] for label in labels}
        trace_all = []
        for i, w in enumerate(windows, start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            fold_name = f"{i}_{w.get('_계절','')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            if len(train_all) < 500 or len(test_all) < 100:
                continue
            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(tr_for_sel, candidate_cols, 0.3, True, True)
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + [
                "목표_발전출력_kW", "_카파", "_청천_kW"
            ]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 500 or len(test) < 100:
                continue

            if need_tuning:
                _, trace = tuning.tune_fold("LightGBM", train, feature_cols, "_카파", capacity_kw, seed)
                trace_all.extend({"폴드": fold_name, **t} for t in trace)

            y_test = test["목표_발전출력_kW"].to_numpy()
            clear_test = test["_청천_kW"].to_numpy()

            for label in labels:
                if label == "raw":
                    m = ultra.make_model("LightGBM", seed)
                    m.fit(train[feature_cols], train["목표_발전출력_kW"])
                    pred = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
                elif label == "청천지수":
                    m = ultra.make_model("LightGBM", seed)
                    m.fit(train[feature_cols], train["_카파"])
                    pred = np.clip(m.predict(test[feature_cols]) * clear_test, 0, capacity_kw)
                elif label == "청천지수+튜닝":
                    pass  # 아래에서 tuned_params 확정 후 일괄 처리
                fold_rows.setdefault(label, [])
                if label != "청천지수+튜닝":
                    fold_rows[label].append(pd.DataFrame({
                        "발행시각": test.index, "폴드": fold_name, "실제_kW": y_test, "예측_kW": pred,
                    }))

        if need_tuning and trace_all:
            trace_df = pd.DataFrame(trace_all)
            param_names = list(tuning.LGBM_SPACE)
            robust = tuning.pick_robust_params(trace_df, [], param_names)
            tuned_params = {k: robust.iloc[0][k] for k in param_names}
            # 정수형이어야 하는 파라미터 캐스팅(랜덤서치 값이 numpy 타입일 수 있음)
            for k in ["n_estimators", "num_leaves", "max_depth", "min_child_samples"]:
                if k in tuned_params and tuned_params[k] is not None:
                    tuned_params[k] = int(tuned_params[k])

            for i, w in enumerate(windows, start=1):
                start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
                fold_name = f"{i}_{w.get('_계절','')}"
                train_all = daylight[daylight.index < start]
                test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
                if len(train_all) < 500 or len(test_all) < 100:
                    continue
                tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
                chosen = harness.select_features_in_fold(tr_for_sel, candidate_cols, 0.3, True, True)
                feature_cols = base_cols + chosen
                required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + [
                    "목표_발전출력_kW", "_카파", "_청천_kW"
                ]
                train = train_all.dropna(subset=required)
                test = test_all.dropna(subset=required)
                if len(train) < 500 or len(test) < 100:
                    continue
                from lightgbm import LGBMRegressor
                m = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **tuned_params)
                m.fit(train[feature_cols], train["_카파"])
                pred = np.clip(m.predict(test[feature_cols]) * test["_청천_kW"].to_numpy(), 0, capacity_kw)
                fold_rows["청천지수+튜닝"].append(pd.DataFrame({
                    "발행시각": test.index, "폴드": fold_name, "실제_kW": test["목표_발전출력_kW"].to_numpy(), "예측_kW": pred,
                }))

        for label in labels:
            if not fold_rows[label]:
                continue
            df = pd.concat(fold_rows[label], ignore_index=True)
            df["오차"] = df["실제_kW"] - df["예측_kW"]
            df["수평_h"] = H
            df["후보"] = label
            results.append(df)
        print(f"  [초단기 +{H}h] 후보 {labels} 완료")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── 단기 ───────────────────────────────────────────────────────────
SHORT_TARGETS = {1: ["기본", "튜닝"], 24: ["기본"], 48: ["기본", "튜닝"]}


def run_short(config, capacity_kw, seed) -> pd.DataFrame:
    df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    windows = config["cross_validation_windows"]
    results = []
    for H, labels in SHORT_TARGETS.items():
        frame = harness.build_frame(df, H, candidate_cols)
        daylight = frame[frame["목표_낮시간"] > 0]
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]

        need_tuning = "튜닝" in labels
        trace_all = []
        fold_rows = {label: [] for label in labels}
        fold_feats = {}
        for i, w in enumerate(windows, start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            fold_name = f"{i}_{w.get('_계절','')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            if len(train_all) < 200 or len(test_all) < 30:
                continue
            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(tr_for_sel, candidate_cols, 0.3, True, True)
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue
            fold_feats[fold_name] = (train, test, feature_cols)

            m = harness.make_model("XGBoost", seed)
            m.fit(train[feature_cols], train["목표_발전출력_kW"])
            pred = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
            fold_rows["기본"].append(pd.DataFrame({
                "발행시각": test.index, "폴드": fold_name, "실제_kW": test["목표_발전출력_kW"].to_numpy(), "예측_kW": pred,
            }))
            if need_tuning:
                _, trace = tuning.tune_fold("XGBoost", train, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
                trace_all.extend({"폴드": fold_name, **t} for t in trace)

        if need_tuning and trace_all:
            trace_df = pd.DataFrame(trace_all)
            param_names = list(tuning.XGB_SPACE)
            robust = tuning.pick_robust_params(trace_df, [], param_names)
            tparams = {k: robust.iloc[0][k] for k in param_names}
            for k in ["n_estimators", "max_depth", "min_child_weight"]:
                if k in tparams and tparams[k] is not None:
                    tparams[k] = int(tparams[k])
            from xgboost import XGBRegressor
            for fold_name, (train, test, feature_cols) in fold_feats.items():
                m = XGBRegressor(objective="reg:absoluteerror", random_state=seed, n_jobs=4, verbosity=0, **tparams)
                m.fit(train[feature_cols], train["목표_발전출력_kW"])
                pred = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
                fold_rows["튜닝"].append(pd.DataFrame({
                    "발행시각": test.index, "폴드": fold_name, "실제_kW": test["목표_발전출력_kW"].to_numpy(), "예측_kW": pred,
                }))

        for label in labels:
            if not fold_rows[label]:
                continue
            d = pd.concat(fold_rows[label], ignore_index=True)
            d["오차"] = d["실제_kW"] - d["예측_kW"]
            d["수평_h"] = H
            d["후보"] = label
            results.append(d)
        print(f"  [단기 +{H}h] 후보 {labels} 완료")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── 일간 ───────────────────────────────────────────────────────────
def run_daily(config, seed) -> pd.DataFrame:
    data, candidate_cols = daily_mod.build_daily_dataset()
    target_col = daily_mod.ACTUAL
    capacity_daily = float(config["site"]["capacity_kw"]) * 24
    windows = config["cross_validation_windows"]

    daily_ref = pd.read_csv(DAILY_OOF_REF, parse_dates=["날짜"], encoding="utf-8-sig")
    hourly_sum = daily_ref.set_index("날짜")["시간모델합계_kWh"]

    # 튜닝파라미터는 전체특성 기준으로 한 번만 탐색(사용자 지시: 2x2 요인설계,
    # 거버넌스·튜닝을 독립 요인으로 취급 — 거버넌스특성+튜닝파라미터도 이 값 재사용)
    trace_all = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절','')}"
        train = data[data.index < start]
        if len(train) < 60:
            continue
        medians = train[candidate_cols].median(numeric_only=True)
        _, trace = tuning.tune_fold("LightGBM", train.fillna(medians), candidate_cols, target_col, capacity_daily, seed)
        trace_all.extend({"폴드": fold_name, **t} for t in trace)
    trace_df = pd.DataFrame(trace_all)
    param_names = list(tuning.LGBM_SPACE)
    robust = tuning.pick_robust_params(trace_df, [], param_names)
    tuned_params = {k: robust.iloc[0][k] for k in param_names}
    for k in ["n_estimators", "num_leaves", "max_depth", "min_child_samples"]:
        if k in tuned_params and tuned_params[k] is not None:
            tuned_params[k] = int(tuned_params[k])

    from lightgbm import LGBMRegressor
    combos = [
        ("전체특성+기본파라미터", "전체", None),
        ("거버넌스특성+기본파라미터", "거버넌스", None),
        ("전체특성+튜닝파라미터", "전체", tuned_params),
        ("거버넌스특성+튜닝파라미터", "거버넌스", tuned_params),
    ]

    results = []
    for label, feat_mode, params in combos:
        direct_rows = []
        for i, w in enumerate(windows, start=1):
            start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            fold_name = f"{i}_{w.get('_계절','')}"
            train = data[data.index < start]
            test = data[(data.index >= start) & (data.index <= end)]
            if len(train) < 60 or len(test) < 10:
                continue
            if feat_mode == "전체":
                feats = candidate_cols
            else:
                cut = int(len(train) * 0.8)
                inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]
                med_inner = inner_tr[candidate_cols].median(numeric_only=True)
                feats, _ = daily_gov.select_features_governed(inner_tr, inner_va, candidate_cols, target_col, med_inner, seed)
                if not feats:
                    feats = candidate_cols
            medians = train[feats].median(numeric_only=True)
            kwargs = params or {}
            m = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **kwargs)
            m.fit(train[feats].fillna(medians), train[target_col])
            pred = np.clip(m.predict(test[feats].fillna(medians)), 0, capacity_daily)
            for date, actual, p in zip(test.index, test[target_col], pred):
                direct_rows.append({"날짜": date, "폴드": fold_name, "실제_일간총량_kWh": actual, "직접모델_kWh": p})
        direct_df = pd.DataFrame(direct_rows)
        if not len(direct_df):
            continue
        merged = direct_df.merge(hourly_sum.rename("시간모델합계_kWh"), left_on="날짜", right_index=True, how="inner")
        cand_cols = ["직접모델_kWh", "시간모델합계_kWh"]
        w_res = optimize_nonnegative_weights(merged["실제_일간총량_kWh"], merged[cand_cols])
        weights = np.asarray(w_res["가중치"], float)
        merged["계층조정_kWh"] = np.clip(merged[cand_cols].to_numpy() @ weights, 0, capacity_daily)
        merged["오차"] = merged["실제_일간총량_kWh"] - merged["계층조정_kWh"]
        merged["후보"] = label
        merged["가중치_직접"] = weights[0]
        merged["가중치_시간합계"] = weights[1]
        results.append(merged)
        print(f"  [일간 {label}] n={len(merged)} 가중치(직접/시간합계)={weights[0]:.2f}/{weights[1]:.2f}")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    print("=== 초단기 ===")
    ultra_df = run_ultra(config, capacity_kw, seed)
    ultra_df.to_csv(OUT / "초단기_행단위.csv", index=False, encoding="utf-8-sig")

    print("\n=== 단기 ===")
    short_df = run_short(config, capacity_kw, seed)
    short_df.to_csv(OUT / "단기_행단위.csv", index=False, encoding="utf-8-sig")

    print("\n=== 일간 ===")
    daily_df = run_daily(config, seed)
    daily_df.to_csv(OUT / "일간_행단위.csv", index=False, encoding="utf-8-sig")

    verdict_rows = []

    # 초단기·단기: 그룹(수평)마다 raw/기본을 기준선으로 채택기준 적용
    for tier_df, group_col, err_col, base_label in [
        (ultra_df, "수평_h", "오차", "raw"), (short_df, "수평_h", "오차", "기본"),
    ]:
        if not len(tier_df):
            continue
        for H, sub in tier_df.groupby(group_col):
            base_sub = sub[sub["후보"] == base_label]
            if not len(base_sub):
                continue
            base_metric = energy_metrics(base_sub["실제_kW"], base_sub["예측_kW"])
            base_season = season_mae_map(base_sub, "폴드", "오차")
            for label, cand_sub in sub.groupby("후보"):
                if label == base_label:
                    continue
                cand_metric = energy_metrics(cand_sub["실제_kW"], cand_sub["예측_kW"])
                cand_season = season_mae_map(cand_sub, "폴드", "오차")
                v = decide(base_metric, cand_metric, base_season, cand_season, label)
                v.update({"티어": "초단기" if group_col == "수평_h" and tier_df is ultra_df else "단기", "수평_h": H,
                           "복잡도": COMPLEXITY.get(label, 9)})
                verdict_rows.append(v)

    # 일간: 전체특성+기본파라미터를 기준선으로
    if len(daily_df):
        base_sub = daily_df[daily_df["후보"] == "전체특성+기본파라미터"]
        base_metric = energy_metrics(base_sub["실제_일간총량_kWh"], base_sub["계층조정_kWh"])
        base_season = season_mae_map(base_sub, "폴드", "오차")
        for label, cand_sub in daily_df.groupby("후보"):
            if label == "전체특성+기본파라미터":
                continue
            cand_metric = energy_metrics(cand_sub["실제_일간총량_kWh"], cand_sub["계층조정_kWh"])
            cand_season = season_mae_map(cand_sub, "폴드", "오차")
            v = decide(base_metric, cand_metric, base_season, cand_season, label)
            v.update({"티어": "일간", "수평_h": None, "복잡도": COMPLEXITY.get(label, 9)})
            verdict_rows.append(v)

    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")

    print("\n\n=== ⑤ 최종 판정표(사전동결 기준 적용) ===")
    print(verdict_df.to_string(index=False))

    # 규칙6: 같은 그룹에서 '채택후보'가 여럿이면 복잡도 최소인 것만 최종채택으로 표시
    verdict_df["최종채택"] = False
    for (티어, 수평), grp in verdict_df.groupby(["티어", "수평_h"], dropna=False):
        cands = grp[grp["판정"] == "채택후보"]
        if len(cands):
            winner_idx = cands["복잡도"].idxmin()
            verdict_df.loc[winner_idx, "최종채택"] = True
    verdict_df.to_csv(OUT / "판정표.csv", index=False, encoding="utf-8-sig")

    print("\n=== 최종채택(규칙6 동률 tie-break 적용 후) ===")
    print(verdict_df[verdict_df["최종채택"]].to_string(index=False))
    print(f"\n저장 완료: {OUT}")
    print("\n※ 이 결과는 '후보·기준 사전동결 + 동일 5계절 rolling-origin' 재검증이다.")
    print("  완전히 새로운 데이터로 다시 외부검증하기 전까지는 이 틀 안에서의 최선일 뿐이다.")


if __name__ == "__main__":
    main()
