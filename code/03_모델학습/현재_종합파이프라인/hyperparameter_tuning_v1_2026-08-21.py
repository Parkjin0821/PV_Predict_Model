# -*- coding: utf-8 -*-
"""④ 살아남은 모델 구조에 하이퍼파라미터 튜닝 — 로드맵 08-21 4단계.

## 원칙(사용자 확정, 반드시 지킬 것)
튜닝은 "무료 성능 향상"이 아니라 **과적합 위험이 있는 추가 최적화**다.
- **각 시험폴드는 절대 안 본다.** 하이퍼파라미터 후보를 고를 때는
  학습구간을 다시 80/20으로 쪼갠 내부 홀드아웃(inner_va)만 쓴다
  (트리 앙상블 가중치·특성거버넌스 순열중요도와 동일하게 반복해 온
  이 프로젝트의 표준 관례).
- **시험폴드는 "이미 정해진 파라미터를 딱 한 번 채점"할 때만 쓴다.**
  시험결과를 보면서 파라미터를 흔들면 그 순간부터 성능이 부풀려진다.
- 폴드마다 최적 파라미터가 다를 수 있다 — **폴드별 재선택**을 그대로
  적용한다(이 프로젝트의 특성재선택과 동일 철학).

## 대상: 지금까지 살아남은(채택된) 모델 구조만
- 초단기(+1~4h): **LightGBM**
- 단기(+1h/+24h/+48h): **XGBoost**
- 일간: **LightGBM**(직접모델 성분 — 거버넌스 특성은 아직 ⑤에서 확정
  예정이라 지금은 기존 51개 특성 그대로 사용, 튜닝된 하이퍼파라미터만
  ⑤에 투입할 후보로 남긴다)

## 탐색 방법
그리드 전체 탐색은 조합이 너무 많아(수천 개) 폴드마다 시간이 오래
걸린다. **랜덤서치**(폴드당 20회 무작위 조합, 시드 고정으로 재현 가능)
로 실용적 범위 안에서 후보를 뽑고, 내부 홀드아웃 MAE가 가장 낮은
조합을 그 폴드의 최종 파라미터로 확정한다.

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python hyperparameter_tuning_v1_2026-08-21.py
```
API 호출 없음(로컬 재학습만). 세 티어 다 하면 조합이 많아(초단기
4수평+단기 3수평+일간 1 = 8개 대상 × 5폴드 × 20회 탐색 ≈ 800회 학습)
수십 분 걸릴 수 있다.

## 산출물
`outputs/하이퍼파라미터튜닝_v1_2026-08-21/`
- `{티어}_폴드별_최적파라미터.csv` (선택된 하이퍼파라미터 전부)
- `{티어}_비교.csv` (기존 고정값 vs 튜닝값, 시험폴드 성능 — 딱 한 번만 채점)
- `요약.txt`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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

_spec_d = importlib.util.spec_from_file_location("daily_v2", ROOT / "train_daily_official_v2_2026-08-21.py")
daily_mod = importlib.util.module_from_spec(_spec_d)
sys.modules["daily_v2"] = daily_mod
_spec_d.loader.exec_module(daily_mod)

OUT = ROOT / "outputs" / "하이퍼파라미터튜닝_v1_2026-08-21"
N_RANDOM_DRAWS = 20

LGBM_SPACE = {
    "n_estimators": [150, 220, 300, 400, 500],
    "learning_rate": [0.015, 0.02, 0.03, 0.04, 0.06],
    "num_leaves": [7, 15, 31, 50],
    "max_depth": [3, 4, 5, 6, -1],
    "min_child_samples": [10, 14, 20, 30, 50],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.1, 0.3, 1.0, 2.0],
}
XGB_SPACE = {
    "n_estimators": [150, 220, 300, 400, 500],
    "learning_rate": [0.015, 0.02, 0.03, 0.04, 0.06],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 8],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.1, 0.3, 1.0, 2.0],
}


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    return {k: v[rng.integers(len(v))] for k, v in space.items()}


def mae(y, p) -> float:
    return float(np.abs(np.asarray(y, float) - np.asarray(p, float)).mean())


def tune_fold(model_kind: str, train: pd.DataFrame, feature_cols: list[str], target_col: str,
              capacity: float, base_seed: int) -> tuple[dict, list[dict]]:
    """학습구간을 80/20으로 쪼개 내부 홀드아웃에서만 랜덤서치. 시험폴드는 절대 안 봄.

    ★08-21 수정(사용자 지적 반영)★: 이전엔 1등 파라미터만 남기고 나머지
    19개 후보·점수를 버렸다. 재현성을 위해 **후보 전체와 내부 홀드아웃
    점수를 전부 반환**한다. `base_seed`를 매 폴드 동일하게 넣으므로
    (RNG가 시드로만 결정) **모든 폴드가 정확히 같은 20개 후보 순서를
    평가한다** — 이 성질을 이용해 ⑤에서 "후보 인덱스별 5폴드 평균
    내부점수가 가장 낮은 조합"을 단일 배포 파라미터로 고를 수 있다
    (폴드마다 다른 1등을 그대로 배포하는 대신, 전 폴드에 걸쳐 두루
    안정적인 조합을 선택 — 사용자가 지적한 "폴드별 값을 배포에 그대로
    못 쓴다" 문제의 해결 규칙)."""
    cut = int(len(train) * 0.8)
    inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]
    space = LGBM_SPACE if model_kind == "LightGBM" else XGB_SPACE
    rng = np.random.default_rng(base_seed)

    best_params, best_mae = None, float("inf")
    trace = []
    for draw_idx in range(N_RANDOM_DRAWS):
        params = sample_params(space, rng)
        if model_kind == "LightGBM":
            m = LGBMRegressor(objective="regression_l1", random_state=base_seed, verbosity=-1, n_jobs=4, **params)
        else:
            m = XGBRegressor(objective="reg:absoluteerror", random_state=base_seed, n_jobs=4, verbosity=0, **params)
        m.fit(inner_tr[feature_cols], inner_tr[target_col])
        pred = np.clip(m.predict(inner_va[feature_cols]), 0, capacity)
        score = mae(inner_va[target_col], pred)
        trace.append({"후보번호": draw_idx, "내부홀드아웃_MAE": score, **params})
        if score < best_mae:
            best_mae, best_params = score, params
    return best_params, trace


def fit_eval(model_kind: str, params: dict | None, train: pd.DataFrame, test: pd.DataFrame,
             feature_cols: list[str], target_col: str, capacity: float, seed: int) -> dict:
    kwargs = params or {}
    if model_kind == "LightGBM":
        m = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **kwargs)
    else:
        m = XGBRegressor(objective="reg:absoluteerror", random_state=seed, n_jobs=4, verbosity=0, **kwargs)
    m.fit(train[feature_cols], train[target_col])
    pred = np.clip(m.predict(test[feature_cols]), 0, capacity)
    y = test[target_col].to_numpy()
    e = y - pred
    denom = float(np.abs(y).sum())
    return {"n": int(len(y)), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "WAPE_pct": float(np.abs(e).sum() / denom * 100) if denom > 0 else None}


DEFAULT_LGBM_ULTRA = dict(n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
                           subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3)
DEFAULT_XGB_SHORT = dict(n_estimators=220, learning_rate=0.04, max_depth=6, min_child_weight=5,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
DEFAULT_LGBM_DAILY = dict(n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=6,
                           min_child_samples=14, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)


def run_ultra_short(config, capacity_kw, seed) -> tuple[list[dict], list[dict], list[dict]]:
    quarter = ultra.load_15min_base()
    hourly_df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS
    param_rows, cmp_rows, trace_rows = [], [], []
    for H in [1, 2, 3, 4]:
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
            chosen = harness.select_features_in_fold(tr_for_sel, candidate_cols, 0.3, True, True)
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 500 or len(test) < 100:
                continue

            best_params, trace = tune_fold("LightGBM", train, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
            base_perf = fit_eval("LightGBM", DEFAULT_LGBM_ULTRA, train, test, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
            tuned_perf = fit_eval("LightGBM", best_params, train, test, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
            param_rows.append({"수평_h": H, "폴드": fold_name, **best_params})
            cmp_rows.append({"수평_h": H, "폴드": fold_name,
                              "기본_MAE": base_perf["MAE"], "기본_RMSE": base_perf["RMSE"],
                              "튜닝_MAE": tuned_perf["MAE"], "튜닝_RMSE": tuned_perf["RMSE"]})
            trace_rows.extend({"수평_h": H, "폴드": fold_name, **t} for t in trace)
            print(f"  [초단기 +{H}h {fold_name}] 기본 MAE={base_perf['MAE']:.3f} → 튜닝 MAE={tuned_perf['MAE']:.3f}")
    return param_rows, cmp_rows, trace_rows


def run_short_term(config, capacity_kw, seed) -> tuple[list[dict], list[dict], list[dict]]:
    df = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    param_rows, cmp_rows, trace_rows = [], [], []
    for H in [1, 24, 48]:
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
            chosen = harness.select_features_in_fold(tr_for_sel, candidate_cols, 0.3, True, True)
            feature_cols = base_cols + chosen
            required = [c for c in feature_cols if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue

            best_params, trace = tune_fold("XGBoost", train, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
            base_perf = fit_eval("XGBoost", DEFAULT_XGB_SHORT, train, test, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
            tuned_perf = fit_eval("XGBoost", best_params, train, test, feature_cols, "목표_발전출력_kW", capacity_kw, seed)
            param_rows.append({"수평_h": H, "폴드": fold_name, **best_params})
            cmp_rows.append({"수평_h": H, "폴드": fold_name,
                              "기본_MAE": base_perf["MAE"], "기본_RMSE": base_perf["RMSE"],
                              "튜닝_MAE": tuned_perf["MAE"], "튜닝_RMSE": tuned_perf["RMSE"]})
            trace_rows.extend({"수평_h": H, "폴드": fold_name, **t} for t in trace)
            print(f"  [단기 +{H}h {fold_name}] 기본 MAE={base_perf['MAE']:.3f} → 튜닝 MAE={tuned_perf['MAE']:.3f}")
    return param_rows, cmp_rows, trace_rows


def run_daily(config, seed) -> tuple[list[dict], list[dict], list[dict]]:
    data, candidate_cols = daily_mod.build_daily_dataset()
    target_col = daily_mod.ACTUAL
    capacity_daily = float(config["site"]["capacity_kw"]) * 24
    param_rows, cmp_rows, trace_rows = [], [], []
    for i, w in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절','')}"
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index <= end)]
        if len(train) < 60 or len(test) < 10:
            continue
        medians = train[candidate_cols].median(numeric_only=True)
        train_f = train.fillna(medians)
        test_f = test.fillna(medians)

        best_params, trace = tune_fold("LightGBM", train_f, candidate_cols, target_col, capacity_daily, seed)
        base_perf = fit_eval("LightGBM", DEFAULT_LGBM_DAILY, train_f, test_f, candidate_cols, target_col, capacity_daily, seed)
        tuned_perf = fit_eval("LightGBM", best_params, train_f, test_f, candidate_cols, target_col, capacity_daily, seed)
        param_rows.append({"폴드": fold_name, **best_params})
        cmp_rows.append({"폴드": fold_name, "기본_MAE": base_perf["MAE"], "기본_RMSE": base_perf["RMSE"],
                          "튜닝_MAE": tuned_perf["MAE"], "튜닝_RMSE": tuned_perf["RMSE"]})
        trace_rows.extend({"폴드": fold_name, **t} for t in trace)
        print(f"  [일간 {fold_name}] 기본 MAE={base_perf['MAE']:.1f} → 튜닝 MAE={tuned_perf['MAE']:.1f}")
    return param_rows, cmp_rows, trace_rows


def pick_robust_params(trace_df: pd.DataFrame, group_cols: list[str], param_names: list[str]) -> pd.DataFrame:
    """★08-21 신규(사용자 지적 반영)★: 폴드별 1등을 그대로 배포하지 않고,
    **후보번호별로 전 폴드 평균 내부홀드아웃 MAE**가 가장 낮은 조합을
    "단일 배포용 파라미터"로 고른다. 동일 시드로 매 폴드 똑같은 후보
    순서를 평가하므로(tune_fold 참고) 후보번호로 안전하게 폴드 간
    평균을 낼 수 있다 — 특정 폴드에만 잘 맞는 파라미터가 아니라 5계절
    전반에 두루 안정적인 조합을 고르는 규칙."""
    key_cols = group_cols + ["후보번호"]
    agg = trace_df.groupby(key_cols)["내부홀드아웃_MAE"].agg(평균내부MAE="mean", 평가폴드수="size").reset_index()
    param_lookup = trace_df.drop_duplicates(subset=key_cols)[key_cols + param_names]
    agg = agg.merge(param_lookup, on=key_cols, how="left")
    if group_cols:
        idx = agg.groupby(group_cols)["평균내부MAE"].idxmin()
    else:
        idx = [agg["평균내부MAE"].idxmin()]
    return agg.loc[idx].reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    summary_lines = ["=== ④ 하이퍼파라미터 튜닝 요약 (내부 홀드아웃으로 선택, 시험폴드는 1회만 채점) ===\n"]

    print("=== 초단기(LightGBM) 튜닝 ===")
    u_params, u_cmp, u_trace = run_ultra_short(config, capacity_kw, seed)
    pd.DataFrame(u_params).to_csv(OUT / "초단기_폴드별_최적파라미터.csv", index=False, encoding="utf-8-sig")
    u_cmp_df = pd.DataFrame(u_cmp)
    u_cmp_df.to_csv(OUT / "초단기_비교.csv", index=False, encoding="utf-8-sig")
    u_trace_df = pd.DataFrame(u_trace)
    u_trace_df.to_csv(OUT / "초단기_전체탐색이력.csv", index=False, encoding="utf-8-sig")
    u_robust = pick_robust_params(u_trace_df, ["수평_h"], list(LGBM_SPACE))
    u_robust.to_csv(OUT / "초단기_단일배포후보.csv", index=False, encoding="utf-8-sig")
    if len(u_cmp_df):
        agg = u_cmp_df.groupby("수평_h")[["기본_MAE", "튜닝_MAE", "기본_RMSE", "튜닝_RMSE"]].mean().round(3)
        summary_lines.append("[초단기] 폴드평균(폴드별 1등 파라미터로 시험 1회 채점)\n" + agg.to_string() + "\n")
        summary_lines.append("[초단기] 단일 배포후보(수평별, 전 폴드 평균 내부MAE 최소)\n" + u_robust.to_string(index=False) + "\n")

    print("\n=== 단기(XGBoost) 튜닝 ===")
    s_params, s_cmp, s_trace = run_short_term(config, capacity_kw, seed)
    pd.DataFrame(s_params).to_csv(OUT / "단기_폴드별_최적파라미터.csv", index=False, encoding="utf-8-sig")
    s_cmp_df = pd.DataFrame(s_cmp)
    s_cmp_df.to_csv(OUT / "단기_비교.csv", index=False, encoding="utf-8-sig")
    s_trace_df = pd.DataFrame(s_trace)
    s_trace_df.to_csv(OUT / "단기_전체탐색이력.csv", index=False, encoding="utf-8-sig")
    s_robust = pick_robust_params(s_trace_df, ["수평_h"], list(XGB_SPACE))
    s_robust.to_csv(OUT / "단기_단일배포후보.csv", index=False, encoding="utf-8-sig")
    if len(s_cmp_df):
        agg = s_cmp_df.groupby("수평_h")[["기본_MAE", "튜닝_MAE", "기본_RMSE", "튜닝_RMSE"]].mean().round(3)
        summary_lines.append("[단기] 폴드평균(폴드별 1등 파라미터로 시험 1회 채점)\n" + agg.to_string() + "\n")
        summary_lines.append("[단기] 단일 배포후보(수평별, 전 폴드 평균 내부MAE 최소)\n" + s_robust.to_string(index=False) + "\n")

    print("\n=== 일간(LightGBM 직접모델) 튜닝 ===")
    d_params, d_cmp, d_trace = run_daily(config, seed)
    pd.DataFrame(d_params).to_csv(OUT / "일간_폴드별_최적파라미터.csv", index=False, encoding="utf-8-sig")
    d_cmp_df = pd.DataFrame(d_cmp)
    d_cmp_df.to_csv(OUT / "일간_비교.csv", index=False, encoding="utf-8-sig")
    d_trace_df = pd.DataFrame(d_trace)
    d_trace_df.to_csv(OUT / "일간_전체탐색이력.csv", index=False, encoding="utf-8-sig")
    d_robust = pick_robust_params(d_trace_df, [], list(LGBM_SPACE))
    d_robust.to_csv(OUT / "일간_단일배포후보.csv", index=False, encoding="utf-8-sig")
    if len(d_cmp_df):
        agg = d_cmp_df[["기본_MAE", "튜닝_MAE", "기본_RMSE", "튜닝_RMSE"]].mean().round(1)
        summary_lines.append("[일간] 폴드평균(폴드별 1등 파라미터로 시험 1회 채점)\n" + agg.to_string() + "\n")
        summary_lines.append("[일간] 단일 배포후보(전 폴드 평균 내부MAE 최소)\n" + d_robust.to_string(index=False) + "\n")

    summary_text = "\n".join(summary_lines)
    (OUT / "요약.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print(f"\n저장 완료: {OUT}")
    print("\n※ '_폴드별_최적파라미터.csv'는 폴드마다 다른 1등 값이라 배포에 그대로 못 쓴다 —")
    print("  실제 후보는 '_단일배포후보.csv'(전 폴드 평균 내부MAE 최소, 시험폴드 미접촉)다.")
    print("  '_전체탐색이력.csv'에 20개 후보×폴드별 내부MAE가 전부 남아있어 재현 가능하다.")
    print("  공식 채택은 ⑤에서 다른 후보(거버넌스 특성·청천지수)와 함께 종합 판단한다.")


if __name__ == "__main__":
    main()
