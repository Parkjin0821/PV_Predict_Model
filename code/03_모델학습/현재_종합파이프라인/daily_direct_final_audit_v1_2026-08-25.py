# -*- coding: utf-8 -*-
"""일간 D+1 직접모델 최종 감사.

핵심 정정
1) 정책 B에 따라 부분가용 과거일의 발전량을 lag/rolling 계산 전에 NaN 처리한다.
2) LightGBM의 native missing을 사용하며 중앙값으로 대체하지 않는다.
3) 실제 배포 전체특성에서 한 그룹씩 제거하는 LOGO를 재현 가능한 표로 남긴다.
4) 시험일은 v5 정상 목표일로 고정하고 모든 후보가 완전히 같은 321일을 평가한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

WORKSPACE = Path(__file__).resolve().parent
PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
ROOT = PROJECT / "03_모델학습" / "현재_종합파이프라인"
OUT = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25"
MODEL_DIR = OUT / "fold_models"
ACTUAL = "실제_일간발전량_kWh"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


e2e = load_module("daily_audit_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
tuning = load_module("daily_audit_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
fhv = load_module("daily_audit_fhv", "fold_hour_prevalidation_v1_2026-08-24.py")
OFFICIAL_WINDOWS = fhv.OFFICIAL_B_WINDOWS

B_SUFFIXES = {
    "기상청관측_현지기압_hPa", "기상청관측_해면기압_hPa", "기상청관측_지면온도_C",
    "기상청관측_풍향_deg", "기상청관측_일조시간_hr",
}
C_SUFFIXES = {"기상청관측_강수량_mm", "기상청관측_적설_cm"}
D_SUFFIXES = {
    "plant_input_power_kw", "mean_input_voltage_v", "mean_frequency_hz",
    "mean_power_factor", "mean_communication_ok", "inverters_available",
}
A_SUFFIXES = {
    "기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_전운량_pct",
    "기상청관측_일사량_W_m2", "기상청관측_풍속_m_s",
}


def classify(col: str) -> str:
    if col in {"해발고도_m", "설비용량_kW"}:
        return "A"
    if col in {
        "7일전_일간발전량_kWh", "7일전_결측여부", "2일전_일간발전량_kWh", "2일전_결측여부",
        "2일전기준_7일이동평균_kWh", "2일전기준_30일이동평균_kWh", "2일전기준_30일표준편차_kWh",
        "목표일_연주기_sin", "목표일_연주기_cos", "목표일_월", "목표일_DIFSWRF_유효개수",
    }:
        return "A"
    if col.startswith("목표일예보_POP_"):
        return "C"
    if col.startswith("목표일예보_"):
        return "A"
    if col.startswith("2일전평균_"):
        suffix = col[len("2일전평균_"):]
        if suffix in B_SUFFIXES:
            return "B"
        if suffix in C_SUFFIXES:
            return "C"
        if suffix in D_SUFFIXES:
            return "D"
        if suffix in A_SUFFIXES:
            return "A"
        raise ValueError(f"미분류 특성: {col}")
    if col == "목표일_요일":
        return "E"
    raise ValueError(f"미분류 특성: {col}")


def corrected_dataset(capacity_kw: float):
    data, features = e2e.build_daily_dataset_v5(capacity_kw)
    raw = pd.read_parquet(e2e.V5_DIR / "집계_일간_실제발전량_v5.parquet").copy()
    raw.index = pd.to_datetime(raw.index)
    partial = raw.get("부분가용일", 0).fillna(0).astype(float)
    history = raw["일간발전량_kWh"].where(partial < 1)

    for lag, name in ((7, "7일전"), (2, "2일전")):
        s = history.shift(lag).reindex(data.index)
        data[f"{name}_일간발전량_kWh"] = s
        data[f"{name}_결측여부"] = s.isna().astype(float)
    shifted = history.shift(2)
    data["2일전기준_7일이동평균_kWh"] = shifted.rolling(7, min_periods=4).mean().reindex(data.index)
    data["2일전기준_30일이동평균_kWh"] = shifted.rolling(30, min_periods=15).mean().reindex(data.index)
    data["2일전기준_30일표준편차_kWh"] = shifted.rolling(30, min_periods=15).std().reindex(data.index)
    return data.sort_index(), features, int((partial > 0).sum())


def cast_params(params):
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, np.generic):
            v = v.item()
        if k in {"n_estimators", "num_leaves", "max_depth", "min_child_samples"}:
            v = int(v)
        out[k] = v
    return out


def predict_oof(data, features, capacity_kw, seed, label, save_models=False):
    rows, audits = [], []
    cap_day = capacity_kw * 24
    for fold, start_s, end_s in OFFICIAL_WINDOWS:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s) + pd.Timedelta(days=1)
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index < end)]
        if len(train) < 60 or len(test) < 10:
            raise RuntimeError(f"표본 부족: {label}/{fold} train={len(train)} test={len(test)}")
        # NaN은 LightGBM이 직접 분기한다. 시험구간/전체기간 통계로 채우지 않는다.
        params, trace = tuning.tune_fold("LightGBM", train, features, ACTUAL, cap_day, seed)
        params = cast_params(params)
        model = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **params)
        model.fit(train[features], train[ACTUAL])
        pred = np.clip(model.predict(test[features]), 0, cap_day)
        artifact = ""
        reload_diff = np.nan
        if save_models:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            artifact_path = MODEL_DIR / f"일간_{fold}_{label}.joblib"
            bundle = {
                "model": model, "features": features, "params": params,
                "target_transform": "daily_kWh", "missing_strategy": "LightGBM_native_missing+flags",
                "history_policy": "부분가용 과거일은 lag/rolling 계산 전 NaN",
                "policy": "B_구간제외", "capacity_profile": "inverter_registered_sum_219",
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }
            joblib.dump(bundle, artifact_path)
            loaded = joblib.load(artifact_path)
            pred_reload = np.clip(loaded["model"].predict(test[loaded["features"]]), 0, cap_day)
            reload_diff = float(np.max(np.abs(pred - pred_reload)))
            artifact = str(artifact_path)
        for date, y, p in zip(test.index, test[ACTUAL].to_numpy(), pred):
            rows.append({"날짜": date, "폴드": fold, "실제_kWh": y, "예측_kWh": p, "후보": label, "모델파일": artifact})
        audits.append({
            "후보": label, "폴드": fold, "학습행수": len(train), "시험행수": len(test),
            "학습최종일": train.index.max(), "시험최초일": test.index.min(),
            "학습시험분리": bool(train.index.max() < test.index.min()),
            "특성수": len(features), "재적재최대차이": reload_diff,
            "내부탐색후보수": len(trace), "NaN중앙값대체": False,
        })
    return pd.DataFrame(rows), pd.DataFrame(audits)


def metrics(frame):
    err = frame["실제_kWh"] - frame["예측_kWh"]
    return float(err.abs().mean()), float(np.sqrt(np.mean(err ** 2)))


def compare(base, cand, label):
    merged = base.merge(cand, on=["날짜", "폴드", "실제_kWh"], suffixes=("_기준", "_후보"), validate="one_to_one")
    b = merged.rename(columns={"예측_kWh_기준": "예측_kWh"})
    c = merged.rename(columns={"예측_kWh_후보": "예측_kWh"})
    b_mae, b_rmse = metrics(b)
    c_mae, c_rmse = metrics(c)
    fold_rows = []
    for fold, g in merged.groupby("폴드", sort=False):
        gb = g.rename(columns={"예측_kWh_기준": "예측_kWh"})
        gc = g.rename(columns={"예측_kWh_후보": "예측_kWh"})
        bm, br = metrics(gb)
        cm, cr = metrics(gc)
        fold_rows.append({"후보": label, "폴드": fold, "n": len(g), "기준_MAE": bm, "후보_MAE": cm,
                          "MAE악화율_pct": (cm-bm)/bm*100, "기준_RMSE": br, "후보_RMSE": cr,
                          "RMSE악화율_pct": (cr-br)/br*100})
    folds = pd.DataFrame(fold_rows)
    mae_gain = (b_mae-c_mae)/b_mae*100
    rmse_gain = (b_rmse-c_rmse)/b_rmse*100
    worst = float(folds[["MAE악화율_pct", "RMSE악화율_pct"]].max().max())
    adopt = bool(mae_gain >= 1 and rmse_gain >= 1 and worst < 5)
    return {
        "후보": label, "n": len(merged), "기준_MAE": b_mae, "기준_RMSE": b_rmse,
        "후보_MAE": c_mae, "후보_RMSE": c_rmse, "MAE개선율_pct": mae_gain,
        "RMSE개선율_pct": rmse_gain, "최대폴드악화율_pct": worst, "채택": adopt,
        "판정": "통과" if adopt else "기각",
    }, folds, merged


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(cfg["site"]["capacity_kw"])
    seed = int(cfg["random_seed"])
    if capacity_kw != 219:
        raise RuntimeError(f"활성 용량이 219kW가 아님: {capacity_kw}")
    data, all_features, partial_days = corrected_dataset(capacity_kw)
    groups = {g: [] for g in "ABCDE"}
    for col in all_features:
        groups[classify(col)].append(col)
    assert sorted(sum(groups.values(), [])) == sorted(all_features)

    print(f"일간 정상 목표일={len(data)}, 부분가용 역사일={partial_days}, 전체특성={len(all_features)}")
    print("그룹수:", {g: len(v) for g, v in groups.items()})
    baseline, audit_base = predict_oof(data, all_features, capacity_kw, seed, "전체58", False)
    verdicts, fold_tables, merged_tables, predictions = [], [], [], [baseline]
    candidate_frames = {}
    for g in "BCDE":
        features = [c for c in all_features if c not in groups[g]]
        label = f"{g}제외_{len(features)}"
        print(f"LOGO {label} 실행")
        frame, audit = predict_oof(data, features, capacity_kw, seed, label, False)
        candidate_frames[g] = (frame, features)
        predictions.append(frame)
        audit_base = pd.concat([audit_base, audit], ignore_index=True)
        verdict, folds, merged = compare(baseline, frame, label)
        verdict["제외그룹"] = g
        verdict["특성수"] = len(features)
        verdicts.append(verdict)
        fold_tables.append(folds)
        merged["후보"] = label
        merged_tables.append(merged)
        print(verdict)

    verdict_df = pd.DataFrame(verdicts)
    passed = verdict_df.loc[verdict_df["채택"], "제외그룹"].tolist()
    if len(passed) > 1:
        combined_features = [c for c in all_features if all(c not in groups[g] for g in passed)]
        final_label = "제외" + "".join(passed) + f"_{len(combined_features)}"
        final_frame, final_audit = predict_oof(data, combined_features, capacity_kw, seed, final_label, True)
        final_verdict, final_folds, _ = compare(baseline, final_frame, final_label)
        if not final_verdict["채택"]:
            best = verdict_df[verdict_df["채택"]].sort_values(["MAE개선율_pct", "RMSE개선율_pct"], ascending=False).iloc[0]
            chosen_group = best["제외그룹"]
            final_frame0, combined_features = candidate_frames[chosen_group]
            final_label = f"{chosen_group}제외_{len(combined_features)}"
            final_frame, final_audit = predict_oof(data, combined_features, capacity_kw, seed, final_label, True)
        audit_base = pd.concat([audit_base, final_audit], ignore_index=True)
    elif len(passed) == 1:
        chosen_group = passed[0]
        _, combined_features = candidate_frames[chosen_group]
        final_label = f"{chosen_group}제외_{len(combined_features)}"
        final_frame, final_audit = predict_oof(data, combined_features, capacity_kw, seed, final_label, True)
        audit_base = pd.concat([audit_base, final_audit], ignore_index=True)
    else:
        combined_features = all_features
        final_label = "전체58"
        final_frame, final_audit = predict_oof(data, combined_features, capacity_kw, seed, final_label, True)
        audit_base = pd.concat([audit_base, final_audit], ignore_index=True)

    final_mae, final_rmse = metrics(final_frame)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "LOGO_행단위예측.csv", index=False, encoding="utf-8-sig")
    verdict_df.to_csv(OUT / "LOGO_판정.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_tables, ignore_index=True).to_csv(OUT / "LOGO_폴드별.csv", index=False, encoding="utf-8-sig")
    pd.concat(merged_tables, ignore_index=True).to_csv(OUT / "LOGO_동일행비교.csv", index=False, encoding="utf-8-sig")
    audit_base.to_csv(OUT / "감사.csv", index=False, encoding="utf-8-sig")
    final_frame.to_csv(OUT / "최종_직접모델_OOF.csv", index=False, encoding="utf-8-sig")
    state = {
        "완료": True, "공식기준": "v5·정책B·219kW·정정5폴드", "시험일수": len(final_frame),
        "부분가용역사일수": partial_days, "부분가용과거발전량처리": "lag/rolling 계산 전 NaN",
        "결측처리": "LightGBM native missing + 결측여부 표시(중앙값 대체 안 함)",
        "LOGO통과제외그룹": passed, "최종구성": final_label, "최종특성수": len(combined_features),
        "최종특성목록": combined_features, "MAE_kWh": final_mae, "RMSE_kWh": final_rmse,
        "재적재전부일치": bool(final_audit["재적재최대차이"].fillna(0).le(1e-12).all()),
        "다음": "D-1 10시 고정 시간모델 일합계와 누출 없는 계층조정 판정",
    }
    (OUT / "최종상태.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
