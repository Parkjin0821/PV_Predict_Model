# -*- coding: utf-8 -*-
"""일간 D+1 계층조정 최종 감사.

일간 직접모델 OOF와 D-1 10시 고정 발행의 06~19시 14개 시간예측 합계를
같은 정상 목표일에서 비교한다. 결합 가중치는 각 시험폴드보다 과거에 나온
OOF만 사용해 정하므로 현재/미래 시험정답을 보지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19")
ROOT = PROJECT / "03_모델학습" / "현재_종합파이프라인"
DIRECT_OUT = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25"
OUT = ROOT / "outputs" / "일간_계층조정_최종감사_v1_2026-08-25"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


e2e = load_module("daily_hier_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc = load_module("daily_hier_dpc", "defect_policy_comparison_v1_2026-08-21.py")
harness = dpc.harness
fhv = load_module("daily_hier_fhv", "fold_hour_prevalidation_v1_2026-08-24.py")
OFFICIAL_WINDOWS = fhv.OFFICIAL_B_WINDOWS
N_INVERTERS = 5
ISSUE_HOUR = 10
TARGET_HOURS = list(range(6, 20))


def metrics(y, p):
    e = np.asarray(y, float) - np.asarray(p, float)
    return float(np.abs(e).mean()), float(np.sqrt(np.mean(e ** 2)))


def hourly_profile_oof(capacity_kw, seed):
    rows, audits = [], []
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    for target_hour in TARGET_HOURS:
        horizon = 15 + target_hour  # build_frame의 H-1 규약: D-1 10시 -> D 06~19시
        frame = e2e.add_difswrf_flag(dpc.load_short_frame(horizon))
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간", "추정_일조시간_hr")]
        issue = frame[frame.index.hour == ISSUE_HOUR].copy()
        issue["_대상시각"] = issue.index + pd.to_timedelta(horizon - 1, unit="h")
        issue["_대상일"] = issue["_대상시각"].dt.normalize()
        if not (issue["_대상시각"].dt.hour == target_hour).all():
            raise AssertionError(f"대상시각 규약 불일치: {target_hour}")

        for fold, start_s, end_s in OFFICIAL_WINDOWS:
            start, end = pd.Timestamp(start_s), pd.Timestamp(end_s) + pd.Timedelta(days=1)
            full = issue["_목표_가용인버터수"] >= N_INVERTERS
            train_all = issue[(issue["_대상일"] < start) & full]
            test_all = issue[(issue["_대상일"] >= start) & (issue["_대상일"] < end) & full]
            if len(train_all) < 60 or len(test_all) < 10:
                raise RuntimeError(f"표본 부족: {fold}/{target_hour}시 train={len(train_all)} test={len(test_all)}")
            selector_train = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(
                selector_train, candidate_cols, threshold=0.3,
                apply_multicollinearity=True, apply_deploy_filter=True,
            )
            chosen = [c for c in chosen if c != "추정_일조시간_hr"]
            features = base_cols + chosen
            native_missing_ok = set(harness.NATIVE_MISSING_OK) | set(dpc.EXTRA_NATIVE_MISSING_OK)
            required = [c for c in features if c not in native_missing_ok] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 60 or len(test) < 10:
                bad = {c: int(test_all[c].isna().sum()) for c in required if c in test_all and test_all[c].isna().any()}
                raise RuntimeError(f"모델준비 표본 부족: {fold}/{target_hour}시 train={len(train)} test={len(test)} 결측={bad}")
            model = harness.make_model("XGBoost", seed)
            model.fit(train[features], train["목표_발전출력_kW"])
            pred = np.clip(model.predict(test[features]), 0, capacity_kw)
            for issued, target_at, target_date, y, p in zip(
                test.index, test["_대상시각"], test["_대상일"], test["목표_발전출력_kW"], pred
            ):
                rows.append({"폴드": fold, "발행시각": issued, "대상시각": target_at, "날짜": target_date,
                             "대상시": target_hour, "실제_kW": y, "예측_kW": p})
            audits.append({"폴드": fold, "대상시": target_hour, "horizon": horizon,
                           "학습행수": len(train), "시험행수": len(test), "특성수": len(features),
                           "학습최종대상일": train["_대상일"].max(), "시험최초대상일": test["_대상일"].min(),
                           "학습시험분리": bool(train["_대상일"].max() < test["_대상일"].min()),
                           "정책": "B_구간제외", "발행시각": "D-1 10:00 고정"})
        print(f"시간모델 대상 {target_hour:02d}시 완료")
    row = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if row.duplicated(["폴드", "날짜", "대상시"]).any():
        raise AssertionError("시간모델 행 중복")
    daily = row.groupby(["폴드", "날짜"], as_index=False).agg(
        시간예측합계_kWh=("예측_kW", "sum"), 시간실제합계_kWh=("실제_kW", "sum"), 시각수=("대상시", "size")
    )
    daily = daily[daily["시각수"] == len(TARGET_HOURS)].copy()
    return row, daily, audit


def optimize_weight(prior):
    if len(prior) < 30:
        return 1.0, "과거OOF 30일 미만→직접모델만"
    y = prior["실제_kWh"].to_numpy(float)
    d = prior["직접예측_kWh"].to_numpy(float)
    h = prior["시간예측합계_kWh"].to_numpy(float)
    grid = np.linspace(0, 1, 1001)
    loss = np.array([np.abs(y - (w*d + (1-w)*h)).mean() for w in grid])
    w = float(grid[int(np.argmin(loss))])
    return w, "과거폴드OOF MAE 최소"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(cfg["site"]["capacity_kw"])
    seed = int(cfg["random_seed"])
    if capacity_kw != 219:
        raise RuntimeError(f"활성 용량이 219kW가 아님: {capacity_kw}")
    direct = pd.read_csv(DIRECT_OUT / "최종_직접모델_OOF.csv", encoding="utf-8-sig", parse_dates=["날짜"])
    direct = direct.rename(columns={"예측_kWh": "직접예측_kWh"})
    hour_rows, hourly_daily, audit = hourly_profile_oof(capacity_kw, seed)
    merged = direct[["폴드", "날짜", "실제_kWh", "직접예측_kWh"]].merge(
        hourly_daily, on=["폴드", "날짜"], how="inner", validate="one_to_one"
    )
    merged["실제합계차이_kWh"] = merged["실제_kWh"] - merged["시간실제합계_kWh"]

    fold_order = [w[0] for w in OFFICIAL_WINDOWS]
    pred_parts, weight_rows = [], []
    prior_parts = []
    for fold in fold_order:
        test = merged[merged["폴드"] == fold].copy()
        prior = pd.concat(prior_parts, ignore_index=True) if prior_parts else merged.iloc[0:0]
        w, reason = optimize_weight(prior)
        test["직접가중치"] = w
        test["시간합계가중치"] = 1-w
        test["계층조정예측_kWh"] = w*test["직접예측_kWh"] + (1-w)*test["시간예측합계_kWh"]
        pred_parts.append(test)
        weight_rows.append({"폴드": fold, "과거OOF일수": len(prior), "직접가중치": w,
                            "시간합계가중치": 1-w, "결정근거": reason})
        prior_parts.append(test)
    result = pd.concat(pred_parts, ignore_index=True)

    rows = []
    for name, col in (("직접모델", "직접예측_kWh"), ("시간모델합계", "시간예측합계_kWh"),
                      ("과거OOF계층조정", "계층조정예측_kWh")):
        mae, rmse = metrics(result["실제_kWh"], result[col])
        wape = float(np.abs(result["실제_kWh"]-result[col]).sum()/result["실제_kWh"].abs().sum()*100)
        rows.append({"구성": name, "n": len(result), "MAE_kWh": mae, "RMSE_kWh": rmse, "WAPE_pct": wape})
    perf = pd.DataFrame(rows)
    direct_mae = float(perf.loc[perf["구성"] == "직접모델", "MAE_kWh"].iloc[0])
    direct_rmse = float(perf.loc[perf["구성"] == "직접모델", "RMSE_kWh"].iloc[0])
    rec_mae = float(perf.loc[perf["구성"] == "과거OOF계층조정", "MAE_kWh"].iloc[0])
    rec_rmse = float(perf.loc[perf["구성"] == "과거OOF계층조정", "RMSE_kWh"].iloc[0])
    mae_gain = (direct_mae-rec_mae)/direct_mae*100
    rmse_gain = (direct_rmse-rec_rmse)/direct_rmse*100
    fold_rows = []
    for fold, g in result.groupby("폴드", sort=False):
        dm, dr = metrics(g["실제_kWh"], g["직접예측_kWh"])
        cm, cr = metrics(g["실제_kWh"], g["계층조정예측_kWh"])
        fold_rows.append({"폴드": fold, "n": len(g), "직접_MAE": dm, "조정_MAE": cm,
                          "MAE악화율_pct": (cm-dm)/dm*100, "직접_RMSE": dr, "조정_RMSE": cr,
                          "RMSE악화율_pct": (cr-dr)/dr*100})
    fold_perf = pd.DataFrame(fold_rows)
    worst = float(fold_perf[["MAE악화율_pct", "RMSE악화율_pct"]].max().max())
    adopt = bool(mae_gain >= 1 and rmse_gain >= 1 and worst < 5)

    # 최종 배포 후보 가중치는 모든 과거 OOF로만 산출하되, 같은 OOF의 공식 성능으로 재사용하지 않는다.
    deployment_w, _ = optimize_weight(result)
    state = {
        "완료": True, "공식기준": "v5·정책B·219kW·정정5폴드",
        "시간모델운영규약": "D-1 10시 1회 발행, D 06~19시 14개 XGBoost 예측 합계",
        "공통시험일수": len(result), "직접_MAE_kWh": direct_mae, "직접_RMSE_kWh": direct_rmse,
        "계층조정_MAE_kWh": rec_mae, "계층조정_RMSE_kWh": rec_rmse,
        "MAE개선율_pct": mae_gain, "RMSE개선율_pct": rmse_gain,
        "최대폴드악화율_pct": worst, "계층조정채택": adopt,
        "최종공식구성": "과거OOF계층조정" if adopt else "일간 직접모델(전체58특성)",
        "향후배포후보_직접가중치": deployment_w,
        "주의": "향후배포후보 가중치는 OOF 전체에서 학습했으므로 이번 공식 성능 산출에는 사용하지 않음",
        "실제일간총량_vs_14시간합계_최대차이_kWh": float(result["실제합계차이_kWh"].abs().max()),
        "감사전부통과": bool(audit["학습시험분리"].all()),
    }
    hour_rows.to_csv(OUT / "시간모델_행단위.csv", index=False, encoding="utf-8-sig")
    hourly_daily.to_csv(OUT / "시간모델_일합계.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "시간모델_누출감사.csv", index=False, encoding="utf-8-sig")
    result.to_csv(OUT / "계층조정_동일일예측.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weight_rows).to_csv(OUT / "폴드별_과거OOF가중치.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(OUT / "구성별_성능.csv", index=False, encoding="utf-8-sig")
    fold_perf.to_csv(OUT / "폴드별_성능.csv", index=False, encoding="utf-8-sig")
    (OUT / "최종상태.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(perf.to_string(index=False))
    print(pd.DataFrame(weight_rows).to_string(index=False))
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
