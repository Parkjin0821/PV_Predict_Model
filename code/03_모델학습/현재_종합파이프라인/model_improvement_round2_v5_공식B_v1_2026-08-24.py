# -*- coding: utf-8 -*-
"""⑥ 기존 1~4번 개선후보(2차, v3/240kW/구경계 기준)를 3차 공식모델
기준(v5·공식 정책B·219kW·정정 공식 5폴드)으로 재판정한다.

## 왜 다시 하나
2차(`model_improvement_round2_v1_2026-08-21.py --step 1..4`)는 v3
데이터·구경계 5폴드(`config.json`의 `cross_validation_windows`, 가을
10-21 시작)로 ①NWP 분위사상 ②구름전이·강수 OOF 잔차보정 ③일사예보
날씨군집화 ④트리 구조개선을 판정했다. 그 뒤 v5(인버터5 결함 시간단위
복구)·공식 정책B(결함구간 제외)·219kW·정정 공식 5폴드(가을 11-17
시작)로 전 모델을 재학습(⑤)했으므로, 같은 4개 후보를 그 새 기준에서
다시 판정해야 "2차 판정이 3차에서도 유효한가"를 확인할 수 있다.

## 재사용(재구현 금지 원칙)
- v5 데이터 적재: `defect_policy_comparison_v1_2026-08-21`
  (`load_ultra_frame`/`load_short_frame`, `harness`/`ultra` 재노출)
- 공식 5폴드: `fold_hour_prevalidation_v1_2026-08-24.OFFICIAL_B_WINDOWS`
  (가을 11-17 정정판)
- 후보 로직 자체(QM/OOF잔차보정/날씨군집/구조탐색 후보·판정규칙):
  `model_improvement_round2_v1_2026-08-21`의 `apply_quantile_mapping`,
  `add_weather_clusters`, `fit_oof_residual`, `apply_residual`,
  `make_model`, `verdict`, `PipelineState`, `candidate_state`,
  `add_target_observations_ultra/_short`, `NWP_OBS_PAIRS`,
  `CLUSTER_SOURCE`, `STEP_NAMES` — 전부 그대로 가져다 쓴다.
- **구조선택만 08-24 결정에 따라 `choose_structure`(단일 80/20분할)
  대신 `choose_structure_kfold`(시간순 walk-forward 내부 4분할평균)를
  쓴다** — 3차 재학습(⑤)이 이미 이걸 기본값으로 채택했으므로 재판정도
  실제 프로덕션과 같은 방법으로 해야 공정하다.
- 일간 v5 데이터셋: `e2e_retrain_v5_공식B_v1_2026-08-24.build_daily_dataset_v5`
  (Codex 08-24 지적사항 반영판 그대로 사용).
- 하이퍼파라미터 튜닝(일간 기준선): `hyperparameter_tuning_v1_2026-08-21.tune_fold`

## 공식 B 정책
초단기·단기 학습·시험 모두 대상시각 가용인버터수=5인 행만 사용
(defect_policy_comparison/e2e_retrain_v5와 동일 정의). 일간은
목표일 자체가 부분가용인 날만 제외(v5 일간 데이터셋 자체가 이미 그렇게
구성됨).

## 실행 범위
1~3번은 초단기(+1~4h)·단기(+1/+24/+48h)에만 적용(시간단위 자료의 NWP
보정·구름전이 로직이라 일간에 적용하면 의미가 달라짐 — 2차와 동일
원칙). 4번(구조개선)만 일간까지 포함한다. 4단계 전부 한 프로세스에서
누적 적용(직전 단계 채택상태를 다음 단계가 이어받음, 2차와 동일 절차).

## 산출물 (`outputs/⑥재판정_v5_공식B_v1_2026-08-24/`)
- `{단계}_{개선안}/판정표.csv`, `계절안정성.csv`, `동일행_예측정답.csv`
- `판정_전체.csv`(4단계 통합), `채택상태.json`
- **이 재판정 결과로 3차 공식모델(⑤ 산출물)을 갱신하지 않는다** — 별도
  후보채택 여부만 확정한다. 채택되면 다음 세션에서 공식 파이프라인에
  반영할지 사용자에게 보고 후 결정.

로컬 재학습만(API 없음).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "⑥재판정_v5_공식B_v1_2026-08-24"
STATE_PATH = OUT / "채택상태.json"
DIF = "DIFSWRF_bsrn정제"
N_INVERTERS = 5


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpc = _load("r6_dpc", "defect_policy_comparison_v1_2026-08-21.py")
harness, ultra = dpc.harness, dpc.ultra
improvement = _load("r6_improvement", "model_improvement_round2_v1_2026-08-21.py")
fhv = _load("r6_fhv", "fold_hour_prevalidation_v1_2026-08-24.py")
clearsky = _load("r6_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
tuning = _load("r6_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
e2e = _load("r6_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")

OFFICIAL_WINDOWS = fhv.OFFICIAL_B_WINDOWS
ULTRA_TARGET = improvement.ULTRA_TARGET  # {1:"raw",2:"raw",3:"kappa",4:"kappa"} — 2차와 동일 정의


def load_ultra_frame_v5(horizon: int) -> pd.DataFrame:
    frame = dpc.load_ultra_frame(horizon)
    quarter = pd.read_parquet(dpc.V5_DIR / "집계_15분_자료_v5.parquet").rename(columns=ultra.RENAME_15MIN)
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    improvement.add_target_observations_ultra(frame, quarter, hourly, horizon)
    return frame


def load_short_frame_v5(horizon: int) -> pd.DataFrame:
    frame = dpc.load_short_frame(horizon)
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(dpc.V5_DIR / "집계_1시간_자료_v5.parquet")
    common = hourly.index.intersection(v5_1h.index)
    hourly = hourly.loc[common].copy()
    hourly["plant_output_kw"] = v5_1h.loc[common, "발전출력_kW"]
    hourly["inverters_available"] = v5_1h.loc[common, "가용인버터수"]
    improvement.add_target_observations_short(frame, hourly, horizon)
    return frame


def predict_fold_v5(tier: str, train0: pd.DataFrame, test0: pd.DataFrame, features0: list[str],
                    state, target_mode: str, capacity: float, seed: int):
    train, test, features, qm_log, cluster_log = improvement.prepare_features(train0, test0, features0, state, seed)
    target = "_카파" if target_mode == "kappa" else "목표_발전출력_kW"
    if state.tree_structure:
        params, structure_name = improvement.choose_structure_kfold(
            tier, train, features, target, target_mode, capacity, seed)
    else:
        params, structure_name = None, "기본"
    corr = (improvement.fit_oof_residual(train, features, target, target_mode, tier, capacity, seed, params)
            if state.residual_correction else None)
    model = improvement.make_model(tier, seed, params)
    model.fit(train[features], train[target])
    pred = improvement.predict_scale(model.predict(test[features]), test, target_mode, capacity)
    if corr is not None:
        pred = improvement.apply_residual(pred, test, corr, capacity)
    return pred, {"QM특성": qm_log, "군집원천특성": cluster_log, "구조": structure_name,
                  "잔차그룹수": len(corr["groups"]) if corr else 0}


def run_power_tier_v5(tier: str, horizon: int, capacity: float, seed: int, current, candidate) -> pd.DataFrame:
    if tier == "초단기":
        frame = load_ultra_frame_v5(horizon)
        candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
        target_mode = ULTRA_TARGET[horizon]
        min_elev = clearsky.MIN_ELEVATION_DEG if target_mode == "kappa" else 0.0
        frame["_청천_kW"] = np.clip(capacity * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
        frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]
        min_train, min_test = 300, 30
    else:
        frame = load_short_frame_v5(horizon)
        candidate_cols = harness.FEATURE_SETS["전체후보"]
        target_mode = "raw"
        daylight = frame[frame["목표_낮시간"] > 0]
        min_train, min_test = 200, 30

    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간") and not c.startswith("__target_obs__")]
    native_ok = harness.NATIVE_MISSING_OK | {DIF}
    rows = []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train_all = daylight[(daylight.index < start) & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
        test_all = daylight[(daylight.index >= start) & (daylight.index < end)
                            & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
        if len(train_all) < min_train or len(test_all) < min_test:
            print(f"  [{tier} +{horizon}h {fold}] 표본부족(train={len(train_all)}, test={len(test_all)}) — 건너뜀")
            continue
        chosen = harness.select_features_in_fold(
            train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
        if target_mode == "kappa":
            required += ["_청천_kW", "_카파"]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=required)
        if len(train) < min_train or len(test) < min_test:
            continue
        p0, log0 = predict_fold_v5(tier, train, test, features, current, target_mode, capacity, seed)
        p1, log1 = predict_fold_v5(tier, train, test, features, candidate, target_mode, capacity, seed)
        for ts, y, a, b in zip(test.index, test["목표_발전출력_kW"], p0, p1):
            rows.append({"티어": tier, "수평_h": horizon, "폴드": fold, "발행시각": ts,
                        "실제": y, "현행예측": a, "후보예측": b,
                        "현행상세": json.dumps(log0, ensure_ascii=False),
                        "후보상세": json.dumps(log1, ensure_ascii=False)})
        print(f"  [{tier} +{horizon}h {fold}] 완료 (학습{len(train):,}/시험{len(test):,})")
    return pd.DataFrame(rows)


def run_daily_structure_v5(capacity: float, seed: int, current, candidate) -> pd.DataFrame:
    data, features = e2e.build_daily_dataset_v5(capacity)
    target = "실제_일간발전량_kWh"
    daily_capacity = capacity * 24
    rows = []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train, test = data[data.index < start], data[(data.index >= start) & (data.index < end)]
        if len(train) < 60 or len(test) < 10:
            print(f"  [일간 {fold}] 표본부족(train={len(train)}, test={len(test)}) — 건너뜀")
            continue
        med = train[features].median(numeric_only=True)
        train, test = train.copy(), test.copy()
        train[features] = train[features].fillna(med)
        test[features] = test[features].fillna(med)

        base_params, _ = tuning.tune_fold("LightGBM", train, features, target, daily_capacity, seed)
        base_params = e2e._cast(base_params, ["n_estimators", "num_leaves", "max_depth", "min_child_samples"])
        base = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **base_params)
        base.fit(train[features], train[target])
        p0 = np.clip(base.predict(test[features]), 0, daily_capacity)

        if candidate.tree_structure:
            params, name = improvement.choose_structure_kfold(
                "일간", train.rename(columns={target: "목표_발전출력_kW"}), features,
                "목표_발전출력_kW", "raw", daily_capacity, seed)
            model = improvement.make_model("일간", seed, params)
            model.fit(train[features], train[target])
            p1 = np.clip(model.predict(test[features]), 0, daily_capacity)
        else:
            p1, name = p0.copy(), "공식폴드내튜닝(v5)"
        for ts, y, a, b in zip(test.index, test[target], p0, p1):
            rows.append({"티어": "일간", "수평_h": "D+1", "폴드": fold, "발행시각": ts,
                        "실제": y, "현행예측": a, "후보예측": b, "현행상세": "공식폴드내튜닝(v5)",
                        "후보상세": name})
        print(f"  [일간 {fold}] 완료 (학습{len(train):,}/시험{len(test):,})")
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity}kW) / 정책: B_구간제외 / "
          f"폴드: 공식 5폴드(가을 11-17 정정) — 3차 공식모델 기준 재판정\n")

    states = improvement.default_state()
    all_verdicts: list[dict] = []
    all_fold_rows: list[dict] = []

    for step in (1, 2, 3, 4):
        targets = [("초단기", h) for h in (1, 2, 3, 4)] + [("단기", h) for h in (1, 24, 48)]
        if step == 4:
            targets.append(("일간", "D+1"))
        step_all_rows = []
        for tier, horizon in targets:
            key = f"{tier}_{horizon}"
            current = improvement.PipelineState(**states[key])
            cand = improvement.candidate_state(current, step)
            print(f"[{improvement.STEP_NAMES[step]}] {tier} {horizon} 시작", flush=True)
            if tier == "일간":
                rows = run_daily_structure_v5(capacity, seed, current, cand)
            else:
                rows = run_power_tier_v5(tier, int(horizon), capacity, seed, current, cand)
            if rows.empty:
                raise RuntimeError(f"시험행 없음: {key}")
            result, folds = improvement.verdict(rows)
            result.update({"단계": step, "개선안": improvement.STEP_NAMES[step], "티어": tier,
                          "수평_h": horizon, "시험행수": len(rows)})
            all_verdicts.append(result)
            for fr in folds:
                fr.update({"단계": step, "티어": tier, "수평_h": horizon})
                all_fold_rows.append(fr)
            rows["단계"] = step
            rows["개선안"] = improvement.STEP_NAMES[step]
            step_all_rows.append(rows)
            if result["채택"]:
                states[key] = asdict(cand)
            print(f"  -> {result['판정사유']} / MAE {result['MAE개선율_pct']:.2f}% / "
                  f"RMSE {result['RMSE개선율_pct']:.2f}%", flush=True)

        step_dir = OUT / f"{step}_{improvement.STEP_NAMES[step]}"
        step_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(step_all_rows, ignore_index=True).to_csv(
            step_dir / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([v for v in all_verdicts if v["단계"] == step]).to_csv(
            step_dir / "판정표.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([f for f in all_fold_rows if f["단계"] == step]).to_csv(
            step_dir / "계절안정성.csv", index=False, encoding="utf-8-sig")

    payload = {"완료단계": 4, "티어수평별": states,
              "동결규칙": "동일행, MAE·RMSE 모두 1% 이상 개선, 계절 MAE 5% 악화 없음",
              "기준": "3차 공식모델(v5·공식B·219kW·정정 공식5폴드), 구조선택=choose_structure_kfold",
              "인식론적한계": "동일 5계절을 반복 사용한 후보검증이며 완전히 새로운 독립시험이 아니다."}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict_df = pd.DataFrame(all_verdicts)
    verdict_df.to_csv(OUT / "판정_전체.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== ⑥ 재판정 결과(v5·공식B·219kW 기준) ===")
    print(verdict_df.to_string(index=False))
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
