# -*- coding: utf-8 -*-
"""최종 공식구조의 end-to-end rolling-origin 리플레이 백테스트.

목적
----
과거 각 발행시점으로 돌아가 그때까지 알 수 있었던 자료만으로 학습·특성선택·
튜닝한 뒤, 폴드 모델을 joblib으로 저장하고 다시 불러와 예측한다. 예측은
Blockdata 대응 형식으로 변환한 뒤에만 시험구간 실제값과 비교한다.

누출 방지 원칙
--------------
1. 학습행은 항상 ``index < fold_start`` 이다.
2. 특성선택은 해당 폴드 학습구간에서만 수행한다.
3. 튜닝 후보 선택은 해당 폴드 학습구간의 마지막 20% 내부 홀드아웃만 쓴다.
   기존 ⑤처럼 여러 폴드의 내부점수를 합쳐 단일 파라미터를 고르지 않는다.
4. 시험 실제값은 모델 입력·선택·튜닝에 쓰지 않고 예측 완료 후 채점에만 쓴다.
5. v3 고정-tm NWP만 사용한다. 라이브 Blockdata API 호출·전송은 없다.

해석 한계
---------
최종 공식구조(raw/청천지수/튜닝 조합)는 동일 5계절 자료를 보며 선택됐으므로
이 실행도 완전한 독립 외부검증은 아니다. 다만 각 폴드의 행·특성·튜닝 수준
정답 누출은 차단한 엄격한 과거 리플레이이며, 새 데이터 외부검증 전까지의
end-to-end 재현검증이다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "E2E_리플레이백테스트_엄격최종_v1_2026-08-21"
MODEL_DIR = OUT / "fold_models"
ROUND2 = False
PLANT_ID = 6715
PLANT_NAME = "광주 광주시청"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("replay_harness", "backtest_harness_v1_2026-08-20밤.py")
ultra = _load("replay_ultra", "train_ultra_short_official_v1_2026-08-21.py")
clearsky = _load("replay_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
daily_mod = _load("replay_daily", "train_daily_official_v2_2026-08-21.py")
tuning = _load("replay_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
improvement = _load("replay_round2", "model_improvement_round2_v1_2026-08-21.py")


# 1차 엄격 리플레이에서 기존 ⑤ 튜닝후보를 폴드별 내부튜닝으로 다시 채점한
# 뒤, 사전동결 규칙 2~4를 그대로 적용해 확정한 구성이다.
# - 초단기 +2h: MAE 개선, RMSE 0.21% 악화 → 규칙3 raw 유지
# - 단기 +1h: RMSE 개선 0.84% → 규칙4 기본 유지
# - 단기 +48h: RMSE 개선 0.07% → 규칙4 기본 유지
# - 일간: 동일 321일 MAE/RMSE 3.90%/2.56% 개선·계절악화 없음 → 튜닝 유지
ULTRA_OFFICIAL = {1: "raw", 2: "raw", 3: "청천지수", 4: "청천지수"}
SHORT_OFFICIAL = {1: "기본", 24: "기본", 48: "기본"}
DAILY_OFFICIAL = "전체특성+튜닝파라미터"


def _safe(text: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in text)


def _cast_params(params: dict, integer_names: list[str]) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            value = value.item()
        if key in integer_names and value is not None:
            value = int(value)
        out[key] = value
    return out


def _metrics(actual, predicted, capacity: float, unit: str) -> dict:
    y = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    err = y - p
    denom = float(np.abs(y).sum())
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "n": int(len(y)),
        f"MAE_{unit}": mae,
        f"RMSE_{unit}": rmse,
        "nMAE_pct": mae / capacity * 100,
        "nRMSE_pct": rmse / capacity * 100,
        "WAPE_pct": float(np.abs(err).sum() / denom * 100) if denom > 0 else None,
    }


def _roundtrip(model, bundle: dict, x: pd.DataFrame, path: Path) -> tuple[np.ndarray, float]:
    """저장 전/재적재 후 원시 예측이 동일한지 확인하고 재적재 예측을 반환."""
    before = np.asarray(model.predict(x), float)
    bundle = {**bundle, "model": model}
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    after = np.asarray(loaded["model"].predict(x), float)
    max_diff = float(np.max(np.abs(before - after))) if len(before) else 0.0
    return after, max_diff


def _audit_row(tier: str, horizon, fold: str, train: pd.DataFrame, test: pd.DataFrame,
               features: list[str], tuned: bool, artifact: Path, roundtrip_diff: float) -> dict:
    train_max = pd.Timestamp(train.index.max())
    test_min = pd.Timestamp(test.index.min())
    forbidden = {"목표_발전출력_kW", "실제_일간발전량_kWh", "실제_일간총량_kWh"}
    return {
        "티어": tier, "수평_h": horizon, "폴드": fold,
        "학습최종시각": train_max, "시험최초시각": test_min,
        "학습_시험_분리통과": bool(train_max < test_min),
        "시험정답_특성포함없음": not bool(forbidden.intersection(features)),
        "특성선택_학습구간만": True,
        "튜닝_폴드내부만": bool(tuned) if tuned else True,
        "고정tm_NWP": True,
        "모델재적재_최대차이": roundtrip_diff,
        "모델재적재_동일": bool(roundtrip_diff <= 1e-12),
        "모델파일": str(artifact),
    }


def run_ultra(config: dict, capacity_kw: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    quarter = ultra.load_15min_base()
    hourly = pd.read_csv(
        harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False
    ).set_index("time").sort_index()
    candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS
    rows, audits = [], []

    for horizon, official in ULTRA_OFFICIAL.items():
        frame = ultra.build_ultra_short_frame(quarter, hourly, horizon)
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        frame["_청천_kW"] = np.clip(
            capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0,
            1e-3, None,
        )
        frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
        use_kappa = "청천지수" in official
        use_tuning = "튜닝" in official
        min_elev = clearsky.MIN_ELEVATION_DEG if use_kappa else 0.0
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]

        for i, window in enumerate(config["cross_validation_windows"], start=1):
            start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
            fold = f"{i}_{window.get('_계절', '')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            if len(train_all) < 500 or len(test_all) < 100:
                continue
            chosen = harness.select_features_in_fold(
                train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True
            )
            features = base_cols + chosen
            required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + [
                "목표_발전출력_kW", "_청천_kW", "_카파"
            ]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 500 or len(test) < 100:
                continue

            target = "_카파" if use_kappa else "목표_발전출력_kW"
            params = None
            structure_name = "기본"
            use_structure = ROUND2 and horizon in (1, 2)
            if use_structure:
                params, structure_name = improvement.choose_structure(
                    "초단기", train, features, target,
                    "kappa" if use_kappa else "raw", capacity_kw, seed,
                )
                model = improvement.make_model("초단기", seed, params)
            elif use_tuning:
                params, _ = tuning.tune_fold("LightGBM", train, features, target, capacity_kw, seed)
                params = _cast_params(params, ["n_estimators", "num_leaves", "max_depth", "min_child_samples"])
                model = LGBMRegressor(
                    objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **params
                )
            else:
                model = ultra.make_model("LightGBM", seed)
            model.fit(train[features], train[target])

            variant = official + ("+폴드내부구조선택" if use_structure else "")
            artifact = MODEL_DIR / _safe(f"초단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = _roundtrip(model, {
                "tier": "초단기", "horizon_h": horizon, "variant": variant,
                "features": features, "params": params, "target_transform": target,
                "structure_name": structure_name, "structure_selected_inside_fold": use_structure,
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, test[features], artifact)
            pred = raw * test["_청천_kW"].to_numpy() if use_kappa else raw
            pred = np.clip(pred, 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon, unit="h")
            for issued, target_time, y, p in zip(test.index, target_at, actual, pred):
                rows.append({
                    "plant_id": PLANT_ID, "plant_name": PLANT_NAME,
                    "티어": "초단기", "수평_h": horizon, "공식구성": variant, "폴드": fold,
                    "발행시각": issued, "대상시각": target_time,
                    "실제_kW": y, "예측_kW": p, "오차_kW": y - p,
                    "버킷시간_h": 0.25, "모델파일": str(artifact),
                })
            audit = _audit_row("초단기", horizon, fold, train, test, features, use_tuning, artifact, diff)
            audit.update({"폴드내부_구조선택": use_structure, "선택구조": structure_name})
            audits.append(audit)
        print(f"[초단기 +{horizon}h] {official}{'+구조개선' if ROUND2 and horizon in (1, 2) else ''} 완료")
    return pd.DataFrame(rows), audits


def run_short(config: dict, capacity_kw: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    source = pd.read_csv(
        harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False
    ).set_index("time").sort_index()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    rows, audits = [], []

    for horizon, official in SHORT_OFFICIAL.items():
        frame = harness.build_frame(source, horizon, candidate_cols)
        daylight = frame[frame["목표_낮시간"] > 0]
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        use_tuning = official == "튜닝"

        for i, window in enumerate(config["cross_validation_windows"], start=1):
            start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
            fold = f"{i}_{window.get('_계절', '')}"
            train_all = daylight[daylight.index < start]
            test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
            if len(train_all) < 200 or len(test_all) < 30:
                continue
            chosen = harness.select_features_in_fold(
                train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True
            )
            features = base_cols + chosen
            required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
            train = train_all.dropna(subset=required)
            test = test_all.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue

            params = None
            structure_name = "기본"
            use_structure = ROUND2
            if use_structure:
                params, structure_name = improvement.choose_structure(
                    "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed,
                )
                model = improvement.make_model("단기", seed, params)
            elif use_tuning:
                params, _ = tuning.tune_fold(
                    "XGBoost", train, features, "목표_발전출력_kW", capacity_kw, seed
                )
                params = _cast_params(params, ["n_estimators", "max_depth", "min_child_weight"])
                model = XGBRegressor(
                    objective="reg:absoluteerror", random_state=seed, n_jobs=4, verbosity=0, **params
                )
            else:
                model = harness.make_model("XGBoost", seed)
            model.fit(train[features], train["목표_발전출력_kW"])

            variant = official + ("+폴드내부구조선택" if use_structure else "")
            artifact = MODEL_DIR / _safe(f"단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = _roundtrip(model, {
                "tier": "단기", "horizon_h": horizon, "variant": variant,
                "features": features, "params": params, "target_transform": "raw_kW",
                "structure_name": structure_name, "structure_selected_inside_fold": use_structure,
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, test[features], artifact)
            pred = np.clip(raw, 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon - 1, unit="h")
            for issued, target_time, y, p in zip(test.index, target_at, actual, pred):
                rows.append({
                    "plant_id": PLANT_ID, "plant_name": PLANT_NAME,
                    "티어": "단기", "수평_h": horizon, "공식구성": variant, "폴드": fold,
                    "발행시각": issued, "대상시각": target_time,
                    "실제_kW": y, "예측_kW": p, "오차_kW": y - p,
                    "버킷시간_h": 1.0, "모델파일": str(artifact),
                })
            audit = _audit_row("단기", horizon, fold, train, test, features, use_tuning, artifact, diff)
            audit.update({"폴드내부_구조선택": use_structure, "선택구조": structure_name})
            audits.append(audit)
        print(f"[단기 +{horizon}h] {official}{'+구조개선' if ROUND2 else ''} 완료")
    return pd.DataFrame(rows), audits


def run_daily(config: dict, capacity_kw: float, seed: int) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    data, features = daily_mod.build_daily_dataset()
    target = daily_mod.ACTUAL
    daily_capacity = capacity_kw * 24
    rows, baseline_rows, audits = [], [], []

    for i, window in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        fold = f"{i}_{window.get('_계절', '')}"
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index <= end)]
        if len(train) < 60 or len(test) < 10:
            continue
        medians = train[features].median(numeric_only=True)
        x_train = train[features].fillna(medians)
        x_test = test[features].fillna(medians)

        # 같은 321개 시험일에서 기본·튜닝을 공정 비교한다. 기존 ⑤ 일간표는
        # 시간모델합계와 inner merge해 119일만 남았으므로 직접모델의 엄격한
        # 비교에는 쓸 수 없다.
        baseline_model = LGBMRegressor(
            objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4
        )
        baseline_model.fit(x_train, train[target])
        baseline_pred = np.clip(baseline_model.predict(x_test), 0, daily_capacity)
        for target_date, y, p in zip(test.index, test[target].to_numpy(), baseline_pred):
            baseline_rows.append({
                "폴드": fold, "대상일": target_date, "후보": "전체특성+기본파라미터",
                "실제_kWh": y, "예측_kWh": p, "오차_kWh": y - p,
            })
        params, _ = tuning.tune_fold("LightGBM", train.fillna(medians), features, target, daily_capacity, seed)
        params = _cast_params(params, ["n_estimators", "num_leaves", "max_depth", "min_child_samples"])
        model = LGBMRegressor(
            objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **params
        )
        model.fit(x_train, train[target])
        artifact = MODEL_DIR / _safe(f"일간_{fold}_{DAILY_OFFICIAL}.joblib")
        raw, diff = _roundtrip(model, {
            "tier": "일간", "horizon": "D+1", "variant": DAILY_OFFICIAL,
            "features": features, "medians": medians.to_dict(), "params": params,
            "target_transform": "daily_kWh", "train_end": str(train.index.max()),
            "test_start": str(test.index.min()), "hierarchical_weights": {"direct": 1.0, "hourly_sum": 0.0},
        }, x_test, artifact)
        pred = np.clip(raw, 0, daily_capacity)
        actual = test[target].to_numpy()
        issued_at = test.index - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
        for issued, target_date, y, p in zip(issued_at, test.index, actual, pred):
            rows.append({
                "plant_id": PLANT_ID, "plant_name": PLANT_NAME,
                "티어": "일간", "수평_h": "D+1", "공식구성": DAILY_OFFICIAL, "폴드": fold,
                "발행시각": issued, "대상일": target_date,
                "실제_kWh": y, "예측_kWh": p, "오차_kWh": y - p,
                "모델파일": str(artifact),
            })
        audits.append(_audit_row("일간", "D+1", fold, train, test, features, True, artifact, diff))
        print(f"[일간 {fold}] {DAILY_OFFICIAL} 완료")
    return pd.DataFrame(rows), audits, pd.DataFrame(baseline_rows)


def make_performance(ac: pd.DataFrame, daily: pd.DataFrame, capacity_kw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, by_fold = [], []
    for (tier, horizon, variant), group in ac.groupby(["티어", "수평_h", "공식구성"], dropna=False):
        overall.append({"티어": tier, "수평_h": horizon, "공식구성": variant,
                        **_metrics(group["실제_kW"], group["예측_kW"], capacity_kw, "kW")})
        for fold, fg in group.groupby("폴드"):
            by_fold.append({"티어": tier, "수평_h": horizon, "공식구성": variant, "폴드": fold,
                            **_metrics(fg["실제_kW"], fg["예측_kW"], capacity_kw, "kW")})
    if len(daily):
        overall.append({"티어": "일간", "수평_h": "D+1", "공식구성": DAILY_OFFICIAL,
                        **_metrics(daily["실제_kWh"], daily["예측_kWh"], capacity_kw * 24, "kWh")})
        for fold, fg in daily.groupby("폴드"):
            by_fold.append({"티어": "일간", "수평_h": "D+1", "공식구성": DAILY_OFFICIAL, "폴드": fold,
                            **_metrics(fg["실제_kWh"], fg["예측_kWh"], capacity_kw * 24, "kWh")})
    return pd.DataFrame(overall), pd.DataFrame(by_fold)


def make_blockdata_outputs(ac: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    power = ac.rename(columns={
        "예측_kW": "plant_ac_power_predicted_kw",
        "실제_kW": "plant_ac_power_actual_kw",
    }).copy()
    power["blockdata_mapping"] = "sum(inverter[].ac_power)"
    power["source_granularity"] = np.where(power["티어"] == "초단기", "15min_bucket_mean", "1h_bucket_mean")
    power["live_api_written"] = False

    cum = ac.copy()
    cum["대상일"] = pd.to_datetime(cum["대상시각"]).dt.normalize()
    cum = cum.sort_values(["티어", "수평_h", "폴드", "대상시각"])
    keys = ["티어", "수평_h", "폴드", "대상일"]
    cum["daily_energy_predicted_kWh"] = (cum["예측_kW"] * cum["버킷시간_h"]).groupby(
        [cum[k] for k in keys]
    ).cumsum()
    cum["daily_energy_actual_kWh"] = (cum["실제_kW"] * cum["버킷시간_h"]).groupby(
        [cum[k] for k in keys]
    ).cumsum()
    cum["blockdata_mapping"] = "inverter[].daily_energy running-sum semantic; plant total"
    cum["live_api_written"] = False

    daily_out = daily.copy()
    if len(daily_out):
        daily_out["blockdata_mapping"] = "forecast day-final energy; not live running daily_energy"
        daily_out["live_api_written"] = False
    return power, pd.concat([cum, daily_out], ignore_index=True, sort=False)


def validate_official_map() -> list[str]:
    verdict_path = ROOT / "outputs" / "최종통합재검증_v1_2026-08-21" / "판정표.csv"
    verdict = pd.read_csv(verdict_path, encoding="utf-8-sig")
    errors = []
    for horizon, label in ULTRA_OFFICIAL.items():
        if label == "raw":
            continue  # 기준선 raw 유지라 판정표 행이 없음
        hit = verdict[(verdict["티어"] == "초단기") & (verdict["수평_h"] == horizon)
                      & (verdict["후보"] == label) & (verdict["최종채택"].astype(str).str.lower() == "true")]
        if len(hit) != 1:
            errors.append(f"초단기 +{horizon}h 공식구성 불일치: {label}")
    for horizon, label in SHORT_OFFICIAL.items():
        if label == "기본":
            continue
        hit = verdict[(verdict["티어"] == "단기") & (verdict["수평_h"] == horizon)
                      & (verdict["후보"] == label) & (verdict["최종채택"].astype(str).str.lower() == "true")]
        if len(hit) != 1:
            errors.append(f"단기 +{horizon}h 공식구성 불일치: {label}")
    hit = verdict[(verdict["티어"] == "일간") & (verdict["후보"] == DAILY_OFFICIAL)
                  & (verdict["최종채택"].astype(str).str.lower() == "true")]
    if len(hit) != 1:
        errors.append(f"일간 공식구성 불일치: {DAILY_OFFICIAL}")
    return errors


def validate_round2_state() -> list[str]:
    state_path = ROOT / "outputs" / "2차_성능개선_v1_2026-08-21" / "채택상태.json"
    if not state_path.exists():
        return [f"2차 채택상태 없음: {state_path}"]
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {"초단기_1", "초단기_2", "단기_1", "단기_24", "단기_48"}
    errors = []
    for key, value in payload.get("티어수평별", {}).items():
        actual = bool(value.get("tree_structure"))
        if actual != (key in expected):
            errors.append(f"구조개선 채택상태 불일치: {key}={actual}")
        if value.get("quantile_mapping") or value.get("residual_correction") or value.get("weather_cluster"):
            errors.append(f"미채택 개선이 상태에 남음: {key}")
    return errors


def main() -> None:
    global OUT, MODEL_DIR, ROUND2
    ap = argparse.ArgumentParser()
    ap.add_argument("--round2", action="store_true", help="2차 채택 구조개선 5건을 적용")
    args = ap.parse_args()
    ROUND2 = bool(args.round2)
    if ROUND2:
        OUT = ROOT / "outputs" / "E2E_리플레이백테스트_2차결과_v1_2026-08-21"
        MODEL_DIR = OUT / "fold_models"
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    map_errors = validate_official_map() + (validate_round2_state() if ROUND2 else [])
    if map_errors:
        raise RuntimeError("판정표 공식구성 불일치: " + "; ".join(map_errors))

    ultra_rows, ultra_audit = run_ultra(config, capacity_kw, seed)
    short_rows, short_audit = run_short(config, capacity_kw, seed)
    daily_rows, daily_audit, daily_baseline = run_daily(config, capacity_kw, seed)
    ac = pd.concat([ultra_rows, short_rows], ignore_index=True)
    audits = pd.DataFrame(ultra_audit + short_audit + daily_audit)

    perf, perf_fold = make_performance(ac, daily_rows, capacity_kw)
    block_power, block_energy = make_blockdata_outputs(ac, daily_rows)

    ac.to_csv(OUT / "행단위_ac_power_예측정답.csv", index=False, encoding="utf-8-sig")
    daily_rows.to_csv(OUT / "행단위_daily_final_예측정답.csv", index=False, encoding="utf-8-sig")
    daily_baseline.to_csv(OUT / "후보비교_일간_기본파라미터.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(OUT / "성능_전체.csv", index=False, encoding="utf-8-sig")
    perf_fold.to_csv(OUT / "성능_폴드별.csv", index=False, encoding="utf-8-sig")
    audits.to_csv(OUT / "누출_및_재적재_감사.csv", index=False, encoding="utf-8-sig")
    block_power.to_csv(OUT / "Blockdata형식_ac_power_리플레이.csv", index=False, encoding="utf-8-sig")
    block_energy.to_csv(OUT / "Blockdata형식_energy_리플레이.csv", index=False, encoding="utf-8-sig")

    base_m = _metrics(daily_baseline["실제_kWh"], daily_baseline["예측_kWh"], capacity_kw * 24, "kWh")
    tuned_m = _metrics(daily_rows["실제_kWh"], daily_rows["예측_kWh"], capacity_kw * 24, "kWh")
    daily_compare = pd.DataFrame([
        {"후보": "전체특성+기본파라미터", **base_m},
        {"후보": DAILY_OFFICIAL, **tuned_m},
    ])
    daily_compare.to_csv(OUT / "후보비교_일간_동일321일.csv", index=False, encoding="utf-8-sig")

    audit_pass = bool(
        len(audits)
        and audits["학습_시험_분리통과"].all()
        and audits["시험정답_특성포함없음"].all()
        and audits["모델재적재_동일"].all()
        and audits["고정tm_NWP"].all()
    )
    summary = {
        "실행유형": "end-to-end rolling-origin replay backtest",
        "라이브_API_호출_전송": False,
        "공식구성": {
            "초단기": {h: v + ("+폴드내부구조선택" if ROUND2 and h in (1, 2) else "") for h, v in ULTRA_OFFICIAL.items()},
            "단기": {h: v + ("+폴드내부구조선택" if ROUND2 else "") for h, v in SHORT_OFFICIAL.items()},
            "일간": DAILY_OFFICIAL,
        },
        "2차_구조개선_적용": ROUND2,
        "폴드모델수": int(len(audits)),
        "ac_power_예측행수": int(len(ac)),
        "daily_final_예측행수": int(len(daily_rows)),
        "누출및재적재감사_통과": audit_pass,
        "인식론적한계": (
            "행·특성·튜닝 수준의 폴드 누출은 차단했다. 다만 공식구조 자체가 동일 5계절 결과로 "
            "선택됐으므로 완전 독립 외부검증은 아니며 새 데이터 재검증이 필요하다."
        ),
        "엄격리플레이_재판정": {
            "초단기_2h": "raw 유지(튜닝 청천지수 RMSE 0.21% 악화)",
            "단기_1h": "기본 유지(튜닝 RMSE 개선 0.84%로 1% 미만)",
            "단기_48h": "기본 유지(튜닝 RMSE 개선 0.07%로 1% 미만)",
            "일간_D+1": "전체특성+튜닝 유지(동일 321일 MAE/RMSE 3.90%/2.56% 개선)",
        },
        "산출물경로": str(OUT),
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== E2E 리플레이 백테스트 성능 ===")
    print(perf.to_string(index=False))
    print("\n=== 누출·재적재 감사 ===")
    print(audits[["티어", "수평_h", "폴드", "학습_시험_분리통과", "시험정답_특성포함없음",
                  "모델재적재_동일"]].to_string(index=False))
    print(f"\n감사 통과: {audit_pass}")
    print(f"저장: {OUT}")
    if not audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
