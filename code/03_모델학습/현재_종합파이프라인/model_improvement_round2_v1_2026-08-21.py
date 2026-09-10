# -*- coding: utf-8 -*-
"""2차 성능개선 1~4 단계별 무누출 rolling-origin 검증기.

실행 순서
---------
    python model_improvement_round2_v1_2026-08-21.py --step 1
    python model_improvement_round2_v1_2026-08-21.py --step 2
    python model_improvement_round2_v1_2026-08-21.py --step 3
    python model_improvement_round2_v1_2026-08-21.py --step 4

각 단계는 직전 ``채택상태.json``만 읽고, 동일 시험행에서 현행 누적구성과
새 후보를 다시 학습해 비교한다. 시험 정답은 채점에만 사용하며 채택 규칙은
AGENTS.md에 사전동결된 규칙(두 지표 1% 이상 개선, 계절 MAE 5% 악화 없음)을
그대로 적용한다.

적용 범위
---------
1. 폴드 내부 NWP 분위사상: 초단기·단기. 학습폴드의 예보/관측 쌍만 사용.
2. 구름전이·강수상태 OOF 잔차보정: 초단기·단기. 학습폴드 마지막 20%의
   시간순 OOF 잔차만 사용.
3. 일사량 예보 기반 날씨 군집화: 초단기·단기. KMeans/표준화는 학습폴드만 fit.
4. 트리 구조개선: 초단기 LightGBM·단기 XGBoost·일간 LightGBM. 구조는
   학습폴드 내부 80/20 홀드아웃에서만 선택.

일간은 이미 시간 집계된 하루 단위 자료라 1~3의 시간별 NWP 보정·전이상태를
그대로 적용하면 의미가 달라진다. 따라서 4번만 공정하게 적용한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "2차_성능개선_v1_2026-08-21"
STATE_PATH = OUT / "채택상태.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("round2_harness", "backtest_harness_v1_2026-08-20밤.py")
ultra = _load("round2_ultra", "train_ultra_short_official_v1_2026-08-21.py")
clearsky = _load("round2_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
daily_mod = _load("round2_daily", "train_daily_official_v2_2026-08-21.py")
tuning = _load("round2_tuning", "hyperparameter_tuning_v1_2026-08-21.py")

ULTRA_TARGET = {1: "raw", 2: "raw", 3: "kappa", 4: "kappa"}
NWP_OBS_PAIRS = {
    "DSWRF": "기상청관측_일사량_W_m2",
    "DSWRFLX_bsrn정제": "기상청관측_일사량_W_m2",
    "TCDC": "기상청관측_전운량_pct",
    "LCDC": "기상청관측_전운량_pct",
    "TMP": "기상청관측_기온_C",
    "REH": "기상청관측_상대습도_pct",
    "WSD": "기상청관측_풍속_m_s",
}
CLUSTER_SOURCE = ["DSWRF", "DSWRFLX_bsrn정제", "TCDC", "LCDC", "POP", "SKY", "목표_태양고도_deg"]
STEP_NAMES = {
    1: "NWP_분위사상",
    2: "구름전이_강수_OOF잔차보정",
    3: "일사예보_날씨군집",
    4: "트리_구조개선",
}


@dataclass
class PipelineState:
    quantile_mapping: bool = False
    residual_correction: bool = False
    weather_cluster: bool = False
    tree_structure: bool = False


def default_state() -> dict[str, dict]:
    keys = [f"초단기_{h}" for h in [1, 2, 3, 4]] + [f"단기_{h}" for h in [1, 24, 48]] + ["일간_D+1"]
    return {k: asdict(PipelineState()) for k in keys}


def load_state(step: int) -> dict[str, dict]:
    if step == 1:
        return default_state()
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"직전 단계 상태가 없습니다: {STATE_PATH}")
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if int(payload.get("완료단계", 0)) != step - 1:
        raise RuntimeError(f"단계 순서 오류: 완료={payload.get('완료단계')} 요청={step}")
    return payload["티어수평별"]


def metrics(y, p) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = y - p
    return {"n": int(len(y)), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt(np.mean(e ** 2)))}


def empirical_qm_fit(x: pd.Series, y: pd.Series, n_quantiles: int = 101):
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 100 or pair.iloc[:, 0].nunique() < 5 or pair.iloc[:, 1].nunique() < 5:
        return None
    q = np.linspace(0, 1, n_quantiles)
    xq = np.quantile(pair.iloc[:, 0].to_numpy(), q)
    yq = np.quantile(pair.iloc[:, 1].to_numpy(), q)
    keep = np.r_[True, np.diff(xq) > 1e-12]
    return xq[keep], yq[keep]


def empirical_qm_apply(values: pd.Series, fitted) -> pd.Series:
    if fitted is None:
        return values.copy()
    xq, yq = fitted
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.interp(arr, xq, yq, left=yq[0], right=yq[-1])
    out[~np.isfinite(arr)] = np.nan
    return pd.Series(out, index=values.index)


def apply_quantile_mapping(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    tr, te, logs = train.copy(), test.copy(), []
    for nwp, obs in NWP_OBS_PAIRS.items():
        hidden = f"__target_obs__{obs}"
        if nwp not in features or hidden not in tr:
            continue
        fitted = empirical_qm_fit(tr[nwp], tr[hidden])
        if fitted is None:
            continue
        tr[nwp] = empirical_qm_apply(tr[nwp], fitted)
        te[nwp] = empirical_qm_apply(te[nwp], fitted)
        logs.append(nwp)
    return tr, te, logs


def add_weather_clusters(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int):
    cols = [c for c in CLUSTER_SOURCE if c in features and c in train]
    if len(cols) < 2 or len(train) < 200:
        return train.copy(), test.copy(), features, []
    med = train[cols].median(numeric_only=True)
    scaler = StandardScaler().fit(train[cols].fillna(med))
    ztr = scaler.transform(train[cols].fillna(med)); zte = scaler.transform(test[cols].fillna(med))
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(ztr)
    tr, te = train.copy(), test.copy()
    tr_labels, te_labels = km.predict(ztr), km.predict(zte)
    added = []
    for k in range(4):
        name = f"날씨군집_{k}"
        tr[name] = (tr_labels == k).astype(int); te[name] = (te_labels == k).astype(int)
        added.append(name)
    return tr, te, features + added, cols


def make_model(tier: str, seed: int, params: dict | None = None):
    if tier in ("초단기", "일간"):
        if params is None:
            if tier == "초단기":
                return ultra.make_model("LightGBM", seed)
            return LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4)
        return LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **params)
    if params is None:
        return harness.make_model("XGBoost", seed)
    return XGBRegressor(objective="reg:absoluteerror", random_state=seed, n_jobs=4, verbosity=0, **params)


def structural_candidates(tier: str):
    if tier in ("초단기", "일간"):
        return [
            ("얕은규제", dict(n_estimators=420, learning_rate=0.025, num_leaves=15, max_depth=5,
                           min_child_samples=45, subsample=0.85, colsample_bytree=0.85,
                           reg_alpha=0.1, reg_lambda=1.0)),
            ("리프규제", dict(n_estimators=320, learning_rate=0.03, num_leaves=31, max_depth=-1,
                           min_child_samples=60, subsample=0.9, colsample_bytree=0.8,
                           reg_alpha=0.2, reg_lambda=1.5)),
        ]
    return [
        ("깊이4_규제", dict(n_estimators=420, learning_rate=0.025, max_depth=4, min_child_weight=8,
                         gamma=0.05, subsample=0.85, colsample_bytree=0.85,
                         reg_alpha=0.1, reg_lambda=1.5, tree_method="hist")),
        ("lossguide", dict(n_estimators=360, learning_rate=0.03, max_depth=0, max_leaves=31,
                        grow_policy="lossguide", min_child_weight=8, gamma=0.05,
                        subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1,
                        reg_lambda=1.5, tree_method="hist")),
    ]


def predict_scale(raw: np.ndarray, frame: pd.DataFrame, target_mode: str, capacity: float):
    pred = raw * frame["_청천_kW"].to_numpy() if target_mode == "kappa" else raw
    return np.clip(pred, 0, capacity)


def choose_structure_kfold(tier: str, train: pd.DataFrame, features: list[str], target: str,
                           target_mode: str, capacity: float, seed: int, k: int = 4):
    """★08-24 신규★ `choose_structure`의 단일 80/20 분할을 시간순 walk-forward
    내부 K분할 평균으로 대체한 안정화판(재판정 결과, 08-24 — 사용자 확정).

    ## 왜 필요했나
    5번 v5 재학습에서 초단기 +1h·+2h가 2차 대비 3~5% 나빠졌다. A/B/C
    정책·폴드경계·DIFSWRF 플래그를 전부 소거법으로 배제한 뒤 남은 원인은
    **`choose_structure`의 단일 80/20 분할**이었다 — v5가 이전에 없던
    데이터(정오대 등)를 복구해 넣으면서 폴드별 학습표본 크기가 달라졌고,
    그 결과 `cut = int(len(train)*0.8)` 분할 지점이 밀려 일부 폴드(초단기
    +1h 겨울, +2h 봄)에서 2차와 다른(더 나쁜) 구조가 선택됐다(2차
    "얕은규제" → 3차 "리프규제"). 단일 시점 분할은 그 한 조각의 우연에
    취약하다는 게 확인된 것.

    ## 방식
    학습구간을 시간순으로 `k+1`등분해, 블록 i(i=1..k)를 검증셋으로 삼고
    그 이전 전체(블록 0~i-1)를 내부학습셋으로 쓰는 **walk-forward
    내부검증**을 k번 반복한다(미래 데이터로 과거를 검증하지 않음 —
    무누출 원칙 유지). 각 구조 후보의 k개 내부검증 MAE **평균**으로
    구조를 고른다 — 단일 조각의 우연에 덜 흔들린다.
    """
    n = len(train)
    if n < 500:
        return None, "기본"
    edges = np.linspace(0, n, k + 2).astype(int)
    options = [("기본", None)] + structural_candidates(tier)
    scores: dict[str, list[float]] = {name: [] for name, _ in options}
    used_splits = 0
    for i in range(1, k + 1):
        tr_end, va_start, va_end = edges[i], edges[i], edges[i + 1]
        if tr_end < 200 or (va_end - va_start) < 30:
            continue
        used_splits += 1
        inner_tr, inner_va = train.iloc[:tr_end], train.iloc[va_start:va_end]
        for name, params in options:
            model = make_model(tier, seed, params)
            model.fit(inner_tr[features], inner_tr[target])
            pred = predict_scale(model.predict(inner_va[features]), inner_va, target_mode, capacity)
            scores[name].append(metrics(inner_va["목표_발전출력_kW"], pred)["MAE"])
    if used_splits == 0:
        return choose_structure(tier, train, features, target, target_mode, capacity, seed)
    avg = {name: float(np.mean(v)) for name, v in scores.items() if v}
    best_name = min(avg, key=avg.get)
    best_params = dict(options)[best_name]
    return best_params, best_name


def choose_structure(tier: str, train: pd.DataFrame, features: list[str], target: str,
                     target_mode: str, capacity: float, seed: int):
    cut = int(len(train) * 0.8)
    inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]
    if len(inner_va) < 30:
        return None, "기본"
    options = [("기본", None)] + structural_candidates(tier)
    scored = []
    for name, params in options:
        model = make_model(tier, seed, params)
        model.fit(inner_tr[features], inner_tr[target])
        pred = predict_scale(model.predict(inner_va[features]), inner_va, target_mode, capacity)
        scored.append((metrics(inner_va["목표_발전출력_kW"], pred)["MAE"], name, params))
    _, name, params = min(scored, key=lambda x: x[0])
    return params, name


def state_columns(frame: pd.DataFrame):
    cloud = next((c for c in ["TCDC", "SKY", "기상청관측_전운량_pct"] if c in frame), None)
    precip = next((c for c in ["POP", "기상청관측_강수량_mm"] if c in frame), None)
    return cloud, precip


def fit_oof_residual(train: pd.DataFrame, features: list[str], target: str, target_mode: str,
                     tier: str, capacity: float, seed: int, params: dict | None):
    cut = int(len(train) * 0.8)
    inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]
    cloud, precip = state_columns(train)
    if len(inner_va) < 60 or cloud is None:
        return {"global": 0.0, "groups": {}, "q": [0.0, 0.0], "cloud": cloud, "precip": precip}
    model = make_model(tier, seed, params)
    model.fit(inner_tr[features], inner_tr[target])
    pred = predict_scale(model.predict(inner_va[features]), inner_va, target_mode, capacity)
    residual = inner_va["목표_발전출력_kW"].to_numpy() - pred
    delta_train = inner_tr[cloud].diff().abs().dropna()
    q1, q2 = (delta_train.quantile([1/3, 2/3]).to_numpy() if len(delta_train) else np.array([0.0, 0.0]))
    delta = inner_va[cloud].diff().abs().fillna(0).to_numpy()
    trans = np.where(delta <= q1, 0, np.where(delta <= q2, 1, 2))
    if precip is None:
        wet = np.zeros(len(inner_va), dtype=int)
    elif precip == "POP":
        wet = (inner_va[precip].fillna(0).to_numpy() >= 50).astype(int)
    else:
        wet = (inner_va[precip].fillna(0).to_numpy() > 0).astype(int)
    groups = {}
    for a in range(3):
        for b in range(2):
            mask = (trans == a) & (wet == b)
            if mask.sum() >= 30:
                groups[f"{a}_{b}"] = float(np.median(residual[mask]))
    return {"global": float(np.median(residual)), "groups": groups, "q": [float(q1), float(q2)],
            "cloud": cloud, "precip": precip}


def apply_residual(pred: np.ndarray, frame: pd.DataFrame, corr: dict, capacity: float):
    cloud, precip = corr["cloud"], corr["precip"]
    if cloud is None:
        return pred
    q1, q2 = corr["q"]
    delta = frame[cloud].diff().abs().fillna(0).to_numpy()
    trans = np.where(delta <= q1, 0, np.where(delta <= q2, 1, 2))
    if precip is None:
        wet = np.zeros(len(frame), dtype=int)
    elif precip == "POP":
        wet = (frame[precip].fillna(0).to_numpy() >= 50).astype(int)
    else:
        wet = (frame[precip].fillna(0).to_numpy() > 0).astype(int)
    offsets = np.array([corr["groups"].get(f"{a}_{b}", corr["global"]) for a, b in zip(trans, wet)])
    offsets = np.clip(offsets, -0.15 * capacity, 0.15 * capacity)
    return np.clip(pred + offsets, 0, capacity)


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, features: list[str], state: PipelineState,
                     seed: int):
    tr, te, qm_log = train.copy(), test.copy(), []
    if state.quantile_mapping:
        tr, te, qm_log = apply_quantile_mapping(tr, te, features)
    cluster_log = []
    out_features = list(features)
    if state.weather_cluster:
        tr, te, out_features, cluster_log = add_weather_clusters(tr, te, out_features, seed)
    return tr, te, out_features, qm_log, cluster_log


def predict_fold(tier: str, train0: pd.DataFrame, test0: pd.DataFrame, features0: list[str],
                 state: PipelineState, target_mode: str, capacity: float, seed: int):
    train, test, features, qm_log, cluster_log = prepare_features(train0, test0, features0, state, seed)
    target = "_카파" if target_mode == "kappa" else "목표_발전출력_kW"
    params, structure_name = (choose_structure(tier, train, features, target, target_mode, capacity, seed)
                              if state.tree_structure else (None, "기본"))
    corr = (fit_oof_residual(train, features, target, target_mode, tier, capacity, seed, params)
            if state.residual_correction else None)
    model = make_model(tier, seed, params)
    model.fit(train[features], train[target])
    pred = predict_scale(model.predict(test[features]), test, target_mode, capacity)
    if corr is not None:
        pred = apply_residual(pred, test, corr, capacity)
    return pred, {"QM특성": qm_log, "군집원천특성": cluster_log, "구조": structure_name,
                  "잔차그룹수": len(corr["groups"]) if corr else 0}


def add_target_observations_short(frame: pd.DataFrame, source: pd.DataFrame, horizon: int):
    for obs in set(NWP_OBS_PAIRS.values()):
        if obs in source:
            frame[f"__target_obs__{obs}"] = source[obs].shift(-horizon)


def add_target_observations_ultra(frame: pd.DataFrame, quarter: pd.DataFrame,
                                  hourly: pd.DataFrame, horizon: int):
    hour_key = quarter.index.floor("1h")
    for obs in set(NWP_OBS_PAIRS.values()):
        if obs not in hourly:
            continue
        aligned = hourly[obs].shift(-horizon).reindex(hour_key)
        aligned.index = quarter.index
        frame[f"__target_obs__{obs}"] = aligned


def run_power_tier(tier: str, horizon: int, config: dict, capacity: float, seed: int,
                   current: PipelineState, candidate: PipelineState):
    if tier == "초단기":
        quarter = ultra.load_15min_base()
        source = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
        frame = ultra.build_ultra_short_frame(quarter, source, horizon)
        add_target_observations_ultra(frame, quarter, source, horizon)
        candidate_cols = ultra.EQUIP_STARTLABEL + ultra.harness.sel.OBSERVED_COLUMNS + ultra.harness.sel.FORECAST_COLUMNS
        min_elev = clearsky.MIN_ELEVATION_DEG if ULTRA_TARGET[horizon] == "kappa" else 0.0
        frame["_청천_kW"] = np.clip(capacity * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000, 1e-3, None)
        frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]
        target_mode = ULTRA_TARGET[horizon]
        min_train, min_test = 500, 100
    else:
        source = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()
        candidate_cols = harness.FEATURE_SETS["전체후보"]
        frame = harness.build_frame(source, horizon, candidate_cols)
        add_target_observations_short(frame, source, horizon)
        daylight = frame[frame["목표_낮시간"] > 0]
        target_mode = "raw"
        min_train, min_test = 200, 30
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간") and not c.startswith("__target_obs__")]
    rows = []
    for i, w in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold = f"{i}_{w.get('_계절', '')}"
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
        if len(train_all) < min_train or len(test_all) < min_test:
            continue
        chosen = harness.select_features_in_fold(train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
        if target_mode == "kappa":
            required += ["_청천_kW", "_카파"]
        train, test = train_all.dropna(subset=required), test_all.dropna(subset=required)
        if len(train) < min_train or len(test) < min_test:
            continue
        p0, log0 = predict_fold(tier, train, test, features, current, target_mode, capacity, seed)
        p1, log1 = predict_fold(tier, train, test, features, candidate, target_mode, capacity, seed)
        for ts, y, a, b in zip(test.index, test["목표_발전출력_kW"], p0, p1):
            rows.append({"티어": tier, "수평_h": horizon, "폴드": fold, "발행시각": ts,
                         "실제": y, "현행예측": a, "후보예측": b,
                         "현행상세": json.dumps(log0, ensure_ascii=False),
                         "후보상세": json.dumps(log1, ensure_ascii=False)})
    return pd.DataFrame(rows)


def run_daily_structure(config: dict, capacity: float, seed: int, current: PipelineState, candidate: PipelineState):
    data, features = daily_mod.build_daily_dataset(); target = daily_mod.ACTUAL
    rows = []
    for i, w in enumerate(config["cross_validation_windows"], start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold = f"{i}_{w.get('_계절', '')}"
        train, test = data[data.index < start], data[(data.index >= start) & (data.index <= end)]
        if len(train) < 60 or len(test) < 10:
            continue
        med = train[features].median(numeric_only=True)
        train, test = train.copy(), test.copy()
        train[features] = train[features].fillna(med); test[features] = test[features].fillna(med)
        # 공식 일간 기준선은 폴드 내부 랜덤탐색 튜닝이다.
        base_params, _ = tuning.tune_fold("LightGBM", train, features, target, capacity * 24, seed)
        for k in ["n_estimators", "num_leaves", "max_depth", "min_child_samples"]:
            if k in base_params:
                base_params[k] = int(base_params[k])
        base = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **base_params)
        base.fit(train[features], train[target])
        p0 = np.clip(base.predict(test[features]), 0, capacity * 24)
        if candidate.tree_structure:
            params, name = choose_structure("일간", train.rename(columns={target: "목표_발전출력_kW"}), features,
                                            "목표_발전출력_kW", "raw", capacity * 24, seed)
            model = make_model("일간", seed, params)
            model.fit(train[features], train[target])
            p1 = np.clip(model.predict(test[features]), 0, capacity * 24)
        else:
            p1, name = p0.copy(), "공식폴드내튜닝"
        for ts, y, a, b in zip(test.index, test[target], p0, p1):
            rows.append({"티어": "일간", "수평_h": "D+1", "폴드": fold, "발행시각": ts,
                         "실제": y, "현행예측": a, "후보예측": b, "현행상세": "공식폴드내튜닝",
                         "후보상세": name})
    return pd.DataFrame(rows)


def verdict(rows: pd.DataFrame):
    b = metrics(rows["실제"], rows["현행예측"]); c = metrics(rows["실제"], rows["후보예측"])
    mae_gain = (b["MAE"] - c["MAE"]) / b["MAE"] * 100
    rmse_gain = (b["RMSE"] - c["RMSE"]) / b["RMSE"] * 100
    worst = -np.inf; worst_fold = ""
    fold_rows = []
    for fold, fg in rows.groupby("폴드"):
        fb = metrics(fg["실제"], fg["현행예측"]); fc = metrics(fg["실제"], fg["후보예측"])
        worsen = (fc["MAE"] - fb["MAE"]) / fb["MAE"] * 100
        fold_rows.append({"폴드": fold, "현행_MAE": fb["MAE"], "후보_MAE": fc["MAE"], "MAE악화율_pct": worsen})
        if worsen > worst:
            worst, worst_fold = worsen, fold
    if mae_gain <= 0 or rmse_gain <= 0:
        decision, reason = False, "MAE·RMSE 동시개선 실패"
    elif mae_gain < 1 or rmse_gain < 1:
        decision, reason = False, "개선율 1% 미만"
    elif worst >= 5:
        decision, reason = False, f"계절불안정({worst_fold} MAE {worst:.2f}% 악화)"
    else:
        decision, reason = True, "동결규칙 2~5 통과"
    return {"현행_MAE": b["MAE"], "현행_RMSE": b["RMSE"], "후보_MAE": c["MAE"],
            "후보_RMSE": c["RMSE"], "MAE개선율_pct": mae_gain, "RMSE개선율_pct": rmse_gain,
            "최대계절_MAE악화율_pct": worst, "채택": decision, "판정사유": reason}, fold_rows


def candidate_state(base: PipelineState, step: int):
    out = PipelineState(**asdict(base))
    setattr(out, ["quantile_mapping", "residual_correction", "weather_cluster", "tree_structure"][step - 1], True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4]); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity = float(config["site"]["capacity_kw"]); seed = int(config["random_seed"])
    states = load_state(args.step)
    all_rows, verdict_rows, fold_rows = [], [], []
    targets = [("초단기", h) for h in [1, 2, 3, 4]] + [("단기", h) for h in [1, 24, 48]]
    if args.step == 4:
        targets.append(("일간", "D+1"))
    for tier, horizon in targets:
        key = f"{tier}_{horizon}"
        current = PipelineState(**states[key]); candidate = candidate_state(current, args.step)
        print(f"[{STEP_NAMES[args.step]}] {tier} {horizon} 시작", flush=True)
        if tier == "일간":
            rows = run_daily_structure(config, capacity, seed, current, candidate)
        else:
            rows = run_power_tier(tier, int(horizon), config, capacity, seed, current, candidate)
        if rows.empty:
            raise RuntimeError(f"시험행 없음: {key}")
        result, folds = verdict(rows)
        result.update({"단계": args.step, "개선안": STEP_NAMES[args.step], "티어": tier, "수평_h": horizon,
                       "시험행수": len(rows)})
        verdict_rows.append(result)
        for fr in folds:
            fr.update({"단계": args.step, "티어": tier, "수평_h": horizon})
            fold_rows.append(fr)
        rows["단계"] = args.step; rows["개선안"] = STEP_NAMES[args.step]
        all_rows.append(rows)
        if result["채택"]:
            states[key] = asdict(candidate)
        print(f"  -> {result['판정사유']} / MAE {result['MAE개선율_pct']:.2f}% / RMSE {result['RMSE개선율_pct']:.2f}%", flush=True)
    step_dir = OUT / f"{args.step}_{STEP_NAMES[args.step]}"
    step_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(step_dir / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(verdict_rows).to_csv(step_dir / "판정표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(step_dir / "계절안정성.csv", index=False, encoding="utf-8-sig")
    payload = {"완료단계": args.step, "최근개선안": STEP_NAMES[args.step], "티어수평별": states,
               "동결규칙": "동일행, MAE·RMSE 모두 1% 이상 개선, 계절 MAE 5% 악화 없음",
               "인식론적한계": "동일 5계절을 반복 사용한 후보검증이며 완전히 새로운 독립시험이 아니다."}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(verdict_rows).to_string(index=False))
    print(f"저장: {step_dir}")


if __name__ == "__main__":
    main()
