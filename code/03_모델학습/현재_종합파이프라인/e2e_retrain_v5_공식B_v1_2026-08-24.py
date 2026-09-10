# -*- coding: utf-8 -*-
"""⑤ v5·공식 B·219kW 전 모델 재학습 — 특성선택→구조선택/튜닝→저장→재적재.

## 범위(사용자 지시 "5번 재학습"만, 6~10번은 별도)
초단기(+1~4h)·단기(+1/+24/+48h)·일간(D+1) 전부. **기존 2차 공식모델과
같은 구성**(초단기 +1h·+2h=raw+구조선택, +3h·+4h=청천지수 기본,
단기 전체=기본+구조선택, 일간=전체특성+폴드내부튜닝)을 **그대로 유지한
채** 데이터만 v5·정책만 공식 B·용량만 219kW로 바꿔 다시 학습한다.
①~④ 후보(분위사상·OOF잔차보정·날씨군집·raw vs 청천지수 등) 재판정은
⑥에서 별도로 한다 — 여기서 구성 자체를 새로 고르지 않는다.

## 재사용(재구현 금지 원칙)
- 폴드정의: `fold_hour_prevalidation_v1_2026-08-24.OFFICIAL_B_WINDOWS`
  (가을 시작 11-17 정정판, 4번 게이트 통과 확인됨)
- v5 데이터 적재·프레임 구성: `defect_policy_comparison_v1_2026-08-21`
  (`load_ultra_frame`/`load_short_frame`, harness·ultra 모듈 포함)
- 구조선택·모델: `model_improvement_round2_v1_2026-08-21`
  (`choose_structure`/`make_model`)
- 하이퍼파라미터 튜닝(일간): `hyperparameter_tuning_v1_2026-08-21.tune_fold`
- 청천지수: `ultra_short_clearsky_v1_2026-08-21`
- DIFSWRF 결측처리: `difswrf_missing_strategy_comparison_v1_2026-08-24`가
  채택한 C(원값 NaN 유지 + 결측여부 플래그)를 그대로 적용

## 공식 B 정책
학습·시험 모두 **대상시각(초단기·단기) 가용인버터수=5**인 행만 사용한다
(defect_policy_comparison과 동일 정의). 추정타깃 포함 0건.

## 일간 v5 패치(Codex 08-24 지적사항 그대로 구현)
- 타깃·설비특성 출처를 v5로 교체(`집계_일간_실제발전량_v5.parquet`,
  `집계_1시간_자료_v5.parquet`의 `가용인버터수`).
- **7일전/2일전 발전량 파생컬럼의 부분가용 결측을 dropna로 버리지 않고
  NaN 유지 + `_결측여부` 플래그를 추가**한다(LightGBM 자체 처리).
  지속성 기준선(7일전값)이 없는 날은 지속성 비교에서만 별도 제외한다.
- NWP 일집계를 `sum(min_count=1)`로 고쳐 하루 전체 결측이 0일사로
  둔갑하는 걸 방지(Codex 지적사항).

## 산출물 (`outputs/E2E_v5_공식B_v1_2026-08-24/`)
- `성능_전체.csv`, `성능_폴드별.csv`, `누출_및_재적재_감사.csv`
- `행단위_ac_power_예측정답.csv`, `행단위_daily_예측정답.csv`
- `요약.json`, `fold_models/`(joblib)

로컬 재학습만(API 없음).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
V5_DIR = ROOT / "outputs" / "v5_복구_2026-08-21"
OUT = ROOT / "outputs" / "E2E_v5_공식B_v1_2026-08-24"
MODEL_DIR = OUT / "fold_models"
PLANT_ID = 6715
PLANT_NAME = "광주 광주시청"
N_INVERTERS = 5
DIF = "DIFSWRF_bsrn정제"
DIF_MASK = f"{DIF}_결측여부"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpc = _load("retrain_dpc", "defect_policy_comparison_v1_2026-08-21.py")
harness, ultra = dpc.harness, dpc.ultra
improvement = _load("retrain_round2", "model_improvement_round2_v1_2026-08-21.py")
tuning = _load("retrain_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
clearsky = _load("retrain_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
fhv = _load("retrain_fhv", "fold_hour_prevalidation_v1_2026-08-24.py")
OFFICIAL_WINDOWS = fhv.OFFICIAL_B_WINDOWS

ULTRA_OFFICIAL = {1: "raw", 2: "raw", 3: "청천지수", 4: "청천지수"}
ULTRA_STRUCTURE = {1: True, 2: True, 3: False, 4: False}  # 2차 공식모델과 동일 구성 유지


def _safe(text: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in text)


def _cast(params, integer_names):
    if params is None:
        return None
    out = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        if k in integer_names and v is not None:
            v = int(v)
        out[k] = v
    return out


def _roundtrip(model, bundle, x, path: Path):
    before = np.asarray(model.predict(x), float)
    bundle = {**bundle, "model": model}
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    after = np.asarray(loaded["model"].predict(x), float)
    diff = float(np.max(np.abs(before - after))) if len(before) else 0.0
    return after, diff


def add_difswrf_flag(frame: pd.DataFrame) -> pd.DataFrame:
    if DIF in frame.columns:
        frame = frame.copy()
        frame[DIF_MASK] = frame[DIF].isna().astype(float)
    return frame


def run_ultra(capacity_kw: float, seed: int, policy: str = "B_구간제외",
              horizons: tuple = (1, 2, 3, 4), model_dir: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """policy: "B_구간제외"(공식) | "A_무처리" | "C_용량가중추정보정"(민감도 재검증용)."""
    model_dir = model_dir or MODEL_DIR
    rows, audits = [], []
    for horizon in horizons:
        official = ULTRA_OFFICIAL[horizon]
        frame = add_difswrf_flag(dpc.load_ultra_frame(horizon))
        candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        use_kappa = "청천지수" in official
        frame["_청천_kW"] = np.clip(
            capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
        # 정책C(민감도 재검증용): 부분가용 목표의 실측 kW를 용량가중 보정계수로
        # 스케일업한 뒤(추정 라벨, 정답데이터 비변경) 그 값으로 카파를 계산한다
        # — raw/kappa 어느 쪽이든 이 컬럼 하나로 정책별 타깃을 통일해 다룬다.
        if policy == "C_용량가중추정보정":
            scaled = (frame["목표_발전출력_kW"] * frame["_목표_용량가중보정계수"]).clip(upper=capacity_kw)
            frame["_정책타깃_kW"] = np.where(
                frame["_목표_가용인버터수"] >= N_INVERTERS, frame["목표_발전출력_kW"], scaled)
        else:
            frame["_정책타깃_kW"] = frame["목표_발전출력_kW"]
        frame["_카파"] = frame["_정책타깃_kW"] / frame["_청천_kW"]
        min_elev = clearsky.MIN_ELEVATION_DEG if use_kappa else 0.0
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]

        for i, (fold, s, e) in enumerate(OFFICIAL_WINDOWS, start=1):
            start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
            full_mask = daylight["_목표_가용인버터수"] >= N_INVERTERS
            train_pool = daylight[daylight.index < start]
            train_b = train_pool if policy != "B_구간제외" else train_pool[full_mask.loc[train_pool.index]]
            test_b = daylight[(daylight.index >= start) & (daylight.index < end) & full_mask]  # 평가는 항상 완전가용만(정책 무관 고정)
            if len(train_b) < 300 or len(test_b) < 30:
                print(f"  [초단기 +{horizon}h {fold}] 표본부족(train={len(train_b)}, test={len(test_b)}) — 건너뜀")
                continue
            # select_features_in_fold는 "목표_발전출력_kW" 컬럼명을 타깃으로 고정
            # 참조하므로, 정책별 실제 학습타깃(_정책타깃_kW)을 그 이름으로 임시
            # 대입해 넘긴다(원본 컬럼은 안 건드림).
            sel_input = train_b.drop(columns=["목표_발전출력_kW"]).assign(
                목표_발전출력_kW=train_b["_정책타깃_kW"]).dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            native_ok = harness.NATIVE_MISSING_OK | {DIF}
            required = [c for c in features if c not in native_ok] + ["_정책타깃_kW", "_청천_kW", "_카파"]
            train = train_b.dropna(subset=required)
            test = test_b.dropna(subset=[c for c in features if c not in native_ok]
                                 + ["목표_발전출력_kW", "_청천_kW"])
            if len(train) < 300 or len(test) < 30:
                continue

            target = "_카파" if use_kappa else "_정책타깃_kW"
            structure_name, params = "기본", None
            if ULTRA_STRUCTURE[horizon]:
                params, structure_name = improvement.choose_structure_kfold(
                    "초단기", train, features, target, "kappa" if use_kappa else "raw", capacity_kw, seed)
                model = improvement.make_model("초단기", seed, params)
            else:
                model = ultra.make_model("LightGBM", seed)
            model.fit(train[features], train[target])

            variant = official + ("+구조선택" if ULTRA_STRUCTURE[horizon] else "") + f"[{policy}]"
            artifact = model_dir / _safe(f"초단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = _roundtrip(model, {
                "tier": "초단기", "horizon_h": horizon, "variant": variant, "features": features,
                "params": params, "target_transform": target, "structure_name": structure_name,
                "정책": policy, "용량프로필": "inverter_registered_sum_219",
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, test[features], artifact)
            pred = raw * test["_청천_kW"].to_numpy() if use_kappa else raw
            pred = np.clip(pred, 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon, unit="h")
            for issued, tt, y, p in zip(test.index, target_at, actual, pred):
                rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "초단기",
                            "수평_h": horizon, "공식구성": variant, "폴드": fold,
                            "발행시각": issued, "대상시각": tt, "실제_kW": y, "예측_kW": p,
                            "버킷시간_h": 0.25, "모델파일": str(artifact)})
            audits.append({"티어": "초단기", "수평_h": horizon, "폴드": fold,
                           "학습최종시각": train.index.max(), "시험최초시각": test.index.min(),
                           "학습_시험_분리통과": bool(train.index.max() < test.index.min()),
                           "추정타깃포함": bool(policy == "C_용량가중추정보정"), "정책": policy,
                           "모델재적재_동일": bool(diff <= 1e-12), "재적재차이": diff,
                           "특성수": len(features), "선택구조": structure_name,
                           "학습행수": len(train), "시험행수": len(test)})
            print(f"[초단기 +{horizon}h {fold}] {variant} 완료 (학습{len(train):,}/시험{len(test):,})")
    return pd.DataFrame(rows), audits


def run_short(capacity_kw: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    rows, audits = [], []
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    for horizon in (1, 24, 48):
        frame = add_difswrf_flag(dpc.load_short_frame(horizon))
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        daylight = frame[frame["목표_낮시간"] > 0]

        for fold, s, e in OFFICIAL_WINDOWS:
            start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
            train_b = daylight[(daylight.index < start) & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
            test_b = daylight[(daylight.index >= start) & (daylight.index < end)
                              & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
            if len(train_b) < 200 or len(test_b) < 30:
                print(f"  [단기 +{horizon}h {fold}] 표본부족(train={len(train_b)}, test={len(test_b)}) — 건너뜀")
                continue
            chosen = harness.select_features_in_fold(
                train_b.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            # 08-25 재감사로 발견·수정: run_ultra()와 동일하게 DIF를 native_ok에
            # 포함해 C전략(원값 NaN 유지+결측여부 플래그, native missing 위임)을
            # 적용한다. 좁은 재감사 결과 공식 5폴드에서는 DIF가 한 번도 선택되지
            # 않아(0/15) 이 수정이 기존 배포 수치를 바꾸지는 않지만, 코드 자체는
            # C전략과 일치시켜둔다(`short_difswrf_c전략_연결_재감사_v1_2026-08-25.py`).
            native_ok = harness.NATIVE_MISSING_OK | {DIF}
            required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
            train = train_b.dropna(subset=required)
            test = test_b.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue

            params, structure_name = improvement.choose_structure_kfold(
                "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
            model = improvement.make_model("단기", seed, params)
            model.fit(train[features], train["목표_발전출력_kW"])

            variant = "기본+구조선택"
            artifact = MODEL_DIR / _safe(f"단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = _roundtrip(model, {
                "tier": "단기", "horizon_h": horizon, "variant": variant, "features": features,
                "params": params, "target_transform": "raw_kW", "structure_name": structure_name,
                "정책": "B_구간제외", "용량프로필": "inverter_registered_sum_219",
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, test[features], artifact)
            pred = np.clip(raw, 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon - 1, unit="h")
            for issued, tt, y, p in zip(test.index, target_at, actual, pred):
                rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "단기",
                            "수평_h": horizon, "공식구성": variant, "폴드": fold,
                            "발행시각": issued, "대상시각": tt, "실제_kW": y, "예측_kW": p,
                            "버킷시간_h": 1.0, "모델파일": str(artifact)})
            audits.append({"티어": "단기", "수평_h": horizon, "폴드": fold,
                           "학습최종시각": train.index.max(), "시험최초시각": test.index.min(),
                           "학습_시험_분리통과": bool(train.index.max() < test.index.min()),
                           "추정타깃포함": False, "정책": "B_구간제외",
                           "모델재적재_동일": bool(diff <= 1e-12), "재적재차이": diff,
                           "특성수": len(features), "선택구조": structure_name,
                           "학습행수": len(train), "시험행수": len(test)})
            print(f"[단기 +{horizon}h {fold}] {variant} 완료 (학습{len(train):,}/시험{len(test):,})")
    return pd.DataFrame(rows), audits


# ── 일간(v5 패치) ────────────────────────────────────────────────────
NWP_AGG_SUM_COLS = {"DSWRF": "sum", "DSWRFLX_bsrn정제": "sum", "DIFSWRF_bsrn정제": "sum",
                    "POP": "max", "WSD": "max", "TCDC": "max", "TMP": "max"}
NWP_AGG_MEAN_COLS = ["DSWRF", "DSWRFLX_bsrn정제", DIF, "TCDC", "LCDC", "MCDC", "HCDC",
                     "REH", "POP", "SKY", "TMP", "WSD"]
NWP_AGG_MIN_COLS = {"REH": "min", "TMP": "min"}
EQUIPMENT_COLS = [
    "plant_input_power_kw", "mean_input_voltage_v", "mean_frequency_hz",
    "mean_power_factor", "mean_communication_ok",
    "기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_강수량_mm",
    "기상청관측_전운량_pct", "기상청관측_일사량_W_m2", "기상청관측_일조시간_hr",
    "기상청관측_풍속_m_s", "기상청관측_풍향_deg", "기상청관측_현지기압_hPa",
    "기상청관측_해면기압_hPa", "기상청관측_적설_cm", "기상청관측_지면온도_C",
]


def build_daily_dataset_v5(capacity_kw: float) -> tuple[pd.DataFrame, list[str]]:
    daily_actual = pd.read_parquet(V5_DIR / "집계_일간_실제발전량_v5.parquet").copy()
    daily_actual.index = pd.to_datetime(daily_actual.index)
    data = pd.DataFrame(index=daily_actual.index)
    ACTUAL = "실제_일간발전량_kWh"
    data[ACTUAL] = daily_actual["일간발전량_kWh"]
    data["목표일_부분가용여부"] = daily_actual.get("부분가용일", 0).reindex(data.index).fillna(0)

    # ★Codex 08-24 지적 반영★: 부분가용 과거일을 dropna로 버리지 않고
    # NaN 유지 + 결측여부 플래그를 추가한다.
    for lag, name in [(7, "7일전"), (2, "2일전")]:
        raw = data[ACTUAL].shift(lag)
        data[f"{name}_일간발전량_kWh"] = raw
        data[f"{name}_결측여부"] = raw.isna().astype(float)
    data["7일전지속성예측_kWh"] = data["7일전_일간발전량_kWh"]
    data["2일전기준_7일이동평균_kWh"] = data[ACTUAL].shift(2).rolling(7, min_periods=4).mean()
    data["2일전기준_30일이동평균_kWh"] = data[ACTUAL].shift(2).rolling(30, min_periods=15).mean()
    data["2일전기준_30일표준편차_kWh"] = data[ACTUAL].shift(2).rolling(30, min_periods=15).std()

    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(V5_DIR / "집계_1시간_자료_v5.parquet")
    common = hourly.index.intersection(v5_1h.index)
    hourly = hourly.loc[common].copy()
    hourly["inverters_available"] = v5_1h.loc[common, "가용인버터수"]
    hourly["날짜"] = hourly.index.normalize()

    # ★Codex 08-24 지적 반영★: sum(min_count=1) — 하루 전체 결측이 0일사로
    # 둔갑하지 않게 한다.
    nwp_sum = hourly.groupby("날짜")[list(NWP_AGG_SUM_COLS)].sum(min_count=1)
    nwp_sum.columns = [f"목표일예보_{c}_sum" for c in nwp_sum.columns]
    nwp_mean = hourly.groupby("날짜")[NWP_AGG_MEAN_COLS].mean()
    nwp_mean.columns = [f"목표일예보_{c}_mean" for c in nwp_mean.columns]
    nwp_min = hourly.groupby("날짜")[list(NWP_AGG_MIN_COLS)].min()
    nwp_min.columns = [f"목표일예보_{c}_min" for c in nwp_min.columns]
    nwp_maxcols = ["TCDC", "POP", "WSD", "TMP"]
    nwp_max = hourly.groupby("날짜")[nwp_maxcols].max()
    nwp_max.columns = [f"목표일예보_{c}_max" for c in nwp_max.columns]
    nwp_valid = hourly.groupby("날짜")[DIF].apply(lambda s: s.notna().sum())
    data = data.join(nwp_sum, how="left").join(nwp_mean, how="left").join(nwp_min, how="left").join(nwp_max, how="left")
    data["목표일_DIFSWRF_유효개수"] = nwp_valid.reindex(data.index)
    data[f"목표일예보_{DIF}_sum_결측여부"] = data[f"목표일예보_{DIF}_sum"].isna().astype(float)

    equipment_daily = hourly[EQUIPMENT_COLS + ["inverters_available", "날짜"]].groupby("날짜").mean(numeric_only=True).shift(2)
    equipment_daily.columns = [f"2일전평균_{c}" for c in equipment_daily.columns]
    data = data.join(equipment_daily, how="left")

    day = data.index.dayofyear
    data["목표일_연주기_sin"] = np.sin(2 * np.pi * day / 365.25)
    data["목표일_연주기_cos"] = np.cos(2 * np.pi * day / 365.25)
    data["목표일_월"] = data.index.month
    data["목표일_요일"] = data.index.dayofweek
    data["해발고도_m"] = float(hourly["site_elevation_dem_m"].dropna().median())
    data["설비용량_kW"] = capacity_kw

    features = [c for c in data.columns if c not in {ACTUAL, "7일전지속성예측_kWh", "목표일_부분가용여부"}]
    # ★공식 B★: 목표일 자체가 부분가용이면(정답이 불확실) 제외. 과거
    # lag/평균 입력의 부분가용은 NaN+플래그로 남겨두고 행을 버리지 않는다.
    data = data[data["목표일_부분가용여부"] < 1]
    data = data.dropna(subset=[ACTUAL])
    data.index.name = "예측대상일"
    return data, features


def run_daily(capacity_kw: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    data, features = build_daily_dataset_v5(capacity_kw)
    ACTUAL = "실제_일간발전량_kWh"
    daily_capacity = capacity_kw * 24
    rows, audits = [], []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index < end)]
        if len(train) < 60 or len(test) < 10:
            print(f"  [일간 {fold}] 표본부족(train={len(train)}, test={len(test)}) — 건너뜀")
            continue
        medians = train[features].median(numeric_only=True)
        x_train, x_test = train[features].fillna(medians), test[features].fillna(medians)

        params, _ = tuning.tune_fold("LightGBM", train.fillna(medians), features, ACTUAL, daily_capacity, seed)
        params = _cast(params, ["n_estimators", "num_leaves", "max_depth", "min_child_samples"])
        model = improvement.make_model("일간", seed, params)
        model.fit(x_train, train[ACTUAL])

        variant = "전체특성+폴드내부튜닝(v5)"
        artifact = MODEL_DIR / _safe(f"일간_{fold}_{variant}.joblib")
        raw, diff = _roundtrip(model, {
            "tier": "일간", "horizon": "D+1", "variant": variant, "features": features,
            "medians": medians.to_dict(), "params": params, "target_transform": "daily_kWh",
            "정책": "B_구간제외", "용량프로필": "inverter_registered_sum_219",
            "train_end": str(train.index.max()), "test_start": str(test.index.min()),
        }, x_test, artifact)
        pred = np.clip(raw, 0, daily_capacity)
        actual = test[ACTUAL].to_numpy()
        issued_at = test.index - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
        for issued, target_date, y, p in zip(issued_at, test.index, actual, pred):
            rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "일간",
                        "수평_h": "D+1", "공식구성": variant, "폴드": fold,
                        "발행시각": issued, "대상일": target_date, "실제_kWh": y, "예측_kWh": p,
                        "모델파일": str(artifact)})
        audits.append({"티어": "일간", "수평_h": "D+1", "폴드": fold,
                       "학습최종시각": train.index.max(), "시험최초시각": test.index.min(),
                       "학습_시험_분리통과": bool(train.index.max() < test.index.min()),
                       "추정타깃포함": False, "정책": "B_구간제외",
                       "모델재적재_동일": bool(diff <= 1e-12), "재적재차이": diff,
                       "특성수": len(features), "선택구조": "튜닝",
                       "학습행수": len(train), "시험행수": len(test)})
        print(f"[일간 {fold}] {variant} 완료 (학습{len(train):,}/시험{len(test):,})")
    return pd.DataFrame(rows), audits


def make_performance(ac: pd.DataFrame, daily: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    rows = []
    for (tier, h), g in ac.groupby(["티어", "수평_h"]):
        e = g["실제_kW"] - g["예측_kW"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        rows.append({"티어": tier, "수평_h": h, "n": len(g), "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
                    "nMAE_pct": round(mae / capacity_kw * 100, 3), "nRMSE_pct": round(rmse / capacity_kw * 100, 3)})
    if len(daily):
        e = daily["실제_kWh"] - daily["예측_kWh"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        wape = float(e.abs().sum() / daily["실제_kWh"].abs().sum() * 100)
        rows.append({"티어": "일간", "수평_h": "D+1", "n": len(daily),
                    "MAE_kWh": round(mae, 2), "RMSE_kWh": round(rmse, 2), "WAPE_pct": round(wape, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) / 정책: B_구간제외 / 폴드: 공식 5폴드(가을 11-17 정정)\n")

    only = sys.argv[1] if len(sys.argv) > 1 else "all"  # "all"|"daily"(초단기·단기 재사용/생략용)
    if only in ("all", "ultra_short"):
        print("=== 초단기 ===")
        ac_u, aud_u = run_ultra(capacity_kw, seed)
        print("\n=== 단기 ===")
        ac_s, aud_s = run_short(capacity_kw, seed)
        ac_u.to_csv(OUT / "_임시_초단기.csv", index=False, encoding="utf-8-sig")
        ac_s.to_csv(OUT / "_임시_단기.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(aud_u).to_csv(OUT / "_임시_감사_초단기.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(aud_s).to_csv(OUT / "_임시_감사_단기.csv", index=False, encoding="utf-8-sig")
    else:
        ac_u = pd.read_csv(OUT / "_임시_초단기.csv", encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
        ac_s = pd.read_csv(OUT / "_임시_단기.csv", encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
        aud_u = pd.read_csv(OUT / "_임시_감사_초단기.csv", encoding="utf-8-sig").to_dict("records")
        aud_s = pd.read_csv(OUT / "_임시_감사_단기.csv", encoding="utf-8-sig").to_dict("records")
        print("=== 초단기·단기: 이전 실행 결과 재사용 ===")

    print("\n=== 일간 ===")
    daily, aud_d = run_daily(capacity_kw, seed)

    ac = pd.concat([ac_u, ac_s], ignore_index=True) if len(ac_u) or len(ac_s) else pd.DataFrame()
    audits = pd.DataFrame(aud_u + aud_s + aud_d)

    perf = make_performance(ac, daily, capacity_kw)
    ac.to_csv(OUT / "행단위_ac_power_예측정답.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "행단위_daily_예측정답.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(OUT / "성능_전체.csv", index=False, encoding="utf-8-sig")
    audits.to_csv(OUT / "누출_및_재적재_감사.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 성능(v5·공식B·219kW) ===")
    print(perf.to_string(index=False))
    print("\n=== 감사 요약 ===")
    print(f"조합 수: {len(audits)} / 학습시험분리 전부통과: {bool(audits['학습_시험_분리통과'].all())} "
          f"/ 재적재동일 전부통과: {bool(audits['모델재적재_동일'].all())} "
          f"/ 추정타깃포함: {int(audits['추정타깃포함'].sum())}건")

    summary = {
        "실행유형": "v5·공식B(구간제외)·219kW 전 모델 재학습(⑤)",
        "폴드": [w[0] for w in OFFICIAL_WINDOWS],
        "용량프로필": config["site"].get("capacity_profile"),
        "용량kW": capacity_kw,
        "조합수": int(len(audits)),
        "학습시험분리_전부통과": bool(audits["학습_시험_분리통과"].all()),
        "재적재_전부통과": bool(audits["모델재적재_동일"].all()),
        "추정타깃포함_건수": int(audits["추정타깃포함"].sum()),
        "라이브_API_호출": False,
        "범위_비고": "⑤만 수행(구성은 2차 공식모델과 동일하게 유지, ①~④ 후보 재판정은 ⑥에서 별도)",
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
