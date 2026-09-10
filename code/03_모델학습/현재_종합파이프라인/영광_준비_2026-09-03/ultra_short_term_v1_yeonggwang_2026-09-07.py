# -*- coding: utf-8 -*-
"""영광 초단기(+1h~+4h) 총출력모델(09-07) - 김제 v1~v3 방법론 통합 재사용.

## 재사용한 것(코드 로직만 - 데이터는 전부 영광 자체)
김제 `ultra_short_term_v1/v2/v3_gimje_*`가 이미 검증한 방법론을 한 파일로
합쳐서 영광에 적용한다(재구현 아니라 그대로 재사용, 사용자에게 09-07
확인받은 원칙 - "코드 틀만 재사용, 데이터·모델은 영광 자체로 독립"):
- 타깃 raw(kW), 스마트지속성(kt×청천전력), Haurwitz 청천GHI: 광주
  2026-08-20밤 검증결론 재사용(v1과 동일 근거).
- 리드타임 파라미터화(+1~4h, v2 방식).
- LightGBM 구조튜닝 3후보(raw/shallow_reg/leaf_reg, v3 방식) - 매 폴드
  학습구간 내부 80/20 홀드아웃에서만 구조를 고른다(시험폴드 leakage 없음).

## 영광이 김제·부안과 다른 점(★중요, 실측 확인함★)
영광 자체 관측소 ASOS252는 **실제 일사량계를 보유**한다(09-07 실측
확인: `일사량_W_m2` 결측률 45.8%, 유효값 9,219건이 물리적으로 타당한
범위(0~1,080 W/m2, 평균 324)로 분포 - 부안·김제가 재사용하는 ASOS243은
일사계가 아예 없어 100% 결측이었던 것과 다름). 그래서 `obs_ghi_wm2`를
후보 특성에 **추가**한다 - 김제·부안엔 없던 영광만의 독립적 이점이지,
다른 지역 자료를 끌어온 게 아니다.

영광은 김제와 달리 알려진 "결함구간"(defect_period) 설정이 없어(02_전처리/
영광 폴더에 그런 config 없음, 완전가용률 99.947%로 이미 확인됨) 그
제외 단계는 생략한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
PLANT_5MIN = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
    r"\영광_발전소_5분_공식후보.parquet"
)
ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31\ASOS"
    r"\기상청_ASOS252_시간환경_20240825_20260804.csv"
)
OUT_DIR = HERE / "outputs" / "영광_초단기_v1_2026-09-07"

CAPACITY_KW = 634.0  # 인버터 등록용량 합계(50kW*12+34kW*1) - 광주 219kW·부안 1000kW·김제 1100kW와 동일 관례
SEED = 42
INITIAL_TRAIN_DAYS = 90
TEST_BLOCK_DAYS = 30
MIN_ROWS_PER_FOLD = 30
VIF_CORR_MIN_ROWS = 10
LEAD_HOURS = [1, 2, 3, 4]
HOLDOUT_FRAC = 0.2

ASOS_COLS = {
    "시각": "observation_time_kst", "지점번호": "station",
    "기온_C": "obs_temp_c", "강수량_mm": "obs_rain_mm", "풍속_m_s": "obs_wind_ms",
    "상대습도_pct": "obs_rh_pct", "전운량_10분위": "obs_cloud_tenths", "전운량_pct": "obs_cloud_pct",
    "일사량_W_m2": "obs_ghi_wm2",  # ★영광만 실측 가능(ASOS252에 일사계 있음)★
}
BASE_FEATURES = [
    "solar_elevation_now", "solar_elevation_target",
    "obs_temp_c", "obs_rh_pct", "obs_cloud_pct", "obs_wind_ms", "obs_ghi_wm2",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_MINUTES = [0, 15, 30, 60]

STRUCTURES = {
    "raw": dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=5,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=1.0),
    "shallow_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=7, max_depth=3,
                        min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
                        reg_alpha=0.3, reg_lambda=2.0),
    "leaf_reg": dict(n_estimators=250, learning_rate=0.04, num_leaves=31, max_depth=-1,
                     min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
                     reg_alpha=0.5, reg_lambda=3.0),
}

VALID_QUALITY = {"complete_observed", "complete_with_short_interpolation",
                 "complete_with_night_zero", "complete_with_idle_zero"}


def haurwitz_clearsky_ghi_wm2(elevation_deg: np.ndarray) -> np.ndarray:
    elev = np.clip(elevation_deg, 0, 90)
    zenith_rad = np.deg2rad(90.0 - elev)
    cosz = np.cos(zenith_rad)
    cosz_safe = np.where(cosz > 1e-3, cosz, np.nan)
    ghi = 1098.0 * cosz_safe * np.exp(-0.059 / cosz_safe)
    return np.where(elevation_deg <= 0, 0.0, np.nan_to_num(ghi, nan=0.0))


def load_base() -> tuple[pd.DataFrame, dict]:
    plant = pd.read_parquet(PLANT_5MIN)
    plant["grid_time_kst"] = pd.to_datetime(plant["grid_time_kst"])
    plant = plant.sort_values("grid_time_kst").reset_index(drop=True)

    diffs = plant["grid_time_kst"].diff().dropna().unique()
    if not (len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=5)):
        raise RuntimeError(f"5분 연속그리드가 아님: {diffs[:5]}")

    plant.loc[~plant["quality_status"].isin(VALID_QUALITY), "plant_ac_power_kw"] = np.nan

    asos = pd.read_csv(ASOS_CSV).rename(columns=ASOS_COLS)
    asos["observation_time_kst"] = pd.to_datetime(asos["observation_time_kst"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    keep = ["observation_time_kst"] + [v for v in ASOS_COLS.values() if v not in ("observation_time_kst", "station")]
    asos = asos[keep].sort_values("observation_time_kst").drop_duplicates("observation_time_kst")

    df = pd.merge_asof(plant, asos, left_on="grid_time_kst", right_on="observation_time_kst",
                       direction="backward", tolerance=pd.Timedelta("3h"))

    for m in LAG_MINUTES:
        slots = m // 5
        df[f"power_lag_{m}min"] = df["plant_ac_power_kw"].shift(slots)

    df["solar_elevation_now"] = df["solar_elevation_deg"]
    df["clearsky_ghi_now_wm2"] = haurwitz_clearsky_ghi_wm2(df["solar_elevation_now"].to_numpy())
    df["clearsky_power_now_kw"] = CAPACITY_KW * df["clearsky_ghi_now_wm2"] / 1000.0
    kt_valid = df["clearsky_power_now_kw"] > 1.0
    df["kt_now"] = np.where(kt_valid, df["power_lag_0min"] / df["clearsky_power_now_kw"], np.nan)
    df["kt_now"] = np.clip(df["kt_now"], 0, 1.5)

    hour = df["grid_time_kst"].dt.hour + df["grid_time_kst"].dt.minute / 60.0
    doy = df["grid_time_kst"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day"] = df["grid_time_kst"].dt.normalize()

    return df, {"결함구간_제외행수(5분)": 0, "비고": "영광은 알려진 결함구간 없음(완전가용률 99.947%)"}


def build_horizon_frame(df: pd.DataFrame, lead_hours: int) -> tuple[pd.DataFrame, str, list[str]]:
    lead_slots = int(lead_hours * 60 / 5)
    target_col = f"target_power_t+{lead_hours}h_kw"
    d = df.copy()
    d[target_col] = d["plant_ac_power_kw"].shift(-lead_slots)
    d["solar_elevation_target"] = d["solar_elevation_deg"].shift(-lead_slots)
    d["clearsky_ghi_target_wm2"] = haurwitz_clearsky_ghi_wm2(d["solar_elevation_target"].to_numpy())
    d["clearsky_power_target_kw"] = CAPACITY_KW * d["clearsky_ghi_target_wm2"] / 1000.0
    d = d[d["solar_elevation_target"] > 0].copy()
    features = [f"power_lag_{m}min" for m in LAG_MINUTES] + BASE_FEATURES
    return d, target_col, features


def pooled_score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = y_true - pred
    return {"n": int(len(y_true)), "MAE_kW": round(float(np.mean(np.abs(err))), 2),
            "RMSE_kW": round(float(np.sqrt(np.mean(err ** 2))), 2)}


def expanding_folds_full_coverage(days: pd.DatetimeIndex, initial: int, block: int) -> list[tuple]:
    n = len(days)
    folds, end = [], initial
    while end < n:
        test_end = min(end + block, n)
        folds.append((days[:end], days[end:test_end]))
        end = test_end
    return folds


def fit_predict(params: dict, train: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=-1)
    model.fit(train[features], train[target])
    return np.clip(model.predict(eval_df[features]), 0, None)


def select_structure_internal_holdout(train_days, d, target_col, features, rng) -> str:
    days = np.array(sorted(train_days))
    n_holdout = max(1, int(len(days) * HOLDOUT_FRAC))
    holdout_days = set(rng.choice(days, size=n_holdout, replace=False))
    inner_train_days = [dd for dd in days if dd not in holdout_days]

    inner_train = d[d["day"].isin(inner_train_days)].dropna(subset=features + [target_col])
    inner_holdout = d[d["day"].isin(holdout_days)].dropna(subset=features + [target_col])
    if len(inner_train) < 30 or len(inner_holdout) < 10:
        return "raw"

    best_name, best_mae = "raw", np.inf
    for name, params in STRUCTURES.items():
        pred = fit_predict(params, inner_train, inner_holdout, features, target_col)
        mae = float(np.mean(np.abs(inner_holdout[target_col].to_numpy() - pred)))
        if mae < best_mae:
            best_mae, best_name = mae, name
    return best_name


def run_walkforward_tuned(d: pd.DataFrame, target_col: str, features: list[str], folds: list[tuple]) -> dict:
    rng = np.random.RandomState(SEED)
    all_true, all_model, all_simple, all_smart = [], [], [], []
    fold_rows = []
    structure_votes: dict[str, int] = {}
    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출"
        need = features + [target_col, "power_lag_0min", "kt_now", "clearsky_power_target_kw"]
        train = d[d["day"].isin(train_days)].dropna(subset=features + [target_col])
        test = d[d["day"].isin(test_days)].dropna(subset=need)
        if len(train) < MIN_ROWS_PER_FOLD or len(test) < 1:
            continue

        chosen = select_structure_internal_holdout(train_days, d, target_col, features, rng)
        structure_votes[chosen] = structure_votes.get(chosen, 0) + 1
        pred_model = fit_predict(STRUCTURES[chosen], train, test, features, target_col)

        y_true = test[target_col].to_numpy()
        pred_simple = test["power_lag_0min"].to_numpy()
        pred_smart = np.clip(test["kt_now"].to_numpy() * test["clearsky_power_target_kw"].to_numpy(), 0, None)

        all_true.append(y_true); all_model.append(pred_model)
        all_simple.append(pred_simple); all_smart.append(pred_smart)
        fold_rows.append({"폴드": i, "시험일수": len(test_days), "시험행수": len(test),
                          "선택된구조": chosen,
                          "모델_MAE": pooled_score(y_true, pred_model)["MAE_kW"],
                          "스마트지속성_MAE": pooled_score(y_true, pred_smart)["MAE_kW"]})

    if not all_true:
        return {"폴드수": 0, "폴드별": [], "오류": "유효 폴드 없음"}
    y_true_all = np.concatenate(all_true)
    return {
        "폴드수": len(fold_rows), "폴드별": fold_rows, "구조선택_투표분포": structure_votes,
        "pooled_모델": pooled_score(y_true_all, np.concatenate(all_model)),
        "pooled_단순지속성": pooled_score(y_true_all, np.concatenate(all_simple)),
        "pooled_스마트지속성": pooled_score(y_true_all, np.concatenate(all_smart)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, meta = load_base()

    def improve(model_mae, base_mae):
        return round((1 - model_mae / base_mae) * 100, 1) if base_mae else None

    horizon_results = {}
    for h in LEAD_HOURS:
        d, target_col, features = build_horizon_frame(base, h)
        days = pd.DatetimeIndex(np.sort(d["day"].unique()))
        folds = expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)
        perf = run_walkforward_tuned(d, target_col, features, folds)
        entry = {"리드타임": f"+{h}h", "일수_total": int(len(days)), "성능": perf}
        if perf.get("폴드수", 0) > 0:
            entry["성능"]["MAE_개선율_vs단순지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_단순지속성"]["MAE_kW"])
            entry["성능"]["MAE_개선율_vs스마트지속성_pct"] = improve(perf["pooled_모델"]["MAE_kW"], perf["pooled_스마트지속성"]["MAE_kW"])
            entry["nMAE_pct"] = round(perf["pooled_모델"]["MAE_kW"] / CAPACITY_KW * 100, 3)
        horizon_results[f"+{h}h"] = entry
        print(f"[+{h}h] 완료 - pooled MAE(모델)={perf.get('pooled_모델',{}).get('MAE_kW')}kW, "
              f"nMAE={entry.get('nMAE_pct')}%, 구조분포={perf.get('구조선택_투표분포')}")

    result = {
        **meta, "capacity_kw_사용값": CAPACITY_KW,
        "특징": "영광 ASOS252 실측일사량(obs_ghi_wm2) 후보 포함 - 부안·김제(ASOS243, 일사계 없음)와의 차이점",
        "평가대상": "각 리드타임의 대상시각 태양고도>0인 행만",
        "리드타임별_결과": horizon_results,
        "_방법론출처": "타깃(raw kW)·스마트지속성·Haurwitz청천전력·구조튜닝은 김제 v1~v3(09-01) 코드 로직 재사용 "
                     "(데이터·학습·결과는 전부 영광 자체, 다른 지역 자료 미사용). "
                     "obs_ghi_wm2 포함은 영광 고유(실측 일사계 보유).",
        "_판정": "잠정치 - promote_to_official 대상 아님(historical archive join candidate 한계 상속, 라이브 shadow 검증 전).",
    }
    (OUT_DIR / "영광_초단기_1to4h_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "리드타임별_결과"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
