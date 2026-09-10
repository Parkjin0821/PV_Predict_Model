# -*- coding: utf-8 -*-
"""6번: 정규화 학습 → 모델 후보 비교 → 시간순(rolling-origin) 교차검증.

08-20 저녁 감사에서 6번으로 이월한 항목들을 여기서 한 번에 반영한다.

## 이 스크립트가 반영하는 감사 결론
1. **rolling-origin(확장창) 교차검증**: 단일 85:15 분할(여름 편중) 대신
   `config.json`의 `cross_validation_windows`(4계절 5폴드)로 검증한다.
2. **타깃 정규화 비교**: raw(kW) 타깃 vs 청천지수(clear-sky index) 정규화
   타깃 두 방식을 실제로 학습해 비교한다. 청천지수 kt = 발전량 /
   (설비용량 × 청천GHI(t)/1000), 청천GHI는 Haurwitz(1945) 모델(오늘 밤
   `interpolate_nwp_3h_to_1h_v1.py`와 동일 공식).
3. **smart persistence 기준모델**: "청천지수가 유지된다"고 가정
   (kt(t-1)을 그대로 써서 pred(t) = kt(t-1) × 청천전력(t))하는 표준
   기준모델. 단순지속성(직전값 그대로)도 같이 비교.
4. **skill score**: 1 − RMSE_model/RMSE_smart_persistence.
5. **nMAE·nRMSE**: 설비용량(240kW) 대비 정규화 지표.
6. **제도기준 성능**: KPX 재생에너지 예측제도 관행(이투뉴스 220309 등,
   미확정 2차자료 — AGENTS.md 08-20 저녁 절 참고)인 "이용률(용량대비
   발전량) 10% 이상 시간대"만 따로 성능을 낸다. 우리 발전소는 240kW로
   제도 직접 대상(20MW 초과)은 아니며, 이 지표는 참고용 비교 기준이다.
7. **모델 후보**: LightGBM, XGBoost (트리 기반 2종부터 시작 — LSTM/GRU는
   `train_hourly_deep.py`에 이미 뼈대가 있으나 이번 1차 비교에서는 제외,
   필요시 후속 진행).

## 특성 구성
- 기준선(항상 포함, 상관분석 대상 아님): 발전량 자기이력 lag(1~24h)·이동
  통계, 태양고도, 일/연주기성.
- 08-20 밤 최종 확정 외생변수 14개(`AGENTS.md` "밤(3차 수정, ★최종★)" 절):
  Blockdata 3개 + KMA 11개, 시간정렬 v3 규약(장비=shift1, 관측=shift0,
  예보=shift-1) 그대로 재사용.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "sel_v3", ROOT / "select_features_by_correlation_threshold_v3_2026-08-20밤.py"
)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

OUT_DIR = ROOT / "outputs" / "6번_시간단위_공식모델_v1_2026-08-20"

FINAL_14 = [
    "plant_input_power_kw", "DSWRF", "기상청관측_일조시간_hr", "mean_power_factor", "REH",
    "mean_input_voltage_v", "추정_출력온도", "기상청관측_상대습도_pct", "기상청관측_전운량_pct",
    "DSWRFLX_bsrn정제", "LCDC", "TCDC", "POP", "SKY",
]

SOLAR_CONSTANT = 1367.0
LATITUDE = 35.14428133


def clearsky_ghi(index: pd.DatetimeIndex) -> pd.Series:
    """Haurwitz(1945) 청천 전천일사 — interpolate_nwp_3h_to_1h_v1.py와 동일 공식."""
    n = index.dayofyear.to_numpy(float)
    local_hour = index.hour.to_numpy(float) + index.minute.to_numpy(float) / 60
    gamma = 2 * np.pi / 365 * (n - 1 + (local_hour - 12) / 24)
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    lat = math.radians(LATITUDE)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    true_solar_min = local_hour * 60 + eqtime + 4 * sel.__dict__.get("LONGITUDE", 126.84058771) - 60 * 9
    hour_angle = np.deg2rad(true_solar_min / 4 - 180)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    mu0 = np.clip(sin_elev, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi = np.where(mu0 > 0, 1098.0 * mu0 * np.exp(-0.059 / np.where(mu0 > 0, mu0, 1)), 0.0)
    return pd.Series(np.nan_to_num(ghi, nan=0.0, posinf=0.0, neginf=0.0), index=index)


def build_frame(df: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    cand = sel.build_candidate_frame(df)
    base = sel.make_base_features(df)
    frame = base.copy()
    for c in FINAL_14:
        frame[c] = cand[c]

    ghi_cs = clearsky_ghi(df.index)
    clearsky_power_kw = capacity_kw * (ghi_cs / 1000.0)
    frame["_청천전력_kW"] = clearsky_power_kw
    frame["_kt"] = np.where(clearsky_power_kw > 1.0, df["plant_output_kw"] / clearsky_power_kw, np.nan)
    frame["_kt_직전"] = frame["_kt"].shift(1)  # 스마트 지속성용(직전 완결값의 kt)
    frame["_지속성_직전출력_kW"] = df["plant_output_kw"].shift(1)  # 단순 지속성 기준모델

    frame["목표_발전출력_kW"] = df["plant_output_kw"]
    frame["목표_낮시간"] = (df["solar_elevation_deg"] > 0).astype(float)
    frame["목표_이용률_pct"] = df["plant_output_kw"] / capacity_kw * 100.0
    return frame


def metrics(y: np.ndarray, p: np.ndarray, capacity_kw: float) -> dict:
    e = y - p
    mae = float(np.abs(e).mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    return {
        "n": int(len(y)), "MAE_kW": mae, "RMSE_kW": rmse,
        "nMAE_pct": round(mae / capacity_kw * 100, 3), "nRMSE_pct": round(rmse / capacity_kw * 100, 3),
    }


def run_fold(fold_name: str, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str],
             capacity_kw: float, seed: int) -> dict:
    results = {"폴드": fold_name, "학습표본수": len(train), "시험표본수": len(test)}

    y_train_raw = train["목표_발전출력_kW"].to_numpy()
    y_test_raw = test["목표_발전출력_kW"].to_numpy()

    # --- 기준모델 ---
    persistence_pred = np.clip(test["_지속성_직전출력_kW"].to_numpy(), 0, capacity_kw)
    smart_pred = np.clip((test["_kt_직전"] * test["_청천전력_kW"]).to_numpy(), 0, capacity_kw)
    results["단순지속성"] = metrics(y_test_raw, persistence_pred, capacity_kw)
    results["스마트지속성"] = metrics(y_test_raw, smart_pred, capacity_kw)
    smart_rmse = results["스마트지속성"]["RMSE_kW"]

    models = {
        "LightGBM": LGBMRegressor(
            n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=220, learning_rate=0.04, max_depth=6, min_child_weight=5,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=0,
        ),
    }

    for model_name, model in models.items():
        # (A) raw kW 타깃
        m = model.__class__(**model.get_params())
        m.fit(train[feature_cols], y_train_raw)
        pred_raw = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
        met_raw = metrics(y_test_raw, pred_raw, capacity_kw)
        met_raw["SkillScore_vs스마트지속성"] = round(1 - met_raw["RMSE_kW"] / smart_rmse, 4) if smart_rmse > 0 else None
        results[f"{model_name}_raw"] = met_raw

        # (B) 청천지수(kt) 정규화 타깃
        tr = train.dropna(subset=["_kt"])
        m2 = model.__class__(**model.get_params())
        m2.fit(tr[feature_cols], tr["_kt"])
        kt_pred = m2.predict(test[feature_cols])
        pred_kt = np.clip(kt_pred * test["_청천전력_kW"].to_numpy(), 0, capacity_kw)
        met_kt = metrics(y_test_raw, pred_kt, capacity_kw)
        met_kt["SkillScore_vs스마트지속성"] = round(1 - met_kt["RMSE_kW"] / smart_rmse, 4) if smart_rmse > 0 else None
        results[f"{model_name}_청천지수정규화"] = met_kt

    # --- 제도기준(이용률>=10%) 하위 성능 (LightGBM raw 기준으로 대표 산출) ---
    util_mask = test["목표_이용률_pct"] >= 10.0
    if util_mask.sum() > 30:
        best_model = LGBMRegressor(
            n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3, random_state=seed, n_jobs=4, verbosity=-1,
        )
        best_model.fit(train[feature_cols], y_train_raw)
        pred_all = np.clip(best_model.predict(test[feature_cols]), 0, capacity_kw)
        y_util, p_util = y_test_raw[util_mask.to_numpy()], pred_all[util_mask.to_numpy()]
        m_util = metrics(y_util, p_util, capacity_kw)
        pct_err = np.clip(np.abs(y_util - p_util) / (y_util + 1e-9) * 100, None, 200)
        m_util["오차율_pct_평균"] = round(float(pct_err.mean()), 2)
        results["LightGBM_raw_제도기준(이용률10%이상)"] = m_util

    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    df = pd.read_csv(sel.DATA_CSV, parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    frame = build_frame(df, capacity_kw)

    feature_cols = [c for c in frame.columns if not c.startswith("_") and c not in
                    ("목표_발전출력_kW", "목표_낮시간", "목표_이용률_pct")]
    # DSWRFLX_bsrn정제는 6~8월 KMA 원본 결손(문자 "nan")으로 특정 구간에서
    # 거의 전부 결측인 게 이미 알려진 사실이다(AGENTS.md 아침 절). 이 한
    # 컬럼 때문에 dropna로 계절 폴드 전체가 날아가지 않도록, 이 컬럼은
    # dropna 필수 목록에서 빼고 LightGBM/XGBoost의 결측 네이티브 처리에
    # 맡긴다(둘 다 결측 분기 학습 지원 — 표준 기능, 임의 대체 아님).
    NATIVE_MISSING_OK = {"DSWRFLX_bsrn정제"}
    required = [c for c in feature_cols if c not in NATIVE_MISSING_OK] + [
        "목표_발전출력_kW", "_kt", "_kt_직전", "_지속성_직전출력_kW", "_청천전력_kW",
    ]
    daylight = frame[frame["목표_낮시간"] > 0].copy()
    print(f"낮시간 전체 표본: {len(daylight)}, 특성수: {len(feature_cols)}")

    all_results = []
    windows = config["cross_validation_windows"]
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절', '')}"
        train = daylight[daylight.index < start].dropna(subset=required)
        test_required = [c for c in feature_cols if c not in NATIVE_MISSING_OK] + [
            "목표_발전출력_kW", "_kt_직전", "_청천전력_kW",
        ]
        test = daylight[(daylight.index >= start) & (daylight.index <= end)].dropna(subset=test_required)
        if len(train) < 200 or len(test) < 30:
            print(f"[{fold_name}] 표본 부족으로 건너뜀 (train={len(train)}, test={len(test)})")
            continue
        print(f"\n=== 폴드 {fold_name}: 학습<{start.date()}, 검증 {start.date()}~{end.date()} ===")
        result = run_fold(fold_name, train, test, feature_cols, capacity_kw, seed)
        all_results.append(result)
        for key in ["단순지속성", "스마트지속성", "LightGBM_raw", "LightGBM_청천지수정규화", "XGBoost_raw", "XGBoost_청천지수정규화"]:
            r = result.get(key)
            if r:
                extra = f" skill={r.get('SkillScore_vs스마트지속성')}" if "SkillScore_vs스마트지속성" in r else ""
                print(f"  {key:30s} MAE={r['MAE_kW']:.2f} RMSE={r['RMSE_kW']:.2f} nRMSE={r['nRMSE_pct']:.2f}%{extra}")

    (OUT_DIR / "폴드별_상세결과.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 폴드 평균 요약
    summary_rows = []
    for key in ["단순지속성", "스마트지속성", "LightGBM_raw", "LightGBM_청천지수정규화", "XGBoost_raw", "XGBoost_청천지수정규화"]:
        maes = [r[key]["MAE_kW"] for r in all_results if key in r]
        rmses = [r[key]["RMSE_kW"] for r in all_results if key in r]
        nrmses = [r[key]["nRMSE_pct"] for r in all_results if key in r]
        skills = [r[key]["SkillScore_vs스마트지속성"] for r in all_results if key in r and r[key].get("SkillScore_vs스마트지속성") is not None]
        summary_rows.append({
            "구성": key, "폴드수": len(maes),
            "평균MAE_kW": round(float(np.mean(maes)), 3) if maes else None,
            "평균RMSE_kW": round(float(np.mean(rmses)), 3) if rmses else None,
            "평균nRMSE_pct": round(float(np.mean(nrmses)), 3) if nrmses else None,
            "평균SkillScore": round(float(np.mean(skills)), 4) if skills else None,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "폴드평균_요약.csv", index=False, encoding="utf-8-sig")
    print("\n\n=== 5폴드 평균 요약 ===")
    print(summary.to_string(index=False))
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
