from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
SEGMENT_SOURCE = HERE / "retrain_단기_v1_segmented_2026-09-09.py"
REGIONAL_SOURCE = PIPELINE_ROOT / "retrain_regional_short_segmented_candidate_v1_2026-09-09.py"
V4_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
    r"\gwangju_1hour_model_dataset_official_v4_segmented_candidate_2026-09-09.csv"
)
NC_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\kma_nwp_d1d2_live_v1_2026-09-09\kma_nwp_d1d2_live.sqlite3"
)
OUT = HERE / "outputs" / "단기_재학습_후보_2026-09-09" / "+48h"
CAPACITY_KW = 240.0
NC_VARIABLES = ["DSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]
ISSUE_FEATURES = [
    "plant_input_power_kw",
    "mean_power_factor",
    "mean_input_voltage_v",
    "발전출력_24시간전_kW",
    "발전출력_24시간이동평균_kW",
    "기상청관측_일조시간_hr",
    "기상청관측_상대습도_pct",
    "기상청관측_전운량_pct",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_nc() -> pd.DataFrame:
    with sqlite3.connect(NC_DB) as con:
        raw = pd.read_sql_query(
            """
            SELECT issue_date, target_time_kst, lead_hours, variable,
                   value, is_missing, first_received_at, dry_run
            FROM nwp_d1d2_values
            WHERE dry_run=0 AND lead_hours BETWEEN 37 AND 59
            """,
            con,
        )
    raw.loc[raw["is_missing"].eq(1), "value"] = np.nan
    raw["target_time_kst"] = (
        pd.to_datetime(raw["target_time_kst"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.tz_localize(None)
        .dt.floor("h")
    )
    raw["first_received_at"] = pd.to_datetime(raw["first_received_at"], errors="coerce")
    wide = raw.pivot_table(
        index=["issue_date", "target_time_kst", "lead_hours"],
        columns="variable",
        values="value",
        aggfunc="last",
    ).reset_index()
    return wide.rename(columns={v: f"forecast_{v}" for v in NC_VARIABLES})


def build_training_frame(segment) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(V4_CSV, encoding="utf-8-sig", low_memory=False)
    source["time"] = pd.to_datetime(source["time"], errors="coerce")
    source = source.dropna(subset=["time"]).set_index("time").sort_index()
    nc = load_nc()

    frames = []
    for lead in range(37, 60):
        # build_frame_segment_aware uses target shift -(horizon-1), hence lead+1.
        framed = segment.build_frame_segment_aware(source.copy(), lead + 1).reset_index()
        framed = framed.rename(columns={framed.columns[0]: "issue_time_kst"})
        framed["issue_time_kst"] = pd.to_datetime(framed["issue_time_kst"]).dt.floor("h")
        framed["issue_date"] = framed["issue_time_kst"].dt.strftime("%Y%m%d")
        framed["target_time_kst"] = framed["issue_time_kst"] + pd.to_timedelta(lead, unit="h")
        framed["lead_hours"] = float(lead)
        frames.append(framed)

    base = pd.concat(frames, ignore_index=True)
    merged = base.merge(
        nc,
        on=["issue_date", "target_time_kst", "lead_hours"],
        how="inner",
        validate="one_to_one",
    )
    t = merged["target_time_kst"]
    merged["target_hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    merged["target_hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    merged["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25)
    merged["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
    merged["issue_day"] = merged["issue_time_kst"].dt.normalize()

    candidates = [
        *ISSUE_FEATURES,
        *[f"forecast_{v}" for v in NC_VARIABLES],
        "target_hour_sin",
        "target_hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    availability = {c: float(merged[c].notna().mean()) for c in candidates if c in merged}
    features = [c for c in candidates if availability.get(c, 0.0) >= 0.90]
    audit = {
        "raw_nc_issue_dates": int(nc["issue_date"].nunique()),
        "joined_issue_dates": int(merged["issue_date"].nunique()),
        "joined_rows_before_complete_case": int(len(merged)),
        "feature_nonmissing_rate": availability,
        "excluded_below_90pct": [c for c in candidates if c not in features],
    }
    return merged, {"features": features, **audit}


def score(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = y - pred
    denom = float(np.abs(y).sum())
    return {
        "MAE_kW": float(np.abs(err).mean()),
        "RMSE_kW": float(np.sqrt(np.mean(err**2))),
        "WAPE_pct": float(np.abs(err).sum() / denom * 100) if denom else None,
        "test_rows": int(len(y)),
    }


def model_for(name: str, harness):
    return LinearRegression() if name == "선형회귀" else harness.make_model(name, 42)


def walk_forward(data: pd.DataFrame, features: list[str], regional, harness) -> list[dict]:
    days = pd.DatetimeIndex(sorted(data["issue_day"].unique()))
    initial = max(30, len(days) // 3)
    results = []
    for name in ["LightGBM", "XGBoost", "선형회귀"]:
        pooled_y, pooled_p, fold_rows = [], [], []
        for fold_no, (train_days, test_days) in enumerate(regional.folds(days, initial), start=1):
            train = data[data["issue_day"].isin(train_days)]
            test = data[data["issue_day"].isin(test_days)]
            if len(train) < 100 or len(test) < 5:
                continue
            model = model_for(name, harness)
            model.fit(train[features], train[segment_target := "목표_발전출력_kW"])
            pred = np.clip(model.predict(test[features]), 0, CAPACITY_KW)
            fold_metric = score(test[segment_target], pred)
            fold_rows.append({
                "fold": fold_no,
                "train_issue_days": int(len(train_days)),
                "test_issue_days": int(len(test_days)),
                **fold_metric,
            })
            pooled_y.extend(test[segment_target].tolist())
            pooled_p.extend(pred.tolist())
        if pooled_y:
            results.append({"model": name, **score(pooled_y, pooled_p), "folds": fold_rows})
    if len(results) != 3:
        raise RuntimeError(f"동일조건 3모델 비교 불가: 유효 모델 {len(results)}/3")
    return results


def main() -> None:
    segment = load_module(SEGMENT_SOURCE, "gwangju_segmented")
    regional = load_module(REGIONAL_SOURCE, "regional_short")
    harness = regional.load_harness()
    frame, frame_audit = build_training_frame(segment)
    features = frame_audit["features"]
    required = [*features, "목표_발전출력_kW"]
    clean = frame.dropna(subset=required).copy()
    if clean["issue_day"].nunique() < 31 or len(clean) < 100:
        raise RuntimeError(
            f"표본 부족: issue_days={clean['issue_day'].nunique()}, rows={len(clean)}"
        )

    comparison = walk_forward(clean, features, regional, harness)
    selected = min(comparison, key=lambda row: row["MAE_kW"])["model"]
    winners = {
        metric: min(comparison, key=lambda row: float("inf") if row[metric] is None else row[metric])["model"]
        for metric in ["MAE_kW", "RMSE_kW", "WAPE_pct"]
    }
    fold_mae_wins = {
        name: sum(
            1
            for fold_no in range(max(len(row["folds"]) for row in comparison))
            if name
            == min(
                (r for r in comparison if fold_no < len(r["folds"])),
                key=lambda r: r["folds"][fold_no]["MAE_kW"],
            )["model"]
        )
        for name in [r["model"] for r in comparison]
    }

    final_model = model_for(selected, harness)
    final_model.fit(clean[features], clean["목표_발전출력_kW"])
    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "model.joblib"
    tmp_path = OUT / "model.joblib.tmp"
    joblib.dump(
        {
            "model": final_model,
            "features": features,
            "region": "광주",
            "horizon": "+48h",
            "model_name": selected,
        },
        tmp_path,
    )
    joblib.load(tmp_path)
    os.replace(tmp_path, model_path)

    defect_column_present = "is_defect_period" in frame.columns
    strict_repeat_win = len(set(winners.values())) == 1 and winners["MAE_kW"] == selected
    manifest = {
        "status": "공식_후보_운영연결보류",
        "region": "광주",
        "tier": "단기",
        "horizon": "+48h",
        "horizon_type": "pooled_window",
        "lead_hours_range": "37~59h (실제 DB 가용 리드만 사용)",
        "definition": "광주 세그먼트 인식 시간프레임과 기존 KIM NC를 issue_date·target_time·lead로 정확히 결합한 프로젝트 자체 실험",
        "source_segmented_csv": str(V4_CSV),
        "source_nc_db": str(NC_DB),
        **frame_audit,
        "training_rows": int(len(clean)),
        "training_issue_dates": int(clean["issue_date"].nunique()),
        "features": features,
        "selected_model": selected,
        "walk_forward": comparison,
        "strict_promotion_check": {
            "metric_winners": winners,
            "fold_mae_wins": fold_mae_wins,
            "same_model_wins_all_pooled_metrics": strict_repeat_win,
            "decision": "승격 보류",
            "reason": "산발적 issue_date 표본과 실전 shadow 미검증; 여러 기간에서 반복 우세를 추가 확인해야 함",
        },
        "leakage_audit": {
            "max_target_relative_lag_used": 0,
            "segment_grouped_shift_and_rolling": True,
            "gap_crossing_blocked_by_segment_id": True,
            "nc_join_keys": ["issue_date", "target_time_kst", "lead_hours"],
            "target_is_observed_at_exact_nc_target_time": True,
            "backfill_first_received_at_not_used_as_historical_availability": True,
            "note": "NC는 백필 수신시각이므로 first_received_at을 과거 발행가능시각으로 역투영하지 않음; 모델 런의 lead 정의와 정확한 목표시각 결합만 사용",
        },
        "defect_audit": {
            "status": "미실시" if not defect_column_present else "기존 라벨만 재사용",
            "is_defect_period_column_present": defect_column_present,
            "promotion_blocker": "광주 결함기간 독립감사 완료 전 공식 승격 금지",
        },
        "methodology_provenance": {
            "domestic_basis_scope": "국내 연구가 지지하는 시간대 재구성·기상변수 활용·과거자료 기반 시계열 검증 원칙만 준용",
            "project_specific_method": "세그먼트 인식 프레임을 archive 없이 산발적 NC issue-date와 결합한 37~59h 풀링은 국내 직접 근거 미확인; 프로젝트 자체 실험",
            "no_forced_literature_equivalence": True,
            "references": [
                {
                    "title": "시계열 모형과 기상변수를 활용한 태양광 발전량 예측 연구",
                    "journal": "응용통계연구 31(1), 2018",
                    "doi": "10.5351/KJAS.2018.31.1.139",
                    "used_for": "시간대별 자료 재구성과 동일 환경 성능비교 원칙",
                },
                {
                    "title": "머신러닝 기반의 예측 시장 참여를 위한 태양광 발전량 예측 알고리즘 및 수익성에 관한 연구",
                    "journal": "한국태양에너지학회 논문집 42(6), 2022",
                    "doi": "10.7836/kses.2022.42.6.173",
                    "used_for": "예보·예측 기상자료를 이용한 시간별 태양광 예측 맥락",
                },
            ],
        },
        "운영_연결": "보류; 기존 공식모델·스케줄러 미변경",
        "api_calls": 0,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "created_at_kst": datetime.now().astimezone().isoformat(),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
