# -*- coding: utf-8 -*-
"""김제 월간 총발전량 후보: 기존 월간 기준선 비교 함수를 그대로 재사용한다."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd

PIPE = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\03_모델학습\현재_종합파이프라인")
COMPARE = PIPE / "compare_monthly_baseline_candidates_v1_2026-09-10.py"
DAILY = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_라이브연계_v1_2026-09-14\김제_발전소_일간_라이브연계_D1검증용.csv")
OUT = PIPE / "김제_준비_2026-09-01" / "outputs" / "김제_월간_후보_v1_2026-09-14"


def load_compare_module():
    spec = importlib.util.spec_from_file_location("monthly_compare", COMPARE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 비교 모듈 로드 실패: {COMPARE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_monthly() -> pd.DataFrame:
    daily = pd.read_csv(DAILY, parse_dates=["date_kst"]).set_index("date_kst")
    valid = daily["quality_status"].eq("valid_daylight_ge90pct")
    daily.loc[~valid, "daily_energy_kwh"] = float("nan")
    monthly = daily["daily_energy_kwh"].resample("MS").agg(total="sum", n="count")
    monthly["days"] = monthly.index.days_in_month
    monthly["y"] = monthly["total"].where(monthly["n"].eq(monthly["days"]))
    monthly["extra"] = daily["daylight_valid_ratio"].resample("MS").mean()
    return monthly


def main() -> None:
    compare = load_compare_module()
    OUT.mkdir(parents=True, exist_ok=True)
    compare.OUT = OUT
    monthly = load_monthly()
    result = compare.run_region("김제", monthly, "daylight_ratio_lag1")
    if not result:
        raise RuntimeError("김제 월간 후보 생성 중단: 비교 가능한 완결월이 부족합니다.")

    frame, features = compare.build_features(monthly, "daylight_ratio_lag1")
    train = frame.dropna(subset=features + ["y"])
    model = compare.GradientBoostingRegressor(
        loss="huber", n_estimators=50, max_depth=2, learning_rate=.04,
        min_samples_leaf=3, random_state=20260819,
    ).fit(train[features], train["y"])
    joblib.dump(model, OUT / "model.joblib")

    manifest = {
        "region": "김제",
        "status": "공식_후보_운영연결보류",
        "tier": "중장기_월간",
        "target": "월간 총발전량_kWh",
        "training_source": str(DAILY),
        "complete_months": int(monthly["y"].notna().sum()),
        "features": features,
        "model": "GradientBoostingRegressor(loss=huber)",
        "validation": result,
        "method": "기존 월간 기준선 비교 함수 재사용; 시간순 walk-forward; 폴드별 과거월만 사용",
        "baselines": ["전월지속성", "전년동월", "계절평균", "선형회귀", "Ridge", "LightGBM"],
        "season_standard": "기상학적 4계절(봄3~5·여름6~8·가을9~11·겨울12~2), Q1~Q4와 분리",
        "adoption_rule": "동일표본 MAE/RMSE/WAPE와 계절별 반복 우세 전까지 운영 승격 금지",
        "academic_note": "국내 문헌은 기상·계절·모델 비교를 지지하나 월간 GB-Huber 직접 근거는 확인되지 않아 자체 비교실험으로만 표기",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "complete_months": manifest["complete_months"], "verdict": result["판정"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
