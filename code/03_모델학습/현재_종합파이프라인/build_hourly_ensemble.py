"""1시간 후보모델의 시간순 교차검증 예측으로 수평별 자동 가중치를 계산한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_common import optimize_nonnegative_weights, regression_metrics, write_json
from pv_pipeline import ROOT, load_config


BASE = ROOT / "outputs" / "1시간_기준모델"
DEEP = ROOT / "outputs" / "1시간_딥러닝"
OUT = ROOT / "outputs" / "1시간_앙상블"
KEYS = ["예측발행시각", "예측대상시각", "예측수평_시간"]
ACTUAL = "실제_1시간평균출력_kW"


def merge_predictions(base_path, deep_path) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    deep = pd.read_parquet(deep_path)
    merged = base.merge(deep, on=KEYS, how="inner", suffixes=("_기준", "_딥러닝"), validate="one_to_one")
    actual_columns = [name for name in merged.columns if name.startswith(ACTUAL)]
    merged[ACTUAL] = merged[actual_columns[0]]
    if len(actual_columns) > 1:
        difference = np.abs(merged[actual_columns[0]] - merged[actual_columns[1]])
        if float(difference.max()) > 1e-4:
            raise ValueError("기준모델과 딥러닝 실제값이 일치하지 않습니다.")
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    capacity = float(config["site"]["capacity_kw"])
    oof = merge_predictions(BASE / "교차검증_비표본예측.parquet", DEEP / "교차검증_비표본예측.parquet")
    test = merge_predictions(BASE / "최종시험_예측.parquet", DEEP / "최종시험_예측.parquet")
    candidates = ["동시간1주전예측_kW", "LightGBM예측_kW", "장단기기억모델예측_kW", "게이트순환모델예측_kW"]
    weights_payload, score_rows, output_parts, oof_output_parts = {}, [], [], []
    for horizon in range(1, 49):
        train = oof[oof["예측수평_시간"].eq(horizon)].dropna(subset=candidates + [ACTUAL])
        result = optimize_nonnegative_weights(train[ACTUAL].to_numpy(), train[candidates].to_numpy())
        weights = np.asarray(result["가중치"], dtype=float)
        weights_payload[str(horizon)] = {
            **result,
            "모델별가중치": {name.replace("예측_kW", ""): float(value) for name, value in zip(candidates, weights)},
        }
        train = train.copy()
        train["자동가중앙상블예측_kW"] = np.clip(train[candidates].to_numpy() @ weights, 0, capacity)
        oof_output_parts.append(train)
        part = test[test["예측수평_시간"].eq(horizon)].dropna(subset=candidates + [ACTUAL]).copy()
        part["자동가중앙상블예측_kW"] = np.clip(part[candidates].to_numpy() @ weights, 0, capacity)
        output_parts.append(part)
        if horizon >= 24:
            for name in candidates + ["자동가중앙상블예측_kW"]:
                score_rows.append({
                    "예측수평_시간": horizon,
                    "모델": name.replace("예측_kW", ""),
                    **regression_metrics(part[ACTUAL], part[name]),
                })
    output = pd.concat(output_parts, ignore_index=True)
    oof_output = pd.concat(oof_output_parts, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    output.to_parquet(OUT / "최종시험_공통표본_예측.parquet")
    oof_output.to_parquet(OUT / "교차검증_자동가중예측.parquet")
    scores.to_csv(OUT / "공식24_48시간_공통표본_성능표.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "예측수평별_자동가중치.json", weights_payload)
    print(scores[scores["모델"].eq("자동가중앙상블")].to_string(index=False))


if __name__ == "__main__":
    main()
