"""초단기 후보모델의 교차검증 예측으로 수평별 자동 가중치를 계산한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_common import optimize_nonnegative_weights, regression_metrics, write_json
from pv_pipeline import ROOT, load_config


BASE = ROOT / "outputs" / "초단기_기준모델"
DEEP = ROOT / "outputs" / "초단기_딥러닝"
OUT = ROOT / "outputs" / "초단기_앙상블"


def merge_predictions(base_path, deep_path) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    deep = pd.read_parquet(deep_path)
    keys = ["예측발행시각", "예측대상시각", "예측수평_분"]
    merged = base.merge(deep, on=keys, how="inner", suffixes=("_기준", "_딥러닝"), validate="one_to_one")
    actual_columns = [name for name in merged.columns if name.startswith("실제_15분평균출력_kW")]
    merged["실제_15분평균출력_kW"] = merged[actual_columns[0]]
    if len(actual_columns) > 1:
        difference = np.abs(merged[actual_columns[0]] - merged[actual_columns[1]])
        # 딥러닝 목표값은 float32로 저장되므로 1e-4kW 이내 반올림 차이는 동일값이다.
        if float(difference.max()) > 1e-4:
            raise ValueError("기준모델과 딥러닝 실제값이 일치하지 않습니다.")
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    oof = merge_predictions(BASE / "교차검증_비표본예측.parquet", DEEP / "교차검증_비표본예측.parquet")
    test = merge_predictions(BASE / "최종시험_예측.parquet", DEEP / "최종시험_예측.parquet")
    candidates = ["지속성예측_kW", "LightGBM예측_kW", "장단기기억모델예측_kW", "게이트순환모델예측_kW"]
    weights_payload, score_rows, output_parts = {}, [], []
    for horizon in config["ultra_short"]["horizon_minutes"]:
        train = oof[oof["예측수평_분"].eq(horizon)].dropna(subset=candidates + ["실제_15분평균출력_kW"])
        result = optimize_nonnegative_weights(
            train["실제_15분평균출력_kW"].to_numpy(),
            train[candidates].to_numpy(),
        )
        weights = np.asarray(result["가중치"], dtype=float)
        weights_payload[str(horizon)] = {
            **result,
            "모델별가중치": {name.replace("예측_kW", ""): float(value) for name, value in zip(candidates, weights)},
        }
        part = test[test["예측수평_분"].eq(horizon)].dropna(
            subset=candidates + ["실제_15분평균출력_kW"]
        ).copy()
        part["자동가중앙상블예측_kW"] = np.clip(part[candidates].to_numpy() @ weights, 0, config["site"]["capacity_kw"])
        output_parts.append(part)
        for name in candidates + ["자동가중앙상블예측_kW"]:
            score_rows.append(
                {
                    "예측수평_분": int(horizon),
                    "모델": name.replace("예측_kW", ""),
                    **regression_metrics(part["실제_15분평균출력_kW"], part[name]),
                }
            )
    output = pd.concat(output_parts, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    output.to_parquet(OUT / "최종시험_공통표본_예측.parquet")
    scores.to_csv(OUT / "최종시험_공통표본_성능표.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "예측수평별_자동가중치.json", weights_payload)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
