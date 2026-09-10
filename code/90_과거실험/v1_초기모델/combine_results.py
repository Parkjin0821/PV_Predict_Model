"""Combine preliminary model metrics into one auditable scorecard."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
out = root / "outputs"
base_payload = json.loads((out / "preliminary_model_metrics.json").read_text(encoding="utf-8"))
metrics = dict(base_payload["metrics"])
metrics["SARIMAX_ARIMA"] = json.loads((out / "preliminary_arima_metrics.json").read_text(encoding="utf-8"))["metrics"]
metrics.update(json.loads((out / "preliminary_neural_metrics.json").read_text(encoding="utf-8"))["metrics"])
families = {
    "Persistence_current": "기준모델", "Previous_day_same_hour": "기준모델",
    "Multiple_linear_regression": "다중회귀", "SARIMAX_ARIMA": "ARIMA 계열",
    "LightGBM": "부스팅", "XGBoost": "부스팅", "RNN": "딥러닝", "LSTM": "딥러닝",
}
base_rmse = metrics["Persistence_current"]["RMSE_kW"]
rows = []
for model, values in metrics.items():
    row = {"모델": model, "계열": families[model], **values}
    row["지속성대비_RMSE개선율_pct"] = (1 - values["RMSE_kW"] / base_rmse) * 100
    row["R2_0.7_진단기준"] = "충족" if values["R2"] >= 0.7 else "미충족"
    rows.append(row)
score = pd.DataFrame(rows).sort_values("RMSE_kW")
score.to_csv(out / "preliminary_all_model_scorecard.csv", index=False, encoding="utf-8-sig")
(out / "preliminary_all_model_scorecard.json").write_text(
    json.dumps({"comparison_note": "동일한 1,185개 시험 표본; 기상 미결합 진단모델",
                "target": base_payload["target"], "forecast_horizon": base_payload["forecast_horizon"],
                "models": score.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
print(score[["모델", "계열", "MAE_kW", "RMSE_kW", "R2", "Pearson_r", "지속성대비_RMSE개선율_pct"]].to_string(index=False))
