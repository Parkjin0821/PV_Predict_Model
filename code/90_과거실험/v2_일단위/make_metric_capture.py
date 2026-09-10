from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

KOREAN_FONT = Path("C:/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    font_manager.fontManager.addfont(KOREAN_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=KOREAN_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    rows = []
    for site, label in [("gwangju", "광주"), ("gimje", "김제")]:
        path = ROOT / "outputs" / "models" / site / "model_metadata_and_metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for model, m in data["metrics"].items():
            rows.append((label, model, m["ME_kWh"], m["MAE_kWh"], m["MPE_pct"],
                         m["MAPE_pct"], m["MSE_kWh2"], m["RMSE_kWh"], m["R2"]))
    rows.sort(key=lambda r: (r[0], -r[-1]))
    header = ["지역", "모델", "ME\n(kWh)", "MAE\n(kWh)", "MPE\n(%)", "MAPE\n(%)",
              "MSE\n(kWh²)", "RMSE\n(kWh)", "R²"]
    body = [[r[0], r[1].replace("Multiple_linear_regression", "다중회귀")
             .replace("Persistence_previous_day", "전일값")
             .replace("Previous_week_same_day", "전주동일요일")
             .replace("SARIMAX_ARIMA", "ARIMA"),
             f"{r[2]:,.1f}", f"{r[3]:,.1f}", f"{r[4]:,.1f}", f"{r[5]:,.1f}",
             f"{r[6]:,.0f}", f"{r[7]:,.1f}", f"{r[8]:.3f}"] for r in rows]

    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.axis("off")
    ax.set_title("일 단위 익일 태양광 발전량 예측 성능 (기상 예보 미결합 기준선)",
                 fontsize=20, fontweight="bold", pad=18)
    table = ax.table(cellText=body, colLabels=header, cellLoc="center", loc="center",
                     colWidths=[.06, .19, .085, .085, .075, .075, .11, .095, .07])
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        if row == 0:
            cell.set_facecolor("#F58220"); cell.set_text_props(color="white", weight="bold")
        elif body[row-1][0] == "광주":
            cell.set_facecolor("#FFF8F2" if row % 2 else "#FFFFFF")
        else:
            cell.set_facecolor("#F4F6F8" if row % 2 else "#FFFFFF")
    fig.text(.5, .035,
             "오차 = 실제값 - 예측값. 양(+)의 ME·MPE는 과소예측. 목표 R² 0.7 미달 — 익일 일사량·운량 등 기상 예보 결합 필요.",
             ha="center", fontsize=11, color="#444444")
    out = ROOT / "previews" / "daily_model_metric_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
