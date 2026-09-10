from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\u-cube\Documents\ChatGPT\유큐브")
SOURCE_CSV = ROOT / "gwangju_pv_multihorizon_v3_workspace" / "outputs" / "gwangju_1hour_model_dataset.csv"
MODEL_ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전")
OUTPUT = ROOT / ".tmp_factor_data.csv"


def load_ground_temperature() -> pd.DataFrame:
    stats = json.loads((MODEL_ROOT / "normalization_stats.json").read_text(encoding="utf-8"))
    mean = float(stats["standardization"]["지면온도"]["mean"])
    std = float(stats["standardization"]["지면온도"]["std"])
    frames = [
        pd.read_parquet(MODEL_ROOT / "train_standardized.parquet", columns=["생성일", "지면온도"]),
        pd.read_parquet(MODEL_ROOT / "test_standardized.parquet", columns=["생성일", "지면온도"]),
    ]
    weather = pd.concat(frames, ignore_index=True)
    weather["기준시각"] = pd.to_datetime(weather["생성일"])
    weather["지면온도(℃)"] = weather["지면온도"].astype(float) * std + mean
    return weather.groupby("기준시각", as_index=False)["지면온도(℃)"].median()


def main() -> None:
    source = pd.read_csv(SOURCE_CSV, parse_dates=["time"])
    source = source.rename(columns={"time": "기준시각"})
    ground = load_ground_temperature()
    source = source.merge(ground, on="기준시각", how="left", validate="one_to_one")

    out = pd.DataFrame(
        {
            "기준시각": source["기준시각"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "연": source["기준시각"].dt.year,
            "월": source["기준시각"].dt.month,
            "일": source["기준시각"].dt.day,
            "시간": source["기준시각"].dt.hour,
            "분": source["기준시각"].dt.minute,
            "지역": "광주광역시 서구 치평동",
            "위도": source["site_latitude"],
            "경도": source["site_longitude"],
            "인버터 평균온도(℃)": source["mean_inverter_temperature_c"],
            "입력전력(kW)": source["plant_input_power_kw"],
            "발전출력(kW)": source["plant_output_kw"],
            "인버터 효율 추정": np.nan,
            "효율 판정": "",
            "전체 설비용량(kW)": source["reported_capacity_kw"],
            "인버터 수(대)": source["reported_inverter_count"],
            "해발(m)": source["site_elevation_dem_m"],
            "지형 경사(도)": source["terrain_slope_deg_dem_approx"],
            "지형 방향(도)": source["terrain_aspect_deg_dem_approx"],
            "지면온도(℃)": source["지면온도(℃)"],
            "초상온도(℃)": np.nan,
            "초상온도 상태": "기상청 공식 제공 확인·원자료 미수집",
            "자료 비고": "지형 방향은 패널 방위각이 아님",
        }
    )
    out.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"saved={OUTPUT}")
    print(f"rows={len(out)}")
    print(f"ground_temperature_missing={int(out['지면온도(℃)'].isna().sum())}")
    print(f"inverter_temperature_missing={int(out['인버터 평균온도(℃)'].isna().sum())}")


if __name__ == "__main__":
    main()
