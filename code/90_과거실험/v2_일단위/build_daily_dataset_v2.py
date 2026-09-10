"""Build daily PV datasets for Gwangju and Gimje from read-only XLSX logs."""

from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))
import numpy as np
import pandas as pd

TOOLS = Path(r"C:\Users\u-cube\Documents\ChatGPT\유큐브\광주_pv_model_workspace")
sys.path.insert(0, str(TOOLS))
from scan_xlsx_logs import NS_MAIN, cell_value, col_index, first_sheet_path, parse_timestamp, shared_strings


SITES = {
    "gwangju": {"name": "광주 광주시청", "inverters": 5, "capacity_kw_report": 240.0,
                 "location_report": "광주 서구 치평동", "weather_station_candidate": "156 광주"},
    "gimje": {"name": "김제 (유)금성썬에너지", "inverters": 10, "capacity_kw_report": 999.0,
              "location_report": "전북 김제시 명덕동", "weather_station_candidate": None},
}


def read_daily(path: Path, inverter: int) -> list[dict]:
    wanted = ["생성일", "출력전력", "온도", "일일", "누적", "주기별발전량", "상태", "통신에러"]
    results, agg = [], None
    prev_ts = prev_power = prev_cumulative = None

    def new_agg(day):
        return {"date": day, "inverter": inverter, "samples": 0, "output_count": 0,
                "output_sum": 0.0, "output_max_kw": None, "integrated_energy_kwh": 0.0,
                "integrated_intervals": 0, "daily_energy_max_kwh": None,
                "daily_energy_last_kwh": None, "cumulative_min_kwh": None,
                "cumulative_max_kwh": None, "cumulative_negative_steps": 0,
                "period_generation_max": None, "status_nonempty": 0,
                "comm_error_nonempty": 0, "valid_temperature_count": 0,
                "valid_temperature_sum": 0.0, "first_timestamp": None, "last_timestamp": None}

    def finish(a):
        if a is None:
            return
        a["output_power_mean_kw"] = a["output_sum"] / a["output_count"] if a["output_count"] else None
        a["temperature_mean_c"] = (a["valid_temperature_sum"] / a["valid_temperature_count"]
                                   if a["valid_temperature_count"] else None)
        del a["output_sum"], a["valid_temperature_sum"]
        results.append(a)

    with zipfile.ZipFile(path) as zf:
        strings = shared_strings(zf)
        sheet_path = first_sheet_path(zf)
        headers = indices = None
        with zf.open(sheet_path) as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != f"{{{NS_MAIN}}}row":
                    continue
                values = {col_index(c.attrib.get("r", "A1")): cell_value(c, strings)
                          for c in elem.findall(f"{{{NS_MAIN}}}c")}
                if headers is None:
                    width = max(values, default=-1) + 1
                    headers = [str(values.get(i) or "") for i in range(width)]
                    indices = {name: headers.index(name) for name in wanted}
                    elem.clear(); continue
                ts = parse_timestamp(values.get(indices["생성일"]))
                if ts is None:
                    elem.clear(); continue
                if agg is None or agg["date"] != ts.date():
                    finish(agg)
                    agg = new_agg(ts.date())
                    prev_ts = prev_power = prev_cumulative = None
                agg["samples"] += 1
                agg["first_timestamp"] = agg["first_timestamp"] or ts.isoformat(sep=" ")
                agg["last_timestamp"] = ts.isoformat(sep=" ")

                def num(name):
                    try: return float(values.get(indices[name]))
                    except (TypeError, ValueError): return None

                power = num("출력전력")
                if power is not None and math.isfinite(power) and power >= 0:
                    agg["output_count"] += 1; agg["output_sum"] += power
                    agg["output_max_kw"] = power if agg["output_max_kw"] is None else max(agg["output_max_kw"], power)
                    if prev_ts is not None and prev_power is not None:
                        hours = (ts - prev_ts).total_seconds() / 3600
                        if 0 < hours <= 0.25:  # do not bridge gaps over 15 minutes
                            agg["integrated_energy_kwh"] += (prev_power + power) / 2 * hours
                            agg["integrated_intervals"] += 1
                    prev_ts, prev_power = ts, power

                daily = num("일일")
                if daily is not None and math.isfinite(daily) and daily >= 0:
                    agg["daily_energy_last_kwh"] = daily
                    agg["daily_energy_max_kwh"] = daily if agg["daily_energy_max_kwh"] is None else max(agg["daily_energy_max_kwh"], daily)
                cumulative = num("누적")
                if cumulative is not None and math.isfinite(cumulative):
                    agg["cumulative_min_kwh"] = cumulative if agg["cumulative_min_kwh"] is None else min(agg["cumulative_min_kwh"], cumulative)
                    agg["cumulative_max_kwh"] = cumulative if agg["cumulative_max_kwh"] is None else max(agg["cumulative_max_kwh"], cumulative)
                    if prev_cumulative is not None and cumulative < prev_cumulative:
                        agg["cumulative_negative_steps"] += 1
                    prev_cumulative = cumulative
                period = num("주기별발전량")
                if period is not None and math.isfinite(period):
                    agg["period_generation_max"] = period if agg["period_generation_max"] is None else max(agg["period_generation_max"], period)
                temp = num("온도")
                if temp is not None and -50 <= temp <= 80:
                    agg["valid_temperature_count"] += 1; agg["valid_temperature_sum"] += temp
                if values.get(indices["상태"]) not in (None, ""): agg["status_nonempty"] += 1
                if values.get(indices["통신에러"]) not in (None, ""): agg["comm_error_nonempty"] += 1
                elem.clear()
    finish(agg)
    return results


def build_site(site: str, cfg: dict) -> dict:
    site_rows = []
    for inverter in range(1, cfg["inverters"] + 1):
        site_rows.extend(read_daily(ROOT / "input" / site / f"inverter_{inverter}.xlsx", inverter))
    device = pd.DataFrame(site_rows)
    device["date"] = pd.to_datetime(device["date"])
    output_dir = ROOT / "outputs"
    device.to_csv(output_dir / f"{site}_daily_by_inverter.csv", index=False, encoding="utf-8-sig")

    plant = (device.groupby("date", as_index=False)
        .agg(inverters_reporting=("inverter", "nunique"), samples=("samples", "sum"),
             min_samples_per_inverter=("samples", "min"), max_samples_per_inverter=("samples", "max"),
             daily_energy_kwh=("daily_energy_max_kwh", "sum"),
             integrated_energy_kwh=("integrated_energy_kwh", "sum"),
             max_plant_power_kw=("output_max_kw", "sum"),
             cumulative_negative_steps=("cumulative_negative_steps", "sum"),
             status_nonempty=("status_nonempty", "sum"), comm_error_nonempty=("comm_error_nonempty", "sum"),
             temperature_mean_c=("temperature_mean_c", "mean")))
    plant["site_id"] = site
    plant["site_name"] = cfg["name"]
    plant["expected_inverters"] = cfg["inverters"]
    plant["complete_inverters"] = plant["inverters_reporting"].eq(cfg["inverters"])
    plant["plant_capacity_kw_report"] = cfg["capacity_kw_report"]
    plant["capacity_confirmed"] = False
    plant["location_report"] = cfg["location_report"]
    plant["weather_station_candidate"] = cfg["weather_station_candidate"]
    plant["daily_capacity_factor_pct"] = plant["daily_energy_kwh"] / (cfg["capacity_kw_report"] * 24) * 100
    plant["meter_to_integrated_ratio"] = np.divide(plant["daily_energy_kwh"], plant["integrated_energy_kwh"],
        out=np.full(len(plant), np.nan), where=plant["integrated_energy_kwh"].to_numpy() > 0)
    typical_min_samples = float(plant.loc[plant["complete_inverters"], "min_samples_per_inverter"].median())
    plant["sample_coverage_ratio"] = plant["min_samples_per_inverter"] / typical_min_samples
    plant["quality_ok"] = (plant["complete_inverters"] & plant["daily_energy_kwh"].gt(0)
                            & plant["sample_coverage_ratio"].ge(0.8)
                            & plant["daily_capacity_factor_pct"].between(0, 100))

    # Optional site/geographic/equipment fields. Blank means not provided; never inferred.
    optional = ["plant_latitude", "plant_longitude", "elevation_m", "terrain_slope_deg",
                "terrain_aspect_deg", "horizon_shading_index", "module_maker", "module_model",
                "module_quantity", "module_tilt_deg", "module_azimuth_deg", "module_efficiency_pct",
                "inverter_maker", "inverter_model", "inverter_capacity_total_kw",
                "inverter_efficiency_pct", "installation_type", "structure_type", "tracking_type",
                "commissioning_date"]
    for col in optional: plant[col] = pd.NA
    plant.to_csv(output_dir / f"{site}_daily_plant.csv", index=False, encoding="utf-8-sig")
    return {
        "site": site, "days": len(plant), "start": plant["date"].min().date().isoformat(),
        "end": plant["date"].max().date().isoformat(), "quality_ok_days": int(plant["quality_ok"].sum()),
        "complete_days": int(plant["complete_inverters"].sum()),
        "daily_energy_min": float(plant["daily_energy_kwh"].min()),
        "daily_energy_max": float(plant["daily_energy_kwh"].max()),
        "median_meter_to_integrated_ratio": float(plant["meter_to_integrated_ratio"].median()),
        "typical_min_samples_per_inverter": typical_min_samples,
    }


def main() -> int:
    (ROOT / "outputs").mkdir(exist_ok=True)
    audit = {site: build_site(site, cfg) for site, cfg in SITES.items()}
    (ROOT / "outputs" / "daily_dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
