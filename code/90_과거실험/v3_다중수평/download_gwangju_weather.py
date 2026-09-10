"""Download public weather forecast archives for the Gwangju City Hall PV model."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external"
OUT.mkdir(parents=True, exist_ok=True)

LAT, LON = 35.1601, 126.8515
START, END = "2024-08-25", "2026-08-04"
BASE_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
]


def get_json(host: str, params: dict) -> dict:
    url = host + "?" + urllib.parse.urlencode(params, safe=",")
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = json.load(response)
    payload["request_url"] = url
    return payload


def main() -> None:
    historical = get_json(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        {
            "latitude": LAT, "longitude": LON, "start_date": START, "end_date": END,
            "hourly": ",".join(BASE_VARS), "timezone": "Asia/Seoul",
        },
    )
    previous_vars = [f"{v}_previous_day{d}" for d in (1, 2) for v in BASE_VARS]
    previous = get_json(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {
            "latitude": LAT, "longitude": LON, "start_date": START, "end_date": END,
            "hourly": ",".join(previous_vars), "timezone": "Asia/Seoul",
        },
    )
    elevation = get_json(
        "https://api.open-meteo.com/v1/elevation",
        {"latitude": LAT, "longitude": LON},
    )
    for name, payload in [
        ("gwangju_historical_forecast_hourly.json", historical),
        ("gwangju_previous_runs_day1_day2_hourly.json", previous),
        ("gwangju_elevation.json", elevation),
    ]:
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "site": "광주광역시청", "address": "광주광역시 서구 내방로 111 (치평동 1200)",
        "latitude_geocoded": LAT, "longitude_geocoded": LON,
        "elevation_dem_m": elevation["elevation"][0],
        "period": [START, END], "timezone": "Asia/Seoul",
        "historical_forecast_rows": len(historical["hourly"]["time"]),
        "previous_run_rows": len(previous["hourly"]["time"]),
        "important_note": (
            "Target-time forecast features use previous_day1 for horizons <=24h and "
            "previous_day2 for horizons >24h, so they were available no later than issue time."
        ),
        "sources": [
            "https://open-meteo.com/en/docs/historical-forecast-api",
            "https://open-meteo.com/en/docs/previous-runs-api",
            "https://api.open-meteo.com/v1/elevation",
            "https://120.gwangju.go.kr/pageLink.do?menuNo=4010",
            "https://newscms.gwangju.go.kr/gallery.es?act=view&bid=0009&cg_code=C02&list_no=5163&mid=a40200000000",
        ],
    }
    (OUT / "gwangju_external_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
