# -*- coding: utf-8 -*-
"""IR105 픽셀값을 sat_features_4regions_continuous.csv에 합치는 보정 스크립트.

## 발견한 버그(09-03 백필 완료 직후)
`extract_sat_features_4regions_v2_incremental_2026-09-03.py`는 raw_nc의
`*.nc` 파일을 채널 구분 없이 전부 글롭해서 처리하는데, 출력 컬럼명이
`{지역}_vi006`로 고정돼 있다. VI006·IR105가 같은 시각(datetime_utc)에
둘 다 있으면 `drop_duplicates(subset=["datetime_utc"], keep="last")`가
파일명 알파벳순(ir105 < vi006)으로 IR105를 먼저 처리하고 VI006이
나중에 덮어써서 **살아남는 건 항상 VI006뿐**이었다 - IR105는 추출은
됐지만 매번 조용히 버려졌다(원본 raw_nc 파일 자체는 안전하게 남아있어
데이터 손실은 없음, 이 스크립트로 복구 가능).

## 이 스크립트가 하는 일
raw_nc의 IR105 파일만 골라 4지역 픽셀값을 뽑고, 기존
`sat_features_4regions_continuous.csv`에 `datetime_utc` 기준으로 병합해
`{지역}_ir105` 컬럼 4개를 새로 채운다. 기존 VI006 컬럼·행은 그대로 두고
컬럼만 추가(merge how="outer"로 VI006에 없던 시각도 보존).

## 실행법
python merge_ir105_into_continuous_v1_2026-09-03.py
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta

KST_OFFSET = timedelta(hours=9)

import h5py
import numpy as np
import pandas as pd
import pyproj

RAW_NC_DIR = r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\01_데이터수집\위성\raw_nc"
OUTPUT_CSV = r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\01_데이터수집\위성\sat_features_4regions_continuous.csv"

SITES = {
    "Gwangju": {"lat": 35.1595, "lon": 126.8526},
    "Buan": {"lat": 35.7317, "lon": 126.7333},
    "Gimje": {"lat": 35.8036, "lon": 126.8808},
    "Yeonggwang": {"lat": 35.2773, "lon": 126.5120},
}

FILENAME_RE = re.compile(r"ir105_ko_(\d{12})\.nc$")


def build_projector(f: h5py.File) -> pyproj.Proj:
    def _get(name, default):
        v = f.attrs.get(name, default)
        return float(np.asarray(v).flat[0])
    return pyproj.Proj(
        f"+proj=lcc +lat_1={_get('standard_parallel1', 30.0)} +lat_2={_get('standard_parallel2', 60.0)} "
        f"+lat_0={_get('origin_latitude', 38.0)} +lon_0={_get('central_meridian', 126.0)} "
        "+x_0=0 +y_0=0 +ellps=WGS84"
    )


def get_site_pixel_indices(f: h5py.File) -> dict[str, tuple[int, int]]:
    ul_x = float(np.asarray(f.attrs["upper_left_easting"]).flat[0])
    ul_y = float(np.asarray(f.attrs["upper_left_northing"]).flat[0])
    pixel_size = float(np.asarray(f.attrs["pixel_size"]).flat[0])
    width = int(np.asarray(f.attrs["image_width"]).flat[0])
    height = int(np.asarray(f.attrs["image_height"]).flat[0])
    x_coords = ul_x + np.arange(width) * pixel_size
    y_coords = ul_y - np.arange(height) * pixel_size
    proj = build_projector(f)
    indices = {}
    for site_name, coord in SITES.items():
        x_m, y_m = proj(coord["lon"], coord["lat"])
        indices[site_name] = (
            int(np.argmin(np.abs(y_coords - y_m))),
            int(np.argmin(np.abs(x_coords - x_m))),
        )
    return indices


def main() -> None:
    ir105_files = sorted(glob.glob(os.path.join(RAW_NC_DIR, "*ir105*.nc")))
    print(f"[*] IR105 파일 {len(ir105_files)}개 발견")

    site_indices = None
    rows = []
    for idx, fp in enumerate(ir105_files, 1):
        m = FILENAME_RE.search(fp)
        if not m:
            continue
        dt_utc = datetime.strptime(m.group(1), "%Y%m%d%H%M")
        dt_kst = dt_utc + KST_OFFSET
        row = {"datetime_kst": dt_kst, "datetime_utc": dt_utc}
        try:
            with h5py.File(fp, "r") as f:
                if site_indices is None:
                    site_indices = get_site_pixel_indices(f)
                data_arr = f["image_pixel_values"]
                for site_name, (y_i, x_i) in site_indices.items():
                    val = data_arr[y_i, x_i]
                    row[f"{site_name}_ir105"] = float(val) if np.isfinite(val) else 0.0
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[X] {os.path.basename(fp)}: {e!r}")
        if idx % 1000 == 0 or idx == len(ir105_files):
            print(f"[{idx}/{len(ir105_files)}] 처리 중...")

    ir105_df = pd.DataFrame(rows)
    print(f"[*] IR105 추출 완료: {len(ir105_df)}행")

    existing = pd.read_csv(OUTPUT_CSV, parse_dates=["datetime_kst", "datetime_utc"], encoding="utf-8-sig")
    print(f"[*] 기존 CSV: {len(existing)}행, 컬럼 {list(existing.columns)}")

    merged = pd.merge(existing, ir105_df, on=["datetime_kst", "datetime_utc"], how="outer")
    merged.sort_values("datetime_kst", inplace=True)

    tmp_path = OUTPUT_CSV + ".tmp"
    merged.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, OUTPUT_CSV)
    print(f"[성공] 병합 완료: {len(merged)}행, 컬럼 {list(merged.columns)} -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
