# -*- coding: utf-8 -*-
"""GK2A 위성 4지역 피처 증분 추출기 - v2(자동 스케줄러용, 09-03).

## 배경
사용자 요청(09-03): raw_nc 원본 수집처럼 4지역 피처 추출도 자동으로 계속
쌓이게 해달라. v1(extract_sat_features_4regions_v1_2026-09-03.py)은
실행할 때마다 raw_nc 전체를 처음부터 다시 읽어 매번 CSV를 통째로
덮어쓰는 구조라, raw_nc가 10분마다 계속 늘어나는 지금 상태로 스케줄러에
얹으면 실행시간이 갈수록 길어진다(수천 개 쌓이면 매번 전부 재처리).

## 이 버전이 다른 점
v1의 검증된 로직(LCC 투영으로 픽셀 인덱스 계산, h5py로 한글경로 직접
열기, 파일명 UTC→KST +9시간 보정 - 전부 09-03에 버그 4건 잡고 실측
검증됨)은 그대로 재사용하되:
1. **증분 처리**: 기존 출력 CSV에 이미 있는 시각(datetime_utc)은
   건너뛰고 신규 파일만 처리 → append(멱등, 자주 돌려도 안전).
2. **h5py 슬라이싱 최적화**: v1은 `image_pixel_values[:]`로 3600×3600
   전체 배열을 메모리에 올린 뒤 4개 픽셀만 뽑았다. 이 버전은
   `dataset[y_i, x_i]`로 필요한 픽셀만 디스크에서 바로 읽는다(값은
   동일 - 인덱싱 결과가 같다는 것만 다르게 읽는 방식일 뿐).
3. 픽셀 인덱스(지역→좌표) 계산은 파일이 달라도 격자 정의가 동일하면
   같은 값이므로, 이번 실행에서 처리할 신규 파일 중 첫 번째 파일에서만
   1회 계산해 재사용(v1은 파일마다 다시 계산).

## 출력 파일을 분리한 이유
v1의 산출물(`sat_features_4regions_202608.csv`)은 08월 시범 스냅샷이라는
원래 의미를 유지하려고 그대로 둔다. 이 증분판은 별도 파일
(`sat_features_4regions_continuous.csv`)에 계속 쌓아 헷갈리지 않게
구분한다(현재는 08-01 시범 28개 + 09-03 신규분이 두 파일에 겹쳐 있을 수
있음 - 정상, 각자 독립적으로 유지).

## 실행법 / 스케줄러
python extract_sat_features_4regions_v2_incremental_2026-09-03.py
`UCUBE_GK2A_Extract4Regions_30min` 스케줄러가 30분마다 자동 실행
(raw_nc 수집 주기 10분보다 느슨하게 잡아 매 실행 신규분이 몇 개씩만
쌓이게 함 - 실행시간이 늘어나지 않게 하는 게 목적).
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
LOG_PATH = r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\01_데이터수집\위성\extract_continuous_log.txt"

SITES = {
    "Gwangju": {"lat": 35.1595, "lon": 126.8526},
    "Buan": {"lat": 35.7317, "lon": 126.7333},
    "Gimje": {"lat": 35.8036, "lon": 126.8808},
    "Yeonggwang": {"lat": 35.2773, "lon": 126.5120},
}

FILENAME_RE = re.compile(r"(\d{12})\.nc$")


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_projector(f: h5py.File) -> pyproj.Proj:
    """v1과 동일 - 파일 자체의 LCC 투영 속성으로 pyproj 객체를 만든다
    (하드코딩 안 함, 재구현 아니라 그대로 재사용)."""
    def _get(name, default):
        v = f.attrs.get(name, default)
        return float(np.asarray(v).flat[0])
    lat1 = _get("standard_parallel1", 30.0)
    lat2 = _get("standard_parallel2", 60.0)
    lat0 = _get("origin_latitude", 38.0)
    lon0 = _get("central_meridian", 126.0)
    return pyproj.Proj(
        f"+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +ellps=WGS84"
    )


def get_site_pixel_indices(f: h5py.File) -> dict[str, tuple[int, int]]:
    """v1과 동일 로직(격자 정의 전역속성에서 픽셀좌표 직접 계산 - dim_y/
    dim_x는 전부 0인 더미라 못 씀, v1에서 실측으로 확인·해결된 부분)."""
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
        x_idx = int(np.argmin(np.abs(x_coords - x_m)))
        y_idx = int(np.argmin(np.abs(y_coords - y_m)))
        indices[site_name] = (y_idx, x_idx)
    return indices


def process_single_nc(file_path: str, site_indices: dict[str, tuple[int, int]]) -> dict | None:
    match = FILENAME_RE.search(file_path)
    if not match:
        return None
    dt_utc = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    dt_kst = dt_utc + KST_OFFSET  # 파일명은 UTC(v1에서 확정된 규약) - KST로 변환
    row_data: dict = {"datetime_kst": dt_kst, "datetime_utc": dt_utc}
    try:
        with h5py.File(file_path, "r") as f:
            data_arr = f["image_pixel_values"]  # h5py Dataset(디스크상 핸들, 아직 안 읽음)
            for site_name, (y_i, x_i) in site_indices.items():
                val = data_arr[y_i, x_i]  # 필요한 픽셀만 읽음(v1은 전체[:] 로드 후 인덱싱)
                row_data[f"{site_name}_vi006"] = float(val) if np.isfinite(val) else 0.0
        return row_data
    except Exception as e:  # noqa: BLE001 - 파일별 실패를 로그만 남기고 계속 진행
        log(f"[X] 파일 처리 에러 ({os.path.basename(file_path)}): {e!r}")
        return None


def load_existing() -> tuple[pd.DataFrame | None, set[str]]:
    if not os.path.isfile(OUTPUT_CSV):
        return None, set()
    df = pd.read_csv(OUTPUT_CSV, parse_dates=["datetime_kst", "datetime_utc"])
    processed = set(df["datetime_utc"].dt.strftime("%Y%m%d%H%M"))
    return df, processed


def main() -> None:
    existing_df, processed = load_existing()
    all_files = sorted(glob.glob(os.path.join(RAW_NC_DIR, "*.nc")))
    new_files = []
    for fp in all_files:
        m = FILENAME_RE.search(fp)
        if m and m.group(1) not in processed:
            new_files.append(fp)

    if not new_files:
        log("신규 파일 없음 - 건너뜀")
        return

    with h5py.File(new_files[0], "r") as f0:
        site_indices = get_site_pixel_indices(f0)

    new_rows = []
    for fp in new_files:
        row = process_single_nc(fp, site_indices)
        if row:
            new_rows.append(row)

    if not new_rows:
        log("신규 파일은 있었으나 전부 처리 실패(위 [X] 로그 참고)")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    combined.drop_duplicates(subset=["datetime_utc"], keep="last", inplace=True)
    combined.sort_values("datetime_kst", inplace=True)

    tmp_path = OUTPUT_CSV + ".tmp"
    combined.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, OUTPUT_CSV)

    log(f"신규 {len(new_rows)}개 추가 완료(누적 {len(combined)}행) -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
