# -*- coding: utf-8 -*-
"""GK2A 위성 4지역(광주/부안/김제/영광) 픽셀 특성 추출 - 09-03 버그수정판.

## 배경
사용자가 준 원본 스크립트(xarray + engine="netcdf4")가 raw_nc 28개
파일 전부에서 `[Errno 22] Invalid argument`로 실패 - 실측으로 원인
2가지를 확인하고 고쳤다.

## 버그 1: netCDF4 라이브러리의 Windows 한글경로 버그(실측 확인)
같은 파일을 ASCII 전용 경로로 복사해서 열면 정상 동작하고, 원래
한글경로(`...\\태양광 발전\\...\\위성\\raw_nc\\...`)로 열면 100%
실패하는 걸 직접 재현해 확인했다 - netCDF4의 C라이브러리가 Windows에서
비ASCII 경로를 못 여는 알려진 문제. **해결**: `h5py`로 직접 열도록
교체(이 프로젝트의 기존 위성 수집기
`collect_gk2a_4sites_bulk_v1_2026-09-01.py`도 h5py를 쓰고 있었음 -
같은 이유로 보임, 재구현 아니라 기존 검증된 접근 재사용).

## 버그 2: `latitude`/`longitude` 변수 자체가 없음(원본 스크립트가
전제한 게 틀림)
GK2A LE1B 파일은 위경도를 직접 안 담고, **Lambert Conformal Conic
투영좌표계 + 격자 정의 속성**으로 위치를 표현한다(standard_parallel1=
30·standard_parallel2=60·origin_latitude=38·central_meridian=126).
원본 스크립트의 `if 'latitude' in ds` 분기는 이 파일들에서 **항상
False**라 전부 "격자 좌표 산출 불가" 폴백(전체 이미지 평균)으로
빠졌을 것.

## 버그 3(수정 1차 시도에서 새로 발견): `dim_y`/`dim_x`도 실제 좌표가
아니라 더미(0으로만 채워짐)
1차 수정은 `dim_y`/`dim_x` 데이터셋을 진짜 투영좌표로 가정하고
`pyproj`로 지역별 위경도를 변환해 그 안에서 최근접 인덱스를 찾도록
고쳤는데, 실행해보니 **4개 지역이 전부 (0,0)으로 나와 값이 똑같았다**
- 실측 확인 결과 `dim_y.min()==dim_y.max()==0.0`(HDF5 '차원 스케일'
연결용 더미 배열일 뿐, 실좌표 아님). **최종 해결**: 파일 전역속성의
격자 정의(좌상단 easting/northing=-899750/899750, pixel_size=500m,
3600×3600)에서 픽셀별 좌표를 직접 계산 - 네 모서리 속성값과 정확히
일치하는지 검증까지 마쳤다(예: 우상단 계산값이 파일의
upper_right_easting과 일치). `pyproj` LCC 변환 자체는 기존
`collect_gk2a_4sites_bulk_v1_2026-09-01.py`의 `GK2ACoordinateMapper`와
동일 파라미터·재사용.

## 버그 4: 파일명 시각이 UTC인데 KST로 착각(원본 스크립트)
GK2A 파일명의 12자리는 **UTC**다(이 프로젝트 기존 검증 수집기
`collect_gk2a_continuous44d_v1_2026-08-21.py`가
`target_utc = target_kst - timedelta(hours=9)`로 명시 - 새로 추정한 게
아니라 이미 확정된 규약 재사용). 원본 스크립트는 이 UTC 문자열을 그대로
naive datetime으로 저장해 **실제로는 9시간 밀린 값**이었다(예: 파일명
0000=UTC자정=KST 09:00인데 "00:00"으로 기록됨 - 발전량 등 KST 기준
자료와 조인하면 9시간 어긋난다). **해결**: 파싱 직후 +9시간 더해
`datetime_kst` 컬럼으로 명확히 저장.

## 실행법
python extract_sat_features_4regions_v1_2026-09-03.py
(pip install h5py pyproj 필요 - 이번에 확인 결과 둘 다 이미 설치돼 있었음)
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
OUTPUT_CSV = r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\01_데이터수집\위성\sat_features_4regions_202608.csv"

SITES = {
    "Gwangju": {"lat": 35.1595, "lon": 126.8526},
    "Buan": {"lat": 35.7317, "lon": 126.7333},
    "Gimje": {"lat": 35.8036, "lon": 126.8808},
    "Yeonggwang": {"lat": 35.2773, "lon": 126.5120},
}


def build_projector(f: h5py.File) -> pyproj.Proj:
    """파일 자체의 LCC 투영 속성으로 pyproj 객체를 만든다(하드코딩 안 함 -
    다른 시기/버전 파일에서 파라미터가 바뀌어도 안전)."""
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
    """★버그수정(09-03, 2차)★: `dim_y`/`dim_x` 데이터셋은 실제 좌표값이
    아니라 전부 0으로 채워진 HDF5 '차원 스케일' 더미(DIMENSION_LIST
    연결용 라벨일 뿐, 실측으로 확인: dim_y.min()==dim_y.max()==0.0) -
    이걸 그대로 쓰면 4개 지역이 전부 (0,0)으로 잡혀서 값이 똑같이
    나온다(1차 수정판에서 실제로 이 증상 재현·확인함). 진짜 좌표는
    전역 속성의 격자 정의(좌상단 easting/northing + pixel_size 500m +
    이미지 크기 3600×3600)에서 직접 계산해야 한다 - 상하좌우 네 모서리
    속성이 이 계산과 정확히 일치하는지 검증까지 마쳤다."""
    ul_x = float(np.asarray(f.attrs["upper_left_easting"]).flat[0])
    ul_y = float(np.asarray(f.attrs["upper_left_northing"]).flat[0])
    pixel_size = float(np.asarray(f.attrs["pixel_size"]).flat[0])
    width = int(np.asarray(f.attrs["image_width"]).flat[0])
    height = int(np.asarray(f.attrs["image_height"]).flat[0])

    x_coords = ul_x + np.arange(width) * pixel_size
    y_coords = ul_y - np.arange(height) * pixel_size  # 북쪽(위)이 큰 값이라 아래로 갈수록 감소

    proj = build_projector(f)
    indices = {}
    for site_name, coord in SITES.items():
        x_m, y_m = proj(coord["lon"], coord["lat"])
        x_idx = int(np.argmin(np.abs(x_coords - x_m)))
        y_idx = int(np.argmin(np.abs(y_coords - y_m)))
        indices[site_name] = (y_idx, x_idx)
    return indices


def process_single_nc(file_path: str, site_indices: dict | None = None):
    match = re.search(r"(\d{12})\.nc$", file_path)
    if not match:
        return None, site_indices

    dt_utc = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    dt_kst = dt_utc + KST_OFFSET  # 파일명은 UTC(위 "버그 4" 참고) - KST로 변환해 저장
    row_data: dict = {"datetime_kst": dt_kst, "datetime_utc": dt_utc}

    try:
        with h5py.File(file_path, "r") as f:
            if site_indices is None:
                site_indices = get_site_pixel_indices(f)

            data_arr = f["image_pixel_values"][:]
            for site_name, (y_i, x_i) in site_indices.items():
                val = data_arr[y_i, x_i]
                row_data[f"{site_name}_vi006"] = float(val) if np.isfinite(val) else 0.0

        return row_data, site_indices

    except Exception as e:  # noqa: BLE001 - 파일별 실패를 계속 진행하며 로그만 남김
        print(f"[X] 파일 처리 에러 ({os.path.basename(file_path)}): {e!r}")
        return None, site_indices


def main() -> None:
    nc_files = sorted(glob.glob(os.path.join(RAW_NC_DIR, "*.nc")))
    print(f"[*] 총 {len(nc_files)}개 NC 파일에 대해 [광주/부안/김제/영광] 피처 추출 시작...")

    results = []
    site_indices = None

    for idx, fpath in enumerate(nc_files, 1):
        data, site_indices = process_single_nc(fpath, site_indices)
        if data:
            results.append(data)
        if idx % 50 == 0 or idx == len(nc_files):
            print(f"[{idx}/{len(nc_files)}] 처리 완료...")

    if not results:
        raise RuntimeError("모든 파일 처리 실패 - 결과 0건(원인은 위 [X] 로그 참고)")

    df = pd.DataFrame(results)
    df.sort_values("datetime_kst", inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[성공] 전처리 완료! 결과 저장: {OUTPUT_CSV}")
    print(f"[성공] 지점별 픽셀 인덱스(첫 파일 기준): {site_indices}")
    print("\n=== 추출된 데이터 요약 (상위 5행) ===")
    print(df.head())


if __name__ == "__main__":
    main()
