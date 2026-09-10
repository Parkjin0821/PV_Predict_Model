# -*- coding: utf-8 -*-
"""정렬 단계 공통 동료 인버터 대조 재분류(공통_품질정책_v1_2026-08-31.json 구현).

영광 통신에러=idle 검증(08-31)에서 만든 방법론을 부안·김제 정렬 스크립트가
공통으로 불러 쓰는 모듈. API 호출 없음, 순수 로컬 판정 로직만 담는다.

★적용 대상★: quality_status=="observed"이면서 comm_error_flag==True인
행만 재분류한다(원래 결측이던 행·정상 관측이던 행은 건드리지 않음).

판정 규칙(영광 실측으로 정한 임계값 - 13대·709일 대조 결과 태양고도>5도인데
flag=True인 행이 2/25만여건뿐이었음, 그 2건은 피어 다수 발전중이었음):
  1. 태양고도 <= daytime_elevation_threshold(기본 5도): 여명·황혼대로 보고
     physical_zero_idle_supported(값 신뢰, 0으로 사용).
  2. 태양고도 > threshold(주간)인데 피어 대부분(peer_majority_frac 이상)이
     peer_producing_kw 이상 발전중: communication_outage_missing(결측 처리,
     값 사용 금지 - 혼자만 통신결측일 가능성).
  3. 태양고도 > threshold인데 피어도 대부분 idle(중앙값<=peer_idle_kw):
     physical_zero_idle_supported(전 발전소가 같은 이유로 쉬고 있었을 가능성).
  4. 그 외(피어 데이터 부족·판정 애매): ambiguous_zero_missing(보수적으로
     결측 처리).

성능: 태양고도 필터를 먼저 적용해 값싸게 대다수(야간·여명대)를 걸러내고,
비싼 피어 대조는 주간 잔여행에만 수행한다(영광 실측 기준 전체의 극소수).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TZ_OFFSET_H = 9


def solar_elevation_deg(ts: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """광주/부안/김제/영광 전부에서 재사용해온 NOAA 근사식(신규 아님)."""
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    time_offset = eqtime + 4 * lon - 60 * TZ_OFFSET_H
    tst = hour * 60 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)
    lat_r = np.deg2rad(lat)
    cos_zenith = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    return 90.0 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))


@dataclass
class PeerComparisonResult:
    quality_status: pd.Series  # 원본과 같은 인덱스, 재분류 반영된 최종값
    reclassified_count: int
    bucket_counts: dict


def classify_with_peer_comparison(
    all_aligned: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    daytime_elevation_threshold: float = 5.0,
    peer_producing_kw: float = 0.5,
    peer_idle_kw: float = 0.2,
    peer_majority_frac: float = 0.75,
    time_col: str = "grid_time_kst",
    inverter_col: str = "inverter_number",
    ac_col: str = "ac_power_kw",
    quality_col: str = "quality_status",
    comm_flag_col: str = "comm_error_flag",
    zero_tolerance_kw: float = 0.05,
) -> PeerComparisonResult:
    """all_aligned: 전 인버터가 concat된 long-format 정렬본(1행=1인버터1슬롯).
    반환하는 quality_status는 all_aligned와 같은 순서/인덱스의 새 Series다 -
    호출부에서 all_aligned[quality_col] = result.quality_status로 대입할 것."""

    if comm_flag_col not in all_aligned.columns:
        raise RuntimeError(f"{comm_flag_col} 없음 - 통신에러 플래그를 먼저 읽어와야 함")

    if all_aligned.duplicated([time_col, inverter_col]).any():
        raise RuntimeError("동일 시각·인버터 중복행이 있어 동료대조를 중단합니다.")

    out = all_aligned[quality_col].astype("string").copy()
    target_mask = (all_aligned[quality_col] == "observed") & all_aligned[comm_flag_col].fillna(False)
    if not target_mask.any():
        return PeerComparisonResult(out, 0, {})
    nonzero_flagged = target_mask & all_aligned[ac_col].abs().gt(zero_tolerance_kw)
    if nonzero_flagged.any():
        raise RuntimeError(
            "통신에러 플래그 관측행 중 0이 아닌 출력이 "
            f"{int(nonzero_flagged.sum())}행입니다. 현재 동료대조는 0출력 판정 전용이므로 중단합니다."
        )

    uniq_t = pd.DatetimeIndex(sorted(all_aligned[time_col].unique()))
    elev = pd.Series(solar_elevation_deg(uniq_t, lat, lon), index=uniq_t)
    row_elev = all_aligned[time_col].map(elev)

    idle_mask = target_mask & (row_elev <= daytime_elevation_threshold)
    out.loc[idle_mask] = "physical_zero_idle_supported"

    daytime_mask = target_mask & (row_elev > daytime_elevation_threshold)
    bucket_counts = {"physical_zero_idle_supported": int(idle_mask.sum())}

    if daytime_mask.any():
        pivot = all_aligned.pivot_table(index=time_col, columns=inverter_col, values=ac_col, aggfunc="first")
        n_peers_total = pivot.shape[1] - 1
        subset = all_aligned.loc[daytime_mask, [time_col, inverter_col]]

        result_by_index = {}
        for t, grp in subset.groupby(time_col):
            if t not in pivot.index:
                for idx in grp.index:
                    result_by_index[idx] = "ambiguous_zero_missing"
                continue
            row = pivot.loc[t]
            for idx, inv in grp[inverter_col].items():
                peers = row.drop(labels=[inv], errors="ignore").dropna()
                if len(peers) < max(3, int(n_peers_total * 0.5)):
                    result_by_index[idx] = "ambiguous_zero_missing"
                    continue
                peer_median = peers.median()
                producing_frac = (peers > peer_producing_kw).mean()
                if producing_frac >= peer_majority_frac and peer_median > peer_producing_kw:
                    result_by_index[idx] = "communication_outage_missing"
                elif peer_median <= peer_idle_kw:
                    result_by_index[idx] = "physical_zero_idle_supported"
                else:
                    result_by_index[idx] = "ambiguous_zero_missing"

        classified = pd.Series(result_by_index, dtype="string").reindex(out.index[daytime_mask])
        if classified.isna().any():
            raise RuntimeError("주간 동료대조 결과가 원본 인덱스와 완전히 대응하지 않습니다.")
        out.loc[classified.index] = classified
        for status in ("communication_outage_missing", "physical_zero_idle_supported", "ambiguous_zero_missing"):
            bucket_counts[status] = bucket_counts.get(status, 0) + int(classified.eq(status).sum())

    reclassified = int(target_mask.sum())
    return PeerComparisonResult(out, reclassified, bucket_counts)
