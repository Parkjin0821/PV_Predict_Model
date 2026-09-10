"""DSWRFLX(직달복사)·DIFSWRF(산란복사)에 BSRN 품질관리 기준을 적용해
물리적으로 불가능한 값을 결측 처리한다.

## 배경 (2026-08-20)
`광주_수치예보_DSWRF_DSWRFLX_DIFSWRF_TCDC_710일.csv`의 DSWRFLX·DIFSWRF는
기상청 KIMR 모델 원본 자체의 결손이다(우리 코드 버그 아님, 실시간 재조회로
재현 확인함):
  - 2026-03말~05월: 물리적으로 불가능한 값(예: 45,824W/m²) 반환
  - 2026-06~08월(65일): API가 문자 그대로 "nan"을 반환 → 파이썬
    float("nan")이 에러 없이 파싱되어 수집기의 결측 감지를 조용히 통과함
DSWRF·TCDC(KIMG 모델)는 같은 기간 결측 0건으로 정상이며 이 문제와 무관하다.

## 처리 방법 (문헌 근거, AGENTS.md 2026-08-20 "처리방법 확정" 절 참고)
BSRN(Baseline Surface Radiation Network) 품질관리 — Long & Dutton 방법의
"physically possible limit(PPL)"을 채택한다. 고정 숫자가 아니라 태양천정각·
대기외복사량 기반 공식을 쓴다:
  DNI(직달, DSWRFLX로 취급): -4 < DNI < Sa
  DHI(산란, DIFSWRF로 취급): -4 < DHI < Sa·0.95·μ0^1.2 + 50
  (Sa = 대기외복사량 = 태양상수 1367W/m² × 이심률보정, μ0 = cos(태양천정각))
DSWRFLX를 DNI(직달법선면일사)로 대응시킨 것은 자체 판단이다 — 기상청이
"직달단파복사속"이라고만 표기하고 수평면/법선면 여부를 명시하지 않았다.
다만 이번에 걸러내는 오염값(수만~10만W/m²)은 어느 쪽으로 가정해도 상한을
압도적으로 초과하므로 이 가정 차이가 결과에 실질적 영향을 주지 않는다.

추가로 자체 판단 보정 하나를 더한다: 태양이 지평선 아래(태양고도<=0)일 때는
DNI 공식이 그대로면 상한이 Sa(약 1300대)로 남아 너무 느슨해지므로, 야간에는
DNI 상한을 0으로 둔다(직달광은 해가 떠야만 존재하므로).

## BRL 역산은 적용하지 않음
결측 구간을 DSWRF에서 역산(BRL 모델)해 채우는 방안은 검토했으나 보류했다
(AGENTS.md 참고 — 패널 설치각도·방위각이 미확보라 직달·산란 구분값을 쓸
데가 없음). 이 스크립트는 결측 처리만 하며 값을 채워 넣지 않는다.

## 출력
원본 파일은 건드리지 않는다. 정제된 DSWRFLX·DIFSWRF만 별도 파일로 저장한다.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


NWP_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_nwp_solar_cloud_710d_v1_2026-08-19"
)
NWP_CSV = NWP_DIR / "광주_수치예보_DSWRF_DSWRFLX_DIFSWRF_TCDC_710일.csv"
OUT_CSV = NWP_DIR / "광주_수치예보_DSWRFLX_DIFSWRF_BSRN정제_710일.csv"
OUT_REPORT = NWP_DIR / "BSRN_정제_요약.txt"

FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]  # KST 대상시각 (D+1 기준)
KST_OFFSET = timedelta(hours=9)

# 확정된 대표좌표 (치평동_대표좌표_출처.json)
LATITUDE = 35.14428133
LONGITUDE = 126.84058771

SOLAR_CONSTANT = 1367.0  # W/m^2, 대기외 태양상수 (BSRN/전통적으로 쓰이는 값)


def solar_position(dt_utc: datetime) -> tuple[float, float]:
    """주어진 UTC 시각에 대한 (태양고도_deg, 대기외복사량 Sa_W_m2)를 계산한다.

    validate_grid_forecast_650d.py의 solar_elevation_approx와 동일 계열
    방법(간이 NOAA식)을 우리 확정 대표좌표로 재구현한 것이다."""
    kst = dt_utc + KST_OFFSET
    lat = math.radians(LATITUDE)
    n = kst.timetuple().tm_yday
    local_hour = kst.hour + kst.minute / 60
    gamma = 2 * math.pi / 365 * (n - 1 + (local_hour - 12) / 24)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    time_offset = eqtime + 4 * LONGITUDE - 60 * 9
    true_solar_min = local_hour * 60 + time_offset
    hour_angle = math.radians(true_solar_min / 4 - 180)

    elev = math.asin(
        math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    )
    elev_deg = math.degrees(elev)

    # 이심률 보정 (Duffie & Beckman 근사식)
    e0 = 1 + 0.033 * math.cos(2 * math.pi * n / 365)
    sa = SOLAR_CONSTANT * e0
    return elev_deg, sa


def bsrn_ppl_bounds(elev_deg: float, sa: float) -> tuple[float, float, float, float]:
    """(DNI 하한, DNI 상한, DHI 하한, DHI 상한) — BSRN physically possible limit."""
    mu0 = max(0.0, math.cos(math.radians(90 - elev_deg)))  # = sin(elev), 야간 0으로 클리핑
    dni_lo, dni_hi = -4.0, (sa if mu0 > 0 else 0.0)
    dhi_lo, dhi_hi = -4.0, sa * 0.95 * (mu0 ** 1.2) + 50.0
    return dni_lo, dni_hi, dhi_lo, dhi_hi


def kst_target_utc(day: str, hour: int) -> datetime:
    d_kst_midnight = datetime.strptime(day, "%Y%m%d")
    target_kst = d_kst_midnight + timedelta(days=1, hours=hour)
    return target_kst - KST_OFFSET


def main() -> int:
    if not NWP_CSV.exists():
        print(f"입력 파일이 없습니다: {NWP_CSV}")
        return 1

    df = pd.read_csv(NWP_CSV, dtype={"발표일": str})
    out = pd.DataFrame({"발표일": df["발표일"]})

    already_missing_lx = 0
    already_missing_dif = 0
    newly_rejected_lx = 0
    newly_rejected_dif = 0
    total_cells = 0

    for hour in FCST_HOURS:
        lx_col = f"DSWRFLX_{hour:02d}h"
        dif_col = f"DIFSWRF_{hour:02d}h"
        lx_out = []
        dif_out = []
        for day, lx_val, dif_val in zip(df["발표일"], df[lx_col], df[dif_col]):
            total_cells += 1
            if pd.isna(lx_val):
                already_missing_lx += 1
            if pd.isna(dif_val):
                already_missing_dif += 1

            utc = kst_target_utc(day, hour)
            elev, sa = solar_position(utc)
            dni_lo, dni_hi, dhi_lo, dhi_hi = bsrn_ppl_bounds(elev, sa)

            if pd.isna(lx_val) or not (dni_lo <= lx_val <= dni_hi):
                if pd.notna(lx_val):
                    newly_rejected_lx += 1
                lx_out.append(np.nan)
            else:
                lx_out.append(lx_val)

            if pd.isna(dif_val) or not (dhi_lo <= dif_val <= dhi_hi):
                if pd.notna(dif_val):
                    newly_rejected_dif += 1
                dif_out.append(np.nan)
            else:
                dif_out.append(dif_val)

        out[lx_col] = lx_out
        out[dif_col] = dif_out

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        f"입력: {NWP_CSV}",
        f"출력: {OUT_CSV}",
        f"셀 수(변수당): {total_cells}",
        "",
        f"DSWRFLX(직달) 기존 결측(문자 nan 등): {already_missing_lx}",
        f"DSWRFLX(직달) 이번에 BSRN 상한 초과로 새로 결측 처리: {newly_rejected_lx}",
        f"DSWRFLX(직달) 최종 결측: {already_missing_lx + newly_rejected_lx} / {total_cells}"
        f" ({100*(already_missing_lx+newly_rejected_lx)/total_cells:.1f}%)",
        "",
        f"DIFSWRF(산란) 기존 결측: {already_missing_dif}",
        f"DIFSWRF(산란) 이번에 BSRN 상한 초과로 새로 결측 처리: {newly_rejected_dif}",
        f"DIFSWRF(산란) 최종 결측: {already_missing_dif + newly_rejected_dif} / {total_cells}"
        f" ({100*(already_missing_dif+newly_rejected_dif)/total_cells:.1f}%)",
    ]
    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
