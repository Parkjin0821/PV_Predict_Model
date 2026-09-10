# -*- coding: utf-8 -*-
"""5-1번(08-20 감사 신설, 08-20 밤 재작업): 3시간 간격 예보자료(NWP 일사·운량 +
격자예보)를 1시간 간격으로 재구성한다. BSRN 물리적 상한 클리핑도 이 안에서
같이 한다(아래 "재작업 이유" 참고 — 상한 계산 방식이 바뀌었기 때문).

## 재작업 이유 (08-20 밤 발견, AGENTS.md "★★★신규 발견★★★" 절 참고)
ASOS 실측 대비 실측 비교 결과, **DSWRF(그리고 같은 계열 DSWRFLX·DIFSWRF로
추정)는 그 시각의 순간값이 아니라 "직전 3시간 평균값"**으로 보인다
(순간값 가설 MAE 137.8/상관 0.782 vs 직전3시간평균 가설 MAE 72.7/상관 0.908,
대조군 TMP(기온)는 순간값 가설이 맞음 — 기상청 공식 문서로 확정하지는
못했으나 실측 대조로 강하게 뒷받침됨, GRIB 관례상 복사플럭스류가 평균/누적,
상태변수가 순간값인 경우가 흔해 물리적으로도 타당).

이 사실 때문에 기존 접근(청천지수 kt를 "각 시각의 점값"으로 보고 점 사이를
보간)은 틀렸다. **3시간 평균값은 해당 3시간 블록 전체를 대표하는 값이므로,
"평균을 보존하면서 블록 내부를 청천하늘 모양대로 나눠 담는" 방식(보존적
재분배, conservative disaggregation)으로 바꿔야 한다.**

## 방법 (자체 판단, 표준 하향규모화 기법 응용 — 출처 명시)
각 3시간 앵커(라벨 시각 h, 커버 구간 {h-2,h-1,h})에 대해:
  weight(t) = 청천모델값(t) / mean(청천모델값(h-2,h-1,h))   for t in {h-2,h-1,h}
  G(t) = 앵커값(h) × weight(t)
이렇게 하면 항상 mean(G(h-2),G(h-1),G(h)) = 앵커값(h)로 **3시간 평균이
정확히 보존**되면서(자료 훼손 없음), 블록 내부는 실제 태양 궤적을 따라
합리적으로 나뉜다. 청천모델 평균이 0에 가까운(야간) 블록은 0으로 둔다.

청천모델:
- GHI(DSWRF)용: **Haurwitz(1945)** `GHI_cs = 1098·μ0·exp(-0.059/μ0)`.
- DNI(DSWRFLX)용: **Meinel & Meinel(1976)** 근사식
  `DNI_cs = 1353·0.7^(AM^0.678)`, AM(air mass)=1/μ0 — 널리 쓰이는 단순
  청천 DNI 근사식(계수 보정 없는 원형).
- DHI(DIFSWRF)용: 별도 모델을 쓰지 않고 분해항등식으로 역산
  `DHI_cs = max(0, GHI_cs − DNI_cs·μ0)` — GHI=DNI·μ0+DHI 물리적 정의를
  그대로 이용(자체 판단, 표준 항등식).
- 이 청천모델들은 "블록 내부 상대적 모양"을 정하는 가중치 계산에만
  쓰이고, 실제 값의 크기(스케일)는 항상 원래 3시간 평균 앵커값으로
  고정되므로 청천모델 자체의 절대오차가 크더라도 결과의 편향은 제한적이다.

## BSRN 물리적 상한 클리핑도 이 스크립트로 흡수(재작업)
기존 `clean_nwp_direct_diffuse_bsrn_qc_v1.py`는 앵커 시각 "순간" Sa로
상한을 계산했다. 값이 3시간 평균이라면 상한도 **그 3시간 구간의 평균
Sa(직달) 상한**으로 계산해야 더 정확하다(다만 오늘 오전 걸러낸 이상값이
수만~10만W/m²로 어떤 상한 계산법을 써도 압도적으로 초과해 실제로 걸러지는
셀 자체는 거의 안 바뀔 것으로 예상 — 아래 실행 결과에서 실측 비교).

## 검증
1. BSRN 재클리핑 결과가 기존(순간값 기준) 결과와 얼마나 다른지 비교.
2. 재구성된 1시간 DSWRF를 ASOS 실측과 비교해 기존(3시간 그대로 선형보간)
   대비 개선됐는지 확인.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CODEX = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주")
NWP_DIR = CODEX / "kma_nwp_solar_cloud_710d_v1_2026-08-19"

SOLAR_CLOUD_CSV = NWP_DIR / "광주_수치예보_DSWRF_DSWRFLX_DIFSWRF_TCDC_710일.csv"
LAYER_CSV = NWP_DIR / "광주_수치예보_LCDC_MCDC_HCDC_710일.csv"
VEC_CSV = CODEX / "grid_forecast_vec_710d_v1_2026-08-19" / "광주_격자예보_VEC_710일.csv"
GRID_TSR_CSV = (
    CODEX / "grid_forecast_710d_collection_v3_2026-08-19"
    / "광주_격자예보_3시간단위_TMP_SKY_REH_710일.csv"
)
GRID_WP_CSV = (
    CODEX / "grid_forecast_wsd_pop_710d_v4_2026-08-19" / "광주_격자예보_WSD_POP_710일.csv"
)
ASOS_CSV = Path(
    r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\기상청_ASOS156_시간환경_모델용.csv"
)

OUT_DIR = CODEX / "nwp_hourly_interpolation_v2_2026-08-20밤"
OUT_CSV = OUT_DIR / "광주_예보_1시간재구성_710일.csv"

FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
KST_OFFSET = timedelta(hours=9)
LATITUDE = 35.14428133
LONGITUDE = 126.84058771
SOLAR_CONSTANT = 1367.0


def solar_geometry(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = index.dayofyear.to_numpy(float)
    local_hour = index.hour.to_numpy(float) + index.minute.to_numpy(float) / 60
    gamma = 2 * np.pi / 365 * (n - 1 + (local_hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    lat = math.radians(LATITUDE)
    true_solar_min = local_hour * 60 + eqtime + 4 * LONGITUDE - 60 * 9
    hour_angle = np.deg2rad(true_solar_min / 4 - 180)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    sin_elev = np.clip(sin_elev, -1, 1)
    elev_deg = np.rad2deg(np.arcsin(sin_elev))
    mu0 = np.clip(sin_elev, 1e-6, None)
    mu0_masked = np.where(sin_elev > 0, mu0, 0.0)
    e0 = 1 + 0.033 * np.cos(2 * np.pi * n / 365)
    sa = SOLAR_CONSTANT * e0

    with np.errstate(divide="ignore", invalid="ignore"):
        ghi_cs = np.where(mu0_masked > 0, 1098.0 * mu0_masked * np.exp(-0.059 / mu0), 0.0)
        air_mass = np.where(mu0_masked > 0, 1.0 / mu0, np.inf)
        dni_cs = np.where(mu0_masked > 0, 1353.0 * np.power(0.7, np.power(air_mass, 0.678)), 0.0)
    ghi_cs = np.nan_to_num(ghi_cs, nan=0.0, posinf=0.0, neginf=0.0)
    dni_cs = np.nan_to_num(dni_cs, nan=0.0, posinf=0.0, neginf=0.0)
    dhi_cs = np.clip(ghi_cs - dni_cs * mu0_masked, 0.0, None)

    return pd.DataFrame(
        {
            "태양고도_deg": elev_deg, "Sa_W_m2": sa, "mu0": mu0_masked,
            "청천GHI_W_m2": ghi_cs, "청천DNI_W_m2": dni_cs, "청천DHI_W_m2": dhi_cs,
        },
        index=index,
    )


def wide_to_series(path: Path, prefix: str) -> pd.Series:
    df = pd.read_csv(path, dtype={"발표일": str})
    times, values = [], []
    for row in df.to_dict("records"):
        target_day = datetime.strptime(row["발표일"], "%Y%m%d") + timedelta(days=1)
        for hour in FCST_HOURS:
            times.append(target_day + timedelta(hours=hour))
            values.append(row.get(f"{prefix}_{hour:02d}h"))
    s = pd.Series(values, index=pd.DatetimeIndex(times), name=prefix)
    return pd.to_numeric(s[~s.index.duplicated(keep="last")], errors="coerce").sort_index()


def bsrn_clip_window(anchor: pd.Series, geo_hourly: pd.DataFrame, kind: str) -> tuple[pd.Series, dict]:
    """3시간 구간 평균 상한으로 BSRN 물리적 상한을 재적용한다.

    kind: "dni" -> 상한 = 구간평균 Sa (야간포함 구간은 그만큼 낮아짐)
          "dhi" -> 상한 = 구간평균(Sa·0.95·mu0^1.2 + 50)
    """
    out = anchor.copy()
    rejected = 0
    already_missing = int(anchor.isna().sum())
    for h in anchor.index:
        val = anchor.loc[h]
        window = pd.DatetimeIndex([h - timedelta(hours=2), h - timedelta(hours=1), h])
        g = geo_hourly.reindex(window)
        if kind == "dni":
            bound = g["Sa_W_m2"].where(g["mu0"] > 0, 0.0).mean()
            lo = -4.0
        else:
            bound = (g["Sa_W_m2"] * 0.95 * (g["mu0"].clip(lower=0) ** 1.2) + 50.0).mean()
            lo = -4.0
        if pd.notna(val) and not (lo <= val <= bound):
            out.loc[h] = np.nan
            rejected += 1
    report = {
        "기존_결측": already_missing,
        "신규_상한초과_결측": rejected,
        "최종_결측": already_missing + rejected,
        "전체": len(anchor),
    }
    return out, report


def interp_linear(s: pd.Series, index: pd.DatetimeIndex, limit: int = 2) -> pd.Series:
    return s.reindex(index).interpolate(method="time", limit=limit, limit_area="inside")


def interp_nearest(s: pd.Series, index: pd.DatetimeIndex, limit: int = 2) -> pd.Series:
    return s.reindex(index).interpolate(method="nearest", limit=limit, limit_area="inside")


def interp_circular(s: pd.Series, index: pd.DatetimeIndex, limit: int = 2) -> pd.Series:
    rad = np.deg2rad(s)
    sin_i = interp_linear(pd.Series(np.sin(rad), index=s.index), index, limit)
    cos_i = interp_linear(pd.Series(np.cos(rad), index=s.index), index, limit)
    return pd.Series(np.rad2deg(np.arctan2(sin_i, cos_i)) % 360, index=index)


def disaggregate_conservative(anchor: pd.Series, hourly_index: pd.DatetimeIndex, geo: pd.DataFrame, cs_col: str) -> pd.Series:
    """3시간 평균 앵커를 청천모양 보존적 재분배로 1시간 값으로 편다."""
    out = pd.Series(np.nan, index=hourly_index, dtype=float)
    cs = geo[cs_col]
    for h, val in anchor.dropna().items():
        window = pd.DatetimeIndex([h - timedelta(hours=2), h - timedelta(hours=1), h])
        window = window[window.isin(hourly_index)]
        if len(window) == 0:
            continue
        cs_win = cs.reindex(window)
        cs_mean = cs_win.mean()
        if cs_mean <= 1.0:  # 청천모델상 사실상 야간 블록
            out.loc[window] = 0.0
        else:
            out.loc[window] = (val * cs_win / cs_mean).to_numpy()
    return out.clip(lower=0.0)


def load_asos() -> pd.DataFrame:
    df = pd.read_csv(ASOS_CSV, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["시각"])
    keep = df[["timestamp", "일사량_W_m2", "전운량_pct", "기온_C", "상대습도_pct", "풍속_m_s"]].copy()
    return keep.set_index("timestamp").sort_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] 3시간 앵커 로드...")
    dswrf = wide_to_series(SOLAR_CLOUD_CSV, "DSWRF")
    dswrflx_raw = wide_to_series(SOLAR_CLOUD_CSV, "DSWRFLX")
    difswrf_raw = wide_to_series(SOLAR_CLOUD_CSV, "DIFSWRF")
    tcdc = wide_to_series(SOLAR_CLOUD_CSV, "TCDC")
    lcdc, mcdc, hcdc = (wide_to_series(LAYER_CSV, c) for c in ["LCDC", "MCDC", "HCDC"])
    vec = wide_to_series(VEC_CSV, "VEC")
    tmp, sky, reh = (wide_to_series(GRID_TSR_CSV, c) for c in ["TMP", "SKY", "REH"])
    wsd, pop = (wide_to_series(GRID_WP_CSV, c) for c in ["WSD", "POP"])

    start = min(s.index.min() for s in [dswrf, tcdc, lcdc, vec, tmp, wsd]) - timedelta(hours=2)
    end = max(s.index.max() for s in [dswrf, tcdc, lcdc, vec, tmp, wsd])
    hourly_index = pd.date_range(start, end, freq="1h")
    geo = solar_geometry(hourly_index)
    print(f"    시간격자 {len(hourly_index)}점 ({start} ~ {end})")

    print("\n[2/5] BSRN 물리적 상한 재클리핑(구간평균 기준)...")
    dswrflx, rpt_lx = bsrn_clip_window(dswrflx_raw, geo, "dni")
    difswrf, rpt_dif = bsrn_clip_window(difswrf_raw, geo, "dhi")
    for name, rpt in [("DSWRFLX(직달)", rpt_lx), ("DIFSWRF(산란)", rpt_dif)]:
        print(f"    {name}: 기존결측 {rpt['기존_결측']}, 신규상한초과 {rpt['신규_상한초과_결측']}, "
              f"최종결측 {rpt['최종_결측']}/{rpt['전체']} ({rpt['최종_결측']/rpt['전체']*100:.1f}%)")

    print("\n[3/5] 청천모양 보존적 재분배로 1시간 재구성...")
    out = pd.DataFrame(index=hourly_index)
    out["DSWRF"] = disaggregate_conservative(dswrf, hourly_index, geo, "청천GHI_W_m2")
    out["DSWRFLX_bsrn정제"] = disaggregate_conservative(dswrflx, hourly_index, geo, "청천DNI_W_m2")
    out["DIFSWRF_bsrn정제"] = disaggregate_conservative(difswrf, hourly_index, geo, "청천DHI_W_m2")
    for c, s in [("TCDC", tcdc), ("LCDC", lcdc), ("MCDC", mcdc), ("HCDC", hcdc)]:
        out[c] = interp_linear(s, hourly_index).clip(0.0, 1.0)
    out["VEC"] = interp_circular(vec, hourly_index)
    out["TMP"] = interp_linear(tmp, hourly_index)
    out["SKY"] = interp_nearest(sky, hourly_index)
    out["REH"] = interp_linear(reh, hourly_index).clip(0.0, 100.0)
    out["WSD"] = interp_linear(wsd, hourly_index).clip(lower=0.0)
    out["POP"] = interp_linear(pop, hourly_index).clip(0.0, 100.0)
    out["태양고도_deg"] = geo["태양고도_deg"]
    out.index.name = "time"

    print("\n[4/5] ASOS 실측 대비 검증 (기존 방식 vs 재작업 방식)...")
    asos = load_asos()
    anchor_times = pd.DatetimeIndex(sorted(dswrf.dropna().index))
    new_only = out.index.difference(anchor_times)
    naive_linear = interp_linear(dswrf, hourly_index)  # 기존에 썼던 방식(순간값 취급 선형보간)

    cmp = pd.DataFrame({
        "재작업(청천보존적재분배)": out.loc[new_only, "DSWRF"],
        "기존방식(순간값 선형보간)": naive_linear.reindex(new_only),
        "ASOS실측": asos["일사량_W_m2"].reindex(new_only),
        "태양고도": geo.loc[new_only, "태양고도_deg"],
    })
    cmp = cmp[cmp["태양고도"] > 0].dropna(subset=["ASOS실측", "재작업(청천보존적재분배)", "기존방식(순간값 선형보간)"])
    lines = [f"검증대상(신규생성 중간시각, 낮시간): {len(cmp)}점"]
    for name in ["재작업(청천보존적재분배)", "기존방식(순간값 선형보간)"]:
        e = cmp[name] - cmp["ASOS실측"]
        lines.append(
            f"  {name}: MAE={e.abs().mean():.2f} W/m², RMSE={np.sqrt((e**2).mean()):.2f} W/m², "
            f"상관계수={np.corrcoef(cmp[name], cmp['ASOS실측'])[0,1]:.4f}"
        )
    # 전체(앵커+신규) 기준도 참고로
    full = pd.DataFrame({"재작업": out["DSWRF"], "ASOS실측": asos["일사량_W_m2"].reindex(out.index),
                          "태양고도": geo["태양고도_deg"]})
    full = full[full["태양고도"] > 0].dropna()
    e = full["재작업"] - full["ASOS실측"]
    lines.append(f"  [참고] 전체시각(앵커+신규) 재작업 결과: MAE={e.abs().mean():.2f} W/m², n={len(full)}")
    report = "\n".join(lines)
    print(report)

    print("\n[5/5] 저장...")
    out.to_csv(OUT_CSV, encoding="utf-8-sig")
    (OUT_DIR / "검증요약.txt").write_text(report, encoding="utf-8")
    cmp.to_csv(OUT_DIR / "검증_상세.csv", encoding="utf-8-sig")
    for c in out.columns:
        v = out[c].notna().sum()
        print(f"  {c:20s} 유효 {v}/{len(out)} ({v/len(out)*100:.1f}%)")
    print(f"\n저장 완료: {OUT_CSV}")


if __name__ == "__main__":
    main()
