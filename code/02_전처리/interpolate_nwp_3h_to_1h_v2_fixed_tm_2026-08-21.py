# -*- coding: utf-8 -*-
"""고정-tm 재수집(710일, 08-20~21) 기반 3시간->1시간 재구성.

interpolate_nwp_3h_to_1h_v1.py(누출 tm 기반)와 동일한 검증된 방법론
(BSRN 물리적 상한 재클리핑 + 청천모양 보존적 재분배)을 그대로 쓰되,
입력을 고정-tm(D 09:00 KST, 리드타임 15~36h) 재수집 파일로 교체한다.
격자예보(VEC/TMP/SKY/REH/WSD, D 05:00 KST 발표)는 애초에 누출이 없었으므로
기존 파일을 그대로 재사용한다.
"""
from __future__ import annotations
import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

CODEX = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주")
FIXED_TM_CSV = (
    CODEX / "kma_nwp_dayahead_fixed_tm_710d_v2_2026-08-20"
    / "광주_익일예보_고정tm_DSWRF_DSWRFLX_DIFSWRF_TCDC_LCDC_MCDC_HCDC_710일.csv"
)
VEC_CSV = CODEX / "grid_forecast_vec_710d_v1_2026-08-19" / "광주_격자예보_VEC_710일.csv"
GRID_TSR_CSV = CODEX / "grid_forecast_710d_collection_v3_2026-08-19" / "광주_격자예보_3시간단위_TMP_SKY_REH_710일.csv"
GRID_WP_CSV = CODEX / "grid_forecast_wsd_pop_710d_v4_2026-08-19" / "광주_격자예보_WSD_POP_710일.csv"
ASOS_CSV = Path(r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\기상청_ASOS156_시간환경_모델용.csv")

OUT_DIR = CODEX / "nwp_hourly_interpolation_v3_fixed_tm_2026-08-21"
OUT_CSV = OUT_DIR / "광주_예보_1시간재구성_고정tm_710일.csv"

FCST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
KST_OFFSET = timedelta(hours=9)
LATITUDE, LONGITUDE, SOLAR_CONSTANT = 35.14428133, 126.84058771, 1367.0


def solar_geometry(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = index.dayofyear.to_numpy(float)
    local_hour = index.hour.to_numpy(float) + index.minute.to_numpy(float) / 60
    gamma = 2 * np.pi / 365 * (n - 1 + (local_hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                        - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    lat = math.radians(LATITUDE)
    true_solar_min = local_hour*60 + eqtime + 4*LONGITUDE - 60*9
    hour_angle = np.deg2rad(true_solar_min/4 - 180)
    sin_elev = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(hour_angle)
    sin_elev = np.clip(sin_elev, -1, 1)
    elev_deg = np.rad2deg(np.arcsin(sin_elev))
    mu0 = np.clip(sin_elev, 1e-6, None)
    mu0_masked = np.where(sin_elev > 0, mu0, 0.0)
    e0 = 1 + 0.033*np.cos(2*np.pi*n/365)
    sa = SOLAR_CONSTANT*e0
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi_cs = np.where(mu0_masked > 0, 1098.0*mu0_masked*np.exp(-0.059/mu0), 0.0)
        air_mass = np.where(mu0_masked > 0, 1.0/mu0, np.inf)
        dni_cs = np.where(mu0_masked > 0, 1353.0*np.power(0.7, np.power(air_mass, 0.678)), 0.0)
    ghi_cs = np.nan_to_num(ghi_cs, nan=0.0, posinf=0.0, neginf=0.0)
    dni_cs = np.nan_to_num(dni_cs, nan=0.0, posinf=0.0, neginf=0.0)
    dhi_cs = np.clip(ghi_cs - dni_cs*mu0_masked, 0.0, None)
    return pd.DataFrame({"태양고도_deg": elev_deg, "Sa_W_m2": sa, "mu0": mu0_masked,
                          "청천GHI_W_m2": ghi_cs, "청천DNI_W_m2": dni_cs, "청천DHI_W_m2": dhi_cs}, index=index)


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
    return out, {"기존_결측": already_missing, "신규_상한초과_결측": rejected,
                 "최종_결측": already_missing + rejected, "전체": len(anchor)}


def interp_linear(s, index, limit=2):
    # ★09-03 수정★: 조회창 안에 유효값이 하나도 없으면(전부 NULL) DB에서
    # 읽어온 값이 전부 None이라 판다스가 이 컬럼을 object dtype으로
    # 만든다 - object dtype은 .interpolate()가 지원 안 해서 예외가 났다
    # (실측 확인: 09-03 새벽 광주 NWP가 08-28 이후 유효값 0건이라 매시간
    # crash, "warming_up"으로 정직하게 처리돼야 할 게 status='error'로
    # 잘못 기록되고 있었음). pd.to_numeric으로 명시 변환하면 전부 NaN인
    # 경우도 정상적으로 float64가 되어 그대로 전부 NaN으로 통과한다
    # (임의값 채움 아님 - 결측은 결측 그대로 유지).
    s = pd.to_numeric(s, errors="coerce")
    return s.reindex(index).interpolate(method="time", limit=limit, limit_area="inside")


def interp_nearest(s, index, limit=2):
    s = pd.to_numeric(s, errors="coerce")
    return s.reindex(index).interpolate(method="nearest", limit=limit, limit_area="inside")


def interp_circular(s, index, limit=2):
    # 09-03 수정: interp_linear와 동일 이유(전부 NULL이면 object dtype).
    s = pd.to_numeric(s, errors="coerce")
    rad = np.deg2rad(s)
    sin_i = interp_linear(pd.Series(np.sin(rad), index=s.index), index, limit)
    cos_i = interp_linear(pd.Series(np.cos(rad), index=s.index), index, limit)
    return pd.Series(np.rad2deg(np.arctan2(sin_i, cos_i)) % 360, index=index)


def disaggregate_conservative(anchor, hourly_index, geo, cs_col):
    out = pd.Series(np.nan, index=hourly_index, dtype=float)
    cs = geo[cs_col]
    for h, val in anchor.dropna().items():
        window = pd.DatetimeIndex([h - timedelta(hours=2), h - timedelta(hours=1), h])
        window = window[window.isin(hourly_index)]
        if len(window) == 0:
            continue
        cs_win = cs.reindex(window)
        cs_mean = cs_win.mean()
        if cs_mean <= 1.0:
            out.loc[window] = 0.0
        else:
            out.loc[window] = (val * cs_win / cs_mean).to_numpy()
    return out.clip(lower=0.0)


def load_asos():
    df = pd.read_csv(ASOS_CSV, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["시각"])
    return df.set_index("timestamp")[["일사량_W_m2"]].sort_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/5] 고정-tm 앵커 로드...")
    dswrf = wide_to_series(FIXED_TM_CSV, "DSWRF")
    dswrflx_raw = wide_to_series(FIXED_TM_CSV, "DSWRFLX")
    difswrf_raw = wide_to_series(FIXED_TM_CSV, "DIFSWRF")
    tcdc = wide_to_series(FIXED_TM_CSV, "TCDC")
    lcdc, mcdc, hcdc = (wide_to_series(FIXED_TM_CSV, c) for c in ["LCDC", "MCDC", "HCDC"])
    vec = wide_to_series(VEC_CSV, "VEC")
    tmp, sky, reh = (wide_to_series(GRID_TSR_CSV, c) for c in ["TMP", "SKY", "REH"])
    wsd, pop = (wide_to_series(GRID_WP_CSV, c) for c in ["WSD", "POP"])

    start = min(s.index.min() for s in [dswrf, tcdc, lcdc, vec, tmp, wsd]) - timedelta(hours=2)
    end = max(s.index.max() for s in [dswrf, tcdc, lcdc, vec, tmp, wsd])
    hourly_index = pd.date_range(start, end, freq="1h")
    geo = solar_geometry(hourly_index)
    print(f"    시간격자 {len(hourly_index)}점 ({start} ~ {end})")

    print("\n[2/5] BSRN 물리적 상한 재클리핑...")
    dswrflx, rpt_lx = bsrn_clip_window(dswrflx_raw, geo, "dni")
    difswrf, rpt_dif = bsrn_clip_window(difswrf_raw, geo, "dhi")
    for name, rpt in [("DSWRFLX", rpt_lx), ("DIFSWRF", rpt_dif)]:
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

    print("\n[4/5] ASOS 실측 대비 검증...")
    asos = load_asos()
    anchor_times = pd.DatetimeIndex(sorted(dswrf.dropna().index))
    new_only = out.index.difference(anchor_times)
    cmp = pd.DataFrame({"재구성": out.loc[new_only, "DSWRF"],
                         "ASOS실측": asos["일사량_W_m2"].reindex(new_only),
                         "태양고도": geo.loc[new_only, "태양고도_deg"]})
    cmp = cmp[cmp["태양고도"] > 0].dropna()
    e = cmp["재구성"] - cmp["ASOS실측"]
    print(f"    신규생성 중간시각(n={len(cmp)}): MAE={e.abs().mean():.2f}W/m² RMSE={np.sqrt((e**2).mean()):.2f}W/m² "
          f"상관={np.corrcoef(cmp['재구성'],cmp['ASOS실측'])[0,1]:.4f}")

    print("\n[5/5] 저장...")
    out.to_csv(OUT_CSV, encoding="utf-8-sig")
    for c in out.columns:
        v = out[c].notna().sum()
        print(f"  {c:20s} 유효 {v}/{len(out)} ({v/len(out)*100:.1f}%)")
    print(f"\n저장 완료: {OUT_CSV}")


if __name__ == "__main__":
    main()
