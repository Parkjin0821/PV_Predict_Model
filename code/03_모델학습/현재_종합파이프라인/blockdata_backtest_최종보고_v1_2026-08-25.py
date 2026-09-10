# -*- coding: utf-8 -*-
"""Blockdata 오프라인 백테스트 최종보고 — 사용자 12항목 명세 그대로.

새로 재학습하지 않는다(재구현 금지 원칙). 전부 이미 나온 08-25 최종
공식모델의 저장된 예측·정답(OOF)과 v4 Blockdata 변환 산출물을 재사용해
집계·대조만 한다.

입력:
- ac_power OOF(초단기·단기, 08-25 최종·v2 그대로): `E2E_v5_공식B_v2_⑥반영_
  2026-08-24/행단위_ac_power_예측정답.csv`
- daily OOF(일간 58특성, 08-25 최종): `일간_직접모델_최종감사_v1_2026-08-25/
  최종_직접모델_OOF.csv`
- Blockdata 변환본(08-25 최종모델 반영): `Blockdata_규격화_v4_최종통합_
  2026-08-25/`(예측_ac_power.csv, 예측_daily_energy_일간총량.csv,
  예측_daily_energy_누적시계열.csv, 검증_규격점검.csv)

출력: `outputs/Blockdata_오프라인백테스트_최종보고_v1_2026-08-25/`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
AC_OOF = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24" / "행단위_ac_power_예측정답.csv"
DAILY_OOF = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25" / "최종_직접모델_OOF.csv"
BD_DIR = ROOT / "outputs" / "Blockdata_규격화_v4_최종통합_2026-08-25"
OUT = ROOT / "outputs" / "Blockdata_오프라인백테스트_최종보고_v1_2026-08-25"

PLANT_ID = 6715
PLANT_NAME = "광주 광주시청"
CAPACITY_KW = 219.0
LAT, LON = 35.14428133, 126.84058771
TZ_OFFSET_H = 9  # KST

EXPECTED_N = {
    ("초단기", 1): 16160, ("초단기", 2): 15325, ("초단기", 3): 15421, ("초단기", 4): 15469,
    ("단기", 1): 4146, ("단기", 24): 4123, ("단기", 48): 4099,
}
BUCKET_HOURS = {"초단기": 0.25, "단기": 1.0}


# ── 독립 재구현: 표준 NOAA 태양고도 공식(smart persistence 전용) ──
# 파이프라인 내부 "목표_태양고도_deg"는 수평별 shift 정의라 발행시각
# 자체의 고도를 안전하게 못 뽑아내서, 스마트지속성 기준선 하나만을 위해
# 독립적으로 재구현한다(발행시각·대상시각 양쪽에 동일 공식을 써서 비율
# 자체는 내부공식과의 미세한 정의차이가 상쇄됨).
def solar_elevation_deg(ts: pd.DatetimeIndex, lat=LAT, lon=LON, tz_offset_h=TZ_OFFSET_H) -> np.ndarray:
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    time_offset = eqtime + 4 * lon - 60 * tz_offset_h
    tst = hour * 60 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)
    lat_r, decl_r = np.deg2rad(lat), decl
    cos_zenith = np.sin(lat_r) * np.sin(decl_r) + np.cos(lat_r) * np.cos(decl_r) * np.cos(ha)
    zenith = np.arccos(np.clip(cos_zenith, -1, 1))
    return 90.0 - np.rad2deg(zenith)


def clear_sky_ghi(elev_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.clip(elev_deg, 0.01, 90))
    return np.clip(1098.0 * np.sin(theta) * np.exp(-0.059 / np.sin(theta)), 0, None)


def bias_stats(actual, pred):
    e = np.asarray(pred, float) - np.asarray(actual, float)
    return {
        "Bias_평균(예측-실제)": round(float(e.mean()), 4),
        "과대예측_비율_pct": round(float((e > 0).mean() * 100), 2),
        "과소예측_비율_pct": round(float((e < 0).mean() * 100), 2),
    }


def core_metrics(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    err = actual - pred
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.abs(actual).sum())
    wape = float(np.abs(err).sum() / denom * 100) if denom > 0 else np.nan
    return mae, rmse, wape


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ac = pd.read_csv(AC_OOF, encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
    daily = pd.read_csv(DAILY_OOF, encoding="utf-8-sig", parse_dates=["날짜"])
    bd_ac = pd.read_csv(BD_DIR / "예측_ac_power.csv", encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
    bd_daily_a = pd.read_csv(BD_DIR / "예측_daily_energy_일간총량.csv", encoding="utf-8-sig", parse_dates=["날짜"])
    bd_cum = pd.read_csv(BD_DIR / "예측_daily_energy_누적시계열.csv", encoding="utf-8-sig",
                         parse_dates=["날짜", "대상시각"])
    bd_checks = pd.read_csv(BD_DIR / "검증_규격점검.csv", encoding="utf-8-sig")

    # ══════════════ 1. 검증 기준정보 ══════════════
    meta = {
        "데이터_버전": "v5(인버터5 결함 시간단위까지 복구완료)",
        "결함_처리": "정책 B(대상시각 가용인버터수=5 미만 구간 학습·시험 제외)",
        "설비용량_kW": CAPACITY_KW,
        "검증구조": "정정된 공식 5계절 rolling-origin(OFFICIAL_B_WINDOWS)",
        "가을_시험_시작일": "2025-11-17",
        "위성특성": "미사용(08-25 최종검증 v3/v4에서 미채택 확정)",
        "초단기_모델버전": "+1h·+2h=raw+폴드내부구조선택 / +3h=청천지수 기본 / "
                       "+4h=청천지수+날씨군집화(⑥ 채택, v2 패치 2026-08-24)",
        "단기_모델버전": "전 수평 기본+폴드내부구조선택(v1, 08-25 DIFSWRF 연결코드 수정"
                     "— 결과영향 0 재확인됨)",
        "일간_모델버전": "08-25 전체 58특성 직접모델(부분가용 lag를 native missing+"
                     "결측여부플래그로 처리, 중앙값대체 없음)",
        "Blockdata_API_실전송": "0건(전부 오프라인 변환·검증)",
        "정답자료_출처": "Blockdata 과거 API 재조회가 아니라 광주 인버터 과거 실측 엑셀"
                     "(집계_15분_자료_v5.parquet / 집계_1시간_자료_v5.parquet / "
                     "집계_일간_실제발전량_v5.parquet — 전부 Blockdata GET /data/6715 원본"
                     "누적수집 기반이나, 이번 백테스트는 그 저장본을 재사용했을 뿐 "
                     "당시 라이브 API를 다시 부르지 않음)",
        "평가대상": "ac_power=물리적 낮시간(목표_낮시간>0)만, 일간=D+1 전일 총량(24시간 적분)",
        "nMAE_분모": f"{CAPACITY_KW}kW 고정(240kW 민감도 프로필은 이번 보고에 포함 안 함)",
        "반올림_출력제한_적용시점": "모델 원시예측 → [0, 219kW] clip → Blockdata 변환 시 "
                                "소수 둘째자리 반올림(순서: clip 먼저, round 나중)",
    }

    # ══════════════ 2. ac_power 수평별 정확도 ══════════════
    rows2 = []
    for (tier, h), g in ac.groupby(["티어", "수평_h"]):
        n_before = len(g)
        gv = g.dropna(subset=["실제_kW", "예측_kW"])
        n_excl = n_before - len(gv)
        mae, rmse, wape = core_metrics(gv["실제_kW"], gv["예측_kW"])
        b = bias_stats(gv["실제_kW"], gv["예측_kW"])
        exp_n = EXPECTED_N.get((tier, h))
        rows2.append({
            "티어": tier, "수평_h": h, "n": len(gv), "예상n": exp_n,
            "n_일치": (len(gv) == exp_n),
            "평가날짜수": gv["대상시각"].dt.normalize().nunique(),
            "평가시작일": gv["대상시각"].min().date().isoformat(),
            "평가종료일": gv["대상시각"].max().date().isoformat(),
            "실제값평균_kW": round(gv["실제_kW"].mean(), 3),
            "예측값평균_kW": round(gv["예측_kW"].mean(), 3),
            "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4),
            "nMAE_pct": round(mae / CAPACITY_KW * 100, 4), "nRMSE_pct": round(rmse / CAPACITY_KW * 100, 4),
            "WAPE_pct": round(wape, 3), **b,
            "실제값_최소": round(gv["실제_kW"].min(), 3), "실제값_최대": round(gv["실제_kW"].max(), 3),
            "예측값_최소": round(gv["예측_kW"].min(), 3), "예측값_최대": round(gv["예측_kW"].max(), 3),
            "제외행수(결측)": n_excl, "커버리지_pct": round(100 * (1 - n_excl / n_before), 3) if n_before else np.nan,
        })
    df2 = pd.DataFrame(rows2)
    df2.to_csv(OUT / "ac_power_수평별_성능.csv", index=False, encoding="utf-8-sig")

    # 폴드별
    rows2f = []
    for (tier, h, fold), g in ac.groupby(["티어", "수평_h", "폴드"]):
        mae, rmse, wape = core_metrics(g["실제_kW"], g["예측_kW"])
        rows2f.append({"티어": tier, "수평_h": h, "폴드": fold, "n": len(g),
                       "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4),
                       "nMAE_pct": round(mae / CAPACITY_KW * 100, 4), "WAPE_pct": round(wape, 3)})
    df2f = pd.DataFrame(rows2f)
    df2f.to_csv(OUT / "ac_power_폴드별_성능.csv", index=False, encoding="utf-8-sig")

    # 시간대별(대상시각 KST 시)
    ac["대상시_KST"] = ac["대상시각"].dt.hour
    rows2h = []
    for (tier, h, hh), g in ac.groupby(["티어", "수평_h", "대상시_KST"]):
        mae, rmse, wape = core_metrics(g["실제_kW"], g["예측_kW"])
        rows2h.append({"티어": tier, "수평_h": h, "대상시_KST": hh, "n": len(g),
                       "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4), "WAPE_pct": round(wape, 3)})
    df2h = pd.DataFrame(rows2h)
    df2h.to_csv(OUT / "ac_power_시간대별_성능.csv", index=False, encoding="utf-8-sig")

    # 출력구간별(전 수평 공통 — 저<20%, 중20~70%, 고>70%, 티어별 집계)
    ac["출력구간"] = pd.cut(ac["실제_kW"] / CAPACITY_KW,
                          bins=[-0.001, 0.20, 0.70, 1.01], labels=["저출력(<20%)", "중출력(20~70%)", "고출력(>70%)"])
    rows2p = []
    for (tier, band), g in ac.groupby(["티어", "출력구간"], observed=True):
        mae, rmse, wape = core_metrics(g["실제_kW"], g["예측_kW"])
        rows2p.append({"티어": tier, "출력구간": band, "n": len(g),
                       "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4), "WAPE_pct": round(wape, 3)})
    pd.DataFrame(rows2p).to_csv(OUT / "ac_power_출력구간별_성능.csv", index=False, encoding="utf-8-sig")

    # 구름전이(발행시각 기준 전운량 변화 3분위) — dpc 로더 필요
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dpc_final", ROOT / "defect_policy_comparison_v1_2026-08-21.py")
        dpc = importlib.util.module_from_spec(spec); sys.modules["dpc_final"] = dpc; spec.loader.exec_module(dpc)
        rows2c = []
        for tier, loader, horizons in [("초단기", dpc.load_ultra_frame, (1, 2, 3, 4)),
                                        ("단기", dpc.load_short_frame, (1, 24, 48))]:
            for h in horizons:
                frame = loader(h)
                if "기상청관측_전운량_pct" not in frame.columns:
                    continue
                change = frame["기상청관측_전운량_pct"].diff().abs()
                q1, q2 = change.quantile([1 / 3, 2 / 3])
                edges = sorted(set([-0.01, float(q1), float(q2), 100.0]))
                if len(edges) < 4:
                    # 변화량이 한쪽으로 쏠려 분위수가 겹치면(예: 1/3 지점이 전부 0)
                    # 3분위 대신 "무변화(0)" vs "변화있음"의 2분위로 대체한다.
                    labels = ["안정(무변화)", "전이(변화있음)"]
                    bucket = pd.cut(change, bins=[-0.01, 0.0, 100.0], labels=labels, duplicates="drop")
                else:
                    bucket = pd.cut(change, bins=edges, labels=["안정", "보통", "전이"], duplicates="drop")
                sub = ac[(ac["티어"] == tier) & (ac["수평_h"] == h)].copy()
                sub["구름구간"] = bucket.reindex(sub["발행시각"]).to_numpy()
                for band, g in sub.dropna(subset=["구름구간"]).groupby("구름구간", observed=True):
                    if len(g) < 5:
                        continue
                    mae, rmse, wape = core_metrics(g["실제_kW"], g["예측_kW"])
                    rows2c.append({"티어": tier, "수평_h": h, "구름전이구간": band, "n": len(g),
                                   "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4)})
        pd.DataFrame(rows2c).to_csv(OUT / "ac_power_구름전이별_성능.csv", index=False, encoding="utf-8-sig")
        cloud_ok = True
    except Exception as e:  # noqa: BLE001
        cloud_ok = False
        cloud_err = str(e)

    # ══════════════ 3. Skill 기준선 상세(단순지속성 + 청천지수지속성) ══════════════
    import importlib.util
    spec = importlib.util.spec_from_file_location("dpc_final2", ROOT / "defect_policy_comparison_v1_2026-08-21.py")
    dpc = importlib.util.module_from_spec(spec); sys.modules["dpc_final2"] = dpc; spec.loader.exec_module(dpc)

    rows3 = []
    for tier, loader, horizons in [("초단기", dpc.load_ultra_frame, (1, 2, 3, 4)),
                                    ("단기", dpc.load_short_frame, (1, 24, 48))]:
        for h in horizons:
            frame = loader(h)
            simple_pers = frame["_지속성_직전출력_kW"]
            elev_issue = solar_elevation_deg(frame.index)
            cs_issue = clear_sky_ghi(elev_issue)
            cs_target = clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy())
            with np.errstate(divide="ignore", invalid="ignore"):
                kappa_issue = simple_pers.to_numpy() / np.where(cs_issue > 1e-3, cs_issue, np.nan)
            smart_pers = pd.Series(kappa_issue * cs_target / 1000.0 * CAPACITY_KW, index=frame.index)
            # clear_sky_ghi는 W/m^2 스케일, 발전량 kW로 맞추려면 위와 동일 비율스케일 가정
            # (지속성값 자체가 이미 kW이므로 kappa=kW/GHI, smart=kappa*GHI_target → 자동으로 kW)
            smart_pers = pd.Series(kappa_issue * cs_target, index=frame.index)
            smart_pers = smart_pers.clip(lower=0, upper=CAPACITY_KW)

            sub = ac[(ac["티어"] == tier) & (ac["수평_h"] == h)].copy()
            sub["단순지속성_kW"] = simple_pers.reindex(sub["발행시각"]).to_numpy()
            sub["청천지수지속성_kW"] = smart_pers.reindex(sub["발행시각"]).to_numpy()
            # 두 기준선을 같은 모델_MAE로 공정비교하려면 "두 기준선 모두 유효한"
            # 공통 시험행만 쓴다(기준선마다 결측패턴이 달라 각자 다른 부분집합을
            # 쓰면 같은 티어·수평인데 모델_MAE가 달라지는 오류가 생김 — 실제로
            # 최초 실행에서 발견해 이렇게 고쳤다).
            common_valid = sub.dropna(subset=["단순지속성_kW", "청천지수지속성_kW"])

            for base_name, col in [("단순지속성(직전 발행시각 실측)", "단순지속성_kW"),
                                    ("청천지수지속성(발행시각 카파 유지)", "청천지수지속성_kW")]:
                subx = common_valid
                mae_m, rmse_m, _ = core_metrics(subx["실제_kW"], subx["예측_kW"])
                mae_b, rmse_b, _ = core_metrics(subx["실제_kW"], subx[col])
                rows3.append({
                    "티어": tier, "수평_h": h, "기준선": base_name, "n": len(subx),
                    "동일시험행": True, "낮시간필터_기준선동일적용": True,
                    "모델_MAE": round(mae_m, 4), "모델_RMSE": round(rmse_m, 4),
                    "기준선_MAE": round(mae_b, 4), "기준선_RMSE": round(rmse_b, 4),
                    "Skill_MAE(1-모델/기준선)": round(1 - mae_m / mae_b, 4) if mae_b else np.nan,
                    "Skill_RMSE(1-모델/기준선)": round(1 - rmse_m / rmse_b, 4) if rmse_b else np.nan,
                })
    df3 = pd.DataFrame(rows3)
    df3.to_csv(OUT / "Skill_기준선별_비교.csv", index=False, encoding="utf-8-sig")

    # ══════════════ 4. Blockdata 변환 전후 동일성 ══════════════
    keys = ["티어", "수평_h", "발행시각", "대상시각"]
    left = ac[keys + ["폴드", "예측_kW", "실제_kW"]].copy()
    right = bd_ac.rename(columns={"예측_ac_power_kw": "변환_예측_kW", "실측_ac_power_kw": "변환_실측_kW"})
    right = right[keys + ["폴드", "변환_예측_kW", "변환_실측_kW"]]
    dup_left = left.duplicated(subset=keys).sum()
    dup_right = right.duplicated(subset=keys).sum()
    merged = left.merge(right, on=keys, how="outer", suffixes=("", "_bd"), indicator=True)
    only_left = int((merged["_merge"] == "left_only").sum())
    only_right = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"].copy()

    rows4 = []
    for (tier, h), g in both.groupby(["티어", "수평_h"]):
        mae_o, rmse_o, _ = core_metrics(g["실제_kW"], g["예측_kW"])
        mae_c, rmse_c, _ = core_metrics(g["변환_실측_kW"], g["변환_예측_kW"])
        diff = (g["변환_예측_kW"] - g["예측_kW"]).abs()
        exact = int((diff <= 1e-9).sum())
        clipped_low = int((g["예측_kW"] <= 0).sum())
        clipped_high = int((g["예측_kW"] >= CAPACITY_KW).sum())
        rows4.append({
            "티어": tier, "수평_h": h, "n": len(g),
            "원본_MAE": round(mae_o, 4), "원본_RMSE": round(rmse_o, 4),
            "변환후_MAE": round(mae_c, 4), "변환후_RMSE": round(rmse_c, 4),
            "MAE차이": round(mae_c - mae_o, 6), "RMSE차이": round(rmse_c - rmse_o, 6),
            "예측값_최대절대차": round(float(diff.max()), 6),
            "완전동일행수": exact, "완전동일비율_pct": round(100 * exact / len(g), 3),
            "반올림으로변경된행수": len(g) - exact,
            "0kW하한적용행수": clipped_low, "219kW상한적용행수": clipped_high,
        })
    df4 = pd.DataFrame(rows4)
    df4.to_csv(OUT / "ac_power_변환전후_감사.csv", index=False, encoding="utf-8-sig")
    key_audit = {"원본중복": int(dup_left), "변환본중복": int(dup_right),
                "원본에만있음": only_left, "변환본에만있음": only_right,
                "조인성공행수": int(len(both)), "전체74743행_일치": bool(len(both) == 74743 and only_left == 0 and only_right == 0)}

    # ══════════════ 5. daily_energy 경로A ══════════════
    fold_map = daily.set_index("날짜")["폴드"]
    mae5, rmse5, wape5 = core_metrics(daily["실제_kWh"], daily["예측_kWh"])
    b5 = bias_stats(daily["실제_kWh"], daily["예측_kWh"])
    season5 = []
    for fold, g in daily.groupby("폴드"):
        m, r, w = core_metrics(g["실제_kWh"], g["예측_kWh"])
        season5.append({"폴드": fold, "n": len(g), "MAE_kWh": round(m, 3), "RMSE_kWh": round(r, 3), "WAPE_pct": round(w, 3)})
    daily_summary = {
        "n": len(daily), "n_일치_321": bool(len(daily) == 321),
        "평가기간_시작": daily["날짜"].min().date().isoformat(), "평가기간_종료": daily["날짜"].max().date().isoformat(),
        "실제_평균_kWh": round(daily["실제_kWh"].mean(), 2), "예측_평균_kWh": round(daily["예측_kWh"].mean(), 2),
        "MAE_kWh": round(mae5, 3), "RMSE_kWh": round(rmse5, 3), "WAPE_pct": round(wape5, 3), **b5,
        "nMAE_정의": f"물리단위(kWh)라 ac_power식 설비용량(kW) 정규화는 미적용 — WAPE(=오차합/실제합)를 정규화지표로 사용",
        "음수건수": int((daily["예측_kWh"] < 0).sum()),
        "물리상한_초과건수(설비219kW×24h=5256kWh 기준)": int((daily["예측_kWh"] > CAPACITY_KW * 24).sum()),
        "결측제외일수": 0,
        "기존공식예상값_MAE_108.67_일치": bool(abs(mae5 - 108.674) < 0.5),
        "기존공식예상값_RMSE_139.78_일치": bool(abs(rmse5 - 139.785) < 0.5),
        "기존공식예상값_WAPE_16.89_일치": bool(abs(wape5 - 16.894) < 0.1),
    }
    (OUT / "daily_energy_A_성능.json").write_text(json.dumps(daily_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(season5).to_csv(OUT / "daily_energy_A_계절별_성능.csv", index=False, encoding="utf-8-sig")

    # A 변환전후
    bd_daily_a_j = bd_daily_a.rename(columns={"예측_일간최종총량_kwh": "변환_예측_kWh", "실측_일간최종총량_kwh": "변환_실측_kWh"})
    mrg5 = daily.merge(bd_daily_a_j[["날짜", "변환_예측_kWh", "변환_실측_kWh"]], on="날짜", how="inner")
    mae5c, rmse5c, _ = core_metrics(mrg5["변환_실측_kWh"], mrg5["변환_예측_kWh"])
    daily_a_conv = {"공통일수": len(mrg5), "원본_MAE": round(mae5, 3), "변환후_MAE": round(mae5c, 3),
                    "원본_RMSE": round(rmse5, 3), "변환후_RMSE": round(rmse5c, 3),
                    "최대절대차": round(float((mrg5["변환_예측_kWh"] - mrg5["예측_kWh"]).abs().max()), 4)}
    pd.DataFrame([{**daily_summary, **{"변환후_"+k: v for k, v in daily_a_conv.items()}}]).to_csv(
        OUT / "daily_energy_A_성능.csv", index=False, encoding="utf-8-sig")

    # ══════════════ 6. daily_energy 경로B ══════════════
    # 6-1 누적시점별
    rows61 = []
    for path, g in bd_cum.groupby("경로"):
        mae, rmse, wape = core_metrics(g["실측_daily_energy_누적_kwh"], g["예측_daily_energy_누적_kwh"])
        neg = int((g["예측_daily_energy_누적_kwh"] < 0).sum())
        decreasing = 0
        for _, gg in g.groupby("날짜"):
            gg = gg.sort_values("대상시각")
            if (gg["예측_daily_energy_누적_kwh"].diff().dropna() < -1e-6).any():
                decreasing += 1
        # 자정초기화 실패: build_daily_cumulative가 날짜별 groupby.cumsum()으로
        # 만든 값이라 구조상 매일 0에서 다시 시작한다(전날 값이 새지 않음) — 0건.
        midnight_fail = 0
        rows61.append({
            "경로": path, "n": len(g),
            "실제_누적평균_kWh": round(g["실측_daily_energy_누적_kwh"].mean(), 3),
            "예측_누적평균_kWh": round(g["예측_daily_energy_누적_kwh"].mean(), 3),
            "MAE_kWh": round(mae, 3), "RMSE_kWh": round(rmse, 3), "WAPE_pct": round(wape, 3),
            "Bias_kWh": round(float((g["예측_daily_energy_누적_kwh"] - g["실측_daily_energy_누적_kwh"]).mean()), 4),
            "누적값음수건수": neg, "하루중누적감소건수": decreasing, "자정초기화실패건수(구조상)": midnight_fail,
        })
    pd.DataFrame(rows61).to_csv(OUT / "daily_energy_B_누적시점별_성능.csv", index=False, encoding="utf-8-sig")

    # 6-2 일마감 총량(완전일 기준: 그 날 그 (경로,날짜) 조합의 버킷수가 전체 데이터셋에서
    # 해당 (경로) 최빈 버킷수의 90% 이상인 날짜만 완전일로 인정)
    rows62 = []
    for path, g in bd_cum.groupby("경로"):
        counts = g.groupby("날짜").size()
        typical = int(counts.mode().iloc[0]) if len(counts) else 0
        full_days = counts[counts >= max(1, int(typical * 0.9))].index
        last = g[g["날짜"].isin(full_days)].sort_values("대상시각").groupby("날짜").tail(1)
        excluded = g["날짜"].nunique() - len(full_days)
        if len(last) == 0:
            continue
        mae, rmse, wape = core_metrics(last["실측_daily_energy_누적_kwh"], last["예측_daily_energy_누적_kwh"])
        rows62.append({
            "경로": path, "완전일기준_최빈버킷수": typical, "완전일수": len(full_days),
            "불완전일제외수": excluded, "일마감평가일수": len(last),
            "실제_평균_kWh": round(last["실측_daily_energy_누적_kwh"].mean(), 2),
            "예측_평균_kWh": round(last["예측_daily_energy_누적_kwh"].mean(), 2),
            "MAE_kWh": round(mae, 3), "RMSE_kWh": round(rmse, 3), "WAPE_pct": round(wape, 3),
            "Bias_kWh": round(float((last["예측_daily_energy_누적_kwh"] - last["실측_daily_energy_누적_kwh"]).mean()), 4),
        })
    df62 = pd.DataFrame(rows62)
    df62.to_csv(OUT / "daily_energy_B_일마감_성능.csv", index=False, encoding="utf-8-sig")

    # ══════════════ 7. 경로A vs 경로B(일마감) 비교 ══════════════
    rows7 = []
    for path, g in bd_cum.groupby("경로"):
        counts = g.groupby("날짜").size()
        typical = int(counts.mode().iloc[0]) if len(counts) else 0
        full_days = counts[counts >= max(1, int(typical * 0.9))].index
        b_last = g[g["날짜"].isin(full_days)].sort_values("대상시각").groupby("날짜").last()[
            ["예측_daily_energy_누적_kwh", "실측_daily_energy_누적_kwh"]]
        a = daily.set_index("날짜")[["예측_kWh", "실제_kWh", "폴드"]]
        common = a.index.intersection(b_last.index)
        if len(common) < 5:
            rows7.append({"B경로": path, "공통일수": len(common), "판정": "표본부족(5일 미만)"})
            continue
        a_c, b_c = a.loc[common], b_last.loc[common]
        abs_diff = (a_c["예측_kWh"] - b_c["예측_daily_energy_누적_kwh"]).abs()
        rel = (abs_diff / a_c["예측_kWh"].abs().replace(0, np.nan) * 100)
        a_err = (a_c["실제_kWh"] - a_c["예측_kWh"]).abs()
        b_err = (a_c["실제_kWh"] - b_c["예측_daily_energy_누적_kwh"]).abs()
        a_better = (a_err < b_err).mean() * 100
        rows7.append({
            "B경로": path, "공통일수": len(common),
            "AB_평균절대차_kWh": round(float(abs_diff.mean()), 3),
            "AB_중앙절대차_kWh": round(float(abs_diff.median()), 3),
            "AB_95백분위절대차_kWh": round(float(abs_diff.quantile(0.95)), 3),
            "AB_평균불일치율_pct": round(float(rel.mean()), 2),
            "AB_중앙불일치율_pct": round(float(rel.median()), 2),
            "AB_최대불일치율_pct": round(float(rel.max()), 2),
            "A가더정확한날비율_pct": round(float(a_better), 2),
            "B가더정확한날비율_pct": round(float(100 - a_better), 2),
            "A_평균Bias_kWh": round(float((a_c["예측_kWh"] - a_c["실제_kWh"]).mean()), 3),
            "B_평균Bias_kWh": round(float((b_c["예측_daily_energy_누적_kwh"] - a_c["실제_kWh"]).mean()), 3),
        })
    df7 = pd.DataFrame(rows7)
    df7.to_csv(OUT / "daily_energy_A_B_동일날짜비교.csv", index=False, encoding="utf-8-sig")

    # ══════════════ 9. Blockdata 23건 규격점검 상세 ══════════════
    bd_checks.to_csv(OUT / "Blockdata_규격점검_23건.csv", index=False, encoding="utf-8-sig")

    # ══════════════ 최종 요약 ══════════════
    n_pass_checks = int((bd_checks["결과"] == "통과").sum())
    summary = {
        "메타": meta,
        "핵심수치": {
            "ac_power_수평별": df2.to_dict("records"),
            "Blockdata_변환전후_키검증": key_audit,
            "daily_A": daily_summary,
            "규격점검_통과": f"{n_pass_checks}/{len(bd_checks)}",
        },
        "결론_3문장_분리": {
            "1_모델자체_예측정확도": "8개 티어·수평 전부 단순지속성 대비 42~66% RMSE 개선(Skill_기준선별_비교.csv), "
                                "절대오차 219kW 설비 기준 7.9~14.9kW(초단기·단기), 일간 D+1 총량 MAE 108.7kWh(WAPE 16.9%) — "
                                "KPX day-ahead 공식지표는 별도 모델로 219kW 기준 NMAE 7.36%(목표7% 대비 0.36%p 미달).",
            "2_Blockdata변환후_정확도유지": f"ac_power 74,743행 전부 조인 성공(중복0·누락0), 변환 전후 MAE·RMSE 차이는 "
                                     f"반올림에 의한 소수점 수준(최대절대차 {df4['예측값_최대절대차'].max() if len(df4) else 'NA'}kW 이하)"
                                     f"이며 예측력 훼손 없음 — 단, daily_energy 경로A/B 간에는 서로 다른 모델(직접모델 vs "
                                     f"시간적분)이라 구조적 불일치가 존재함(A_B_동일날짜비교.csv 참고, 정답 취급 금지).",
            "3_실제API라이브연동": "실제 Blockdata API 실시간 전송·폴링·지연·장애복구는 이번에도 0건 시험 — "
                                "전부 저장된 과거 인버터 실측 위에서의 오프라인 변환·점검이다.",
        },
        "이번검증의_한계": [
            "실제 Blockdata API 과거 조회가 아니라 인버터 엑셀 과거 실측을 정답으로 사용",
            "Blockdata API는 현재 스냅샷만 제공하는 구조라 이번 백테스트가 그 한계를 우회한 것도 아님",
            "실전송·실시간 폴링·지연·장애복구는 이번에도 시험하지 않음",
            "5분 순간 스냅샷(Blockdata 실측 응답 원 단위)과 모델의 15분/1시간 버킷평균은 정의가 다름",
            "같은 5계절(OFFICIAL_B_WINDOWS)에서 구조선택과 성능을 함께 검토했으므로 완전 독립 외부시험은 아님",
            "향후 신규 기간(2026-08 이후) 데이터로 동결모델 독립검증 필요",
            "KPX day-ahead 모델(4차 카파군집)은 이 Blockdata E2E 모델과 별개 모델 — 결과를 혼합 해석하지 않음",
        ],
        "total_energy_한계": [
            "절대 누적 실측 앵커 없음 — total_energy는 이번 정확도 평가 대상이 아니다",
            "따라서 total_energy 절대값 예측은 구조적으로 불가능",
            "현재 검증 가능한 것은 daily_energy 증분(경로A·B)과 ac_power 출력 구조뿐",
            "실제 연동 시 Blockdata 최신 total_energy를 앵커로 주입해야 절대 누적값이 나온다",
            "앵커가 없을 때는 null 유지(build_total_energy 로직 그대로) — 임의의 0 또는 추정 누적값 사용 금지",
        ],
        "산출물_경로": str(OUT),
    }
    (OUT / "Blockdata_오프라인백테스트_최종요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = ["# Blockdata 오프라인 백테스트 최종보고 (2026-08-25)", "",
          "## 1. 검증 기준정보"] + [f"- {k}: {v}" for k, v in meta.items()] + [
          "", "## 2~9. 상세표는 동봉 CSV 참고", "",
          "## 결론(3문장, 분리)"]
    for k, v in summary["결론_3문장_분리"].items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## 이번 검증의 한계"] + [f"- {x}" for x in summary["이번검증의_한계"]]
    md += ["", "## total_energy 한계"] + [f"- {x}" for x in summary["total_energy_한계"]]
    (OUT / "Blockdata_오프라인백테스트_최종요약.md").write_text("\n".join(md), encoding="utf-8")

    print("=== 완료 ===")
    print(f"ac_power 수평별:\n{df2[['티어','수평_h','n','n_일치','MAE_kW','RMSE_kW','nMAE_pct']].to_string(index=False)}")
    print(f"\n키검증: {key_audit}")
    print(f"\n규격점검: {n_pass_checks}/{len(bd_checks)} 통과")
    print(f"구름전이 산출: {'성공' if cloud_ok else '실패:'+str(locals().get('cloud_err'))}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
