"""영광 과거 1시간 발전량·ASOS·고정tm D+1 NWP 결합(API 호출 없음).

김제 combine_gimje_history_power_asos_nwp_v1_2026-08-31.py와 완전히 동일한
방법론 - 경로만 영광으로 교체(ASOS252, NWP 5변수는 김제와 동일 DSWRF만
수집됨). 사용자 09-03 지시("영광 데이터셋 구축, 코드만 주면 내가 알아서
돌릴게")로 작성 - 이 스크립트는 실행하지 않고 코드만 준비한다.

★영광 전용 주의사항(김제와 다른 점)★
1. **결함구간 미확정**: 김제는 이미 확정된 결함구간(2025-05-28~06-11)이
   있어 `config/김제_전처리_규칙_v1_2026-08-31.json`을 필수로 요구했다.
   영광은 아직 그런 감사가 없었다(`시간정렬_v1_2026-09-01` 산출물의
   `meter_reset_events_total: 0`만 확인됨 - 이건 계량기 리셋이 없었다는
   것이지 "결함구간이 아예 없다"는 걸 증명하지 않는다). 이 스크립트는
   `--defect-config`를 **선택적**으로 만들어, 안 주면 `is_defect_period`
   전부 False로 채우고 요약 JSON에 "미감사" 경고를 명시한다 - 없는 근거를
   지어내지 않는다.
2. **인버터 13기, 정격용량 미검증**: `시간정렬_v1_2026-09-01` 요약에
   "inverter_capacity_verified: false"로 이미 명시돼 있음 - 이 스크립트는
   발전량·기상만 결합하고 용량 관련 판단은 하지 않는다(관련 없음).
3. NWP는 김제와 동일하게 DSWRF 1종만 수집(DSWRFLX·DIFSWRF 없음).

★선행조건★: NWP·GRID 백필 710일 전부 완료(이미 완료, 09-03 12:39
AGENTS.md 참고) - 이 스크립트는 그 완료를 실행 시점에 재검증한다(조용히
부분데이터로 결합하지 않음, 김제와 동일 안전장치).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

POWER = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\시간집계_v1_2026-09-01"
    r"\영광_발전소_1시간_공식후보.parquet"
)
ASOS = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31\ASOS"
    r"\기상청_ASOS252_시간환경_20240825_20260804.csv"
)
NWP_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31\NWP"
    r"\영광_익일예보_고정tm_DSWRF_TCDC_LCDC_MCDC_HCDC_710일.csv"
)
GAP_AUDIT_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\기상과거백필_v1_2026-08-31\ASOS"
    r"\결측시간_감사.csv"
)
DEFECT_CONFIG = Path(__file__).resolve().parent / "config" / "영광_전처리_규칙_v1_2026-09-03.json"
OUT = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\영광\과거발전_기상결합_v1_2026-09-03"
)
ASOS_STATION = 252

RADIATION_VARS = ("DSWRF",)  # 영광은 김제와 동일하게 DSWRFLX/DIFSWRF 미수집


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--power", type=Path, default=POWER)
    p.add_argument("--asos", type=Path, default=ASOS)
    p.add_argument("--nwp-csv", type=Path, default=NWP_CSV)
    p.add_argument("--gap-audit-csv", type=Path, default=GAP_AUDIT_CSV)
    p.add_argument("--defect-config", type=Path, default=DEFECT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=OUT)
    return p.parse_args()


def load_defect_window(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, bool]:
    """반환값 3번째(audited)가 False면 감사 자체가 없었다는 뜻 - 호출부가
    is_defect_period를 전부 False로 채우고 요약에 경고를 남긴다."""
    if not path.is_file():
        return None, None, False
    cfg = json.loads(path.read_text(encoding="utf-8"))
    dp = cfg["defect_period"]
    return pd.Timestamp(dp["start_inclusive"]), pd.Timestamp(dp["end_exclusive"]), True


def main():
    a = cli()
    if not a.power.exists():
        raise FileNotFoundError(a.power)
    if not a.asos.exists():
        raise FileNotFoundError(a.asos)
    if not a.nwp_csv.exists():
        raise FileNotFoundError(a.nwp_csv)

    defect_start, defect_end_excl, defect_audited = load_defect_window(a.defect_config)

    power = pd.read_parquet(a.power)
    power["target_time_kst"] = pd.to_datetime(power["grid_time_kst"], errors="raise")
    if power["target_time_kst"].duplicated().any():
        raise RuntimeError("발전량 대상시각 중복")

    asos = pd.read_csv(a.asos)
    asos["observation_time_kst"] = pd.to_datetime(asos["시각"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    if asos["observation_time_kst"].isna().any():
        raise RuntimeError("ASOS 시각 파싱 실패")
    if sorted(pd.to_numeric(asos["지점번호"], errors="coerce").dropna().astype(int).unique()) != [ASOS_STATION]:
        raise RuntimeError(f"ASOS {ASOS_STATION} 외 지점 혼입")
    if asos["observation_time_kst"].duplicated().any():
        raise RuntimeError("ASOS 관측시각 중복")

    nwp = pd.read_csv(a.nwp_csv)
    nwp["issue_date"] = pd.to_datetime(nwp["발표일"].astype(str), format="%Y%m%d", errors="raise")
    if "발행시각_kst" not in nwp:
        raise RuntimeError("NWP 공식 발행시각 열 없음")
    nwp["prediction_issue_time_kst"] = pd.to_datetime(nwp["발행시각_kst"].astype(str), format="%Y%m%d%H%M", errors="raise")
    if nwp["issue_date"].duplicated().any():
        raise RuntimeError("NWP 발표일 중복")
    if not nwp["수집완료"].fillna(False).astype(bool).all():
        raise RuntimeError("NWP 미완료 발표일 존재")
    expected_days = pd.date_range("2024-08-25", "2026-08-04", freq="D")
    if len(nwp) != len(expected_days):
        raise RuntimeError(
            f"NWP 백필이 아직 안 끝났습니다({len(nwp)}/{len(expected_days)}일) - "
            "710일 전부 완료한 뒤 다시 실행할 것(조용히 부분데이터로 결합하지 않음)."
        )
    if not (nwp["prediction_issue_time_kst"].dt.normalize() == nwp["issue_date"]).all():
        raise RuntimeError("NWP 발표일·발행시각 날짜 불일치")
    run = pd.to_datetime(nwp["요청_tm_utc"].astype(str), format="%Y%m%d%H%M", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    if (run > nwp["prediction_issue_time_kst"]).any():
        raise RuntimeError("NWP 모델런이 발행시각보다 미래")

    receipt_candidates = [c for c in ("실제수신시각_kst", "first_received_at_kst", "received_at_kst") if c in nwp.columns]
    if receipt_candidates:
        received = pd.to_datetime(nwp[receipt_candidates[0]], errors="raise")
        if (received > nwp["prediction_issue_time_kst"]).any():
            raise RuntimeError("NWP 실제수신시각이 예측 발행시각보다 미래")
        receipt_validation = "validated"
    else:
        receipt_validation = "unavailable_historical_archive"

    value_cols = [c for c in nwp.columns if re.search(r"_\d{2}h$", c)]
    radiation_cleaned = {v: 0 for v in RADIATION_VARS}
    records = []
    for _, d in nwp.iterrows():
        issue = d["prediction_issue_time_kst"]
        day = d["issue_date"] + pd.Timedelta(days=1)
        for h in range(0, 24, 3):
            rec = {"prediction_issue_time_kst": issue, "target_time_kst": day + pd.Timedelta(hours=h), "requested_tm_utc": d["요청_tm_utc"]}
            for c in value_cols:
                if c.endswith(f"_{h:02d}h"):
                    variable = c[:-4]
                    value = pd.to_numeric(d[c], errors="coerce")
                    if variable in radiation_cleaned and pd.notna(value) and (value < 0 or value > 2000):
                        radiation_cleaned[variable] += 1
                        value = pd.NA
                    rec["forecast_" + variable] = value
            records.append(rec)
    fc = pd.DataFrame(records)
    if fc.duplicated(["prediction_issue_time_kst", "target_time_kst"]).any():
        raise RuntimeError("NWP 발행·대상시각 중복")
    if (fc["target_time_kst"] <= fc["prediction_issue_time_kst"]).any():
        raise RuntimeError("NWP 대상시각이 발행시각 이하")

    joined = fc.merge(power.drop(columns=["grid_time_kst"]), on="target_time_kst", how="left")

    issue_asos = pd.merge_asof(
        fc[["prediction_issue_time_kst"]].drop_duplicates().sort_values("prediction_issue_time_kst"),
        asos.sort_values("observation_time_kst"),
        left_on="prediction_issue_time_kst", right_on="observation_time_kst",
        direction="backward", tolerance=pd.Timedelta("3h"),
    )
    issue_asos = issue_asos.rename(columns={c: "issue_asos_" + c for c in asos.columns if c != "observation_time_kst"})
    joined = joined.merge(issue_asos, on="prediction_issue_time_kst", how="left")

    reference = asos.rename(columns={c: "reference_target_asos_" + c for c in asos.columns if c != "observation_time_kst"}).rename(columns={"observation_time_kst": "target_time_kst"})
    joined = joined.merge(reference, on="target_time_kst", how="left")

    future_leak_asos = int((joined["observation_time_kst"] > joined["prediction_issue_time_kst"]).sum())
    future_leak_run = int((run > nwp["prediction_issue_time_kst"]).sum())
    future_leak_target = int((joined["target_time_kst"] <= joined["prediction_issue_time_kst"]).sum())
    future_leak = future_leak_asos + future_leak_run + future_leak_target
    if future_leak:
        raise RuntimeError("ASOS·NWP 발행/대상시각 미래누출")

    # 결함구간 표시(행 삭제 아님) - 감사 안 됐으면 전부 False + 요약에 경고.
    if defect_audited:
        joined["is_defect_period"] = (joined["target_time_kst"] >= defect_start) & (joined["target_time_kst"] < defect_end_excl)
    else:
        joined["is_defect_period"] = False

    a.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(a.output_dir / "영광_과거발전_ASOS_NWP_결합.parquet", index=False)

    gap_dates = []
    if a.gap_audit_csv.is_file():
        gaps = pd.read_csv(a.gap_audit_csv, encoding="utf-8-sig")
        gap_dates = sorted(gaps["날짜"].astype(str).tolist()) if "날짜" in gaps.columns else []

    forecast_cols = [c for c in joined.columns if c.startswith("forecast_")]
    summary = {
        "status": "historical_archive_join_candidate",
        "rows": len(joined),
        "issue_days": int(joined["prediction_issue_time_kst"].nunique()),
        "targets": int(joined["target_time_kst"].nunique()),
        "power_available_rows": int(joined["plant_ac_power_kw"].notna().sum()),
        "issue_asos_available_rows": int(joined["observation_time_kst"].notna().sum()),
        "defect_period_rows": int(joined["is_defect_period"].sum()),
        "defect_period_window": (
            {"start_inclusive": str(defect_start.date()), "end_exclusive": str(defect_end_excl.date())}
            if defect_audited else None
        ),
        "★defect_period_audited★": defect_audited,
        "defect_period_warning": (
            None if defect_audited else
            "영광은 아직 결함구간 감사가 없었다(계량기리셋 0건만 확인됨 - "
            "통신오류/설비이상 등 다른 유형의 결함은 미확인). is_defect_period는 "
            "전부 False로 채워졌다 - 모델링 전 인버터별 이력 감사(광주 08-27 "
            "audit_inverter_history 패턴)를 먼저 하는 걸 권장."
        ),
        "future_leak_rows": future_leak,
        "future_leak_asos_rows": future_leak_asos,
        "future_leak_nwp_run_rows": future_leak_run,
        "invalid_target_order_rows": future_leak_target,
        "nwp_actual_receipt_time_validation": receipt_validation,
        "nwp_actual_receipt_time_limitation": "과거 백필 CSV에 당시 실제수신시각이 없어 검증 불가; 라이브 SQLite 결합에서만 강제 검사" if receipt_validation != "validated" else None,
        "nwp_radiation_invalid_to_missing": radiation_cleaned,
        "forecast_missing_by_variable": {c: int(joined[c].isna().sum()) for c in forecast_cols},
        "asos_upstream_missing_dates(자동수용_결측시간_감사)": gap_dates,
        "asos_solar_supported": False,
        "reference_target_asos_is_model_input": False,
    }
    (a.output_dir / "영광_과거결합_요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
