"""김제 과거 1시간 발전량·ASOS·고정tm D+1 NWP 결합(API 호출 없음).

부안 combine_buan_history_power_asos_nwp_v1_2026-08-28.py와 완전히 동일한
방법론 - 경로만 김제로 교체, radiation_cleaned은 김제가 실제로 수집한
변수(DSWRF만, DSWRFLX/DIFSWRF는 3단계에서 제외 확정)로 축소.

★선행조건★: 이 스크립트는 NWP 백필(710일)이 완료돼야 실행 가능하다.
문법검사만 통과한 상태로 미리 작성해뒀다 - NWP/GRID 백필이 끝나는 대로
바로 실행할 수 있게 하기 위함(결합 자체는 GRID 없이 ASOS+NWP만으로도
가능하므로, GRID가 더 늦게 끝나면 이 스크립트부터 먼저 돌리고
join_gimje_grid_forecast_v1_2026-08-31.py로 이어붙이면 된다).

입력용 ASOS는 발행시각까지 관측된 최신값만 사용한다. 대상시각 ASOS는
사후 분석용 reference 접두어로만 보존하며 모델 입력으로 사용하지 않는다.
과거 NWP 파일에는 당시 API 실제수신시각이 없으므로 모델런·공식 발행시각은
검증하되 실제수신시각 검증 불가 상태를 산출물에 명시한다(부안과 동일 한계).

★부안 대비 개선점★: 부안 스크립트는 asos_upstream_missing_times를
하드코딩했었다(그때그때 확인한 값을 다시 못 갱신하는 리스크). 김제는
ASOS 백필의 결측시간_감사.csv를 실행시점에 직접 읽어 동적으로 채운다.

★결함구간 컬럼★: 삭제하지 않고 is_defect_period 불리언 컬럼으로만
표시한다(행 자체를 여기서 지우지 않음) - 부안에서 "결합 경로는 결함구간
미제외"였는데 그 사실을 모델 스크립트 쪽에서 매번 기억해야 했던 게
Codex 독립감사에서 걸린 문제였다. 김제는 컬럼으로 항상 보이게 해서
이후 어떤 모델 스크립트든 필터를 빠뜨리면 바로 티가 나게 한다.
target_time_kst 기준으로 판정한다(issue_day 기준으로 걸면 부안에서
실제로 걸렸던 하루밀림 버그가 재발하므로 반드시 target_time_kst 기준).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

POWER = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\시간집계_v1_2026-08-31"
    r"\김제_발전소_1시간_공식후보.parquet"
)
ASOS = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31\ASOS"
    r"\기상청_ASOS243_시간환경_20240825_20260804.csv"
)
NWP_DIR = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31\NWP"
)
GAP_AUDIT_CSV = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\기상과거백필_v1_2026-08-31\ASOS"
    r"\결측시간_감사.csv"
)
DEFECT_CONFIG = Path(__file__).resolve().parent / "config" / "김제_전처리_규칙_v1_2026-08-31.json"
OUT = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\김제\과거발전_기상결합_v1_2026-08-31"
)

RADIATION_VARS = ("DSWRF",)  # 김제는 DSWRFLX/DIFSWRF 미수집(3단계 확정)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--power", type=Path, default=POWER)
    p.add_argument("--asos", type=Path, default=ASOS)
    p.add_argument("--nwp-dir", type=Path, default=NWP_DIR)
    p.add_argument("--gap-audit-csv", type=Path, default=GAP_AUDIT_CSV)
    p.add_argument("--defect-config", type=Path, default=DEFECT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=OUT)
    return p.parse_args()


def load_defect_window(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not path.is_file():
        raise FileNotFoundError(f"결함구간 확정 설정 없음: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    dp = cfg["defect_period"]
    return pd.Timestamp(dp["start_inclusive"]), pd.Timestamp(dp["end_exclusive"])


def main():
    a = cli()
    files = sorted(a.nwp_dir.glob("김제_익일예보_고정tm_*.csv"))
    if not a.power.exists():
        raise FileNotFoundError(a.power)
    if not a.asos.exists():
        raise FileNotFoundError(a.asos)
    if len(files) != 1:
        raise RuntimeError(f"NWP 통합 CSV는 정확히 1개여야 함(백필 완료 전이면 0개일 수 있음): {files}")

    defect_start, defect_end_excl = load_defect_window(a.defect_config)

    power = pd.read_parquet(a.power)
    power["target_time_kst"] = pd.to_datetime(power["grid_time_kst"], errors="raise")
    if power["target_time_kst"].duplicated().any():
        raise RuntimeError("발전량 대상시각 중복")

    asos = pd.read_csv(a.asos)
    # CSV 시각은 +09:00 offset 포함, NWP·발전량은 KST naive이므로 같은 축으로 통일한다.
    asos["observation_time_kst"] = pd.to_datetime(asos["시각"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    if asos["observation_time_kst"].isna().any():
        raise RuntimeError("ASOS 시각 파싱 실패")
    if sorted(pd.to_numeric(asos["지점번호"], errors="coerce").dropna().astype(int).unique()) != [243]:
        raise RuntimeError("ASOS 243 외 지점 혼입")
    if asos["observation_time_kst"].duplicated().any():
        raise RuntimeError("ASOS 관측시각 중복")

    nwp = pd.read_csv(files[0])
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
            "collect_gwangju_nwp_dayahead_fixed_tm_710d_v2_2026-08-20.py(--site-tag 김제)를 "
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

    # 결함구간 표시(행 삭제 아님) - target_time_kst 기준, half-open[start,end).
    joined["is_defect_period"] = (joined["target_time_kst"] >= defect_start) & (joined["target_time_kst"] < defect_end_excl)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(a.output_dir / "김제_과거발전_ASOS_NWP_결합.parquet", index=False)

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
        "defect_period_window": {"start_inclusive": str(defect_start.date()), "end_exclusive": str(defect_end_excl.date())},
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
    (a.output_dir / "김제_과거결합_요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
