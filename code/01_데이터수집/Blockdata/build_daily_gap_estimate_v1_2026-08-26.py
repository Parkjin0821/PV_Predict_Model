# -*- coding: utf-8 -*-
"""2026-08-05~08-24(20일) 일간발전량 구멍을 카운터뺄셈+ASOS실측일사량
배분으로 채운 v6 데이터셋 생성 — 원본 v5는 전혀 건드리지 않는다.

## 방법(사용자 승인, 08-26)
1. **20일 총량(신뢰도 높음)**: 엑셀 마지막 `누적` vs Blockdata 최초
   `total_energy`의 인버터별 차이(`check_total_energy_continuity_v1_
   2026-08-26.py`로 이미 검증 — 5대 전부 "그럴듯함" 판정, 합계
   13,294.66kWh).
2. **일별 배분(추정)**: 이 기간 ASOS 실측 일사량(`backfill_asos_gap_
   20260805_20260824_v1_2026-08-26.py`로 이미 확보, 480/480시간 성공)
   `solar_mj_m2` 일합에 비례 배분. 균등분배보다 낫지만 여전히 추정이다
   — 그래서 `추정치_여부` 플래그를 반드시 남긴다(아래 참고).

## 원본 불변 원칙
`집계_일간_실제발전량_v5.parquet`는 **한 글자도 안 건드린다**. 대신
같은 파일명으로 새 폴더(`outputs/v6_일간구멍보정_2026-08-26/`)에 저장한다
— `e2e_retrain_v5_공식B_v1_2026-08-24.py`·`daily_direct_final_audit_v1_
2026-08-25.py`가 파일명은 하드코딩하지만 디렉토리는 `V5_DIR` 변수를
쓰므로, 나중에 이 v6를 쓸 때는 그 함수들을 재구현하지 않고
`e2e.V5_DIR`을 이 새 폴더로 몽키패치했다가 복원하는 방식을 쓴다
(DSWRFLX/mean_communication_ok 제외 재검증 스크립트에서 이미 검증된
같은 패턴 — `retrain_exclude_dswrflx_v1_2026-08-26.py`의 후보목록
몽키패치 참고).

## 새로 추가하는 컬럼
- `추정치_여부`: 1=이 20일(카운터뺄셈+일사량배분 추정), 0=나머지 전부(실측)
- 그 외 원본 컬럼(`낮시간예상개수` 등 장비 커버리지 통계)은 이 20일에
  대해 원천적으로 알 수 없으므로 NaN으로 남긴다(임의값 금지 원칙 —
  모델 학습 경로는 이 컬럼들을 안 쓰는 것도 이미 확인했다,
  `build_daily_dataset_v5()`는 `일간발전량_kWh`·`부분가용일`만 읽음).
- `부분가용일`은 0으로 둔다 — 이래야 `corrected_dataset()`의
  `history = raw["일간발전량_kWh"].where(partial<1)`에서 이 20일이
  결측 처리되지 않고 lag/rolling 입력으로 실제로 쓰인다(그게 이
  작업의 목적).

## 산출물
`outputs/v6_일간구멍보정_2026-08-26/집계_일간_실제발전량_v5.parquet`
(v5와 같은 스키마 + `추정치_여부` 컬럼 추가)
`outputs/v6_일간구멍보정_2026-08-26/20일_배분_감사.csv` (일자별 배분 근거)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
            r"\03_모델학습\현재_종합파이프라인")
V5_PARQUET = ROOT / "outputs" / "v5_복구_2026-08-21" / "집계_일간_실제발전량_v5.parquet"
OUT_DIR = ROOT / "outputs" / "v6_일간구멍보정_2026-08-26"
OUT_PARQUET = OUT_DIR / "집계_일간_실제발전량_v5.parquet"  # 파일명은 v5와 동일(몽키패치용)

EXCEL_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일")
EXCEL_FILES = {
    n: EXCEL_DIR / f"광주 광주시청 _{n}번 인버터 로그_계산됨.xlsx" if n == 1
    else EXCEL_DIR / f"광주 광주시청_{n}번 인버터 로그_계산됨.xlsx"
    for n in range(1, 6)
}
BLOCK_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25"
    r"\blockdata_history.sqlite3"
)
KMA_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25"
    r"\kma_live_inputs.sqlite3"
)
GAP_START = pd.Timestamp("2026-08-05")
GAP_END = pd.Timestamp("2026-08-24")  # 포함


def per_inverter_total_energy_diff() -> dict[int, float]:
    """check_total_energy_continuity_v1_2026-08-26.py와 동일 로직(그대로
    재사용 — 새로 짜지 않음, 여기 인라인으로 다시 부른 이유는 그 스크립트가
    __main__ 전용이라 함수 재사용이 애매해서 핵심 두 함수만 복제했다)."""
    diffs = {}
    for inv, path in EXCEL_FILES.items():
        df = pd.read_excel(path, usecols=["생성일", "누적"]).dropna(subset=["생성일", "누적"]).sort_values("생성일")
        excel_time, excel_cum = pd.Timestamp(df.iloc[-1]["생성일"]), float(df.iloc[-1]["누적"])

        conn = sqlite3.connect(BLOCK_DB)
        row = conn.execute(
            "SELECT measurement_time, total_energy FROM inverter_measurements "
            "WHERE inverter_number=? AND total_energy IS NOT NULL "
            "ORDER BY measurement_time ASC LIMIT 1", (inv,),
        ).fetchone()
        conn.close()
        if row is None:
            raise RuntimeError(f"인버터 {inv}: Blockdata total_energy 기록 없음")
        bd_total = float(row[1])
        diffs[inv] = bd_total - excel_cum
        print(f"  인버터{inv}: 엑셀누적({excel_time.date()})={excel_cum:,.2f} -> "
              f"BD최초total_energy={bd_total:,.2f}  차이={diffs[inv]:,.2f}")
    return diffs


def daily_solar_weights() -> pd.Series:
    conn = sqlite3.connect(KMA_DB)
    df = pd.read_sql_query(
        "SELECT observation_time, solar_mj_m2 FROM asos_hourly "
        "WHERE observation_time>=? AND observation_time<?",
        conn, params=[GAP_START.isoformat(), (GAP_END + pd.Timedelta(days=1)).isoformat()],
    )
    conn.close()
    if df.empty:
        raise RuntimeError("asos_hourly에 08-05~24 구간 자료가 없다 — backfill 스크립트 먼저 실행할 것.")
    df["observation_time"] = pd.to_datetime(df["observation_time"]).dt.tz_localize(None)
    df["날짜"] = df["observation_time"].dt.normalize()
    daily = df.groupby("날짜")["solar_mj_m2"].sum()
    expected_days = pd.date_range(GAP_START, GAP_END, freq="D")
    missing_days = expected_days.difference(daily.index)
    if len(missing_days):
        raise RuntimeError(f"일사량 자료 없는 날짜 있음: {list(missing_days)} — 배분 못 함, 중단.")
    return daily


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 인버터별 total_energy 차이(20일 총량) 계산...")
    diffs = per_inverter_total_energy_diff()
    total_plant = sum(diffs.values())
    print(f"  5대 합계: {total_plant:,.2f}")

    print("\n[2/4] ASOS 실측 일사량으로 일별 가중치 계산...")
    daily_solar = daily_solar_weights()
    weights = daily_solar / daily_solar.sum()
    print(daily_solar.to_string())

    print("\n[3/4] 인버터별×일자별 배분 -> 발전소 합계로 집계...")
    audit_rows = []
    daily_plant_kwh = pd.Series(0.0, index=weights.index)
    for inv, diff in diffs.items():
        per_day = diff * weights
        daily_plant_kwh += per_day
        for date, val in per_day.items():
            audit_rows.append({"날짜": date, "인버터": inv, "배분_kWh": val,
                               "일사량가중치": weights[date], "인버터_20일총량": diff})
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT_DIR / "20일_배분_감사.csv", index=False, encoding="utf-8-sig")
    print(daily_plant_kwh.round(2).to_string())
    print(f"  배분 총합 검산: {daily_plant_kwh.sum():,.2f} (원본 총량 {total_plant:,.2f}과 "
          f"일치해야 함, 차이={abs(daily_plant_kwh.sum()-total_plant):.6f})")

    print("\n[4/4] v5 원본 + 20일 추정 병합 -> v6 저장(원본 파일 불변)...")
    original = pd.read_parquet(V5_PARQUET).copy()
    original.index = pd.to_datetime(original.index)
    original.index.name = "날짜"
    if "추정치_여부" not in original.columns:
        original["추정치_여부"] = 0

    new_rows = pd.DataFrame({
        "일간발전량_kWh": daily_plant_kwh,
        "낮시간예상개수": np.nan, "낮시간실측개수": np.nan, "낮시간자료충족률": np.nan,
        "가용인버터수_낮시간최소": np.nan, "가용인버터수_낮시간평균": np.nan,
        "부분가용일": 0,  # lag/rolling 입력으로 쓰이게 함(결측 처리 안 되도록)
        "추정치_여부": 1,
    })
    new_rows.index.name = "날짜"

    overlap = original.index.intersection(new_rows.index)
    if len(overlap):
        raise RuntimeError(f"원본과 겹치는 날짜가 있음(있으면 안 됨): {list(overlap)}")

    combined = pd.concat([original, new_rows]).sort_index()
    gap_check = pd.date_range(combined.index.min(), combined.index.max(), freq="D").difference(combined.index)
    combined.to_parquet(OUT_PARQUET)

    print(f"  원본 {len(original)}행 + 추정 {len(new_rows)}행 = 합계 {len(combined)}행")
    print(f"  전체 날짜범위: {combined.index.min().date()} ~ {combined.index.max().date()}")
    print(f"  이어붙인 후에도 남은 빈 날짜: {len(gap_check)}개"
          + (f" {list(gap_check.date)[:5]}..." if len(gap_check) else "(없음 — 완전 연속)"))
    print(f"\n저장 완료: {OUT_PARQUET}")
    print(f"감사파일: {OUT_DIR / '20일_배분_감사.csv'}")
    print(f"\n★원본 확인★ v5 원본은 안 건드림: {V5_PARQUET}")


if __name__ == "__main__":
    main()
