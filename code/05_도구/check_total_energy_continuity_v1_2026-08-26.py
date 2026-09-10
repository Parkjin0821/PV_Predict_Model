# -*- coding: utf-8 -*-
"""엑셀 로그의 `누적` 카운터와 Blockdata의 `total_energy` 카운터가
연속된 같은 물리 카운터인지 확인 — 08-05~08-24 20일 구멍을 뺄셈으로
복원할 수 있는지 판단하기 위한 진단.

## API 호출 0건
`total_energy`·`daily_energy`는 이미 `collect_blockdata_history_v1_
2026-08-25.py`가 08-25부터 수집해 로컬 SQLite에 저장해두고 있었다
(스키마 확인함, PRAGMA table_info로 두 컬럼 존재 확인됨). 그래서 이
스크립트는 엑셀 파일과 그 로컬 SQLite만 읽는다 — 새 API 호출 없음.

## 판정 방법
인버터별로:
1. 엑셀 로그의 마지막 행(`생성일`, `누적`) — 08-04 20시대까지 있음.
2. Blockdata 로컬 DB의 최초 행(`measurement_time`, `total_energy`) —
   08-25 16시대부터 있음.
3. 두 값이 **같은 리셋 없는 누적 카운터**라면:
   `08-05~08-24 총발전량(kWh) = Blockdata_최초_total_energy - 엑셀_마지막_누적`
   이 값을 그 구간 일수로 나눈 "일평균"이 이 발전소의 평소 일간 발전량
   (~150~180kWh/인버터대, 5대 합계 700~900kWh/일대 — 과거 710일 실적
   기준)과 비슷한 범위면 카운터가 진짜 연속이라는 뜻이다. 음수가 나오거나
   말도 안 되게 크면(리셋·단위불일치·기기교체 등) 이 방법을 쓰면 안 된다
   — 그 경우는 그대로 정직하게 보고하고 이 방법을 포기해야 한다.

## 이 스크립트가 "확정"하지 않는 것
카운터 연속성이 확인되더라도, 이건 "20일 총합"만 복원해줄 뿐 일별
분포는 알려주지 않는다. 총합을 날짜별로 배분하려면 별도로 그 기간의
날씨자료(DSWRF 등, 이미 있음)에 비례해 나누는 방법을 써야 한다 —
이 스크립트의 범위 밖이며, 여기서 연속성이 확인된 뒤에 별도로 설계한다.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
# 과거 710일 실적 기준 참고치(집계_일간_실제발전량_v5.parquet, 발전소 5대 합계
# 평균 약 780kWh/일대) — 절대 기준이 아니라 "말이 되는 범위인지" 눈대중용.
PLANT_DAILY_REFERENCE_KWH = 780.0


def last_excel_reading(inverter: int) -> tuple[pd.Timestamp, float]:
    path = EXCEL_FILES[inverter]
    if not path.is_file():
        raise FileNotFoundError(f"엑셀 파일 없음: {path}")
    df = pd.read_excel(path, usecols=["생성일", "누적"])
    df = df.dropna(subset=["생성일", "누적"]).sort_values("생성일")
    last = df.iloc[-1]
    return pd.Timestamp(last["생성일"]), float(last["누적"])


def first_blockdata_reading(inverter: int) -> tuple[pd.Timestamp | None, float | None]:
    if not BLOCK_DB.is_file():
        raise FileNotFoundError(f"라이브 DB 없음: {BLOCK_DB}")
    conn = sqlite3.connect(BLOCK_DB)
    try:
        row = conn.execute(
            "SELECT measurement_time, total_energy FROM inverter_measurements "
            "WHERE inverter_number=? AND total_energy IS NOT NULL "
            "ORDER BY measurement_time ASC LIMIT 1",
            (inverter,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None, None
    return pd.to_datetime(row[0], utc=True), float(row[1])


def main() -> int:
    print("=" * 78)
    print("엑셀 `누적` vs Blockdata `total_energy` 연속성 진단 (API 호출 0건)")
    print("=" * 78)

    rows = []
    for inv in range(1, 6):
        try:
            excel_time, excel_cum = last_excel_reading(inv)
        except Exception as exc:
            print(f"\n[인버터 {inv}] 엑셀 읽기 실패: {exc}")
            continue
        bd_time, bd_total = first_blockdata_reading(inv)

        print(f"\n[인버터 {inv}]")
        print(f"  엑셀 마지막 기록: {excel_time}  누적={excel_cum:,.2f}")
        if bd_time is None:
            print("  Blockdata: total_energy 기록 없음(전부 NULL이거나 행 없음)")
            continue
        print(f"  Blockdata 최초 기록: {bd_time}  total_energy={bd_total:,.2f}")

        gap_days = (bd_time.tz_localize(None) - excel_time).total_seconds() / 86400
        diff = bd_total - excel_cum
        implied_daily = diff / gap_days if gap_days > 0 else float("nan")
        print(f"  구간일수: {gap_days:.2f}일")
        print(f"  차이(총발전량 추정): {diff:,.2f} (단위는 원본 그대로 — 아래서 kWh 여부 판단)")
        print(f"  구간 일평균(diff/일수): {implied_daily:,.2f}")

        plausible = 0 < implied_daily < (PLANT_DAILY_REFERENCE_KWH / 5) * 3
        print(f"  [판정] 인버터 1대 일평균이 대략 0~{(PLANT_DAILY_REFERENCE_KWH/5)*3:.0f} "
              f"범위인가(과거 실적 참고): {'그럴듯함' if plausible else '★이상함 — 카운터 불연속 의심★'}")

        rows.append({
            "인버터": inv, "엑셀마지막시각": str(excel_time), "엑셀마지막누적": excel_cum,
            "BD최초시각": str(bd_time), "BD최초total_energy": bd_total,
            "구간일수": gap_days, "차이": diff, "구간일평균": implied_daily, "그럴듯함": plausible,
        })

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 78)
        print("요약표")
        print("=" * 78)
        print(df.to_string(index=False))
        total_diff = df["차이"].sum()
        print(f"\n5대 합계 차이(=20일 구멍 총발전량 추정, 단위 확인 필요): {total_diff:,.2f}")
        print(f"5대 합계 구간일평균: {total_diff / df['구간일수'].mean():,.2f} "
              f"(과거 5대 합계 평균 참고치: {PLANT_DAILY_REFERENCE_KWH:.0f}/일)")
        if not df["그럴듯함"].all():
            print(
                "\n★경고★ 일부 인버터가 '이상함' 판정입니다 — 카운터 리셋, 단위 불일치"
                "(예: 엑셀은 kWh인데 API는 Wh), 기기 교체 등의 가능성이 있습니다."
                " 이 경우 total_energy 뺄셈 방법을 그대로 쓰면 안 되고, 원인을"
                " 먼저 확인해야 합니다."
            )
    else:
        print("\n비교 가능한 인버터가 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
