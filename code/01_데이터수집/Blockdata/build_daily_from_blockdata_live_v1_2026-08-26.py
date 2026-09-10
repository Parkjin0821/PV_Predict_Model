# -*- coding: utf-8 -*-
"""라이브 Blockdata 5분 실측을 일간 합계로 집계해 v6 뒤에 이어붙인다 — v7.

## 왜 필요한가(08-26 사용자 지적으로 발견한 공백)
`build_daily_gap_estimate_v1_2026-08-26.py`(v6)가 2026-08-05~08-24
20일 구멍을 카운터뺄셈+ASOS배분으로 메웠지만, **그건 과거~08-24까지만**
이다. `shadow_readiness_일간_v1_2026-08-26.py`는 v6를 전혀 모르고 순수
라이브 Blockdata 수집 테이블의 기간만 보고 "32일 필요"를 판정한다 —
v6가 이미 08-24까지 채워놨는데도 라이브 이력을 처음부터 32일 다시
쌓아야 하는 것처럼 판정하는 공백이 있었다. 이 스크립트가 그 연결
고리(v6 → 라이브 실측 → v7)를 만든다.

## 방법 — 재구현 없음(원칙 준수)
`pv_pipeline.py`의 검증된 공식 함수를 그대로 재사용한다.
1. Blockdata 라이브 SQLite `plant_snapshots`(발전소 합산 스냅샷 —
   인버터별로 다시 합산할 필요 없이 이미 `plant_ac_power_kw`·
   `valid_ac_power_count`를 제공)를 5분 격자에 정렬.
2. `pv_pipeline.solar_position()`으로 태양고도 계산 → `물리적낮`
   (v5·v6와 동일한 태양고도>0 기준).
3. 발전소 단위 물리범위 클리핑(0~capacity_kw) — `rebuild_plant_v5_
   recovered_2026-08-21.py`의 168행과 동일 규칙.
4. `pv_pipeline.build_daily_actual()`을 그대로 호출 — 낮시간자료충족률
   90% 미만인 날은 자동으로 NaN(임의보간 없음, 원본과 동일한 엄격 기준).
   2026-08-25는 Blockdata 수집이 16:56부터 시작해 오전 데이터가 없으므로
   이 기준에서 자동으로 NaN 처리된다(의도된 정직한 결과 — 별도 조치 없음).
5. `가용인버터수_낮시간최소/평균`·`부분가용일`은 `valid_ac_power_count`를
   `rebuild_plant_v5`의 "가용인버터수_낮시간평균 < 5 → 부분가용일" 규칙과
   동일하게 적용해 계산(라이브의 "가용인버터수"에 해당하는 필드가
   `valid_ac_power_count`이므로 이름만 다르고 의미는 같음).
6. v6 parquet 뒤에 **v6 마지막 날짜(2026-08-24) 이후만** 이어붙인다
   (겹치는 날짜가 있으면 즉시 에러 — 원본 훼손 방지). `추정치_여부=0`
   으로 저장(카운터뺄셈 추정이 아니라 실측 5분자료 직접 집계이므로
   v6의 20일 추정보다 신뢰도가 높다).

## 원본 불변 원칙
v5·v6 parquet는 한 글자도 안 건드린다. 매 실행마다 v6를 기준으로 다시
읽어 v7을 처음부터 새로 만든다(멱등성 — v7 위에 v7을 쌓지 않음).

## 산출물
`outputs/v7_라이브연계_2026-08-26/집계_일간_실제발전량_v5.parquet`
(v5·v6와 같은 스키마, 파일명도 동일 — V5_DIR 몽키패치 재사용 위함)
`outputs/v7_라이브연계_2026-08-26/라이브_신규일자_감사.csv`
"""
from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PIPE_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19"
                r"\03_모델학습\현재_종합파이프라인")
V6_PARQUET = PIPE_DIR / "outputs" / "v6_일간구멍보정_2026-08-26" / "집계_일간_실제발전량_v5.parquet"
OUT_DIR = PIPE_DIR / "outputs" / "v7_라이브연계_2026-08-26"
OUT_PARQUET = OUT_DIR / "집계_일간_실제발전량_v5.parquet"
BLOCK_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25"
    r"\blockdata_history.sqlite3")
PLANT_ID = 6715
N_INVERTERS = 5


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PIPE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pv_pipeline = _load("v7_pv_pipeline", "pv_pipeline.py")


def load_live_five_minute(capacity_kw: float, latitude: float, longitude: float) -> pd.DataFrame:
    """Blockdata plant_snapshots를 5분 격자로 정렬한 "five"-호환 프레임을 만든다
    (rebuild_plant_v5의 `five`와 같은 역할 컬럼: 발전출력_kW·물리적낮·
    가용인버터수)."""
    with sqlite3.connect(BLOCK_DB) as conn:
        raw = pd.read_sql_query(
            "SELECT snapshot_time, plant_ac_power_kw, valid_ac_power_count "
            "FROM plant_snapshots WHERE plant_id=? ORDER BY snapshot_time",
            conn, params=[PLANT_ID],
        )
    if raw.empty:
        raise RuntimeError("plant_snapshots가 비어있다 — Blockdata 수집기 상태를 먼저 확인할 것.")
    raw["snapshot_time"] = pd.to_datetime(raw["snapshot_time"], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    raw = raw.set_index("snapshot_time").sort_index()

    # ★버그 수정(첫 실행에서 발견)★: 격자 끝을 "마지막 관측시각"으로 잡으면
    # 아직 안 끝난 오늘 하루가 "관측된 만큼만"을 분모로 삼아 충족률이
    # 부풀려진다(예: 16:40까지만 관측했는데 그때까지의 낮시간만 기대개수로
    # 잡아 90%를 넘겨버림 — 저녁 시간대가 통째로 안 셈해짐). 격자 끝을
    # 마지막 관측일의 **자정까지**로 강제 확장해, 아직 안 온 저녁 시간대가
    # 결측으로 정직하게 잡히게 한다.
    grid_end = max(raw.index.max().ceil("5min"), raw.index.max().normalize() + pd.Timedelta(days=1))
    grid = pd.date_range(raw.index.min().floor("5min"), grid_end, freq="5min")
    five = raw[["plant_ac_power_kw", "valid_ac_power_count"]].resample("5min").mean().reindex(grid)
    five = five.rename(columns={"plant_ac_power_kw": "발전출력_kW", "valid_ac_power_count": "가용인버터수"})

    elev, _azi = pv_pipeline.solar_position(grid, latitude, longitude)
    five["물리적낮"] = (elev > 0).astype("int8")

    # ★rebuild_plant_v5 168행과 동일: 발전소 단위 물리범위 클리핑★
    five.loc[~five["발전출력_kW"].between(0, capacity_kw), "발전출력_kW"] = np.nan
    return five


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = pv_pipeline.load_config()
    capacity_kw = float(config["site"]["capacity_kw"])
    latitude = float(config["site"]["latitude"])
    longitude = float(config["site"]["longitude"])

    print("[1/4] Blockdata 라이브 plant_snapshots를 5분 격자로 정렬...")
    five = load_live_five_minute(capacity_kw, latitude, longitude)
    print(f"  격자 {len(five):,}개(5분), 기간 {five.index.min()} ~ {five.index.max()}")

    print("\n[2/4] pv_pipeline.build_daily_actual() 재사용(재구현 없음)...")
    daily = pv_pipeline.build_daily_actual(five)
    day_only = five[five["물리적낮"] == 1]
    daily["가용인버터수_낮시간최소"] = day_only.groupby(day_only.index.normalize())["가용인버터수"].min()
    daily["가용인버터수_낮시간평균"] = day_only.groupby(day_only.index.normalize())["가용인버터수"].mean().round(2)
    daily["부분가용일"] = (daily["가용인버터수_낮시간평균"] < N_INVERTERS).astype("Int64")
    daily["추정치_여부"] = 0  # 실측 5분자료 직접 집계 — 카운터뺄셈 추정이 아님

    print(daily[["일간발전량_kWh", "낮시간자료충족률", "가용인버터수_낮시간평균"]].to_string())

    # ★안전장치(이중 확인)★: 90% 충족률 기준을 통과해도 "오늘(KST)"은
    # 절대 완결일로 인정하지 않는다 — 자정 전에 재실행되면 아직 안 끝난
    # 하루가 우연히 기준을 넘길 수 있으므로, 날짜 자체로 한 번 더 막는다.
    today_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    if today_kst in daily.index:
        daily.loc[today_kst, "일간발전량_kWh"] = np.nan

    print("\n[3/4] v6 기준선과 병합(v6 마지막 날짜 이후만 추가)...")
    v6 = pd.read_parquet(V6_PARQUET).copy()
    v6.index = pd.to_datetime(v6.index)
    v6.index.name = "날짜"
    v6_last = v6.index.max()

    new_rows = daily[daily.index > v6_last].copy()
    overlap = v6.index.intersection(new_rows.index)
    if len(overlap):
        raise RuntimeError(f"v6와 겹치는 날짜가 있음(있으면 안 됨): {list(overlap)}")

    valid_new = new_rows[new_rows["일간발전량_kWh"].notna()]
    combined = pd.concat([v6, new_rows]).sort_index()
    combined.to_parquet(OUT_PARQUET)
    daily.reset_index().to_csv(OUT_DIR / "라이브_신규일자_감사.csv", index=False, encoding="utf-8-sig")

    # build_daily_dataset_v5()는 같은 V5_DIR 폴더에서 시간단위 집계도 찾는다
    # (날씨예보 병합용, 이번 작업과 무관 — 안 건드림). v6가 v5의 사본을
    # 그대로 들고 있던 것과 동일하게, v7도 v6의 사본을 그대로 복사한다
    # (내용 변경 없음, 재구현 아님 — 파일 존재 위치만 맞춤).
    hourly_src = V6_PARQUET.parent / "집계_1시간_자료_v5.parquet"
    if hourly_src.is_file():
        shutil.copy2(hourly_src, OUT_DIR / "집계_1시간_자료_v5.parquet")

    print(f"  v6 {len(v6)}행 + 라이브신규 {len(new_rows)}행(유효 {len(valid_new)}행, "
          f"낮시간충족률<90%로 자동제외 {len(new_rows)-len(valid_new)}행) = {len(combined)}행")
    print(f"  v6 마지막날짜: {v6_last.date()}")
    if len(valid_new):
        print(f"  라이브 신규 유효날짜: {list(valid_new.index.date)}")
    else:
        print("  ★아직 완전한 라이브 신규일자 없음★ — Blockdata 수집이 08-25 오후부터 "
              "시작돼 08-25는 오전 결측으로 90% 기준 미달, 08-26 이후 완결일이 쌓이면 "
              "다음 실행에서 자동 반영됨(이 스크립트를 재실행만 하면 됨, 코드 수정 불필요).")

    print("\n[4/4] 완전성 확인...")
    full_range = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
    gap = full_range.difference(combined.index)
    print(f"  전체 날짜범위: {combined.index.min().date()} ~ {combined.index.max().date()}, "
          f"빈 날짜 {len(gap)}개" + (f" {list(gap.date)[:5]}..." if len(gap) else "(없음)"))
    print(f"\n저장 완료: {OUT_PARQUET}")
    print(f"★원본 확인★ v5·v6는 안 건드림. 이 스크립트는 멱등성 — 매번 v6부터 새로 계산.")


if __name__ == "__main__":
    main()
