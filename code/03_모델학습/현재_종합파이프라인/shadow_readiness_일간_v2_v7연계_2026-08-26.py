# -*- coding: utf-8 -*-
"""일간 D+1 라이브 운영 준비도 게이트 — v2, v7(라이브 이어붙임) 인식판.

## v1과 다른 점(사용자 지적으로 발견한 공백 수정)
`shadow_readiness_일간_v1_2026-08-26.py`는 **v6/v7 데이터셋을 전혀 모르고**
순수 라이브 Blockdata 수집 테이블(`inverter_measurements`)의 최소~최대
시각 차이만 보고 "32일 필요"를 판정했다. 하지만 `build_daily_gap_
estimate_v1_2026-08-26.py`(v6)가 이미 2026-08-05~08-24 20일 구멍을
카운터뺄셈+ASOS배분으로 메웠고, `build_daily_from_blockdata_live_v1_
2026-08-26.py`(v7)가 그 뒤에 라이브 실측을 계속 이어붙인다 — v1은 이
둘을 안 보고 "라이브 이력을 처음부터 32일 다시 쌓아야 한다"는 잘못된
판정을 내리고 있었다.

## 실제로 필요한 것 — 재구현 없음, 정확한 조건으로 교체
`daily_direct_final_audit_v1_2026-08-25.py::corrected_dataset()`가 쓰는
lag 규칙을 그대로 따라간다(코드 재구현 아님, 같은 규칙을 데이터 존재
여부 확인에만 적용):
- `history.shift(7)` → **목표일(D+1)의 7일전** 날짜에 유효한(추정 포함)
  일간발전량이 있어야 함
- `history.shift(2)` → **목표일의 2일전** 날짜에 유효한 일간발전량이
  있어야 함(30일 이동통계도 이 날짜를 기준으로 계산됨)
- 30일 이동창(목표일 2일전까지 30일)의 실제 채움 비율도 정보용으로
  같이 보고한다(이건 native missing으로도 처리 가능해 하드 블로커는
  아님 — 학습 데이터 자체도 100% 채워진 적은 없었다).

v6의 추정 20일(카운터뺄셈+ASOS배분)은 **lag 입력으로는 정상 사용**
(원래 설계 그대로, 08-25 절 참고) — 그래서 이 게이트도 추정치_여부와
무관하게 "일간발전량_kWh notna"만 확인한다(추정이든 실측이든 lag
입력으로는 둘 다 유효).

## 왜 이게 32일보다 훨씬 빠른가
v6+v7이 2026-08-24까지 연속(추정 포함)이므로, target_day=D+1 기준
7일전·2일전 요구는 당분간 항상 과거 이력 범위 안에 있다. 유일한 남은
공백은 **2026-08-25 하루**(Blockdata 수집이 그날 오후부터 시작돼 90%
낮시간 기준을 못 넘김 — `build_daily_from_blockdata_live_v1_2026-08-26.py`
참고)뿐이고, 이마저도 target_day가 이틀만 더 지나면(2026-08-28 이후)
2일전 요구 범위 밖으로 자연히 벗어난다. 즉 "32일 자연누적"이 아니라
**실질적으로 1~2일 후 자동 해소**다.

## 실행 전 필수
`build_daily_from_blockdata_live_v1_2026-08-26.py`를 먼저(또는 매번)
실행해 v7을 최신화할 것 — 이 게이트는 v7 parquet을 읽기만 하고 직접
갱신하지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "outputs" / "운영모델_v2_2026-08-25" / "운영모델_일간_D+1.joblib"
V7_PARQUET = ROOT / "outputs" / "v7_라이브연계_2026-08-26" / "집계_일간_실제발전량_v5.parquet"
V6_PARQUET = ROOT / "outputs" / "v6_일간구멍보정_2026-08-26" / "집계_일간_실제발전량_v5.parquet"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


asm = _load("daily_ready_common_v2", "live_feature_assembler_단기_v1_2026-08-26.py")
shadow = _load("daily_ready_shadow_v2", "shadow_predict_단기_v1_2026-08-26.py")


def _load_daily_history() -> tuple[pd.DataFrame, str]:
    if V7_PARQUET.is_file():
        raw = pd.read_parquet(V7_PARQUET)
        source = "v7(라이브연계)"
    elif V6_PARQUET.is_file():
        raw = pd.read_parquet(V6_PARQUET)
        source = "v6(구멍보정만, v7 미생성)"
    else:
        raise RuntimeError("v6/v7 일간 데이터셋이 없다 — build_daily_gap_estimate_v1_2026-08-26.py"
                          "와 build_daily_from_blockdata_live_v1_2026-08-26.py를 먼저 실행할 것.")
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index(), source


def audit() -> dict:
    bundle = joblib.load(BUNDLE)
    now = pd.Timestamp.now(tz=asm.KST)
    now_naive = now.tz_localize(None)

    history, source = _load_daily_history()
    valid_days = set(history.index[history["일간발전량_kWh"].notna()])

    target_day = (now_naive + pd.Timedelta(days=1)).normalize()
    lag7_date = target_day - pd.Timedelta(days=7)
    lag2_date = target_day - pd.Timedelta(days=2)
    lag7_ok = lag7_date in valid_days
    lag2_ok = lag2_date in valid_days

    window_days = pd.date_range(lag2_date - pd.Timedelta(days=29), lag2_date, freq="D")
    window_filled = sum(1 for d in window_days if d in valid_days)

    with sqlite3.connect(asm.BLOCK_DB) as conn:
        t0, t1 = conn.execute(
            "SELECT min(measurement_time), max(measurement_time) FROM inverter_measurements").fetchone()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(inverter_measurements)")}
    raw_span_days = 0.0
    if t0 and t1:
        raw_span_days = (pd.to_datetime(t1, utc=True) - pd.to_datetime(t0, utc=True)).total_seconds() / 86400

    with sqlite3.connect(asm.KMA_DB) as conn:
        nwp = conn.execute(
            "SELECT count(DISTINCT target_time_kst), "
            "sum(CASE WHEN is_missing=0 THEN 1 ELSE 0 END) FROM nwp_values "
            "WHERE substr(target_time_kst,1,10)=? AND first_received_at<=?",
            (target_day.date().isoformat(), now.isoformat()),
        ).fetchone()

    blockers = []
    if not lag7_ok:
        blockers.append(f"7일전({lag7_date.date()}) 일간발전량(실측/추정) 없음")
    if not lag2_ok:
        blockers.append(f"2일전({lag2_date.date()}) 일간발전량(실측/추정) 없음")
    if any(f.endswith("mean_communication_ok") for f in bundle["features"]) and "communication_ok" not in cols:
        blockers.append("일간 번들이 mean_communication_ok를 요구하지만 라이브 API 스키마에 원천필드 없음")
    if not nwp or int(nwp[0] or 0) < 8:
        blockers.append(f"목표일 D+1 NWP 시각 {int((nwp or (0,0))[0] or 0)}/8")
    hard = any("원천필드 없음" in x for x in blockers)

    return {
        "상태": "blocked" if hard else ("warming_up" if blockers else "ready"),
        "점검시각": now.isoformat(), "예측대상일": target_day.date().isoformat(),
        "데이터출처": source,
        "7일전_충족": lag7_ok, "2일전_충족": lag2_ok,
        "30일창_채움일수": window_filled, "30일창_전체": len(window_days),
        "참고_Blockdata원시수집일수": round(raw_span_days, 3),
        "목표일_NWP_시각수": int((nwp or (0, 0))[0] or 0),
        "차단사유": blockers,
        "조치": ("mean_communication_ok 제거 후보를 동일 5계절로 재검증 후 운영번들 재학습" if hard
               else ("2026-08-25 부분일 자체 보정 또는 자연 경과(2일 이내 자동해소) 중 선택"
                     if (not lag7_ok or not lag2_ok) else None)),
    }


def main() -> None:
    result = audit()
    conn = shadow.ensure_db()
    now = result["점검시각"]
    issue = pd.Timestamp(now).tz_localize(None).floor("D") + pd.Timedelta(hours=10)
    reason = "; ".join(result["차단사유"])
    conn.execute(
        "UPDATE shadow_predictions SET status='superseded' "
        "WHERE tier='일간' AND status IN ('warming_up','blocked')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO shadow_predictions "
        "(predicted_at,tier,horizon_h,issue_time,target_time,status,reason,bundle_version) "
        "VALUES (?, '일간', 24, ?, ?, ?, ?, 'v2_2026-08-25+DIFSWRF제외_2026-08-26')",
        (now, str(issue.tz_localize(None)), result["예측대상일"], result["상태"], reason),
    )
    conn.commit(); conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
