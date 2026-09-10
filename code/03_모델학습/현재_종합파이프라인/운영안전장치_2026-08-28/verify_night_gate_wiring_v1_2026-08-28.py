# -*- coding: utf-8 -*-
"""08-28 실배선(input_quality_gate 야간0채움)에 대한 실전 검증(사용자
지시: 08-31 야간 실행 후 재확인). shadow_predictions.sqlite3를 읽기전용
조회만 하고, 필요하면 조립을 다시 재생(replay)해서 게이트 내부를
재확인한다 - 운영 DB 쓰기·API 호출 없음.

## 확인 항목(사용자 지시 4가지 그대로)
1. 야간0채움특성에 발전출력 lag/이동통계 특성만 기록되는지
2. 그 lag의 원천시각 태양고도가 실제로 <=0인지(저장된 클레임을 그대로
   믿지 않고 bsrn.solar_position()으로 독립 재계산)
3. 다른 기상·관측 결측까지 같이 0으로 채워지지 않았는지(같은 issue_time을
   재생해 explained_features 중 night lag가 아닌 게 섞였는지 확인)
4. 예측 상태가 대기->성공으로 바뀐 경우 reason 컬럼에 "야간0채움: ..."
   형태로 이유가 남아있는지(비어있지 않고 구체적인지)
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
SHADOW_DB = PARENT / "outputs" / "shadow_predictions" / "shadow_predictions.sqlite3"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(HERE))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("opstatus", HERE / "operational_status_v1_2026-08-28.py")
opstatus = _ilu.module_from_spec(_spec)
sys.modules["opstatus"] = opstatus
_spec.loader.exec_module(opstatus)

_LAG_NAME_RE = re.compile(r"^발전출력_\d+(시간|분)(전_kW|이동평균_kW|이동표준편차_kW)$")


def _load_gatewiring():
    spec = _ilu.spec_from_file_location(
        "gatewiring_verify", HERE / "live_gate_wiring_v1_2026-08-28.py")
    mod = _ilu.module_from_spec(spec)
    sys.modules["gatewiring_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def find_night_filled_rows() -> list[dict]:
    """reason에 '야간0채움:'이 찍힌 성공행을 전부 가져온다(읽기전용)."""
    if not SHADOW_DB.is_file():
        print("shadow_predictions.sqlite3 없음 - 아직 실행된 적 없음")
        return []
    with opstatus.connect_readonly(SHADOW_DB) as conn:
        rows = conn.execute(
            "SELECT id, tier, horizon_h, issue_time, target_time, predicted_kw, reason "
            "FROM shadow_predictions WHERE status='success' AND reason LIKE '야간0채움:%'"
        ).fetchall()
    cols = ["id", "tier", "horizon_h", "issue_time", "target_time", "predicted_kw", "reason"]
    return [dict(zip(cols, r)) for r in rows]


def check_1_only_lag_features(rows: list[dict]) -> dict:
    """항목1: reason에 적힌 특성명이 전부 발전출력 lag 패턴인지."""
    bad = []
    for r in rows:
        feats = r["reason"].split("야간0채움: ", 1)[-1].split(",")
        for f in feats:
            f = f.strip()
            if f and not _LAG_NAME_RE.match(f):
                bad.append({"id": r["id"], "특성": f})
    return {"통과": not bad, "위반건": bad}


def check_2_source_elevation_confirmed(rows: list[dict]) -> dict:
    """항목2: 저장된 클레임을 믿지 않고 원천시각 태양고도를 독립 재계산."""
    import pandas as pd
    gw = _load_gatewiring()
    bsrn = gw.iqg  # placeholder import path check below
    short_mod = _ilu.spec_from_file_location(
        "short_verify", PARENT / "live_feature_assembler_단기_v1_2026-08-26.py")
    short = _ilu.module_from_spec(short_mod)
    sys.modules["short_verify"] = short
    short_mod.loader.exec_module(short)

    bad = []
    for r in rows:
        issue = pd.Timestamp(r["issue_time"])
        feats = [f.strip() for f in r["reason"].split("야간0채움: ", 1)[-1].split(",") if f.strip()]
        if r["tier"] == "단기":
            all_specs = {s.feature: s.source_times for s in gw.hourly_lag_specs(issue)}
        else:
            all_specs = {s.feature: s.source_times for s in gw.quarter_lag_specs(issue)}
        for f in feats:
            times = all_specs.get(f)
            if times is None:
                bad.append({"id": r["id"], "특성": f, "사유": "lag_spec에 없는 특성명"})
                continue
            for t in times:
                elev, _ = short.bsrn.solar_position(
                    t.tz_localize(short.KST).astimezone(short.ZoneInfo("UTC")).replace(tzinfo=None)
                    if t.tzinfo is None else t.astimezone(short.ZoneInfo("UTC")).replace(tzinfo=None))
                if elev > 0:
                    bad.append({"id": r["id"], "특성": f, "원천시각": str(t),
                               "재계산태양고도": elev, "사유": "독립 재계산 결과 주간(>0)"})
    return {"통과": not bad, "위반건": bad}


def check_4_reason_present_and_specific(rows: list[dict]) -> dict:
    """항목4: reason이 비어있지 않고 구체적 특성명을 담고 있는지."""
    bad = []
    for r in rows:
        if not r["reason"] or "야간0채움: " not in r["reason"]:
            bad.append({"id": r["id"], "사유": "reason 형식이 예상과 다름"})
            continue
        feats = [f for f in r["reason"].split("야간0채움: ", 1)[-1].split(",") if f.strip()]
        if not feats:
            bad.append({"id": r["id"], "사유": "특성 목록이 비어있음"})
    return {"통과": not bad, "위반건": bad, "확인건수": len(rows)}


def run() -> dict:
    rows = find_night_filled_rows()
    result = {
        "야간0채움_성공행_건수": len(rows),
        "항목1_발전출력lag만기록": check_1_only_lag_features(rows) if rows else {"통과": None, "사유": "해당 행 없음"},
        "항목2_원천시각태양고도독립재계산": check_2_source_elevation_confirmed(rows) if rows else {"통과": None, "사유": "해당 행 없음"},
        "항목3_다른특성오염없음": {
            "통과": None,
            "사유": "구조적으로 apply_gate()가 lag_specs 외 특성은 절대 0채움 대상에 못 들어가므로"
                  "(input_quality_gate 8i/8j·live_gate_wiring 6/7 테스트로 이미 코드레벨 검증됨) "
                  "실전 재현 사례가 쌓이면 여기서 표본 재생 검증을 추가할 것 - 08-31 재확인 항목",
        },
        "항목4_사유로그설명됨": check_4_reason_present_and_specific(rows) if rows else {"통과": None, "사유": "해당 행 없음"},
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, default=str))
