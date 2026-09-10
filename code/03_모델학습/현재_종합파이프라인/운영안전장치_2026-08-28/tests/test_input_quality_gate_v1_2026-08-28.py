# -*- coding: utf-8 -*-
"""input_quality_gate_v1_2026-08-28.py 오프라인 테스트.

운영 DB를 쓰지 않는다(합성 row 딕셔너리만 사용). 마지막 한 케이스만
"저장된 실제 라이브 조립 결과"를 읽기 전용으로 재사용해 게이트가 실전
데이터에도 예외 없이 동작하는지 확인한다(신규 API 호출 없음).
"""
from __future__ import annotations

import importlib.util as ilu
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

spec = ilu.spec_from_file_location("iqg", PARENT / "input_quality_gate_v1_2026-08-28.py")
iqg = ilu.module_from_spec(spec)
sys.modules["iqg"] = iqg
spec.loader.exec_module(iqg)
opstatus = iqg.opstatus

CFG = iqg.load_config()


def test_1_no_missing_normal():
    row = {"a": 1.0, "b": 2.0}
    r = iqg.evaluate(row, ["a", "b"], CFG)
    assert r["status"] == opstatus.NORMAL, r
    return True


def test_2_unexplained_nan_blocked():
    row = {"a": math.nan, "b": 2.0}
    r = iqg.evaluate(row, ["a", "b"], CFG)
    assert r["status"] == opstatus.BLOCKED, r
    assert "a" in r["unexplained_features"]
    return True


def test_3_night_zero_explained():
    row = {"발전출력_1시간전_kW": math.nan}
    claim = iqg.ZeroFillClaim("발전출력_1시간전_kW", "태양고도 -5deg, 야간 확인",
                              night_flag=True)
    r = iqg.evaluate(row, ["발전출력_1시간전_kW"], CFG, zero_fill_claims=[claim])
    assert r["status"] == opstatus.DEGRADED, r
    assert "발전출력_1시간전_kW" not in r["unexplained_features"]
    return True


def test_4_daytime_missing_without_night_claim_blocked():
    """night_flag 없이 그냥 결측이면(낮인데 결측) 설명 안 된 것으로 처리."""
    row = {"발전출력_1시간전_kW": math.nan}
    claim = iqg.ZeroFillClaim("발전출력_1시간전_kW", "야간 아님", night_flag=False)
    r = iqg.evaluate(row, ["발전출력_1시간전_kW"], CFG, zero_fill_claims=[claim])
    assert r["status"] == opstatus.BLOCKED, r
    return True


def test_5_rainfall_no_phenomenon_explained():
    row = {"기상청관측_강수량_mm": math.nan}
    claim = iqg.ZeroFillClaim("기상청관측_강수량_mm", "ASOS 무강수 NULL 확인",
                              no_phenomenon_flag=True)
    r = iqg.evaluate(row, ["기상청관측_강수량_mm"], CFG, zero_fill_claims=[claim])
    assert r["status"] == opstatus.DEGRADED, r
    return True


def test_6_interpolation_within_limit_explained():
    row = {"x": math.nan}
    interp = {"x": {"steps": 2, "step_minutes": 5}}
    r = iqg.evaluate(row, ["x"], CFG, interpolation_log=interp)
    assert r["status"] == opstatus.DEGRADED, r
    return True


def test_7_interpolation_over_limit_unexplained():
    row = {"x": math.nan}
    interp = {"x": {"steps": 5, "step_minutes": 5}}  # 25분 > 10분 한도
    r = iqg.evaluate(row, ["x"], CFG, interpolation_log=interp)
    assert r["status"] == opstatus.BLOCKED, r
    assert "x" in r["over_interpolated_features"]
    return True


def test_8_future_nwp_missing_no_auto_fill():
    """미래 NWP는 claim이 있어도(허용 규칙 자체가 false라서) 설명 안 됨."""
    row = {"DSWRF": math.nan}
    claim = iqg.ZeroFillClaim("DSWRF", "그냥 채워봄", no_phenomenon_flag=True)
    r = iqg.evaluate(row, ["DSWRF"], CFG, zero_fill_claims=[claim])
    # DSWRF는 강수/적설 계열이 아니므로 no_phenomenon_flag 규칙이 애초에 안 맞음
    assert r["status"] == opstatus.BLOCKED, r
    return True


def test_8b_insufficient_history_separated_from_unexplained():
    """08-28 외부검토 보완 3: 롤링 특성 워밍업 구간 NaN은 unexplained가
    아니라 별도 버킷(structurally_unavailable_features)에 들어가야 하고,
    사유 문구도 "데이터 결측"이 아니라 "워밍업 구간"이라고 정확히 나와야
    한다."""
    row = {"발전량_24시간이동평균_kW": math.nan}
    claim = iqg.ZeroFillClaim(
        "발전량_24시간이동평균_kW", "파이프라인 가동 3시간째 - 24시간 윈도우 미충족",
        insufficient_history_flag=True, history_available=3, history_required=24)
    r = iqg.evaluate(row, ["발전량_24시간이동평균_kW"], CFG, zero_fill_claims=[claim])
    assert "발전량_24시간이동평균_kW" not in r["unexplained_features"], r
    assert "발전량_24시간이동평균_kW" in r["structurally_unavailable_features"], r
    assert "워밍업" in r["structurally_unavailable_features"]["발전량_24시간이동평균_kW"]
    assert "데이터 결측 아님" in r["structurally_unavailable_features"]["발전량_24시간이동평균_kW"]
    return True


def test_8c_insufficient_history_still_blocks_by_default():
    """기본 정책은 unexplained와 동일하게 blocked - "워밍업이니까 그냥
    통과"는 아니다. 사유만 정확해질 뿐 안전기본값(예측불가)은 그대로."""
    row = {"발전량_24시간이동평균_kW": math.nan}
    claim = iqg.ZeroFillClaim(
        "발전량_24시간이동평균_kW", "워밍업", insufficient_history_flag=True,
        history_available=3, history_required=24)
    r = iqg.evaluate(row, ["발전량_24시간이동평균_kW"], CFG, zero_fill_claims=[claim])
    assert r["status"] == opstatus.BLOCKED, r
    assert "워밍업" in r["status_reason"]
    return True


def test_8d_insufficient_history_does_not_leak_into_unrelated_missing():
    """워밍업 특성과 진짜 결측 특성이 같이 있으면, 진짜 결측이 unexplained로
    먼저 잡혀 전체 status가 그쪽 정책(현재는 동일하게 blocked)을 따르되,
    두 원인이 결과에서 서로 다른 필드로 분리돼 있어야 한다(섞이면 나중에
    원인 분석이 불가능해짐)."""
    row = {"발전량_24시간이동평균_kW": math.nan, "진짜결측특성": math.nan}
    claim = iqg.ZeroFillClaim(
        "발전량_24시간이동평균_kW", "워밍업", insufficient_history_flag=True,
        history_available=3, history_required=24)
    r = iqg.evaluate(row, ["발전량_24시간이동평균_kW", "진짜결측특성"], CFG,
                     zero_fill_claims=[claim])
    assert "진짜결측특성" in r["unexplained_features"], r
    assert "발전량_24시간이동평균_kW" in r["structurally_unavailable_features"], r
    assert "발전량_24시간이동평균_kW" not in r["unexplained_features"], r
    return True


def test_8e_night_zero_fill_claims_point_lag_confirmed_night():
    """08-28 실배선(②): 발전출력 1시간전_kW가 NaN이고 그 시각 태양고도가
    -5deg로 확인되면 night_flag 클레임이 생성되고, evaluate()에 넣으면
    degraded로 설명돼야 한다."""
    elev = {"2026-08-28T21:00:00": -5.0, "2026-08-28T22:00:00": 3.0}
    spec = iqg.LagSpec("발전출력_1시간전_kW", ["2026-08-28T21:00:00"])
    claims = iqg.build_night_zero_fill_claims(
        {"발전출력_1시간전_kW": math.nan}, ["발전출력_1시간전_kW"], [spec], elev.get)
    assert len(claims) == 1, claims
    r = iqg.evaluate({"발전출력_1시간전_kW": math.nan}, ["발전출력_1시간전_kW"], CFG,
                     zero_fill_claims=claims)
    assert r["status"] == opstatus.DEGRADED, r
    assert "발전출력_1시간전_kW" in r["explained_features"], r
    return True


def test_8f_night_zero_fill_claims_daytime_source_no_claim():
    """원천시각 태양고도가 양수(주간)면 클레임을 만들지 않는다 - 안전기본값."""
    elev = {"2026-08-28T14:00:00": 45.0}
    spec = iqg.LagSpec("발전출력_1시간전_kW", ["2026-08-28T14:00:00"])
    claims = iqg.build_night_zero_fill_claims(
        {"발전출력_1시간전_kW": math.nan}, ["발전출력_1시간전_kW"], [spec], elev.get)
    assert claims == [], claims
    return True


def test_8g_night_zero_fill_claims_unresolvable_source_no_claim():
    """원천시각을 조회할 수 없으면(None) 클레임을 만들지 않는다 - 입증 안
    되면 보수적으로 미설명 결측으로 남긴다."""
    spec = iqg.LagSpec("발전출력_1시간전_kW", ["2026-08-28T21:00:00"])
    claims = iqg.build_night_zero_fill_claims(
        {"발전출력_1시간전_kW": math.nan}, ["발전출력_1시간전_kW"], [spec], lambda t: None)
    assert claims == [], claims
    return True


def test_8h_night_zero_fill_claims_rolling_window_requires_all_night():
    """이동평균/표준편차처럼 여러 원천시각이 있는 특성은 전부 야간이어야
    클레임이 생긴다 - 하나라도 주간이면 전체를 미설명으로 남긴다."""
    elev_all_night = {"t1": -5.0, "t2": -3.0, "t3": -1.0}
    spec_ok = iqg.LagSpec("발전출력_3시간이동평균_kW", ["t1", "t2", "t3"])
    claims_ok = iqg.build_night_zero_fill_claims(
        {"발전출력_3시간이동평균_kW": math.nan}, ["발전출력_3시간이동평균_kW"],
        [spec_ok], elev_all_night.get)
    assert len(claims_ok) == 1, claims_ok

    elev_mixed = {"t1": -5.0, "t2": 2.0, "t3": -1.0}  # t2만 주간
    spec_bad = iqg.LagSpec("발전출력_3시간이동평균_kW", ["t1", "t2", "t3"])
    claims_bad = iqg.build_night_zero_fill_claims(
        {"발전출력_3시간이동평균_kW": math.nan}, ["발전출력_3시간이동평균_kW"],
        [spec_bad], elev_mixed.get)
    assert claims_bad == [], claims_bad
    return True


def test_8i_night_zero_fill_claims_skips_non_missing_and_unrequired():
    """이미 값이 있는 특성이나 required_features에 없는 특성은 건너뛴다."""
    elev = {"t1": -5.0}
    spec_has_value = iqg.LagSpec("발전출력_1시간전_kW", ["t1"])
    claims = iqg.build_night_zero_fill_claims(
        {"발전출력_1시간전_kW": 3.2}, ["발전출력_1시간전_kW"], [spec_has_value], elev.get)
    assert claims == [], "값이 이미 있는데 클레임이 생성됨"

    spec_not_required = iqg.LagSpec("발전출력_1시간전_kW", ["t1"])
    claims2 = iqg.build_night_zero_fill_claims(
        {"발전출력_1시간전_kW": math.nan}, [], [spec_not_required], elev.get)
    assert claims2 == [], "required_features에 없는데 클레임이 생성됨"
    return True


def test_8j_night_zero_fill_claims_unrelated_missing_stays_unexplained():
    """야간확인 대상이 아닌 다른 결측 특성은 그대로 unexplained/blocked -
    사용자 지시: '그 외 결측은 기존처럼 차단'."""
    elev = {"t1": -5.0}
    spec = iqg.LagSpec("발전출력_1시간전_kW", ["t1"])
    row = {"발전출력_1시간전_kW": math.nan, "DSWRF": math.nan}
    claims = iqg.build_night_zero_fill_claims(
        row, ["발전출력_1시간전_kW", "DSWRF"], [spec], elev.get)
    r = iqg.evaluate(row, ["발전출력_1시간전_kW", "DSWRF"], CFG, zero_fill_claims=claims)
    assert r["status"] == opstatus.BLOCKED, r
    assert "DSWRF" in r["unexplained_features"], r
    assert "발전출력_1시간전_kW" not in r["unexplained_features"], r
    return True


def test_9_native_missing_not_auto_success():
    """LightGBM이 predict 가능한 상황(값이 실제로는 존재)이어도, 게이트가
    '결측이 있다'고 보면 무조건 blocked/degraded지 normal이 아니어야 한다."""
    row = {"a": math.nan, "b": 5.0}
    r = iqg.evaluate(row, ["a", "b"], CFG)
    assert r["status"] != opstatus.NORMAL
    assert r["status"] == opstatus.BLOCKED
    return True


def test_10_nan_rate_delta_thresholds():
    training = {"plant_output_kw_daytime": 0.095, "기상청관측_일사량_W_m2": 0.0055}
    live_ok = {"plant_output_kw_daytime": 0.125, "기상청관측_일사량_W_m2": 0.02}
    r = iqg.audit_nan_rate_delta(training, live_ok, CFG)
    assert r["종합판정"] == "ok", r  # 0.125 < warning_above(0.15)

    live_warn = {"plant_output_kw_daytime": 0.20, "기상청관측_일사량_W_m2": 0.02}
    r2 = iqg.audit_nan_rate_delta(training, live_warn, CFG)
    assert r2["종합판정"] == "warning", r2

    live_blocked = {"plant_output_kw_daytime": 0.35, "기상청관측_일사량_W_m2": 0.02}
    r3 = iqg.audit_nan_rate_delta(training, live_blocked, CFG)
    assert r3["종합판정"] == "blocked", r3
    return True


def test_11_real_live_snapshot_no_crash():
    """실제 저장된 라이브 조립 결과(읽기 전용)로 게이트가 예외 없이 도는지 확인.
    API 호출 없음 - 이미 저장된 shadow_predictions.sqlite3만 읽는다."""
    import sqlite3
    db = PARENT.parent / "outputs" / "shadow_predictions" / "shadow_predictions.sqlite3"
    if not db.is_file():
        print("  (참고) shadow DB 없음 - 이 서브테스트는 건너뜀")
        return True
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT reason FROM shadow_predictions WHERE reason IS NOT NULL LIMIT 5"
        ).fetchall()
    finally:
        conn.close()
    # reason 문자열에서 결측특성 목록을 흉내내 게이트에 넣어보는 정도의 스모크 테스트
    for (reason,) in rows:
        fake_feats = [f.strip("[]'\" ") for f in str(reason).split(",")][:3]
        fake_feats = [f for f in fake_feats if f]
        row = {f: math.nan for f in fake_feats}
        r = iqg.evaluate(row, fake_feats, CFG)
        assert r["status"] in opstatus.ALL_STATUSES
    return True


TESTS = [
    test_1_no_missing_normal, test_2_unexplained_nan_blocked,
    test_3_night_zero_explained, test_4_daytime_missing_without_night_claim_blocked,
    test_5_rainfall_no_phenomenon_explained, test_6_interpolation_within_limit_explained,
    test_7_interpolation_over_limit_unexplained, test_8_future_nwp_missing_no_auto_fill,
    test_8b_insufficient_history_separated_from_unexplained,
    test_8c_insufficient_history_still_blocks_by_default,
    test_8d_insufficient_history_does_not_leak_into_unrelated_missing,
    test_8e_night_zero_fill_claims_point_lag_confirmed_night,
    test_8f_night_zero_fill_claims_daytime_source_no_claim,
    test_8g_night_zero_fill_claims_unresolvable_source_no_claim,
    test_8h_night_zero_fill_claims_rolling_window_requires_all_night,
    test_8i_night_zero_fill_claims_skips_non_missing_and_unrequired,
    test_8j_night_zero_fill_claims_unrelated_missing_stays_unexplained,
    test_9_native_missing_not_auto_success, test_10_nan_rate_delta_thresholds,
    test_11_real_live_snapshot_no_crash,
]


def main() -> int:
    results = []
    for t in TESTS:
        try:
            ok = t()
            results.append((t.__name__, "PASS" if ok else "FAIL", ""))
        except AssertionError as e:
            results.append((t.__name__, "FAIL", str(e)))
        except Exception as e:
            results.append((t.__name__, "ERROR", "%s: %s" % (type(e).__name__, e)))
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    for name, status, msg in results:
        print("[%s] %s %s" % (status, name, ("- " + msg) if msg else ""))
    print("\n%d/%d PASS" % (n_pass, len(results)))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
