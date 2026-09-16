# -*- coding: utf-8 -*-
"""초단기 shadow 예측 실측 대조 - MAE/RMSE/nMAE + 수평별 유효 표본 수.

`live_feature_assembler_ultrashort_v1_2026-09-08.py`가 쌓아온
`shadow_predictions_ultrashort.sqlite3`의 `성공` 예측 중 target_time이
이미 지난 것만, 그 시각 실제 발전량(`plant_snapshots`, 완전가용 기준은
조립기와 동일)과 대조한다. target_time 근방(±tol_minutes) 실측이 없으면
그 표본은 평가에서 제외한다(임의보간 없음) - "아직 평가 가능한 표본이
없다"는 상태를 정직하게 보고하며, 억지로 표본을 만들지 않는다.

API 호출 없음(전부 저장된 라이브 DB 읽기). 실행할 때마다 그 시점까지
확정된 결과를 다시 계산한다(별도 저장 없이 매번 최신화 - 표본이 계속
늘어나는 값이라 스냅샷을 고정할 이유가 아직 없음).
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHADOW_DB = ROOT / "shadow_predictions_ultrashort.sqlite3"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


assembler = load_module("ultra_assembler", ROOT / "live_feature_assembler_ultrashort_v1_2026-09-08.py")
REGIONS = assembler.REGIONS
CAPACITY = {r: c["capacity"] for r, c in REGIONS.items()}
MIN_INDEPENDENT_DAYS = 5

# ★09-16 수정★: 실측 시리즈를 48시간만 읽고 있었다. 평가는
# `evaluable = due.dropna(subset=["actual_kw"])`로 실측 없는 예측을 버리므로,
# 48시간 창 = 최대 3개 캘린더 날짜 → independent_days가 **구조적으로 3을
# 못 넘었다**. MIN_INDEPENDENT_DAYS=5 게이트는 영원히 도달 불가였고
# (evaluation_status가 항상 "예비치", R²가 항상 null), 누적 성공예측
# 11,124건 중 8,884건(79.9%)이 매일 그냥 버려지고 있었다.
# 실측(plant_snapshots)은 광주 08-25·김제 09-01부터 전부 보유하고 있으므로
# 데이터가 없어서가 아니라 창이 좁아서 생긴 문제였다.
# shadow 이력 전체를 덮도록 넓힌다(조회는 5분 스냅샷 수천 행이라 가볍다).
EVAL_LOOKBACK_HOURS = 24 * 60


def load_daytime_zero_anomaly_times(cfg: dict, solar_module,
                                    lookback_hours: int = EVAL_LOOKBACK_HOURS,
                                    min_run_minutes: float = 15.0) -> pd.DatetimeIndex:
    """주간 전 인버터 합계 0kW가 연속된 구간의 시각을 반환한다.

    원본을 삭제하지 않고 기상 기반 모델 성능판정에서만 분리한다. 실제 설비
    정지와 원천 0값 송신을 DB만으로 구분할 수 없으므로 '이상 후보'다.
    """
    con = sqlite3.connect(cfg["plant_db"])
    since = (pd.Timestamp.now(tz=assembler.KST).tz_localize(None)
             - pd.Timedelta(hours=lookback_hours)).isoformat()
    df = pd.read_sql_query(
        "SELECT snapshot_time, plant_ac_power_kw, quality_status, "
        "expected_inverter_count, valid_ac_power_count FROM plant_snapshots "
        "WHERE plant_id=? AND snapshot_time>=? ORDER BY snapshot_time",
        con, params=(cfg["plant_id"], since))
    con.close()
    if df.empty:
        return pd.DatetimeIndex([])
    df["time"] = pd.to_datetime(df["snapshot_time"]).dt.tz_localize(None)
    complete = (df["quality_status"].isin(["ok", "warning"])
                & (df["expected_inverter_count"] == cfg["expected_inverters"])
                & (df["valid_ac_power_count"] == cfg["expected_inverters"]))
    daylight = df["time"].map(lambda t: assembler.solar_elev(solar_module, t) > 5.0)
    candidate = complete & daylight & (df["plant_ac_power_kw"].fillna(np.inf) <= 0.5)
    flagged: list[pd.Timestamp] = []
    run: list[pd.Timestamp] = []
    previous = None
    for t, is_candidate in zip(df["time"], candidate):
        if is_candidate and (previous is None or t - previous <= pd.Timedelta(minutes=10)):
            run.append(t)
        elif is_candidate:
            if run and (run[-1] - run[0]).total_seconds() / 60 >= min_run_minutes:
                flagged.extend(run)
            run = [t]
        else:
            if run and (run[-1] - run[0]).total_seconds() / 60 >= min_run_minutes:
                flagged.extend(run)
            run = []
        previous = t if is_candidate else None
    if run and (run[-1] - run[0]).total_seconds() / 60 >= min_run_minutes:
        flagged.extend(run)
    return pd.DatetimeIndex(flagged)


def near_anomaly(times: pd.DatetimeIndex, value, tolerance_minutes: float = 4.0) -> bool:
    if len(times) == 0:
        return False
    t = pd.Timestamp(value)
    pos = times.searchsorted(t)
    candidates = []
    if pos < len(times):
        candidates.append(times[pos])
    if pos > 0:
        candidates.append(times[pos - 1])
    return bool(candidates and min(abs(x - t) for x in candidates)
                <= pd.Timedelta(minutes=tolerance_minutes))


def evaluate(tol_minutes: float = 4.0, min_independent_days: int = MIN_INDEPENDENT_DAYS,
             representative_issues_per_day: int = 2) -> pd.DataFrame:
    con = sqlite3.connect(SHADOW_DB)
    preds = pd.read_sql_query(
        "SELECT id, region, horizon_h, issue_time_kst, target_time_kst, predicted_kw "
        "FROM shadow_ultrashort_predictions WHERE status='성공'", con)
    con.close()
    if preds.empty:
        return pd.DataFrame()
    preds["target_time_kst"] = pd.to_datetime(preds["target_time_kst"])
    now = pd.Timestamp.now(tz=assembler.KST).tz_localize(None)
    due = preds[preds["target_time_kst"] <= now].copy()
    if due.empty:
        return pd.DataFrame()
    due = (due.sort_values("id")
              .drop_duplicates(["region", "horizon_h", "issue_time_kst", "target_time_kst"], keep="last"))
    due["issue_date"] = due["issue_time_kst"].str[:10]
    due["issue_rank_in_day"] = (pd.to_datetime(due["issue_time_kst"])
                                  .groupby([due["region"], due["horizon_h"], due["issue_date"]])
                                  .rank(method="first"))
    due_rep = due[due["issue_rank_in_day"] <= representative_issues_per_day].copy()

    actual_series = {
        r: assembler.load_power_series(cfg["plant_db"], cfg["plant_id"], cfg["expected_inverters"],
                                       lookback_hours=EVAL_LOOKBACK_HOURS)
        for r, cfg in REGIONS.items()
    }

    def lookup_actual(row):
        s = actual_series.get(row["region"])
        if s is None or s.empty:
            return np.nan
        return assembler.lookup_near(s, row["target_time_kst"], tol_minutes=tol_minutes)

    due["actual_kw"] = due.apply(lookup_actual, axis=1)
    due_rep["actual_kw"] = due_rep.apply(lookup_actual, axis=1)
    due["issue_actual_kw"] = due.apply(lambda r: lookup_actual(pd.Series({"region": r["region"], "target_time_kst": pd.to_datetime(r["issue_time_kst"])})), axis=1)
    due_rep["issue_actual_kw"] = due_rep.apply(lambda r: lookup_actual(pd.Series({"region": r["region"], "target_time_kst": pd.to_datetime(r["issue_time_kst"])})), axis=1)
    evaluable = due.dropna(subset=["actual_kw"]).copy()
    representative = due_rep.dropna(subset=["actual_kw"]).copy()
    if evaluable.empty:
        return pd.DataFrame()

    evaluable["error_kw"] = evaluable["predicted_kw"] - evaluable["actual_kw"]
    solar_modules = {"부안": assembler.BUAN_SOLAR, "김제": assembler.GIMJE_SOLAR,
                     "영광": assembler.YEONGGWANG_SOLAR, "광주": assembler.GWANGJU_SOLAR}
    evaluable["target_elevation"] = evaluable.apply(
        lambda r: assembler.solar_elev(solar_modules[r["region"]], r["target_time_kst"]), axis=1)
    evaluable["period"] = np.where(evaluable["target_elevation"] > 0, "day", "night")
    evaluable["target_date"] = evaluable["target_time_kst"].dt.date
    representative["target_elevation"] = representative.apply(
        lambda r: assembler.solar_elev(solar_modules[r["region"]], r["target_time_kst"]), axis=1)
    representative["period"] = np.where(representative["target_elevation"] > 0, "day", "night")
    representative["target_date"] = representative["target_time_kst"].dt.date
    anomaly_times = {
        r: load_daytime_zero_anomaly_times(REGIONS[r], solar_modules[r]) for r in REGIONS
    }
    evaluable["operational_anomaly"] = evaluable.apply(
        lambda r: (near_anomaly(anomaly_times[r["region"]], r["issue_time_kst"])
                   or near_anomaly(anomaly_times[r["region"]], r["target_time_kst"])), axis=1)
    representative["operational_anomaly"] = representative.apply(
        lambda r: (near_anomaly(anomaly_times[r["region"]], r["issue_time_kst"])
                   or near_anomaly(anomaly_times[r["region"]], r["target_time_kst"])), axis=1)
    evaluable["persistence_kw"] = evaluable["issue_actual_kw"]
    target_clear = np.maximum(np.sin(np.deg2rad(evaluable["target_elevation"])), 0.0)
    issue_elev = evaluable.apply(lambda r: assembler.solar_elev(solar_modules[r["region"]], pd.to_datetime(r["issue_time_kst"])), axis=1)
    issue_clear = np.maximum(np.sin(np.deg2rad(issue_elev)), 0.0)
    evaluable["clear_sky_persistence_kw"] = np.where(issue_clear > 1e-6, evaluable["issue_actual_kw"] * target_clear / issue_clear, 0.0)
    rows = []
    for (region, h, period), raw_g in evaluable.groupby(["region", "horizon_h", "period"]):
        g = raw_g[~raw_g["operational_anomaly"]].copy()
        if g.empty:
            continue
        mse = float((g["error_kw"] ** 2).mean())
        mae = float(g["error_kw"].abs().mean())
        rmse = float(np.sqrt(mse))
        cap = CAPACITY[region]
        # R² = 1 - SS_res/SS_tot. n이 작거나(예비치 구간 다수) 실측값 분산이
        # 거의 0(예: 야간 실측이 전부 0.x kW 근처)이면 분모가 0에 가까워져
        # 값이 무의미해진다 - 억지로 숫자를 만들지 않고 null로 정직하게 둔다.
        ss_res = float((g["error_kw"] ** 2).sum())
        ss_tot = float(((g["actual_kw"] - g["actual_kw"].mean()) ** 2).sum())
        independent_days = int(g["target_date"].nunique())
        # 5분 간격 반복예측 수백 건은 서로 독립 표본이 아니다. 독립일수가
        # 게이트에 못 미치면 R²가 날씨 하루에 따라 과대/과소 변동하므로
        # 숫자를 노출하지 않는다. 원시 예측·실측 행은 그대로 보존한다.
        r2 = (round(1 - ss_res / ss_tot, 4)
              if independent_days >= min_independent_days and ss_tot > 1e-6 else None)
        rep_g = representative[(representative["region"] == region) &
                               (representative["horizon_h"] == h) &
                               (representative["period"] == period) &
                               (~representative["operational_anomaly"])]
        raw_mae = float(raw_g["error_kw"].abs().mean())
        rows.append({
            "region": region, "horizon_h": h, "period": period, "n": len(g),
            "raw_n": len(raw_g),
            "operational_anomaly_excluded_n": int(raw_g["operational_anomaly"].sum()),
            "independent_days": independent_days,
            "representative_n": int(len(rep_g)),
            "effective_N": int(rep_g["target_date"].nunique()),
            "representative_issues_per_day": representative_issues_per_day,
            "evaluation_status": "예비치" if independent_days < min_independent_days else "평가가능",
            "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3), "MSE_kW2": round(mse, 3),
            "R2": r2,
            "nMAE_pct": round(mae / cap * 100, 3),
            # ★09-16★ nMAE 분모를 화면에서 바로 확인할 수 있게 노출한다
            # (09-16 4지역 기준을 발전소 API 정격(AC)으로 통일 - AGENTS.md 09-16(13)).
            "capacity_kw": cap,
            "nMAE_basis": "발전소 API 정격(AC)",
            "raw_MAE_kW": round(raw_mae, 3),
            "raw_nMAE_pct": round(raw_mae / cap * 100, 3),
            "persistence_MAE_kW": round(float((g["persistence_kw"] - g["actual_kw"]).abs().mean()), 3),
            "clear_sky_persistence_MAE_kW": round(float((g["clear_sky_persistence_kw"] - g["actual_kw"]).abs().mean()), 3),
        })
    return pd.DataFrame(rows).sort_values(["region", "horizon_h"])


def main() -> None:
    result = evaluate()
    if result.empty:
        con = sqlite3.connect(SHADOW_DB)
        pending = pd.read_sql_query(
            "SELECT MIN(target_time_kst) AS earliest_due "
            "FROM shadow_ultrashort_predictions WHERE status='성공'", con)
        con.close()
        earliest = pending["earliest_due"].iloc[0] if not pending.empty else None
        print(f"아직 target_time이 도래한 평가 가능 표본이 없습니다"
             f"(가장 이른 target_time: {earliest}). 나중에 다시 실행하세요.")
        return
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
