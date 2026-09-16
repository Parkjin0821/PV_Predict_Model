# -*- coding: utf-8 -*-
"""★라이브 특성조립기 — 단기(+1h/+24h/+48h) 전용★

## 범위(의도적으로 좁힘)
단기 3개 수평만 먼저 완전하게 만든다. 초단기(카파변환·+4h 날씨군집화)와
일간(30~32일 이력·corrected_dataset 재사용)은 구조가 더 복잡해 뒤에
같은 패턴으로 확장한다(이 파일 맨 아래 "확장 메모" 참고).

## 원본 공식 추적 결과(재구현 아님 — 전부 기존 스크립트에서 그대로 가져옴)
- **BSRN 물리상한 정제**(DSWRFLX·DIFSWRF → `_bsrn정제`):
  `02_전처리/clean_nwp_direct_diffuse_bsrn_qc_v1.py`의
  `solar_position()`·`bsrn_ppl_bounds()` 그대로 재사용.
- **설비 특성 집계 공식**(`90_과거실험/v3_다중수평/
  build_train_gwangju_multihorizon.py:108-116`에서 원본 확인):
  - `plant_input_power_kw` = Σ(입력전력=dc_power), **min_count=5**
    (5대 전부 있어야 유효값, 하나라도 없으면 NaN)
  - `mean_input_voltage_v` = mean(입력전압=dc_volt)
  - `mean_power_factor` = mean(역률=pf)
  - `plant_output_kw`(지속성·lag용) = Σ(출력전력=ac_power), min_count=5
- **태양고도·주기 특성**: `harness.build_frame()`이 이미 계산 —
  재구현 없이 그대로 씀(golden replay로 이미 검증됨).
- **NWP/동네예보 소스 분리**: DSWRF·DSWRFLX·DIFSWRF·TCDC·LCDC는 NWP
  (`nwp_values`), SKY·REH·POP·WSD·TMP·VEC는 동네예보(`grid_forecast`) —
  Codex 파이프라인 문서(08-25) 설명과 일치.
- **VEC(풍향) 원형보간**(08-26 추가): `interpolate_nwp_3h_to_1h_v2_
  fixed_tm_2026-08-21.py`의 `interp_circular()` 그대로 재사용 — 일반
  선형보간이 아니라 sin/cos 성분으로 분해해 보간 후 각도로 되돌린다.
  수집기(`collect_kma_asos_grid_live_v1_2026-08-25.py`)의 `GRID_VARS`에도
  VEC를 추가해야 실제 값이 들어온다(코드는 연결됐으나 수집 재개는 별도
  선행조건 — AGENTS.md "다음 순서" 참고).
- **추정_일조시간_hr**(08-26 추가): `build_official_hourly_dataset_v3_
  fixed_tm_2026-08-21.py`의 `derive_sunshine()`과 동일 공식(재구현 아님)
  — 별도 API가 아니라 DSWRFLX_bsrn정제로 계산하는 파생특성이다. 현재
  라이브 DSWRFLX가 전 시각 결측이라(AGENTS.md 참고) 코드는 연결됐어도
  결과는 그대로 결측이다.

## ★검증 필요 표시(정직하게 남김)★
- `plant_snapshots.plant_ac_power_kw`/`plant_dc_power_kw`가 위 min_count=5
  합산 규칙과 완전히 같은 방식으로 Codex 수집기에서 계산됐는지는
  **직접 실행해서 대조 못 해봤다**(Blockdata 이력이 아직 1시간 미만이라
  golden replay 같은 완전일치 검증이 불가능했음). 이 파일은 대신
  `inverter_measurements`에서 인버터별 원값을 직접 합산/평균해
  **위 공식을 스스로 재현**하도록 짰다(plant_snapshots의 사전집계값에
  의존하지 않음) — 이러면 Codex 수집기의 집계 로직과 무관하게 원본
  공식과 확실히 일치한다.
- 첫 실제 실행 결과(행수·값 범위)를 반드시 `production_readiness_check`
  류로 한 번 더 감사할 것 — 특히 min_count=5 때문에 인버터 5대가 전부
  안 잡히면 그 시각은 자연히 NaN(가짜값 금지 원칙 유지).

## 사용법
```python
from live_feature_assembler_단기_v1_2026-08-26 import build_live_hourly, assemble_short_row
hourly = build_live_hourly(end_time=pd.Timestamp.now(tz="Asia/Seoul"), lookback_hours=72)
row = assemble_short_row(hourly, horizon=24, bundle=단기h24_번들)
```
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KST = ZoneInfo("Asia/Seoul")
# ★09-16 신규★: ASOS 실측 지연 p95=552.6분(9.2시간) 기준으로 여유를 두고
# 12시간까지 뒤로 탐색한다(assemble_and_predict의 동적 issue_time 탐색).
ASOS_LOOKBACK_HOURS = 12
BLOCK_DB = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\blockdata_live_history_v1_2026-08-25\blockdata_history.sqlite3")
KMA_DB = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3")
DIF = "DIFSWRF_bsrn정제"
DSX = "DSWRFLX_bsrn정제"
GWANGJU_INVERTER_CLIP_CAPACITY_KW = 241.58


def _load(name: str, filename: str, rel: str = "."):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("live_asm_harness", "backtest_harness_v1_2026-08-20밤.py")
bsrn = _load("live_asm_bsrn", "clean_nwp_direct_diffuse_bsrn_qc_v1.py", rel="../../02_전처리")
nwp_interp = _load("live_asm_nwp_interp", "interpolate_nwp_3h_to_1h_v2_fixed_tm_2026-08-21.py", rel="../../02_전처리")
infer = _load("live_asm_infer", "production_inference_utils_v1_2026-08-25.py")
sel_v3 = _load("live_asm_sel", "select_features_by_correlation_threshold_v3_2026-08-20밤.py")
# 08-28 실배선(사용자 지시 ②, 좁은 범위): 발전출력 lag/이동통계 결측 중
# 원천시각이 물리적 야간(태양고도<=0)으로 확인된 것만 0허용, 나머지는
# 기존과 동일하게 차단. 회귀검사(실제 성공행 재생 4/4 완전동일, 단기
# 3종 현재 대기사유·결측특성셋 동일) 통과 후 적용 - AGENTS.md 08-28 참고.
gatewiring = _load("live_asm_gatewiring", "live_gate_wiring_v1_2026-08-28.py",
                   rel="운영안전장치_2026-08-28")
GATE_CFG = gatewiring.iqg.load_config()

# Faiman(2008)/Sandia PVPMC 계수 — derive_gwangju_module_temp_v1.py와 동일값 재사용
_U0, _U1, _DELTA_T_CND = 25.0, 6.84, 3.0
NWP_VARS = ["DSWRF", "DSWRFLX", "DIFSWRF", "TCDC", "LCDC", "MCDC", "HCDC"]
GRID_VARS = ["TMP", "SKY", "REH", "WSD", "POP", "VEC"]  # 08-26: VEC(풍향) 추가

# 추정_일조시간_hr 상수·정의 — 출처: derive_gwangju_future_sunshine_v1.py의
# THRESHOLD_W_M2/DAYLIGHT_COVERAGE_MIN, build_official_hourly_dataset_v3_
# fixed_tm_2026-08-21.py의 derive_sunshine()과 완전히 동일한 공식(재구현
# 아님) — 기상청 정의(직달일사 120W/m^2 이상 누적시간)를 DSWRFLX_bsrn정제로
# 재현한다.
SUNSHINE_THRESHOLD_W_M2 = 120.0
SUNSHINE_DAYLIGHT_COVERAGE_MIN = 0.90


def _equipment_hourly(end_time: pd.Timestamp, lookback_hours: int) -> pd.DataFrame:
    """인버터 원값을 5분→1시간 평균으로 재집계한다.

    학습 파이프라인처럼 인버터별 5분 평균을 먼저 만들고 발전소 합계를
    계산한 뒤 1시간 평균을 낸다. 시간당 마지막 스냅샷 하나만 쓰지 않는다.
    """
    five = _equipment_5min(end_time, lookback_hours)
    if five.empty:
        return five
    count = five["plant_output_kw"].resample("1h").count()
    out = five.resample("1h").mean(numeric_only=True)
    out["measurement_count_5min"] = count
    incomplete = count < 9  # 학습 1시간 집계의 75%(12개 중 9개)
    live_cols = ["plant_output_kw", "plant_input_power_kw", "plant_input_current_a",
                 "mean_input_voltage_v", "mean_power_factor", "mean_frequency_hz"]
    out.loc[incomplete, live_cols] = np.nan
    out.index.name = "time"
    return out


def _equipment_5min(end_time: pd.Timestamp, lookback_hours: int) -> pd.DataFrame:
    """API 응답(스냅샷) 단위로 인버터 5대를 귀속 → 발전소 합산.

    ## 08-27 재설계 배경(Codex 읽기전용 감사로 원인 확정)
    이전 버전은 각 인버터 측정행을 `measurement_time.floor("5min")`으로
    독립적으로 5분 슬롯에 배정했다. 그런데 **같은 API 응답 안에서도
    인버터별 측정시각이 중앙값 113초까지 벌어지고, 그 결과 15.5%는
    같은 응답의 5대가 서로 다른 5분 슬롯으로 갈라졌다**(Codex 실측).
    `min_count=5`(5대 전부 있어야 유효) 문턱 자체는 학습 파이프라인의
    `aggregate_power(required_fraction=0.75)`와 동일 계승이라 문제가
    아니었다 — 문제는 "5분 슬롯 배정 방법"이었다(라이브 Blockdata API
    에만 있는 배치-응답 구조, 엑셀 로그 기반 학습 데이터엔 이 문제
    자체가 없음 — 그래서 재학습 불필요, 라이브 조립기만 수정).

    **새 방법**: 5분 슬롯으로 개별 floor하지 않고, **같은 API 응답
    (`response_hash`)에 속한 인버터를 전부 그 응답의 대표시각
    (`plant_snapshots.snapshot_time`)에 귀속**한다 — 응답 자체가 이미
    "이 5대는 같은 순간을 찍은 것"이라는 가장 확실한 그룹 단위이므로
    시간 기반 재정렬보다 근본적으로 정확하다. 품질게이트는 Codex가
    검증한 값 그대로: 5대 수신·5대 유효·용량합 219kW·측정시각분산
    ≤600초(10분) 전부 통과해야 유효 스냅샷으로 채택.
    (실측: 이 방식으로 낮시간 완전슬롯률 81.9%→시간단위 준비율 91.7%,
    기존 방식과 공통 유효시간의 출력차 평균 0.41kW로 작음 — Codex 절 참고)
    """
    if not BLOCK_DB.is_file():
        raise FileNotFoundError(f"Blockdata 이력 DB 없음: {BLOCK_DB}")
    start = (end_time - timedelta(hours=lookback_hours)).tz_localize(None).isoformat()
    end = end_time.tz_localize(None).isoformat()
    conn = sqlite3.connect(BLOCK_DB)
    try:
        snaps = pd.read_sql_query(
            "SELECT snapshot_time, response_hash, received_inverter_count, "
            "valid_ac_power_count, registered_capacity_sum_kw, measurement_spread_seconds "
            "FROM plant_snapshots WHERE snapshot_time >= ? AND snapshot_time <= ?",
            conn, params=(start, end),
        )
        inv = pd.read_sql_query(
            "SELECT response_hash, inverter_number, ac_power, dc_power, dc_volt, "
            "dc_current, pf, freq FROM inverter_measurements "
            "WHERE measurement_time >= ? AND measurement_time <= ?",
            conn, params=(start, end),
        )
    finally:
        conn.close()
    if snaps.empty or inv.empty:
        return pd.DataFrame()

    # 품질게이트(Codex 실측 임계 그대로): 5대 수신·5대 유효·등록용량합
    # 219kW·측정시각분산 10분 이내만 유효 스냅샷으로 채택.
    # ★09-02 수정★: registered_capacity_sum_kw가 09-01 18:04를 기점으로
    # 219.0 → 240.58로 계단식 전환된 걸 실측 확인(1,297건 219.0 →
    # 그 이후 전부 240.58, 중간값 없음 - 노이즈가 아니라 Blockdata
    # 원천의 등록용량 값 자체가 바뀐 것). 어느 쪽이 "진짜" 정격인지는
    # 아직 배만수 부장님 확인 대기(AGENTS.md 09-02 capacity 감사 참고,
    # 임의 결론 내지 않음) - 그와 별개로 이 게이트의 본래 목적은
    # "5대가 뒤섞이지 않은 진짜 스냅샷인지" 판별이므로, 정격 확정 전까지
    # 임시로 두 값 다 허용해 게이트가 실제 정상 데이터를 막지 않게 한다.
    # 정격이 확정되면 이 조건을 다시 단일 범위로 좁힐 것.
    # ★09-14 주말루프 52회차 추가★: registered_capacity_sum_kw가 09-09
    # 20:38을 기점으로 또 240.58 → 241.58로 계단식 전환된 걸 실측
    # 확인(그 시점부터 09-14 09시 확인 시점까지 5일간 789건 전부
    # 241.58, 중간값 없음 - 위와 동일한 성격의 전환). 이 셋째 값이
    # 허용범위 밖이라 09-09 20:38 이후 광주 품질게이트 통과 스냅샷이
    # 0건이 되어 `live_feature_assembler_단기_v2_pooled_2026-09-10.py`의
    # gwangju() 경로가 KeyError('plant_output_kw')로 죽는 원인이었음
    # (UCUBE_ShortPooled24h_Shadow_Daily1430 09-13 14:30 실패 등 - 로그가
    # 없어 그동안 "ASOS 기근 연쇄영향"으로 오판했었음, 09-14 로깅 추가
    # 후 트레이스백으로 확정). 같은 원칙(정격 미확정, 임의 결론 금지)으로
    # 셋째 값도 임시 허용 - 정격 확정되면 세 범위 전부 재검토할 것.
    good = snaps[
        (snaps["received_inverter_count"] == 5)
        & (snaps["valid_ac_power_count"] == 5)
        & (
            snaps["registered_capacity_sum_kw"].between(218.9, 219.1)
            | snaps["registered_capacity_sum_kw"].between(240.48, 240.68)
            | snaps["registered_capacity_sum_kw"].between(241.48, 241.68)
        )
        & (snaps["measurement_spread_seconds"] <= 600)
    ].copy()
    if good.empty:
        return pd.DataFrame()
    good["snapshot_time"] = pd.to_datetime(good["snapshot_time"], utc=True).dt.tz_convert(KST).dt.tz_localize(None)

    # 학습과 동일하게 개별 인버터 출력의 물리범위를 합산 전에 적용한다
    # (plant_snapshots의 사전집계값을 그대로 믿지 않고, 인버터 원값을
    # 직접 클리핑·합산해 학습 공식과 정확히 같은 경로를 재현한다).
    inv.loc[~inv["ac_power"].between(0.0, 80.0), "ac_power"] = np.nan
    inv.loc[~inv["dc_power"].between(0.0, 80.0), "dc_power"] = np.nan
    per_resp = inv.groupby("response_hash")
    resp_agg = pd.DataFrame({
        "plant_output_kw": per_resp["ac_power"].sum(min_count=5),
        "plant_input_power_kw": per_resp["dc_power"].sum(min_count=5),
        "plant_input_current_a": per_resp["dc_current"].sum(min_count=5),
        "mean_input_voltage_v": per_resp["dc_volt"].mean(),
        "mean_power_factor": per_resp["pf"].mean(),
        "mean_frequency_hz": per_resp["freq"].mean(),
        "inverters_available": per_resp["ac_power"].count(),
    })

    good = good.set_index("response_hash").join(resp_agg, how="inner")
    good = good.set_index("snapshot_time")
    out = pd.DataFrame({
        "plant_output_kw": good["plant_output_kw"],
        "plant_input_power_kw": good["plant_input_power_kw"],
        "plant_input_current_a": good["plant_input_current_a"],
        "mean_input_voltage_v": good["mean_input_voltage_v"],
        "mean_power_factor": good["mean_power_factor"],
        "mean_frequency_hz": good["mean_frequency_hz"],
        "inverters_available": good["inverters_available"],
        # Blockdata 실시간 스키마에는 인버터 온도·통신상태 필드가 없다
        # (inverter_measurements에 없음) — 과거 학습 데이터는 별도 로그
        # 출처였던 것으로 보인다. 후보 컬럼 목록엔 있어야 하므로 결측으로
        # 채운다(native missing — 임의값 금지, 어차피 08-25 확정 8개
        # 운영모델의 features 목록엔 이 두 컬럼이 안 뽑혀있어 예측에는
        # 영향 없음, production_readiness_check로 재확인 가능).
        "mean_inverter_temperature_c": np.nan,
        "mean_communication_ok": np.nan,
    })
    out.index.name = "time"
    return out


def _nwp_hourly(end_time: pd.Timestamp, lookback_hours: int,
                future_hours: int = 0) -> pd.DataFrame:
    if not KMA_DB.is_file():
        raise FileNotFoundError(f"기상청 라이브 DB 없음: {KMA_DB}")
    start_naive = (end_time - timedelta(hours=lookback_hours)).tz_localize(None)
    issue_naive = end_time.tz_localize(None)
    # 보수적 재분배는 3시간 앵커 h가 h-2,h-1,h를 만든다. 요청 목표의
    # 1시간 값을 얻으려면 뒤쪽 앵커가 최대 2시간 더 필요하다.
    weather_end_naive = (end_time + timedelta(hours=future_hours + 2)).tz_localize(None)
    conn = sqlite3.connect(KMA_DB)
    try:
        # 각 target_time_kst별로 "발표시각이 가장 늦은(=가장 최신) 값"만 남긴다
        # (같은 target을 여러 발표일이 다시 예보했을 수 있어, 최신 예보를 우선).
        nwp = pd.read_sql_query(
            "SELECT target_time_kst, variable, value, is_missing, planned_issue_at_kst, "
            "first_received_at "
            "FROM nwp_values WHERE target_time_kst >= ? AND target_time_kst <= ? "
            "AND planned_issue_at_kst <= ? AND first_received_at <= ? "
            "AND variable IN ({})".format(",".join("?" * len(NWP_VARS))),
            conn, params=[start_naive.isoformat(), weather_end_naive.isoformat(),
                          end_time.isoformat(), end_time.isoformat(), *NWP_VARS],
        )
        grid = pd.read_sql_query(
            "SELECT target_time_kst, variable, value, is_missing, run_time_kst, "
            "first_received_at FROM grid_forecast "
            "WHERE target_time_kst >= ? AND target_time_kst <= ? "
            "AND run_time_kst <= ? AND first_received_at <= ? AND variable IN ({})".format(
                ",".join("?" * len(GRID_VARS))),
            conn, params=[start_naive.isoformat(), weather_end_naive.isoformat(),
                          end_time.isoformat(), end_time.isoformat(), *GRID_VARS],
        )
        asos = pd.read_sql_query(
            "SELECT observation_time, temperature_c, rainfall_mm, wind_speed_m_s, "
            "wind_direction_deg, humidity_pct, local_pressure_hpa, sea_pressure_hpa, "
            "sunshine_hr, solar_w_m2, snow_cm, cloud_pct, ground_temperature_c "
            "FROM asos_hourly WHERE observation_time >= ? AND observation_time <= ?",
            conn, params=[start_naive.isoformat(), issue_naive.isoformat()],
        )
    finally:
        conn.close()

    def _naive(s: pd.Series) -> pd.Series:
        """DB에 저장된 시각 문자열에 타임존 오프셋이 섞여 있어도(예:
        +09:00) 전부 tz정보 없는(naive) 타임스탬프로 통일한다 — 최초
        실행에서 join 시 tz-naive/tz-aware 혼재 오류를 실제로 잡았다."""
        t = pd.to_datetime(s)
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_localize(None)
        return t

    def pivot_latest(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["target_time_kst"] = _naive(df["target_time_kst"])
        sort_col = "planned_issue_at_kst" if "planned_issue_at_kst" in df.columns else "run_time_kst"
        if sort_col in df.columns:
            df = df.sort_values(sort_col).drop_duplicates(
                ["target_time_kst", "variable"], keep="last")
        df.loc[df["is_missing"] == 1, "value"] = np.nan
        return df.pivot(index="target_time_kst", columns="variable", values="value")

    nwp_p = pivot_latest(nwp)
    grid_p = pivot_latest(grid)

    # 학습 데이터와 동일한 3시간→1시간 재구성. 일사량은 청천곡선 모양을
    # 보존해 3시간 평균을 재분배하고, 운량은 시간선형, 동네예보는 변수별
    # 선형/최근접 보간을 쓴다. 단순 reindex/forward-fill로 바꾸지 않는다.
    anchor_indices = [x.index for x in (nwp_p, grid_p) if not x.empty]
    if anchor_indices:
        start = min(x.min() for x in anchor_indices) - timedelta(hours=2)
        stop = max(x.max() for x in anchor_indices)
        hourly_index = pd.date_range(start, stop, freq="1h")
        geo = nwp_interp.solar_geometry(hourly_index)
        rebuilt_nwp = pd.DataFrame(index=hourly_index)
        if "DSWRF" in nwp_p:
            rebuilt_nwp["DSWRF"] = nwp_interp.disaggregate_conservative(
                nwp_p["DSWRF"], hourly_index, geo, "청천GHI_W_m2")
        if "DSWRFLX" in nwp_p:
            clean, _ = nwp_interp.bsrn_clip_window(nwp_p["DSWRFLX"], geo, "dni")
            rebuilt_nwp["DSWRFLX"] = nwp_interp.disaggregate_conservative(
                clean, hourly_index, geo, "청천DNI_W_m2")
        if "DIFSWRF" in nwp_p:
            clean, _ = nwp_interp.bsrn_clip_window(nwp_p["DIFSWRF"], geo, "dhi")
            rebuilt_nwp["DIFSWRF"] = nwp_interp.disaggregate_conservative(
                clean, hourly_index, geo, "청천DHI_W_m2")
        for c in ("TCDC", "LCDC", "MCDC", "HCDC"):
            if c in nwp_p:
                rebuilt_nwp[c] = nwp_interp.interp_linear(
                    nwp_p[c], hourly_index).clip(0.0, 1.0)
        rebuilt_grid = pd.DataFrame(index=hourly_index)
        for c in ("TMP", "REH", "WSD", "POP"):
            if c in grid_p:
                s = nwp_interp.interp_linear(grid_p[c], hourly_index)
                if c == "REH":
                    s = s.clip(0.0, 100.0)
                elif c == "WSD":
                    s = s.clip(lower=0.0)
                elif c == "POP":
                    s = s.clip(0.0, 100.0)
                rebuilt_grid[c] = s
        if "SKY" in grid_p:
            rebuilt_grid["SKY"] = nwp_interp.interp_nearest(grid_p["SKY"], hourly_index)
        if "VEC" in grid_p:
            # 풍향은 일반 선형보간 금지 — sin/cos 성분으로 분해해 보간한 뒤
            # 각도로 되돌리는 원형보간(interpolate_nwp_3h_to_1h_v2의
            # interp_circular과 완전히 동일한 함수, 재구현 아님).
            rebuilt_grid["VEC"] = nwp_interp.interp_circular(grid_p["VEC"], hourly_index)
        nwp_p = rebuilt_nwp
        grid_p = rebuilt_grid
    if not asos.empty:
        asos["observation_time"] = _naive(asos["observation_time"])
        asos_p = asos.set_index("observation_time")
    else:
        asos_p = pd.DataFrame()

    out = pd.concat([nwp_p, grid_p.add_prefix("GRID_")], axis=1)
    out = out.join(asos_p, how="outer")
    out.index.name = "time"
    return out.sort_index()


def _derive_estimated_sunshine_hours(hourly: pd.DataFrame) -> pd.Series:
    """일별 `추정_일조시간_hr` — 기상청 정의(직달일사강도 120W/m^2 이상인
    시간의 누적)를 실측 대신 1시간 재구성 `DSWRFLX_bsrn정제`로 재현한다.
    `build_official_hourly_dataset_v3_fixed_tm_2026-08-21.py`의
    `derive_sunshine()`과 동일 공식(재구현 아님) — 낮시간(태양고도>0) 대비
    유효(non-NaN)비율이 90% 미만인 날은 임의로 채우지 않고 결측 유지한다.

    ★라이브 스트리밍 보정★: 배치 원본은 날짜별 24시간이 항상 있지만
    라이브 lookback/future 창은 하루 일부만 포함할 수 있다. 프레임 안에
    들어온 낮시간 수를 분모로 쓰면 일부 2시간만 있어도 100%로 오판하므로,
    각 날짜의 00~23시 태양고도를 별도로 계산한 `전체 낮시간 수`를 분모로
    쓴다. 창 경계일은 안전하게 결측이 되며 낮은 가짜 일조시간을 만들지 않는다.
    """
    day = hourly.index.normalize()
    tmp = hourly[["solar_elevation_deg", DSX]].copy()
    tmp["날짜"] = day
    daylight = tmp[tmp["solar_elevation_deg"] > 0]
    rows: dict = {}
    for date in pd.Index(day.unique()).sort_values():
        g = daylight[daylight["날짜"] == date]
        full_day = pd.date_range(pd.Timestamp(date), periods=24, freq="1h")
        full_elev = [bsrn.solar_position(
            t.tz_localize(KST).astimezone(ZoneInfo("UTC")).replace(tzinfo=None))[0]
            for t in full_day]
        n_total = int((np.asarray(full_elev) > 0).sum())
        n_valid = g[DSX].notna().sum()
        coverage = n_valid / n_total if n_total else 0.0
        if coverage >= SUNSHINE_DAYLIGHT_COVERAGE_MIN:
            rows[date] = float((g[DSX].dropna() >= SUNSHINE_THRESHOLD_W_M2).sum())
        else:
            rows[date] = np.nan
    # build_official_hourly_dataset_v3의 `merged.index.normalize().map(sunshine)`과
    # 동일 패턴 — day는 hourly.index와 같은 길이·순서라 위치 정렬이 곧 라벨 정렬이다.
    return pd.Series(day.map(pd.Series(rows, dtype=float)).to_numpy(), index=hourly.index)


def build_live_hourly(end_time: pd.Timestamp, lookback_hours: int = 72,
                      future_hours: int = 0) -> pd.DataFrame:
    """v3_고정tm hourly CSV와 같은 스키마를 라이브 원자료로 재조립한다.
    (golden replay가 검증한 것과 동일하게, 이 프레임을 harness.build_frame()에
    그대로 넣으면 학습 때와 같은 특성이 나온다 — build_frame 자체는 안 바꿈)"""
    equip = _equipment_hourly(end_time, lookback_hours)
    weather = _nwp_hourly(end_time, lookback_hours, future_hours=future_hours)
    hourly = equip.join(weather, how="outer").sort_index()
    if hourly.empty:
        return hourly
    # 목표시각 태양고도는 API가 아니라 천문식으로 계산 가능하므로, 예보
    # 구간 끝까지 시간격자를 보장한다.
    full_index = pd.date_range(hourly.index.min(),
                               end_time.tz_localize(None) + timedelta(hours=future_hours),
                               freq="1h")
    hourly = hourly.reindex(full_index)

    # 태양고도(solar_elevation_deg) — build_frame()이 df에서 직접 찾는
    # 필수 베이스 컬럼(candidate_cols에는 없음, 별도로 반드시 있어야 함).
    # BSRN 정제에도 같은 값을 재사용한다(중복계산 안 함).
    elev_sa = [bsrn.solar_position(t.tz_localize(KST).astimezone(ZoneInfo("UTC")).replace(tzinfo=None))
               for t in hourly.index]
    hourly["solar_elevation_deg"] = [e for e, s in elev_sa]

    # BSRN 물리상한 정제(clean_nwp_direct_diffuse_bsrn_qc_v1과 동일 공식)
    if "DSWRFLX" in hourly.columns or "DIFSWRF" in hourly.columns:
        bounds = [bsrn.bsrn_ppl_bounds(e, s) for e, s in elev_sa]
        dni_lo = np.array([b[0] for b in bounds]); dni_hi = np.array([b[1] for b in bounds])
        dhi_lo = np.array([b[2] for b in bounds]); dhi_hi = np.array([b[3] for b in bounds])
        if "DSWRFLX" in hourly.columns:
            lx = hourly["DSWRFLX"].to_numpy(float)
            bad = (lx < dni_lo) | (lx > dni_hi)
            hourly[DSX] = np.where(bad, np.nan, lx)
        if "DIFSWRF" in hourly.columns:
            dif = hourly["DIFSWRF"].to_numpy(float)
            bad = (dif < dhi_lo) | (dif > dhi_hi)
            hourly[DIF] = np.where(bad, np.nan, dif)

    # 추정_일조시간_hr(파생특성, 별도 API 불필요 — DSWRFLX_bsrn정제로 계산)
    if DSX in hourly.columns:
        hourly["추정_일조시간_hr"] = _derive_estimated_sunshine_hours(hourly)

    # 추정_출력온도(Faiman+Sandia, derive_gwangju_module_temp_v1과 동일 공식)
    # 입력: T_air=동네예보 TMP, WS=동네예보 WSD, G=NWP DSWRF
    if {"GRID_TMP", "GRID_WSD", "DSWRF"}.issubset(hourly.columns):
        t_air = hourly["GRID_TMP"].to_numpy(float)
        ws = hourly["GRID_WSD"].to_numpy(float)
        g = hourly["DSWRF"].to_numpy(float)
        t_module = t_air + g / (_U0 + _U1 * ws)
        hourly["추정_모듈표면온도"] = t_module
        hourly["추정_출력온도"] = t_module + (g / 1000.0) * _DELTA_T_CND

    # 컬럼명 최종 정합(harness/candidate_cols가 기대하는 이름으로) — ASOS
    # 원본 컬럼명 → OBSERVED_COLUMNS 명명 규칙 그대로 매핑.
    rename = {
        "GRID_SKY": "SKY", "GRID_REH": "REH", "GRID_POP": "POP", "GRID_TMP": "TMP",
        "GRID_VEC": "VEC",
        "solar_w_m2": "기상청관측_일사량_W_m2", "temperature_c": "기상청관측_기온_C",
        "rainfall_mm": "기상청관측_강수량_mm", "wind_speed_m_s": "기상청관측_풍속_m_s",
        "wind_direction_deg": "기상청관측_풍향_deg", "humidity_pct": "기상청관측_상대습도_pct",
        "local_pressure_hpa": "기상청관측_현지기압_hPa", "sea_pressure_hpa": "기상청관측_해면기압_hPa",
        "sunshine_hr": "기상청관측_일조시간_hr", "snow_cm": "기상청관측_적설_cm",
        "cloud_pct": "기상청관측_전운량_pct", "ground_temperature_c": "기상청관측_지면온도_C",
    }
    hourly = hourly.rename(columns=rename)

    # ★08-27 버그 수정(train-serve skew)★: ASOS는 무강수·무적설 시각을
    # NULL로 보내는데, 학습 hourly CSV(v3_고정tm)에서는 이 두 컬럼의
    # NaN율이 정확히 0.000(710일 전수 확인, 0값 비율 각각 93.7%·96.7%)
    # — 즉 학습 파이프라인 어딘가(원본 수집·전처리 단계)에서 이미
    # "무강수/무적설=0"으로 인코딩돼 있었다. 라이브만 NaN으로 새면
    # 학습 때 한 번도 못 본 결측 패턴이 생겨 일간 모델(이 두 컬럼을
    # 실제로 쓰는 유일한 티어)이 영구히 대기 상태에 갇힌다. 임의값
    # 주입이 아니라 "ASOS 규격상 무강수·무적설의 정식 표현이 0"이라는
    # 근거로, 그리고 학습분포와 정확히 일치시키기 위해 0으로 채운다
    # (다른 관측 컬럼은 그대로 NaN 유지 — 이 둘만 해당).
    for c in ("기상청관측_강수량_mm", "기상청관측_적설_cm"):
        if c in hourly.columns:
            # ★추가 수정(같은 날 재검증 중 발견)★: SQLite에서 무강수·
            # 무적설 시각이 Python None으로 와서 이 컬럼이 object dtype이
            # 됐다 — fillna(0.0)만으로는 dtype이 float으로 안 올라가서
            # 뒷단의 groupby().mean(numeric_only=True)가 컬럼째로
            # 걸러내는 2차 결함이 있었다(직접 재현·확인함). to_numeric으로
            # 명시적으로 float 캐스팅까지 해야 완전히 고쳐진다.
            hourly[c] = pd.to_numeric(hourly[c], errors="coerce").fillna(0.0)

    # ★안전망★: candidate_cols(STARTLABEL+OBSERVED+FORECAST) 중 위에서
    # 실제로 못 채운 컬럼은 결측으로라도 반드시 만들어둔다. build_frame이
    # candidate_cols 전체를 순회하며 df[c]를 찾으므로, 컬럼 자체가 없으면
    # KeyError로 죽는다(최초 실행에서 실제로 겪은 버그) — 값을 모르면
    # 결측 처리하되(임의값 금지), 컬럼 존재 자체는 보장한다. VEC·
    # 추정_일조시간_hr은 08-26에 코드 연결이 끝났으나(위 참고), VEC는
    # 라이브 수집기가 실제로 값을 채워 보내기 전까지는 여전히 이 안전망을
    # 거쳐 결측으로 채워진다. 이 8개 운영모델의 실제 features에는 두 컬럼
    # 다 안 뽑혀있어(운영모델_목록.json으로 재확인 가능) 예측 정확도에는
    # 영향 없다.
    for c in sel_v3.CANDIDATE_COLUMNS:
        if c not in hourly.columns:
            hourly[c] = np.nan

    hourly.index = hourly.index.tz_localize(None) if hourly.index.tz is not None else hourly.index
    return hourly


def assemble_and_predict(bundle: dict, horizon: int, end_time: pd.Timestamp | None = None,
                         lookback_hours: int = 72) -> dict:
    """조립부터 predict_kw까지 한 번에 — shadow 실행기가 이걸 호출."""
    end_time = end_time or pd.Timestamp.now(tz=KST)
    # ★08-26 갱신★: +48h는 예전엔 "D+1 고정-tm만 저장"을 이유로 무조건
    # 대기 처리했다. `probe_kma_nwp_d2_feasibility_v1_2026-08-26.py`로
    # 같은 D 00UTC 런이 D+2(lead 39~60h)까지 실제 값을 준다는 걸 라이브로
    # 확인했고, `collect_kma_nwp_live_d2_v1_2026-08-26.py`로 D+2 8시각을
    # 수집하는 별도 수집기를 만들었다(같은 nwp_values 테이블에 저장,
    # target_time_kst로 구분되므로 D+1과 충돌 없음). 이 함수의 NWP 조회
    # (`_nwp_hourly`)는 애초에 시간범위 쿼리라 D+1/D+2를 구분하지
    # 않으므로, 특별취급을 없애고 +1h/+24h와 동일한 경로로 흘려보낸다.
    # D+2 수집기를 아직 못 돌린 날에는 특성결측수가 그대로 잡혀 정직하게
    # "대기"로 나온다(가짜 성공 없음 — 아래 공통 결측검사가 처리).
    hourly = build_live_hourly(end_time, lookback_hours, future_hours=horizon)
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    rebuilt = harness.build_frame(hourly, horizon, candidate_cols)
    if DIF in rebuilt.columns:
        rebuilt[f"{DIF}_결측여부"] = rebuilt[DIF].isna().astype(float)

    if rebuilt.empty:
        return {"상태": "실패", "사유": "재조립 프레임이 비어있음(원자료 부족)"}
    # ★★09-16 수정(1차: 고정 2시간 → 2차: 동적탐색으로 교체)★★
    # 이 issue_time = "지금 정시"였는데, 광주 ASOS(station 156)는 관측
    # 시각으로부터 우리 DB 도착까지 실측 지연이 있다(13:00 관측 →
    # 14:12 수신 사례). 매시 :03 실행이 "이번 정시" 값을 요구했으니
    # 그 값은 항상 존재할 수 없는 시간대였다 - 08-26 이 파일 생성 이후
    # 09-16까지 3주 가까이 이 tier(단기 +1h/+24h/+48h)가 **단 한 번도
    # success를 낸 적이 없었다**(shadow DB 실측: error 52건 +
    # warming_up 332건, success 0건).
    #
    # 1차 수정(2시간 고정 오프셋)은 틀렸다 - 9월 전체 지연 실측
    # (asos_hourly first_received_at - observation_time)이 중앙값
    # 72.6분으로는 안정적이지만 **p90=372.6분(6시간+), p95=552.6분
    # (9시간+), 120분 초과가 23%**로 편차가 매우 크다(특정 날짜에
    # 몰린 게 아니라 9월 거의 매일 발생). 고정 2시간으로는 실행
    # 5번 중 1번꼴로 여전히 실패해 "재발 방지"라 부를 수 없었다.
    #
    # 최종 수정: 고정 오프셋을 버리고, ASOS 핵심 관측(기온·습도·운량)이
    # **실제로 채워진 가장 최근 시각**을 매번 동적으로 탐색한다. 지연이
    # 몇 분이든 몇 시간이든 무관하게 안전하고, 그래도 못 찾으면(진짜
    # 장기 결측) 정직하게 대기로 처리한다 - 가짜 성공을 만들지 않는다는
    # 이 파일의 원칙과 일치.
    ASOS_CORE_COLS = [c for c in
                      ("기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_전운량_pct")
                      if c in rebuilt.columns]
    issue_time = None
    candidate = end_time.floor("h").tz_localize(None)
    for _ in range(ASOS_LOOKBACK_HOURS):
        if candidate in rebuilt.index and ASOS_CORE_COLS:
            if rebuilt.loc[candidate, ASOS_CORE_COLS].notna().all():
                issue_time = candidate
                break
        candidate = candidate - pd.Timedelta(hours=1)
    if issue_time is None:
        latest_candidate = end_time.floor("h").tz_localize(None)
        return {"상태": "대기",
                "사유": f"최근 {ASOS_LOOKBACK_HOURS}시간 내 ASOS 핵심관측(기온·습도·운량) 완결 시각 없음",
                "발행시각": str(latest_candidate)}
    missing_feats = [f for f in bundle["features"] if f not in rebuilt.columns]
    if missing_feats:
        return {"상태": "실패", "사유": f"특성 부족: {missing_feats}", "발행시각": str(issue_time)}

    row = rebuilt.loc[[issue_time], bundle["features"]]
    n_missing = int(row.isna().sum().sum())
    missing_values = [f for f in bundle["features"] if pd.isna(row.iloc[0][f])]
    # 08-28 실배선: XGBoost가 NaN을 수학적으로 받을 수 있다는 것과 운영
    # 입력이 준비됐다는 것은 다르다 - 원칙은 그대로 유지한다. 다만 발전출력
    # lag/이동통계 결측 중 원천시각이 물리적 야간(태양고도<=0)으로 확인된
    # 것만 input_quality_gate가 0 허용으로 명시 판정하고, 그 외 결측은
    # 전과 동일하게 무조건 "대기"(success로 저장하지 않음).
    night_zero_filled: list[str] = []
    if missing_values:
        row_dict = row.iloc[0].to_dict()
        lag_specs = gatewiring.hourly_lag_specs(issue_time)
        elev = (hourly["solar_elevation_deg"] if "solar_elevation_deg" in hourly.columns
               else pd.Series(dtype=float))
        gate_result = gatewiring.apply_gate(row_dict, bundle["features"], GATE_CFG,
                                            lag_specs, elev.get)
        if gate_result["status"] == gatewiring.iqg.opstatus.BLOCKED:
            return {
                "상태": "대기", "사유": "운영 입력 미충족 - " + gate_result["status_reason"],
                "발행시각": str(issue_time), "수평_h": horizon,
                "특성결측수": n_missing, "결측특성": gate_result["unexplained_features"],
                "번들버전": bundle.get("번들버전"),
            }
        night_zero_filled = gate_result["night_zero_filled_features"]
        row = pd.DataFrame([gate_result["row"]], index=[issue_time])[bundle["features"]]
    # 번들의 capacity_kw는 학습 당시 정의를 보존하고, 라이브 물리 상한만
    # 현재 인버터 등록용량 합계로 명시적으로 덮어쓴다.
    live_bundle = dict(bundle)
    live_bundle["clip_상한"] = GWANGJU_INVERTER_CLIP_CAPACITY_KW
    pred = infer.predict_kw(live_bundle, row)
    target_time = issue_time + pd.Timedelta(hours=(horizon - 1))
    result = {
        "상태": "성공", "발행시각": str(issue_time), "대상시각": str(target_time),
        "수평_h": horizon, "예측_kW": round(float(pred[0]), 4),
        "특성결측수": n_missing, "번들버전": bundle.get("번들버전"),
    }
    if night_zero_filled:
        result["야간0채움특성"] = night_zero_filled
    return result


# ═══════════════ 확장 메모(다음 작업, 같은 패턴 재사용) ═══════════════
# 초단기: use_kappa=True인 h3·h4는 predict_kw()가 "목표_태양고도_deg"를
#   자동으로 쓰므로 rebuilt에 그 컬럼만 있으면 그대로 재사용 가능.
#   h4(날씨군집)는 bundle["클러스터"]가 있으면 predict_kw가 자동으로
#   assign_weather_clusters()를 호출하므로 이 파일 그대로 재사용된다 —
#   초단기 전용 조립기를 새로 짤 필요 없이, load_ultra_frame 대신 이
#   build_live_hourly()를 15분 단위로만 바꾸면 된다(현재는 1시간 단위).
# 일간: daily_direct_final_audit.corrected_dataset()과 동일한 lag(7일·
#   2일)·30일 이동통계를 라이브 Blockdata 이력(plant_output_kw 일간
#   합계)으로 재현해야 한다 — 최소 30~32일 이력이 쌓인 뒤 작업 시작.
if __name__ == "__main__":
    import json
    print("이 파일은 라이브 SQLite가 있어야 실행됩니다(Codex 환경).")
    print("사용 예: assemble_and_predict(bundle, horizon=24)")
