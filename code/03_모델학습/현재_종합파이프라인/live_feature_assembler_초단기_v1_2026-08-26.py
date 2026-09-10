# -*- coding: utf-8 -*-
"""라이브 특성조립기 — 초단기 15분, +1~+4h.

API를 직접 호출하지 않고 Codex가 이미 수집한 Blockdata/KMA SQLite만 읽는다.
운영모델 golden replay에서 검증된 ``build_ultra_short_frame``과
``predict_kw``를 그대로 재사용한다. 입력 하나라도 빠지면 예측하지 않고
``대기``를 반환한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str, rel: str = "."):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


short = _load("ultra_live_short_common", "live_feature_assembler_단기_v1_2026-08-26.py")
ultra = _load("ultra_live_official", "train_ultra_short_official_v1_2026-08-21.py")
# 08-28 실배선(사용자 지시 ②, 좁은 범위) - live_feature_assembler_단기와
# 동일한 게이트 배선 모듈을 재사용(중복 정의 안 함). short 모듈이 이미
# 같은 걸 로드하지만, 이 파일은 short를 통하지 않고 직접 쓰는 지점이 있어
# 명시적으로 한 번 더 잡아둔다(같은 파일이라 sys.modules 캐시로 재실행
# 비용 없음).
gatewiring = _load("ultra_live_gatewiring", "live_gate_wiring_v1_2026-08-28.py",
                   rel="운영안전장치_2026-08-28")
GATE_CFG = gatewiring.iqg.load_config()


def _quarter_base(end_time: pd.Timestamp, lookback_hours: int) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    five = short._equipment_5min(end_time, lookback_hours)  # 공식 공통 입력층
    if five.empty:
        return pd.DataFrame(), None
    count = five["plant_output_kw"].resample("15min").count()
    q = five.resample("15min").mean(numeric_only=True)
    q["measurement_count_5min"] = count
    incomplete = count < 3
    cols = ["plant_output_kw", "plant_input_power_kw", "plant_input_current_a",
            "mean_input_voltage_v", "mean_power_factor", "mean_frequency_hz"]
    q.loc[incomplete, cols] = np.nan
    q = q.rename(columns={"plant_output_kw": "발전출력_kW"})
    complete = q.index[q["measurement_count_5min"] >= 3]
    issue = (complete.max() + pd.Timedelta(minutes=15)) if len(complete) else None
    return q, issue


def assemble_and_predict(bundle: dict, horizon: int,
                         end_time: pd.Timestamp | None = None,
                         lookback_hours: int = 12) -> dict:
    if horizon not in (1, 2, 3, 4):
        raise ValueError("초단기 수평은 1~4h만 허용")
    end_time = end_time or pd.Timestamp.now(tz=short.KST)
    q, issue_naive = _quarter_base(end_time, lookback_hours)
    if issue_naive is None:
        return {"상태": "대기", "사유": "완결된 15분 Blockdata 구간 없음",
                "수평_h": horizon, "번들버전": bundle.get("번들버전")}
    issue_aware = issue_naive.tz_localize(short.KST)
    if issue_aware > end_time:
        return {"상태": "대기", "사유": "마지막 15분 구간이 아직 완결되지 않음",
                "발행시각": str(issue_naive), "수평_h": horizon,
                "번들버전": bundle.get("번들버전")}

    # 해당 발행시각 이후에 입수한 예보가 섞이지 않도록 issue_aware를
    # 명시적으로 공통 시간단위 조립기에 전달한다.
    hourly = short.build_live_hourly(issue_aware, lookback_hours=max(72, lookback_hours),
                                     future_hours=horizon)
    end_q = issue_naive + pd.Timedelta(hours=horizon)
    q = q.reindex(pd.date_range(q.index.min(), end_q, freq="15min"))
    elev_sa = [short.bsrn.solar_position(
        t.tz_localize(short.KST).astimezone(short.ZoneInfo("UTC")).replace(tzinfo=None))
        for t in q.index]
    q["태양고도_deg"] = [x[0] for x in elev_sa]

    frame = ultra.build_ultra_short_frame(q, hourly, horizon)
    if short.DIF in frame.columns:
        frame[f"{short.DIF}_결측여부"] = frame[short.DIF].isna().astype(float)
    if issue_naive not in frame.index:
        return {"상태": "대기", "사유": "발행시각 행 조립 실패",
                "발행시각": str(issue_naive), "수평_h": horizon,
                "번들버전": bundle.get("번들버전")}
    generated = {f for f in bundle["features"] if f.startswith("날씨군집_")}
    required = [f for f in bundle["features"] if f not in generated]
    if bundle.get("클러스터") is not None:
        required = list(dict.fromkeys(required + list(bundle["클러스터"]["입력컬럼"])))
    absent = [f for f in required if f not in frame.columns]
    if absent:
        return {"상태": "실패", "사유": f"특성 컬럼 부족: {absent}",
                "발행시각": str(issue_naive), "수평_h": horizon}
    row = frame.loc[[issue_naive]]
    missing = [f for f in required if pd.isna(row.iloc[0][f])]
    # 08-28 실배선: 발전출력 lag/이동통계 결측 중 원천시각이 물리적
    # 야간(태양고도<=0)으로 확인된 것만 0허용, 그 외는 전과 동일하게 차단.
    night_zero_filled: list[str] = []
    if missing:
        row_dict = row.iloc[0].to_dict()
        lag_specs = gatewiring.quarter_lag_specs(issue_naive)
        gate_result = gatewiring.apply_gate(row_dict, required, GATE_CFG,
                                            lag_specs, q["태양고도_deg"].get)
        if gate_result["status"] == gatewiring.iqg.opstatus.BLOCKED:
            return {"상태": "대기", "사유": "운영 입력 미충족 - " + gate_result["status_reason"],
                    "발행시각": str(issue_naive), "수평_h": horizon,
                    "특성결측수": len(missing), "결측특성": gate_result["unexplained_features"],
                    "번들버전": bundle.get("번들버전")}
        night_zero_filled = gate_result["night_zero_filled_features"]
        filled_row = gate_result["row"]
        row = pd.DataFrame([{c: filled_row.get(c, row.iloc[0][c]) for c in row.columns}],
                           index=[issue_naive])
    pred = short.infer.predict_kw(bundle, row)
    result = {"상태": "성공", "발행시각": str(issue_naive),
             "대상시각": str(issue_naive + pd.Timedelta(hours=horizon)),
             "수평_h": horizon, "예측_kW": round(float(pred[0]), 4),
             "특성결측수": 0, "번들버전": bundle.get("번들버전")}
    if night_zero_filled:
        result["야간0채움특성"] = night_zero_filled
    return result


if __name__ == "__main__":
    print("shadow_predict_초단기_v1_2026-08-26.py에서 호출하십시오.")
