# -*- coding: utf-8 -*-
"""라이브 특성조립 직후 / 모델 호출 전 공통 품질 게이트(사용자 지시 2번).

## 위치
초단기(live_feature_assembler_초단기)·단기(live_feature_assembler_단기)·
일간(shadow_predict_일간)이 각자 조립한 "특성 1행"과 "그 모델 번들의
features 목록"을 받아 이 게이트를 통과시킨 뒤에만 predict_kw를 부르도록
설계했다. 이번 작업에서는 게이트 함수만 구현하고 기존 조립기 파일들은
수정하지 않는다(사용자 지시: 기존 정상 수집자료·코드 훼손 금지 원칙을
보수적으로 해석 - 실제 배선은 검증 후 별도 승인받아 진행).

## 핵심 원칙(전부 사용자 지시 2번 그대로)
- 선택특성 기준으로 설명되지 않은 NaN이 하나라도 있으면 정상예측 금지.
- native missing이 숫자를 뱉어도 자동 성공 아님 - 그 판단은 이 게이트가 한다.
- 야간 발전출력 결측은 물리적 야간(태양고도<=0)일 때만 0 허용.
- 강수량/적설은 "현상 없음"이 확인된 경우만 0 허용 - 이 함수 자체는
  "이미 0으로 채워져 들어온 값"을 신뢰할지 여부를 검사하는 게 아니라,
  호출부가 0으로 채운 근거(night_flag/no_phenomenon_flag)를 명시적으로
  전달했는지를 강제한다(임의 채움 여부를 사후에 알 수 없으므로 사전 계약).
- 5분 격자 내부결측은 최대 2스텝(10분)까지만 기존 보간 허용, 초과분은
  보간하지 않고 길이만 기록.
- 미래 NWP 결측은 평균/0/관측값 대체 금지.
- 인버터 장기통신두절은 정책B 유지(제외, 임의채움 없음) - 이 함수는
  정책B 적용 자체를 하지 않고, 이미 정책B가 적용된 데이터가 들어온다고
  가정한 뒤 그 결과(NaN)를 그대로 존중한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config" / "input_quality_gate.json"

sys.path.insert(0, str(HERE))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("opstatus", HERE / "operational_status_v1_2026-08-28.py")
opstatus = _ilu.module_from_spec(_spec)
sys.modules["opstatus"] = opstatus
_spec.loader.exec_module(opstatus)


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_rainfall_or_snow_feature(feature_name: str) -> bool:
    """no_phenomenon_flag 허용 대상인지(강수량/적설 계열인지) 이름으로 확인.
    08-28 테스트로 발견된 버그(무관 특성도 통과되던 것) 수정용 가드."""
    return ("강수" in feature_name) or ("적설" in feature_name)


class ZeroFillClaim:
    """호출부가 "이 값을 0으로 채운 근거"를 명시적으로 선언하는 계약.

    임의로 0을 넣고 나중에 정당화하는 게 아니라, 채우기 "전"에 근거를
    선언해야 게이트가 그 0값을 신뢰한다 - 순서를 강제해 사후 정당화를
    막는다.
    """

    def __init__(self, feature: str, reason: str, night_flag: bool | None = None,
                no_phenomenon_flag: bool | None = None,
                insufficient_history_flag: bool | None = None,
                history_available: int | None = None,
                history_required: int | None = None):
        self.feature = feature
        self.reason = reason
        self.night_flag = night_flag
        self.no_phenomenon_flag = no_phenomenon_flag
        # 08-28 외부검토 보완 3: 롤링 특성의 "초기 워밍업 구간이라 히스토리가
        # 아직 안 쌓였다"는 근거를 별도로 선언한다 - night_flag/
        # no_phenomenon_flag와 마찬가지로 호출부가 "채우기 전"에 명시해야
        # 하고(사후 정당화 금지 원칙 동일 적용), 실제 결측(unexplained)과
        # 절대 같은 통계 버킷에 섞이지 않는다.
        self.insufficient_history_flag = insufficient_history_flag
        self.history_available = history_available
        self.history_required = history_required


def evaluate(
    row: dict[str, Any],
    required_features: list[str],
    cfg: dict,
    zero_fill_claims: list[ZeroFillClaim] | None = None,
    interpolation_log: dict | None = None,
    training_nan_rates: dict[str, float] | None = None,
) -> dict:
    """품질 게이트 평가. 부작용 없음(파일에 쓰지 않음) - 호출부가 결과를
    받아 예측 실행 여부와 PredictionRecord 저장을 결정한다.

    Parameters
    ----------
    row: 조립된 특성 1행(feature -> value). None/NaN은 결측으로 취급.
    required_features: 이 모델 번들의 bundle["features"] 그대로.
    cfg: input_quality_gate.json 로드 결과.
    zero_fill_claims: 0으로 채운 특성들의 근거 선언 목록.
    interpolation_log: {feature: {"steps": int, "step_minutes": int}} 형태로
        호출부가 이미 수행한 5분격자 내부보간 이력(있으면).
    training_nan_rates: {feature: 학습기준 NaN율} - 있으면 학습-운영 대조도
        같이 계산(없어도 게이트 판정 자체는 가능).
    """
    claims_by_feature = {c.feature: c for c in (zero_fill_claims or [])}
    interpolation_log = interpolation_log or {}
    allow = cfg.get("allow_zero_fill", {})
    max_steps = int(cfg.get("max_interpolation_steps", 2))

    missing_features: list[str] = []
    unexplained: list[str] = []
    explained: dict[str, str] = {}
    over_interpolated: list[str] = []
    # 08-28 외부검토 보완 3: "실제 결측"과 "정상 워밍업(히스토리 부족)"을
    # 절대 같은 버킷(unexplained)에 섞지 않는다 - 원인이 다르면 통계도
    # 분리해야 나중에 "결측이 늘었다"는 오판을 막을 수 있다.
    structurally_unavailable: dict[str, str] = {}

    for feat in required_features:
        val = row.get(feat, None)
        is_missing = val is None or (isinstance(val, float) and val != val)  # NaN 체크
        if not is_missing:
            continue
        missing_features.append(feat)

        claim = claims_by_feature.get(feat)
        interp = interpolation_log.get(feat)

        if claim is not None:
            ok = False
            if claim.night_flag and allow.get("plant_output_night", False):
                ok = True
                explained[feat] = "물리적 야간 - 발전출력 0 허용: " + claim.reason
            elif claim.no_phenomenon_flag and _is_rainfall_or_snow_feature(feat):
                # 버그수정(08-28 테스트로 발견): feature 이름이 실제로
                # 강수/적설 계열인지 확인하지 않고 config 플래그만 봤더니
                # DSWRF 같은 무관한 특성도 no_phenomenon_flag만 세우면
                # 통과되는 결함이 있었다. 이제 이름까지 확인한다.
                is_rain = "강수" in feat
                is_snow = "적설" in feat
                if (is_rain and allow.get("rainfall_no_phenomenon", False)) or (
                    is_snow and allow.get("snow_no_phenomenon", False)
                ):
                    ok = True
                    explained[feat] = "현상없음 확인 - 0 허용: " + claim.reason
            elif claim.insufficient_history_flag:
                # unexplained에 넣지 않고 별도 버킷으로 - "결측"이 아니라
                # "롤링 윈도우가 아직 안 찼다"는 구조적으로 다른 사실이다.
                structurally_unavailable[feat] = (
                    "롤링 히스토리 워밍업 구간(%s/%s) - 데이터 결측 아님: %s"
                    % (claim.history_available, claim.history_required, claim.reason))
                ok = True  # unexplained 판정 스킵(아래서 별도 정책 적용)
            if not ok:
                unexplained.append(feat)
            continue

        if interp is not None:
            steps = int(interp.get("steps", 0))
            if steps <= max_steps:
                explained[feat] = "5분격자 내부보간 %d스텝(<=%d) 허용" % (steps, max_steps)
            else:
                over_interpolated.append(feat)
                unexplained.append(feat)
            continue

        # claim도 interpolation_log도 없는 순수 미설명 결측
        unexplained.append(feat)

    unexplained_policy = cfg.get("unexplained_nan_policy", "blocked")
    insufficient_history_policy = cfg.get("insufficient_history_policy", "blocked")
    if unexplained:
        status = unexplained_policy if unexplained_policy in opstatus.ALL_STATUSES else opstatus.BLOCKED
        reason = ("설명되지 않은 결측 %d건(전체 결측 %d건 중): %s" %
                 (len(unexplained), len(missing_features), unexplained))
    elif structurally_unavailable:
        # 실제 결측은 없지만 롤링 워밍업으로 값이 아직 없는 특성이 있는
        # 경우 - "결측"이라 부르지 않고 별도 정책으로 판정한다.
        status = (insufficient_history_policy
                 if insufficient_history_policy in opstatus.ALL_STATUSES else opstatus.BLOCKED)
        reason = ("롤링 히스토리 워밍업 구간이라 값이 아직 없는 특성 %d건(데이터 이상 아님): %s"
                 % (len(structurally_unavailable), list(structurally_unavailable)))
    elif missing_features:
        status = opstatus.DEGRADED
        reason = ("전부 사전허용 규칙으로 설명된 결측 %d건 - degraded로 진행 가능"
                 % len(missing_features))
    else:
        status = opstatus.NORMAL
        reason = "결측 없음"

    nan_rate_report = None
    if training_nan_rates:
        nan_rate_report = {}
        for feat, train_rate in training_nan_rates.items():
            live_missing = feat in missing_features
            nan_rate_report[feat] = {
                "학습기준": train_rate,
                "이_행_결측여부": live_missing,
            }

    return {
        "status": status,
        "status_reason": reason,
        "missing_features": missing_features,
        "unexplained_features": unexplained,
        "explained_features": explained,
        "structurally_unavailable_features": structurally_unavailable,
        "over_interpolated_features": over_interpolated,
        "nan_rate_report": nan_rate_report,
        "checked_feature_count": len(required_features),
    }


def audit_nan_rate_delta(training_series: dict[str, float], live_series: dict[str, float],
                         cfg: dict) -> dict:
    """학습-운영 결측률 감사(사용자 지시 2번 "학습-운영 결측률 감사").

    training_series/live_series: {feature: NaN율(0~1)}. 호출부가 이미
    각자의 데이터로 계산해 전달한다(이 함수는 통계 재계산을 하지 않고
    임계 판정만 한다 - 재구현 금지 원칙, 통계는 각 조립기가 이미 갖고
    있는 pandas 프레임에서 직접 뽑는 게 더 정확함).
    """
    thresholds = cfg.get("nan_rate_thresholds", {})
    generic = thresholds.get("generic_feature", {})
    special = thresholds.get("plant_output_kw_daytime", {})

    rows = []
    for feat, train_rate in training_series.items():
        live_rate = live_series.get(feat)
        if live_rate is None:
            continue
        delta = live_rate - train_rate
        if feat == "plant_output_kw_daytime" and special:
            level = "ok"
            if live_rate > float(special.get("blocked_above", 1.0)):
                level = "blocked"
            elif live_rate > float(special.get("warning_above", 1.0)):
                level = "warning"
        else:
            level = "ok"
            if delta > float(generic.get("blocked_delta_above", 1.0)):
                level = "blocked"
            elif delta > float(generic.get("warning_delta_above", 1.0)):
                level = "warning"
        rows.append({
            "feature": feat, "학습NaN율": round(train_rate, 4),
            "라이브NaN율": round(live_rate, 4), "차이": round(delta, 4),
            "판정": level,
        })
    worst = "ok"
    for r in rows:
        if r["판정"] == "blocked":
            worst = "blocked"
            break
        if r["판정"] == "warning":
            worst = "warning"
    return {"항목": rows, "종합판정": worst}


class LagSpec:
    """발전출력 lag/이동통계 특성 1개의 "값이 실제로 어느 과거 시각(들)에서
    왔는지" 명세. point 특성(예: 발전출력_1시간전_kW)은 source_times가
    원소 1개, 이동평균/표준편차는 그 윈도우에 속한 시각 전부."""

    def __init__(self, feature: str, source_times: list):
        self.feature = feature
        self.source_times = list(source_times)


def build_night_zero_fill_claims(
    row: dict[str, Any],
    required_features: list[str],
    lag_specs: list[LagSpec],
    elevation_at,
) -> list[ZeroFillClaim]:
    """08-28 사용자 지시(input_quality_gate 실배선, 좁은 범위): 발전출력
    lag/이동통계 특성이 NaN이고, 그 값이 나왔어야 할 과거 시각(들)이 전부
    물리적 야간(태양고도<=0)으로 확인될 때만 night_flag 클레임을 만든다.

    - `elevation_at(timestamp)`: 그 시각의 태양고도(deg)를 반환하는 조회
      함수(호출부가 이미 계산해둔 solar_elevation_deg/태양고도_deg 시리즈의
      `.get`을 그대로 넘기면 된다 - 여기서 새로 계산하지 않는다, 재사용
      원칙). 조회 실패 시 None을 반환해야 한다.
    - 시각 조회가 안 되거나(None), NaN이거나, 하나라도 태양고도>0(주간)이면
      그 특성은 클레임을 만들지 않는다 - "입증 안 되면 보수적으로 그대로
      미설명 결측으로 남긴다"는 안전기본값(사용자 지시 그대로: 그 외
      결측은 기존처럼 차단).
    - required_features에 없거나 row에서 실제로 NaN이 아닌 특성은 건너뛴다
      (이미 값이 있는데 클레임을 만들 이유가 없음).
    """
    claims: list[ZeroFillClaim] = []
    for spec in lag_specs:
        if spec.feature not in required_features:
            continue
        val = row.get(spec.feature, None)
        is_missing = val is None or (isinstance(val, float) and val != val)
        if not is_missing:
            continue
        elevations = []
        confirmed_night = True
        for t in spec.source_times:
            e = elevation_at(t)
            if e is None or (isinstance(e, float) and e != e) or e > 0:
                confirmed_night = False
                break
            elevations.append(float(e))
        if confirmed_night and elevations:
            claims.append(ZeroFillClaim(
                spec.feature,
                "lag 원천시각 %d개 전부 태양고도<=0 확인(최고 %.2fdeg) - 물리적 야간이라 "
                "0 허용, 임의값 아님" % (len(elevations), max(elevations)),
                night_flag=True))
    return claims
