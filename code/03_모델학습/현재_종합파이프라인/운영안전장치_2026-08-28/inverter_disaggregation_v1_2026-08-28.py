# -*- coding: utf-8 -*-
"""총예측 -> 인버터별 예측 변환 인터페이스(사용자 지시 6번).

근거: AGENTS.md 2026-08-27 "인버터 분해 5계절 교차검증 - 실행 완료·
수평별 방법 확정" 절, config/inverter_disaggregation.json(사전등록 채택표).

## 이번 작업에서 하는 것
- 수평(티어+h)별로 config에 사전등록된 방법(A_정격용량비례 / C_LightGBM비중)을
  선택해 총예측 1개 값을 인버터 5대 값으로 변환하는 순수 함수.
- Σ인버터 = 총예측이 되도록 assert(허용오차 config 그대로).
- 일간 D+1은 미검증이라 config의 "excluded"에 있으면 명시적으로 막는다.

## 이번 작업에서 하지 않는 것(사용자 지시: 공식경로 승격 금지)
- 이 모듈을 실제 예측 파이프라인(shadow_predict_*, live_feature_assembler_*)에
  배선하지 않는다 - 여기서 함수만 준비하고 테스트 산출물만 만든다.
- C_LightGBM비중의 실제 운영용 가중치 번들은 아직 존재하지 않는다(08-27
  CV는 폴드별 성능만 산출했고 최종 가중치를 joblib 등으로 저장하지 않음 -
  outputs/인버터_분해교차검증_v1_2026-08-27/ 확인 결과 가중치 컬럼 자체가
  없음). 따라서 disaggregate_C_weighted()는 weights를 호출부가 명시적으로
  전달해야 동작한다 - 이 스크립트가 가중치를 임의로 만들어내지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config" / "inverter_disaggregation.json"


class DisaggregationError(Exception):
    pass


class ExcludedHorizonError(DisaggregationError):
    pass


class SumMismatchError(DisaggregationError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_method(tier: str, horizon_label: str, cfg: dict) -> str:
    """tier(예: '초단기'/'단기'/'일간') + horizon_label(예: '1h'/'24h'/'D+1')로
    사전등록된 방법을 찾는다. excluded에 있으면 명시적으로 막는다(조용히
    기본값으로 대체하지 않음 - 가짜 성공 금지 원칙)."""
    key = "%s_%s" % (tier, horizon_label)
    if key in cfg.get("excluded", {}):
        raise ExcludedHorizonError(
            "%s는 미검증으로 제외된 수평이다: %s" % (key, cfg["excluded"][key]))
    method = cfg.get("selection", {}).get(key)
    if method is None:
        raise KeyError("%s에 대한 사전등록 방법이 config에 없다 - 임의 기본값 사용 금지" % key)
    return method


def disaggregate_A_capacity_proportional(total_kw: float, capacity_kw: dict[str, float]) -> dict[str, float]:
    total_capacity = sum(capacity_kw.values())
    if total_capacity <= 0:
        raise DisaggregationError("정격용량 합계가 0 이하 - 배분 불가")
    return {inv: total_kw * (cap / total_capacity) for inv, cap in capacity_kw.items()}


def disaggregate_C_weighted(total_kw: float, weights: dict[str, float],
                            weight_tolerance: float = 1e-6) -> dict[str, float]:
    """weights는 인버터별 비중(합계가 1.0에 가까워야 함) - 호출부가 명시
    전달해야 한다(임의 생성 금지). 합계가 1.0에서 크게 벗어나면 그 자체가
    가중치 산출 로직의 결함일 수 있으므로 예외를 낸다."""
    w_sum = sum(weights.values())
    if abs(w_sum - 1.0) > weight_tolerance:
        raise DisaggregationError(
            "가중치 합계가 1.0이 아니다(합계=%.8f) - 가중치 산출 결과를 다시 확인할 것" % w_sum)
    return {inv: total_kw * w for inv, w in weights.items()}


def assert_sum_preserved(total_kw: float, per_inverter_kw: dict[str, float],
                         tolerance_kw: float) -> None:
    s = sum(per_inverter_kw.values())
    diff = abs(s - total_kw)
    if diff > tolerance_kw:
        raise SumMismatchError(
            "Σ인버터(%.10f) != 총예측(%.10f), 차이=%.10f > 허용오차 %.2e"
            % (s, total_kw, diff, tolerance_kw))


def disaggregate(total_kw: float, tier: str, horizon_label: str, cfg: dict,
                 weights_C: dict[str, float] | None = None) -> dict[str, Any]:
    """공개 진입점. 방법 선택 -> 배분 -> 합계보존 assert까지 한 번에 수행.
    반환값에 "방법"을 명시해 어떤 규칙으로 나왔는지 항상 추적 가능하게 한다."""
    method = select_method(tier, horizon_label, cfg)
    if method.startswith("A_"):
        per_inv = disaggregate_A_capacity_proportional(total_kw, cfg["inverter_capacity_kw"])
    elif method.startswith("C_"):
        if weights_C is None:
            raise ValueError(
                "%s는 C_LightGBM비중 방법인데 weights_C가 전달되지 않았다 - "
                "08-28 시점엔 공식 운영용 가중치 번들이 없으므로 호출부가 "
                "명시적으로 전달해야 한다(자동 생성 금지)" % method)
        per_inv = disaggregate_C_weighted(total_kw, weights_C)
    else:
        raise NotImplementedError("정의되지 않은 방법: %s" % method)

    tolerance = float(cfg.get("sum_preservation_tolerance_kw", 1e-6))
    assert_sum_preserved(total_kw, per_inv, tolerance)

    return {
        "티어": tier, "수평": horizon_label, "방법": method,
        "총예측_kW": total_kw, "인버터별_kW": per_inv,
        "합계_kW": sum(per_inv.values()),
    }
