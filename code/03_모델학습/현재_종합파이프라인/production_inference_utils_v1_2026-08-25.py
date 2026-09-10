# -*- coding: utf-8 -*-
"""운영모델 번들 v2 공통 추론 유틸 — Codex 리뷰(08-25) 반영.

`train_production_models_v2_2026-08-25.py`가 저장하는 번들을 받아서
"특성 프레임 하나 → 최종 kW 예측" 까지 마무리하는 순수 함수들만 모은다.
학습 코드가 아니라 **추론 전용**이며, 실시간 특성조립기(4단계)와
예측 실행기(5단계)가 이 모듈을 그대로 import해서 쓰면 된다(재구현 금지).

## 이 모듈이 해결하는 것(Codex 리뷰 지적사항)
1. 초단기 +4h(청천지수+날씨군집화): 번들에 저장된 KMeans·스케일러·
   중앙값·군집입력컬럼을 그대로 재적용해 **학습 당시와 동일한 방식으로**
   새 행의 날씨군집을 배정한다(결측은 학습 중앙값으로 채운 뒤 배정 —
   06-24 ⑥ 채택안과 동일 규칙).
2. 청천지수(κ) 모델(+3h·+4h): κ 예측 → 대상시각 태양고도 → Haurwitz
   청천일사 → 설비용량 적용 → kW 역변환 → [0, capacity] clip까지
   한 함수(`predict_kw`)로 묶는다.
3. 8개 모델 전부 같은 인터페이스(`predict_kw(bundle, frame)`)로 호출
   가능하게 만들어, 예측 실행기가 티어별로 분기 코드를 새로 안 짜도 되게 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Haurwitz(1945) 청천일사량 계수 — ultra_short_clearsky_v1_2026-08-21.clear_sky_ghi와 동일
_HAURWITZ_A = 1098.0
_HAURWITZ_B = 0.059


def clear_sky_ghi(elevation_deg: np.ndarray | float) -> np.ndarray:
    theta = np.deg2rad(np.clip(elevation_deg, 0.01, 90))
    sin_t = np.sin(theta)
    return np.clip(_HAURWITZ_A * sin_t * np.exp(-_HAURWITZ_B / sin_t), 0, None)


def assign_weather_clusters(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """번들에 `클러스터` 키가 있는 모델(초단기 +4h)에서만 호출.
    frame은 아직 군집 컬럼이 없는 상태로 넘긴다 — 여기서 추가해 돌려준다."""
    c = bundle.get("클러스터")
    if c is None:
        return frame
    cols, med, scaler, km = c["입력컬럼"], c["결측대체_중앙값"], c["스케일러"], c["kmeans"]
    out = frame.copy()
    missing_cols = [col for col in cols if col not in out.columns]
    if missing_cols:
        raise KeyError(f"군집 입력컬럼이 프레임에 없음: {missing_cols}")
    x = out[cols].fillna(med)
    z = scaler.transform(x)
    labels = km.predict(z)
    for k in range(c["군집수"]):
        out[f"날씨군집_{k}"] = (labels == k).astype(int)
    return out


def predict_kw(bundle: dict, frame: pd.DataFrame, elevation_col: str = "목표_태양고도_deg") -> np.ndarray:
    """번들 하나 + (해당 모델이 요구하는 특성이 전부 채워진) 프레임 하나를 받아
    최종 kW 예측(물리범위 clip 완료)을 반환한다. 초단기·단기·일간 8개 번들
    전부 이 함수 하나로 처리 가능하다."""
    x = frame.copy()
    if bundle.get("클러스터") is not None:
        x = assign_weather_clusters(bundle, x)

    missing = [f for f in bundle["features"] if f not in x.columns]
    if missing:
        raise KeyError(f"모델이 요구하는 특성이 라이브 프레임에 없음: {missing}")

    raw = bundle["model"].predict(x[bundle["features"]])
    capacity_kw = bundle["capacity_kw"]
    # 일간(kWh) 등 kW가 아닌 단위 모델은 clip 상한이 capacity_kw가 아니다 —
    # 번들에 명시된 clip_상한을 우선 쓰고, 없으면 capacity_kw로 대체(하위호환).
    clip_상한 = bundle.get("clip_상한", capacity_kw)

    if bundle["타깃유형"] == "kappa":
        kappa_info = bundle["카파_역변환"]
        min_elev = kappa_info["최소태양고도_deg"]
        elev = x[elevation_col].to_numpy()
        cs_kw = np.clip(capacity_kw * clear_sky_ghi(elev) / 1000.0, 1e-3, None)
        pred = np.asarray(raw, float) * cs_kw
        pred = np.where(elev < min_elev, 0.0, pred)  # MIN_ELEVATION_DEG 미만은 카파 정의 자체가 불안정 → 야간/저고도 0 처리
    else:
        pred = np.asarray(raw, float)

    return np.clip(pred, 0.0, clip_상한)


def audit_bundle(bundle: dict) -> dict:
    """예측 실행기가 시작 전에 번들이 완전한지(추론에 필요한 모든 부품이
    실제로 들어있는지) 점검할 때 쓰는 헬퍼."""
    checks = {
        "features_존재": "features" in bundle and len(bundle["features"]) > 0,
        "model_존재": "model" in bundle,
        "타깃유형_존재": "타깃유형" in bundle,
        "capacity_kw_존재": "capacity_kw" in bundle,
        "번들버전": bundle.get("번들버전", "★없음(v1, 불완전)★"),
    }
    if bundle.get("타깃유형") == "kappa":
        checks["카파_역변환_존재"] = bundle.get("카파_역변환") is not None
    if bundle.get("tier") == "초단기" and bundle.get("horizon_h") == 4:
        checks["클러스터_존재"] = bundle.get("클러스터") is not None
    checks["전체통과"] = all(v is True for k, v in checks.items() if k != "번들버전")
    return checks
