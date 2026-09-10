# -*- coding: utf-8 -*-
"""인버터분해 C_LightGBM비중 "운영용" 최종모델 학습(사용자 지시 - 작업
④, 사용자가 직접 실행하기로 함: "제가 코드만 작성, 사용자님이 직접
실행").

## 이 스크립트가 하는 일
`inverter_disaggregation_cv_v1_2026-08-27.py`가 이미 5계절 rolling-origin
walk-forward 방식으로 C방법의 채택 여부를 사전등록기준으로 검증했다
(outputs/인버터_분해교차검증_v1_2026-08-27/사전등록규칙_판정표.csv -
초단기_1h·초단기_2h·단기_1h가 C_LightGBM비중 채택, 초단기_3h/4h·단기_24h/
48h는 A_정격용량비례 채택 - 이 스크립트는 후자엔 손대지 않는다. A는
학습이 필요 없는 정격비례 계산이라 config/inverter_disaggregation.json의
inverter_capacity_kw로 이미 충분).

CV는 "검증용으로 데이터를 fold별로 쪼갠 walk-forward 모델"만 만들었고
운영에 그대로 쓸 "전체 이력으로 적합한 최종모델 1개"는 없었다. 이
스크립트는 CV 스크립트의 내부 함수(_model_frame/_feature_columns/
_normalize_shares 등, 전부 import로 재사용 - 재구현 안 함)를 그대로
써서 그 최종모델 3세트(초단기+1h, 초단기+2h, 단기+1h)를 학습해 joblib
번들로 저장한다.

## 이 스크립트가 하지 않는 것
- `config/inverter_disaggregation.json`의 `promote_to_official`을
  true로 바꾸지 않는다 - 번들을 만들기만 하고 실제 배선/공식경로 승격은
  사용자 승인 후 별도 진행.
- 운영 DB·실시간 API 호출 없음 - 전부 이미 저장된 인버터 원본 Excel
  기반 이력(`audit_inverter_history_v1_2026-08-27.py`)과 공식 5계절 OOF
  프레임(`defect_policy_comparison_v1_2026-08-21.py`)만 재사용한다.
- 기존 CV 산출물(outputs/인버터_분해교차검증_v1_2026-08-27/)을 수정하지
  않는다(import만 함, 파일 쓰기 없음).
- 8개 공식 운영모델·joblib은 건드리지 않는다 - 저장 경로가 완전히
  다르다(아래 OUT 참고, 공식 모델 디렉터리 밖).

## 실행 방법(사용자가 직접)
    cd 03_모델학습/현재_종합파이프라인/운영안전장치_2026-08-28
    python inverter_disaggregation_train_v1_2026-08-28.py
학습행이 수만~수십만 행 규모라 CV 스크립트 실행 경험상 수 분 내 끝날
것으로 예상되지만 실제 소요시간은 환경에 따라 다를 수 있다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent  # 03_모델학습/현재_종합파이프라인

CV_SCRIPT = PARENT / "inverter_disaggregation_cv_v1_2026-08-27.py"
AUDIT_SCRIPT = PARENT / "audit_inverter_history_v1_2026-08-27.py"
DPC_SCRIPT = PARENT / "defect_policy_comparison_v1_2026-08-21.py"

# 08-27 CV의 사전등록 판정표(outputs/인버터_분해교차검증_v1_2026-08-27/
# 사전등록규칙_판정표.csv)에서 "최종선택=True & 방법=C_LightGBM비중"인
# 조합만 그대로 옮겼다 - 재계산하지 않고 이미 확정된 채택표를 신뢰한다.
ADOPTED_C_HORIZONS = [
    ("초단기", 1),
    ("초단기", 2),
    ("단기", 1),
]

# ★미승격 경로★ - 공식 운영모델 디렉터리와 완전히 분리된 위치.
OUT = HERE / "outputs" / "인버터분해_운영가중치_v1_2026-08-28"
CONFIG_PATH = HERE / "config" / "inverter_disaggregation.json"


def _load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError("재사용 대상 스크립트가 없다: %s" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_training_frame(cv, dpc, audit_mod, tier: str, horizon: int) -> pd.DataFrame:
    """CV와 동일한 특성조립 로직(cv._model_frame)으로 전체 이력 프레임을
    만든다. CV처럼 fold 경계로 자르지 않고 전부 학습에 쓴다 - 이건 성능
    "검증"이 아니라 "지금 갖고 있는 모든 이력으로 최종 배포모델을
    적합(final fit)"하는 것이기 때문이다(사전등록 채택 여부 판정은 이미
    CV가 끝냈으므로 여기서 다시 하지 않는다)."""
    per_inv = audit_mod.load_all_inverters_5min()
    agg15 = audit_mod.aggregate_inverter_power(per_inv, "15min")
    agg1h = audit_mod.aggregate_inverter_power(per_inv, "1h")
    agg = agg15 if tier == "초단기" else agg1h
    frame = cv._model_frame(dpc, tier, horizon, agg)
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    # CV와 동일 기준: 정책B 결함구간 제외 + 인버터 5대 실제값 전부 있는 행만.
    defect = ((frame["대상시각"] >= cv.DEFECT_START) &
             (frame["대상시각"] < cv.DEFECT_END_EXCLUSIVE))
    frame = frame[~defect & frame[inv_cols].notna().all(axis=1)].copy()
    return frame


def fit_final_models(cv, frame: pd.DataFrame, features: list[str],
                     seed: int = 42) -> dict:
    """인버터 5대 각각에 대해 독립 LGBMRegressor를 적합한다. 하이퍼파라미터는
    CV의 _predict_c()와 완전히 동일하게 맞춰 "검증 때와 다른 모델을 운영에
    쓰는" 불일치를 방지한다."""
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    shares = frame[inv_cols].div(frame[inv_cols].sum(axis=1), axis=0)
    valid = np.isfinite(shares).all(axis=1)
    frame_v, shares_v = frame.loc[valid], shares.loc[valid]
    if len(frame_v) < 300:
        raise RuntimeError("학습행 부족(300 미만): %d행" % len(frame_v))
    models = {}
    for i, col in enumerate(inv_cols):
        model = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              max_depth=5, min_child_samples=40, subsample=0.9,
                              colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=seed + i, n_jobs=-1, verbosity=-1)
        model.fit(frame_v[features], shares_v[col])
        models[str(i + 1)] = model
    return {"models": models, "n_train_rows": len(frame_v)}


def predict_shares(cv, bundle: dict, row_features: dict) -> dict[str, float]:
    """운영 추론용 헬퍼: 학습된 5개 모델로 인버터별 비중을 예측하고,
    CV와 동일한 `_normalize_shares()`로 클리핑+정규화해 합계가 정확히
    1.0이 되게 만든다(음수 제거·클립 포함). 반환값은
    inverter_disaggregation_v1_2026-08-28.py의 disaggregate(weights_C=...)
    에 바로 넘길 수 있는 {인버터번호(str): 비중} 형태다."""
    features = bundle["features"]
    x = pd.DataFrame([row_features])[features]
    raw = np.array([[bundle["models"][str(i)].predict(x)[0] for i in range(1, 6)]])
    normalized = cv._normalize_shares(raw)[0]
    return {str(i + 1): float(normalized[i]) for i in range(5)}


def _sanity_check_sum_to_one(cv, bundle: dict, frame: pd.DataFrame,
                             features: list[str], n_sample: int = 20) -> dict:
    """학습 직후 자체 점검: 임의 표본 n_sample행에 대해 예측 비중 합계가
    1.0에 정확히 맞는지 확인한다(가짜 성공 금지 - "학습됐다"고만 보고하지
    않고 실측으로 증명)."""
    sample = frame.sample(min(n_sample, len(frame)), random_state=0)
    max_diff = 0.0
    negative_before_clip = 0
    for _, row in sample.iterrows():
        shares = predict_shares(cv, bundle, row[features].to_dict())
        s = sum(shares.values())
        max_diff = max(max_diff, abs(s - 1.0))
    return {"표본수": len(sample), "비중합계_최대오차": max_diff}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cv = _load_module("inverter_disagg_cv_20260827", CV_SCRIPT)
    dpc = _load_module("inverter_disagg_dpc_20260828train", DPC_SCRIPT)
    audit_mod = _load_module("inverter_audit_20260828train", AUDIT_SCRIPT)

    summary = []
    for tier, horizon in ADOPTED_C_HORIZONS:
        key = "%s_%dh" % (tier, horizon)
        print("\n=== %s 학습 시작 ===" % key)
        frame = build_training_frame(cv, dpc, audit_mod, tier, horizon)
        features = cv._feature_columns(frame)
        fit = fit_final_models(cv, frame, features)
        bundle = {
            "티어": tier, "수평_h": horizon, "방법": "C_LightGBM비중",
            "features": features, "models": fit["models"],
            "n_train_rows": fit["n_train_rows"],
            "capacity_kw": {str(i + 1): float(cv.CAPACITY[i]) for i in range(5)},
            "학습시각": datetime.now().isoformat(timespec="seconds"),
            "근거": ("08-27 5계절 rolling-origin CV 사전등록기준 통과 - "
                   "outputs/인버터_분해교차검증_v1_2026-08-27/사전등록규칙_판정표.csv"),
            "promote_to_official": False,
            "_주의": "이 번들은 미승격 상태다. 공식 예측 파이프라인에 배선하려면 "
                   "별도 승인·통합 작업이 필요하다.",
        }
        check = _sanity_check_sum_to_one(cv, bundle, frame, features)
        bundle["학습직후_자체점검"] = check
        print("학습행수=%d, 특성수=%d, 비중합계_최대오차=%.2e"
             % (fit["n_train_rows"], len(features), check["비중합계_최대오차"]))
        if check["비중합계_최대오차"] > 1e-9:
            raise AssertionError(
                "%s: 비중 정규화 후에도 합계가 1.0에서 %.2e만큼 벗어남 - "
                "학습을 성공으로 보고하지 않는다" % (key, check["비중합계_최대오차"]))

        bundle_path = OUT / ("inverter_c_weights_%s.joblib" % key)
        joblib.dump(bundle, bundle_path)
        summary.append({
            "수평키": key, "학습행수": fit["n_train_rows"], "특성수": len(features),
            "비중합계_최대오차": check["비중합계_최대오차"],
            "번들경로": str(bundle_path),
        })

    summary_path = OUT / "학습요약.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # 08-28 외부검토 보완 4: 산출물 실제 존재/크기를 실행로그에 증명.
    sys.path.insert(0, str(HERE))
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("opstatus", HERE / "operational_status_v1_2026-08-28.py")
    opstatus = _ilu.module_from_spec(_spec)
    # 버그수정(사용자 실행 08-28에서 발견): sys.modules 등록을 빼먹으면
    # Python 3.14의 dataclass가 `from __future__ import annotations` 타입
    # 힌트를 sys.modules[cls.__module__]로 못 찾아 AttributeError가 난다
    # (실제 학습·저장은 이 줄 이전에 이미 전부 끝난 뒤라 영향 없었음 -
    # 산출물 검증 출력 단계만 실패했었다). 다른 스크립트들과 동일하게
    # exec_module 전에 등록한다.
    sys.modules["opstatus"] = opstatus
    _spec.loader.exec_module(opstatus)
    paths = {row["수평키"]: row["번들경로"] for row in summary}
    paths["학습요약"] = str(summary_path)
    opstatus.verify_and_print_outputs(paths)

    print("\n=== 전체 요약 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n★promote_to_official은 여전히 false입니다 - 공식경로 배선은 "
         "별도 승인 후 진행하세요.★")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
