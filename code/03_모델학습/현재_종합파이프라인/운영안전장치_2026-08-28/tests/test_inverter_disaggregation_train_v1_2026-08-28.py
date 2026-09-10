# -*- coding: utf-8 -*-
"""inverter_disaggregation_train_v1_2026-08-28.py 오프라인 스모크테스트.

실제 인버터 원본 Excel 이력·공식 5계절 OOF 프레임은 읽지 않는다(그건
사용자가 직접 실제 실행할 몫). 여기서는 완전 합성 DataFrame으로
fit_final_models()/predict_shares()/_sanity_check_sum_to_one()의 배관
로직(모델 5개를 독립 학습하고, 예측 후 정규화해 합계가 1.0이 되는지)만
검증한다 - build_training_frame()/main()은 실제 이력 파일이 필요해 이
테스트에서 호출하지 않는다.
"""
from __future__ import annotations

import importlib.util as ilu
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

# CV 스크립트에서 순수 유틸(_normalize_shares, CAPACITY)만 재사용 - 이 모듈은
# import 시점에 실제 Excel 이력을 읽지 않는다(함수 호출 시에만 읽음).
spec_cv = ilu.spec_from_file_location(
    "inverter_disagg_cv_test", PARENT.parent / "inverter_disaggregation_cv_v1_2026-08-27.py")
cv = ilu.module_from_spec(spec_cv)
sys.modules["inverter_disagg_cv_test"] = cv
spec_cv.loader.exec_module(cv)

spec_train = ilu.spec_from_file_location(
    "idt", PARENT / "inverter_disaggregation_train_v1_2026-08-28.py")
idt = ilu.module_from_spec(spec_train)
sys.modules["idt"] = idt
spec_train.loader.exec_module(idt)


def _make_synthetic_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inv_cols = [f"인버터{i}_kW" for i in range(1, 6)]
    capacity = np.array([50.0, 50.0, 30.0, 39.0, 50.0])
    base_total = rng.uniform(20, 180, size=n)
    shares_true = capacity / capacity.sum() + rng.normal(0, 0.02, size=(n, 5))
    shares_true = np.clip(shares_true, 0.01, None)
    shares_true /= shares_true.sum(axis=1, keepdims=True)
    values = shares_true * base_total[:, None]
    df = pd.DataFrame(values, columns=inv_cols)
    df["대상_hour_sin"] = np.sin(rng.uniform(0, 2 * np.pi, size=n))
    df["대상_hour_cos"] = np.cos(rng.uniform(0, 2 * np.pi, size=n))
    df["DSWRF"] = rng.uniform(0, 900, size=n)
    df["TCDC"] = rng.uniform(0, 10, size=n)
    return df


def test_1_fit_final_models_returns_five_models():
    frame = _make_synthetic_frame()
    features = ["대상_hour_sin", "대상_hour_cos", "DSWRF", "TCDC"]
    fit = idt.fit_final_models(cv, frame, features, seed=1)
    assert set(fit["models"].keys()) == {"1", "2", "3", "4", "5"}, fit["models"].keys()
    assert fit["n_train_rows"] == len(frame)
    return True


def test_2_fit_final_models_raises_on_too_few_rows():
    frame = _make_synthetic_frame(n=50)
    features = ["대상_hour_sin", "대상_hour_cos", "DSWRF", "TCDC"]
    try:
        idt.fit_final_models(cv, frame, features, seed=1)
        assert False, "학습행 50건인데 통과됨(최소기준 300건 미달)"
    except RuntimeError as e:
        assert "학습행 부족" in str(e)
        return True


def test_3_predict_shares_sums_to_one_after_normalization():
    frame = _make_synthetic_frame()
    features = ["대상_hour_sin", "대상_hour_cos", "DSWRF", "TCDC"]
    fit = idt.fit_final_models(cv, frame, features, seed=1)
    bundle = {"models": fit["models"], "features": features}
    row = frame.iloc[0][features].to_dict()
    shares = idt.predict_shares(cv, bundle, row)
    assert set(shares.keys()) == {"1", "2", "3", "4", "5"}
    total = sum(shares.values())
    assert abs(total - 1.0) < 1e-9, total
    assert all(v >= 0 for v in shares.values()), shares
    return True


def test_4_sanity_check_reports_near_zero_error():
    frame = _make_synthetic_frame()
    features = ["대상_hour_sin", "대상_hour_cos", "DSWRF", "TCDC"]
    fit = idt.fit_final_models(cv, frame, features, seed=1)
    bundle = {"models": fit["models"], "features": features}
    check = idt._sanity_check_sum_to_one(cv, bundle, frame, features, n_sample=10)
    assert check["표본수"] == 10, check
    assert check["비중합계_최대오차"] < 1e-9, check
    return True


def test_5_predict_shares_compatible_with_disaggregate_interface():
    """predict_shares()의 반환값이 inverter_disaggregation_v1_2026-08-28.py의
    disaggregate(weights_C=...) 계약(합계 1.0, {인버터번호str: 비중})과
    정확히 맞는지 통합 확인."""
    spec_idis = ilu.spec_from_file_location(
        "idis_test", PARENT / "inverter_disaggregation_v1_2026-08-28.py")
    idis = ilu.module_from_spec(spec_idis)
    spec_idis.loader.exec_module(idis)

    frame = _make_synthetic_frame()
    features = ["대상_hour_sin", "대상_hour_cos", "DSWRF", "TCDC"]
    fit = idt.fit_final_models(cv, frame, features, seed=1)
    bundle = {"models": fit["models"], "features": features}
    row = frame.iloc[0][features].to_dict()
    shares = idt.predict_shares(cv, bundle, row)

    cfg = idis.load_config()
    r = idis.disaggregate(100.0, "초단기", "1h", cfg, weights_C=shares)
    assert abs(r["합계_kW"] - 100.0) < 1e-6, r
    return True


TESTS = [
    test_1_fit_final_models_returns_five_models,
    test_2_fit_final_models_raises_on_too_few_rows,
    test_3_predict_shares_sums_to_one_after_normalization,
    test_4_sanity_check_reports_near_zero_error,
    test_5_predict_shares_compatible_with_disaggregate_interface,
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
