# -*- coding: utf-8 -*-
"""단기 `run_short()` DIFSWRF C전략 연결 누락 — 좁은 재감사(연결 부분만).

## 배경
08-25 일간 최종감사 중, 공식 E2E(`e2e_retrain_v5_공식B_v1_2026-08-24.py`)의
`run_ultra()`는 `native_ok = harness.NATIVE_MISSING_OK | {DIF}`로 DIFSWRF를
C전략(원값 NaN 유지 + `_결측여부` 플래그, native missing 위임)대로 처리하지만,
같은 파일의 `run_short()`(233행)는 `harness.NATIVE_MISSING_OK`만 쓰고 DIF를
추가하지 않는다. `DIFSWRF_bsrn정제`가 단기 전체후보(`CANDIDATE_COLUMNS`)에
있고 상관 임계(0.3) 통과 시 특성으로 선택될 수 있으므로, 선택되는 폴드에서는
결측행이 (마스크 특성이 이미 붙어있는데도) dropna로 통째로 삭제된다 —
일간에서 발견된 것과 동일 유형의 버그.

## 이 스크립트가 하는 일(좁은 재감사 — 모델구조 재탐색 아님)
공식 `run_short()`와 완전히 같은 입력·폴드·구조선택·시드를 그대로 재사용하되
`native_ok` 한 줄만 두 가지로 바꿔(버그판 vs C전략판) 나란히 재학습한다.
- 특성선택(`select_features_in_fold`)은 native_ok에 의존하지 않으므로 두
  변형에서 선택 특성 집합은 동일하다 — 오직 학습/시험 행 필터링만 달라진다.
- DIF가 실제로 선택된 폴드가 있는지, 있다면 행수·성능이 얼마나 달라지는지만
  좁게 확인한다.

## 재사용(재구현 금지 원칙)
- `e2e_retrain_v5_공식B_v1_2026-08-24.py`: dpc/harness/improvement 로더,
  `add_difswrf_flag`, `DIF`, `MODEL_DIR` 규칙, `_roundtrip`.
- `fold_hour_prevalidation_v1_2026-08-24.OFFICIAL_B_WINDOWS`(정정 5폴드).
- `model_improvement_round2_v1_2026-08-21.choose_structure_kfold/make_model`.

## 산출물
`outputs/단기_DIFSWRF_C전략_연결_재감사_v1_2026-08-25/`
- `폴드별_비교.csv`: 버그판 vs C전략판의 특성수·DIF선택여부·학습행수·
  시험행수·MAE·RMSE 나란히 비교.
- `요약.json`: 전체 개선율, DIF가 선택된 폴드 수.

로컬 재학습만(API 없음).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "단기_DIFSWRF_C전략_연결_재감사_v1_2026-08-25"
N_INVERTERS = 5


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


e2e = _load("audit_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc, harness, improvement = e2e.dpc, e2e.harness, e2e.improvement
OFFICIAL_WINDOWS = e2e.OFFICIAL_WINDOWS
DIF = e2e.DIF
add_difswrf_flag = e2e.add_difswrf_flag


def run_short_variant(capacity_kw: float, seed: int, include_dif_native: bool) -> tuple[pd.DataFrame, list[dict]]:
    """공식 run_short()와 동일하되 native_ok에 DIF를 넣을지만 바꾼 버전."""
    rows, audits = [], []
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    for horizon in (1, 24, 48):
        frame = add_difswrf_flag(dpc.load_short_frame(horizon))
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        daylight = frame[frame["목표_낮시간"] > 0]

        for fold, s, e in OFFICIAL_WINDOWS:
            start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
            train_b = daylight[(daylight.index < start) & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
            test_b = daylight[(daylight.index >= start) & (daylight.index < end)
                              & (daylight["_목표_가용인버터수"] >= N_INVERTERS)]
            if len(train_b) < 200 or len(test_b) < 30:
                continue
            chosen = harness.select_features_in_fold(
                train_b.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            dif_selected = DIF in features

            native_ok = harness.NATIVE_MISSING_OK | ({DIF} if include_dif_native else set())
            required = [c for c in features if c not in native_ok] + ["목표_발전출력_kW"]
            train = train_b.dropna(subset=required)
            test = test_b.dropna(subset=required)
            if len(train) < 200 or len(test) < 30:
                continue

            params, structure_name = improvement.choose_structure_kfold(
                "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
            model = improvement.make_model("단기", seed, params)
            model.fit(train[features], train["목표_발전출력_kW"])
            pred = np.clip(model.predict(test[features]), 0, capacity_kw)
            actual = test["목표_발전출력_kW"].to_numpy()
            target_at = test.index + pd.to_timedelta(horizon - 1, unit="h")
            variant = "C전략(고침)" if include_dif_native else "버그판(현재공식)"
            for issued, tt, y, p in zip(test.index, target_at, actual, pred):
                rows.append({"수평_h": horizon, "폴드": fold, "변형": variant,
                            "발행시각": issued, "대상시각": tt, "실제_kW": y, "예측_kW": p})
            audits.append({
                "수평_h": horizon, "폴드": fold, "변형": variant,
                "DIF_선택됨": dif_selected, "특성수": len(features), "선택구조": structure_name,
                "학습행수": len(train), "시험행수": len(test),
            })
            print(f"[단기 +{horizon}h {fold}] {variant} DIF선택={dif_selected} "
                  f"학습{len(train):,}/시험{len(test):,} 완료")
    return pd.DataFrame(rows), audits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량={capacity_kw}kW seed={seed}\n")

    print("=== 버그판(현재 공식 run_short — DIF를 native_ok에서 뺌) ===")
    buggy_rows, buggy_audit = run_short_variant(capacity_kw, seed, include_dif_native=False)
    print("\n=== C전략판(DIF를 native_ok에 포함 — 일간과 동일 원칙) ===")
    fixed_rows, fixed_audit = run_short_variant(capacity_kw, seed, include_dif_native=True)

    audit = pd.DataFrame(buggy_audit + fixed_audit)
    audit.to_csv(OUT / "폴드별_비교.csv", index=False, encoding="utf-8-sig")

    def summarize(rows: pd.DataFrame, label: str) -> dict:
        if len(rows) == 0:
            return {"변형": label, "n": 0}
        e = rows["실제_kW"] - rows["예측_kW"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        return {"변형": label, "n": len(rows), "MAE_kW": round(mae, 4), "RMSE_kW": round(rmse, 4)}

    overall = [summarize(buggy_rows, "버그판(현재공식)"), summarize(fixed_rows, "C전략판(고침)")]
    per_horizon = []
    for h in (1, 24, 48):
        b = buggy_rows[buggy_rows["수평_h"] == h]
        f = fixed_rows[fixed_rows["수평_h"] == h]
        per_horizon.append({"수평_h": h, **{f"버그_{k}": v for k, v in summarize(b, "버그").items() if k != "변형"}})
        per_horizon.append({"수평_h": h, **{f"고침_{k}": v for k, v in summarize(f, "고침").items() if k != "변형"}})

    n_dif_selected_folds = int(audit[audit["변형"] == "버그판(현재공식)"]["DIF_선택됨"].sum())
    n_total_folds = int((audit["변형"] == "버그판(현재공식)").sum())
    row_diff = pd.DataFrame(buggy_audit)[["수평_h", "폴드", "학습행수", "시험행수"]].merge(
        pd.DataFrame(fixed_audit)[["수평_h", "폴드", "학습행수", "시험행수"]],
        on=["수평_h", "폴드"], suffixes=("_버그", "_고침"))
    row_diff["학습행_증가"] = row_diff["학습행수_고침"] - row_diff["학습행수_버그"]
    row_diff["시험행_증가"] = row_diff["시험행수_고침"] - row_diff["시험행수_버그"]
    row_diff.to_csv(OUT / "행수_변화.csv", index=False, encoding="utf-8-sig")

    summary = {
        "발견": "run_short()가 DIF를 native_ok에 안 넣어 C전략(원값NaN+마스크)이 "
                "아니라 옛 방식(DIF 결측행 dropna)으로 처리되고 있었음 — run_ultra()는 정상.",
        "DIF가_선택된_폴드수": f"{n_dif_selected_folds}/{n_total_folds}",
        "전체_비교": overall,
        "폴드별_행수_최대_증가": {
            "학습행": int(row_diff["학습행_증가"].max()) if len(row_diff) else 0,
            "시험행": int(row_diff["시험행_증가"].max()) if len(row_diff) else 0,
        },
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
