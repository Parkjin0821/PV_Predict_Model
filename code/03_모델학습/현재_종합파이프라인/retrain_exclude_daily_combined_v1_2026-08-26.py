# -*- coding: utf-8 -*-
"""일간 모델 — 2개 결측특성 동시 제외 재검증 — 08-26.

## 배경
`retrain_exclude_mean_communication_ok_v1_2026-08-26.py`와
`retrain_exclude_dswrflx_v1_2026-08-26.py`가 각각 따로 검증했지만,
실제 운영번들은 **두 특성을 동시에** 빼야 한다(둘 다 라이브에서
구조적으로 못 구함). 따로 뺐을 때 안전하다고 같이 뺐을 때도 안전하다고
보장되지 않으므로(상호작용 가능성), 동시제외 조합을 별도로 검증한다.

## 재구현 없음 — 두 스크립트와 동일하게 `daily_direct_final_audit_v1_
2026-08-25.py`의 `corrected_dataset`/`predict_oof`/`compare`만 재사용.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "일간_결합제외_재검증_v1_2026-08-26"
DSX = "DSWRFLX_bsrn정제"
MCO = "2일전평균_mean_communication_ok"
LABEL_BEFORE = "전체58(기존)"
LABEL_AFTER = "MCO+DSX_동시제외"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


daily_audit = _load("combo_daily", "daily_direct_final_audit_v1_2026-08-25.py")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    data, features, n_partial = daily_audit.corrected_dataset(capacity_kw)
    dsx_cols = [c for c in features if DSX in c]
    if MCO not in features or not dsx_cols:
        raise RuntimeError(f"특성목록 확인 실패: MCO={MCO in features}, DSX관련={dsx_cols}")
    drop_cols = [MCO] + dsx_cols
    print(f"전체특성 {len(features)}개, 동시제외 대상: {drop_cols}")
    features_after = [c for c in features if c not in drop_cols]
    print(f"제외 후 특성 {len(features_after)}개")

    before_rows, _ = daily_audit.predict_oof(data, features, capacity_kw, seed, LABEL_BEFORE)
    after_rows, _ = daily_audit.predict_oof(data, features_after, capacity_kw, seed, LABEL_AFTER)
    verdict, folds, merged = daily_audit.compare(before_rows, after_rows, LABEL_AFTER)

    pd.concat([before_rows.assign(구성=LABEL_BEFORE), after_rows.assign(구성=LABEL_AFTER)],
              ignore_index=True).to_csv(OUT / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "폴드별_비교.csv", index=False, encoding="utf-8-sig")
    (OUT / "요약.json").write_text(
        json.dumps({"제외컬럼": drop_cols, "compare_요약": verdict}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n=== 폴드별 비교(동시제외) ===")
    print(folds.to_string(index=False))
    print("\n=== 요약 ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str))
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
