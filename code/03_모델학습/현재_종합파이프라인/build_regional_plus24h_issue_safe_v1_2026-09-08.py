# -*- coding: utf-8 -*-
"""부안·김제·영광 +24h 전용 모델 - 발행시각 안전(issue-safe), 코덱스
D+1 프레임 재사용.

## 배경(사용자 지시: "부안/김제/영광 신규 모델 구축은 너가 진행해줄래?")
코덱스가 09-08에 "D+1 자료는 +14~+35h 대상시각을 포함하지만 이것을
+24/+48 전용 검증이라고 바꿔 부르지 않는다"고 명확히 선을 그었다 -
전용 모델이 없다는 뜻. 실측 확인 결과 리드타임은 14/17/20/23/26/29/
32/35h(3시간 간격 8개)뿐이라, **+24h는 가장 가까운 23h로 만들 수
있지만 +48h는 데이터 자체가 35h까지밖에 없어 지금은 절대 못 만든다**
(코드 문제 아님 - 없는 미래 NWP를 만들어낼 수 없음, +48h는 별도
데이터수집 확장 또는 WeatherNext3 전환 후에나 가능 - AGENTS.md에 별도
기록).

## 재사용(재구현 안 함)
`rebuild_regional_dayahead_issue_safe_v1_2026-09-08.py`(코덱스 작성)의
`build_safe_frame()`을 그대로 import해서 쓴다 - 발행시각 이하 lag/
rolling만 쓰는 누출방지 로직, 김제 미래일사 제외, phase2 특성목록
재사용 전부 동일. 이 파일이 새로 하는 건 **결과 프레임을 리드타임
23h(±1.5h)로 필터링한 뒤 그 부분집합만으로 walk-forward를 다시 도는
것** 하나뿐이다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "rebuild_regional_dayahead_issue_safe_v1_2026-09-08.py"
TARGET = "plant_ac_power_kw"
TARGET_LEAD_H = 23.0  # 14~35h(3h간격) 중 24h에 가장 가까운 실제 가용값
LEAD_TOLERANCE_H = 1.5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_region(region: str, cfg: dict, base) -> dict:
    frame, features, audit = base.build_safe_frame(cfg)
    lead_h = (frame["target_time_kst"] - frame["issue_time_kst"]).dt.total_seconds() / 3600
    sub = frame[(lead_h - TARGET_LEAD_H).abs() <= LEAD_TOLERANCE_H].copy()
    if sub.empty:
        raise RuntimeError(f"{region}: lead_h={TARGET_LEAD_H}±{LEAD_TOLERANCE_H} 구간에 표본 없음")

    required = features + [TARGET]
    days = pd.DatetimeIndex(np.sort(sub["issue_day"].dropna().unique()))
    model_names = ["LightGBM", "XGBoost", "선형회귀"]
    pooled: dict[str, dict] = {}
    for name in model_names:
        ys, ps = [], []
        for train_days, test_days in base.folds(days, cfg["initial"], cfg["block"]):
            if train_days.max() >= test_days.min():
                raise AssertionError("날짜누출")
            tr = sub[sub["issue_day"].isin(train_days)].dropna(subset=required)
            te = sub[sub["issue_day"].isin(test_days)].dropna(subset=required)
            if len(tr) < 60 or len(te) < 3:
                continue
            m = base.make_model(base.HARNESS, name)
            m.fit(tr[features], tr[TARGET])
            ps.append(np.clip(m.predict(te[features]), 0, cfg["capacity"]))
            ys.append(te[TARGET].to_numpy())
        if not ys:
            raise RuntimeError(f"{region}/{name}(+24h): 유효 폴드 없음 (표본 {len(sub)}행)")
        pooled[name] = {**base.metrics(np.concatenate(ys), np.concatenate(ps)),
                        "test_rows": int(sum(map(len, ys)))}

    selected = min(pooled, key=lambda n: pooled[n]["MAE_kW"])
    clean = sub.dropna(subset=required).sort_values("target_time_kst")
    model = base.make_model(base.HARNESS, selected)
    model.fit(clean[features], clean[TARGET])
    replay = clean.tail(min(64, len(clean)))
    pred = np.asarray(model.predict(replay[features]), float)
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{region}(+24h): 비유한 replay")

    out = cfg["dir"] / "outputs" / "공식_plus24h_issue_safe_v1_2026-09-08"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "model.joblib"
    tmp = out / "model.joblib.tmp"
    payload = {"model": model, "features": features, "region": region, "horizon": "+24h(nearest=23h)",
              "capacity_kw": cfg["capacity"], "model_name": selected}
    joblib.dump(payload, tmp)
    loaded = joblib.load(tmp)
    if not np.allclose(pred, loaded["model"].predict(replay[features]), rtol=0, atol=1e-10):
        raise RuntimeError(f"{region}(+24h): 번들 왕복 불일치")
    os.replace(tmp, final)

    result = {
        "status": "ready_for_live_assembler_after_nwp_gate", "region": region,
        "horizon": "+24h", "actual_lead_h_used": TARGET_LEAD_H, "lead_tolerance_h": LEAD_TOLERANCE_H,
        "selected_model": selected, "feature_count": len(features), "capacity_kw": cfg["capacity"],
        "training_rows": len(clean), "walk_forward": pooled, "leakage_audit": audit,
        "nmae_pct_selected": round(pooled[selected]["MAE_kW"] / cfg["capacity"] * 100, 4),
        "model_sha256": digest(final), "created_at_kst": datetime.now().astimezone().isoformat(),
        "_주의": "+48h는 리드타임 데이터가 35h까지뿐이라 지금 데이터로 절대 구축 불가 - "
               "신규 데이터수집 확장 또는 WeatherNext3 전환 후 재시도 대상.",
    }
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    base = load_module("dayahead_issue_safe_base", BASE_SCRIPT)
    phase2 = base.load_module("plus24h_phase2", base.PHASE2)
    base.HARNESS = phase2._load_harness()
    results = [run_region(r, c, base) for r, c in base.CFG.items()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
