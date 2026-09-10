# -*- coding: utf-8 -*-
"""초단기 +4h 전용 DSWRFLX 제외 재검증 — 최신 공식(청천지수+날씨군집화) 기준.

## 배경(Codex 08-26 지적, 정확함)
`retrain_exclude_dswrflx_v1_2026-08-26.py`는 초단기 4개 수평 전부에
`e2e_retrain_v5_공식B_v1_2026-08-24.py::run_ultra()`를 썼다. 그런데
+4h의 **현재 공식 모델은 v1이 아니라 v2 패치**(`e2e_retrain_v5_공식B_
v2_⑥반영_v1_2026-08-24.py::run_ultra_patch()`) — "청천지수 기본"이
아니라 **"청천지수+날씨군집화"**다(AGENTS.md: "+4h=청천지수+날씨군집화
(v2에서 신규 채택)"). 그래서 이전 스크립트의 +4h 결과는 이미 stale한
모델과 비교한 것이라 무효다. +1h·+2h·+3h·단기 3종은 v1 그대로가 지금도
공식이라 그 결과들은 유효하다(재검증 불필요).

## 두 곳에서 DSWRFLX를 빼야 하는 이유
`run_ultra_patch()`는 (1) 후보특성 선택에 `harness.sel.FORECAST_
COLUMNS`를, (2) 날씨군집화 입력에 `improvement.CLUSTER_SOURCE`
(`["DSWRF","DSWRFLX_bsrn정제","TCDC","LCDC","POP","SKY","목표_태양고도_deg"]`)
를 **따로** 참조한다. 코드 추적 결과, `run_ultra_patch()`가 쓰는
`dpc.load_ultra_frame()`은 `sel.OBSERVED_COLUMNS`/`FORECAST_COLUMNS`
로 프레임 자체를 새로 조립하므로(`hourly_features_for_horizon()`),
`sel.FORECAST_COLUMNS`만 패치해도 DSWRFLX가 프레임에 아예 안 생겨
`CLUSTER_SOURCE`의 `c in train` 조건에서 자동으로 걸러지긴 한다 —
그래도 Codex 요청대로 **양쪽 다 명시적으로 패치**해 의도를 코드에서도
분명히 남긴다(하나만 믿고 가다 나중에 프레임 조립 방식이 바뀌면
조용히 새는 걸 막는 이중 안전장치).

## 재구현 없음
`e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py::run_ultra_patch()`를
그대로 호출한다. 후보특성·군집원천 목록만 호출 전 몽키패치로 걸러냈다가
복원한다. 동일행 정렬은 `retrain_exclude_dswrflx_v1_2026-08-26.py`의
`align_same_rows`와 같은 방식(재사용 목적으로 이 파일에도 인라인으로
그 로직만 복제 — 원본 파일을 import하면 그쪽 __main__ 블록까지 딸려오는
걸 피하려 함).

## 이 스크립트가 하지 않는 것
운영모델(운영모델_v2)을 전혀 안 건드린다. 성능 비교표만 만든다.
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
OUT = ROOT / "outputs" / "DSWRFLX_제외_재검증_초단기4h_v1_2026-08-26"
DSX = "DSWRFLX_bsrn정제"
LABEL_BEFORE = "DSX_포함(v2공식)"
LABEL_AFTER = f"{DSX}_제외"
KEYS = ["티어", "수평_h", "폴드", "발행시각"]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2 = _load("ultra4_v2", "e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py")
harness = v2.harness
improvement = v2.improvement


def _remove_dsx() -> dict:
    before = {
        "전체후보": list(harness.FEATURE_SETS["전체후보"]),
        "FORECAST_COLUMNS": list(harness.sel.FORECAST_COLUMNS),
        "OBSERVED_COLUMNS": list(harness.sel.OBSERVED_COLUMNS),
        "CLUSTER_SOURCE": list(improvement.CLUSTER_SOURCE),
    }
    found = []
    if DSX in harness.FEATURE_SETS["전체후보"]:
        harness.FEATURE_SETS["전체후보"] = [c for c in harness.FEATURE_SETS["전체후보"] if c != DSX]
        found.append("FEATURE_SETS[전체후보]")
    if DSX in harness.sel.FORECAST_COLUMNS:
        harness.sel.FORECAST_COLUMNS = [c for c in harness.sel.FORECAST_COLUMNS if c != DSX]
        found.append("sel.FORECAST_COLUMNS")
    if DSX in harness.sel.OBSERVED_COLUMNS:
        harness.sel.OBSERVED_COLUMNS = [c for c in harness.sel.OBSERVED_COLUMNS if c != DSX]
        found.append("sel.OBSERVED_COLUMNS")
    if DSX in improvement.CLUSTER_SOURCE:
        improvement.CLUSTER_SOURCE = [c for c in improvement.CLUSTER_SOURCE if c != DSX]
        found.append("improvement.CLUSTER_SOURCE")
    if not found:
        raise RuntimeError(f"{DSX}를 아무 목록에서도 못 찾음 — 가정이 틀렸을 수 있다. 중단.")
    print(f"[몽키패치] {DSX} 제거됨: {found}")
    return before


def _restore(before: dict) -> None:
    harness.FEATURE_SETS["전체후보"] = before["전체후보"]
    harness.sel.FORECAST_COLUMNS = before["FORECAST_COLUMNS"]
    harness.sel.OBSERVED_COLUMNS = before["OBSERVED_COLUMNS"]
    improvement.CLUSTER_SOURCE = before["CLUSTER_SOURCE"]


def align_same_rows(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """retrain_exclude_dswrflx_v1_2026-08-26.py::align_same_rows와 동일 로직
    (그 파일에서 검증된 방식 그대로 복제 — 표본 정렬이 깨지면 중단)."""
    before = hourly[hourly["구성"] == LABEL_BEFORE]
    after = hourly[hourly["구성"] == LABEL_AFTER]
    merged = before[KEYS + ["실제_kW", "예측_kW"]].merge(
        after[KEYS + ["실제_kW", "예측_kW"]], on=KEYS,
        suffixes=("_기존", "_제외"), validate="one_to_one")
    mismatch = (merged["실제_kW_기존"] - merged["실제_kW_제외"]).abs() > 1e-9
    if bool(mismatch.any()):
        raise RuntimeError(f"동일 키인데 실측값이 다른 행 {int(mismatch.sum())}건 — 표본 정렬 이상. 중단.")
    coverage = pd.DataFrame([
        {"구성": LABEL_BEFORE, "자체_행수": len(before), "교집합_행수": len(merged), "교집합밖_행수": len(before) - len(merged)},
        {"구성": LABEL_AFTER, "자체_행수": len(after), "교집합_행수": len(merged), "교집합밖_행수": len(after) - len(merged)},
    ])
    return merged, coverage


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) / "
          f"대상: 초단기 +4h만(v2 공식, 청천지수+날씨군집화) / 폴드: 공식 5폴드\n")

    print(f"=== [1/2] {LABEL_BEFORE} ===")
    before_df, before_audit = v2.run_ultra_patch(capacity_kw, seed)
    before_df["구성"] = LABEL_BEFORE

    print(f"\n=== [2/2] {LABEL_AFTER} ===")
    saved = _remove_dsx()
    try:
        after_df, after_audit = v2.run_ultra_patch(capacity_kw, seed)
    finally:
        _restore(saved)
    after_df["구성"] = LABEL_AFTER

    hourly = pd.concat([before_df, after_df], ignore_index=True)
    hourly.to_csv(OUT / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")

    merged, coverage = align_same_rows(hourly)
    coverage.to_csv(OUT / "표본커버리지.csv", index=False, encoding="utf-8-sig")

    rows = []
    for (tier, h), g in merged.groupby(["티어", "수평_h"], sort=False):
        actual = g["실제_kW_기존"].to_numpy()
        rec = {"티어": tier, "수평_h": h, "동일행_n": len(g)}
        for label, col in ((LABEL_BEFORE, "예측_kW_기존"), (LABEL_AFTER, "예측_kW_제외")):
            e = actual - g[col].to_numpy()
            mae, rmse = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
            rec[f"nMAE_pct_{label}"] = round(mae / capacity_kw * 100, 3)
            rec[f"nRMSE_pct_{label}"] = round(rmse / capacity_kw * 100, 3)
        rec["nMAE_pct_변화"] = round(rec[f"nMAE_pct_{LABEL_AFTER}"] - rec[f"nMAE_pct_{LABEL_BEFORE}"], 3)
        rec["nRMSE_pct_변화"] = round(rec[f"nRMSE_pct_{LABEL_AFTER}"] - rec[f"nRMSE_pct_{LABEL_BEFORE}"], 3)
        rows.append(rec)
    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / "성능비교.csv", index=False, encoding="utf-8-sig")

    (OUT / "감사로그.json").write_text(
        json.dumps({"감사_기존": before_audit, "감사_제외": after_audit}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n=== 표본 커버리지(동일행 정렬 결과) ===")
    print(coverage.to_string(index=False))
    print("\n=== 성능비교(동일행 기준, 포함 vs 제외) ===")
    print(perf.to_string(index=False))
    print(f"\n저장 완료: {OUT}")
    print("\n★주의★ 채택 여부를 자동 결정하지 않는다 — 결과를 Claude에게 전달해 판단받을 것.")


if __name__ == "__main__":
    main()
