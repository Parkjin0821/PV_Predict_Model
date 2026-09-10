# -*- coding: utf-8 -*-
"""DSWRFLX_bsrn정제 후보특성 제외 재검증 — 08-26.

## 배경
Codex 진단(`diagnose_kma_nwp_kimr_missing_v1_2026-08-26.py`)으로
DSWRFLX(KIMR)가 2026-06-01부터 라이브에서 원문 자체가 `nan`으로
영구 결측임이 확정됐다(AGENTS.md 08-26절). Google Scholar 사전게이트로
확인한 결과, "GHI에서 DNI를 역산 추정하는 건 권장되지 않는다"는 게
문헌 결론이었다(에어로졸에 따라 DNI 변동폭이 GHI보다 훨씬 큼). 이미
2026-08-20에도 같은 이유로 BRL 역산을 보류하고 DSWRF(GHI)를 주력
입력으로 확정한 전례가 있다. 그래서 이번에도 **새 대체 파생특성을
설계하지 않고**, DSWRFLX_bsrn정제를 후보특성에서 완전히 빼고 기존
공식 파이프라인(구조선택·튜닝·5폴드)을 그대로 재실행해 성능이
유지되는지만 확인한다.

DIFSWRF_bsrn정제(DIF)는 건드리지 않는다 — 이미 확립된 C전략(원값
NaN 유지+결측여부 플래그)으로 높은 결측률에도 대응하도록 설계돼 있고,
사용자가 이번 변경 범위에서 명시적으로 제외했다.

## 재구현 없음(원칙 준수) — 이 스크립트가 새로 만든 로직은 없다
- **초단기·단기**: `e2e_retrain_v5_공식B_v1_2026-08-24.py`의
  `run_ultra()`/`run_short()`를 그대로 호출한다. 두 함수는 후보특성
  목록을 매 호출 시 `harness.FEATURE_SETS["전체후보"]`·
  `harness.sel.OBSERVED_COLUMNS`·`harness.sel.FORECAST_COLUMNS`에서
  새로 읽어오므로(캐시 안 됨), 호출 전에 이 세 곳에서 DSX 하나만
  몽키패치로 제거했다가 복원한다. `select_features_in_fold()`·
  `choose_structure_kfold()`·5폴드 루프·roundtrip 검증은 전부 원본
  그대로 재사용된다.
- **일간**: `daily_direct_final_audit_v1_2026-08-25.py`의
  `corrected_dataset()`(현재 공식 특성공학)로 데이터를 만들고,
  DSX 파생 컬럼(`목표일예보_DSWRFLX_bsrn정제_sum`·`_mean`)만 특성
  목록에서 뺀 뒤, 같은 파일의 `predict_oof()`(기존 LOGO 그룹제거
  재검증에 이미 쓰던 함수)와 `compare()`(기존 기준 대비 후보 비교
  함수)를 그대로 재사용한다 — 새 비교 로직을 만들지 않았다.

## 이 스크립트가 하지 않는 것(중요)
- **운영모델(운영모델_v2)을 덮어쓰지 않는다.** 성능 비교표만 만든다.
  결과를 보고 실제로 채택할지는 Claude가 판단한 뒤 별도로 진행한다.
- 초단기·단기는 `e2e.run_ultra`/`run_short`가 내부적으로 자체 `MODEL_DIR`에
  joblib을 저장하는데, "포함"/"제외" 두 번 호출하면 같은 파일명을 두
  번째 호출(제외) 결과가 덮어쓴다 — 이 스크립트의 목적은 이 파일의
  `outputs/`에 저장하는 성능 비교 CSV/JSON이지 그 중간 joblib 보관이
  아니므로 의도한 동작이다. 일간은 `predict_oof(save_models=False)`
  기본값이라 애초에 joblib을 만들지 않는다.

## Codex 실행 시 참고
- 실행 시간: 초단기 4수평 + 단기 3수평을 "포함"/"제외" 두 번씩(5폴드
  튜닝 포함) + 일간 두 번(5폴드 튜닝 포함) — 로컬 사양에 따라 총
  20~60분 걸릴 수 있다.
- 출력: `outputs/DSWRFLX_제외_재검증_v1_2026-08-26/`
  - `초단기_단기_동일행_예측정답.csv`, `일간_동일행_예측정답.csv`
  - `초단기_단기_성능비교.csv` (동일행 기준 티어·수평별 nMAE/nRMSE + 변화량)
  - `초단기_단기_표본커버리지.csv` (동일행 정렬에서 밀려난 행 수)
  - `일간_성능비교.csv` (MAE/RMSE/WAPE), `일간_폴드별_비교.csv`
    (`daily_audit.compare()` 원본 출력)
  - `감사로그.json` (몽키패치 위치, 폴드별 감사기록)

## 몽키패치 안전성(08-26 코드 추적으로 확인함 — 추측 아님)
후보목록을 호출 전에 갈아끼우는 방식이라, "프레임에는 DSX가 남아있는데
후보목록에서만 빠져서 오히려 `base_cols`로 강제 투입되는" 사고가 나면
제외 실험이 조용히 무효가 된다. 실제 경로를 따라가 그렇지 않음을 확인했다.
- 초단기: `train_ultra_short_official_v1_2026-08-21.py::
  hourly_features_for_horizon()`이 `harness.sel.OBSERVED_COLUMNS`·
  `FORECAST_COLUMNS`를 **호출 시점에** 읽어 프레임을 만든다 → 패치된
  목록이 반영되어 프레임 자체에 DSX가 안 생긴다.
- 단기: `defect_policy_comparison_v1_2026-08-21.py::load_short_frame()`이
  `harness.FEATURE_SETS["전체후보"]`를 **호출 시점에** 읽어
  `harness.build_frame()`에 넘긴다 → 마찬가지로 DSX가 안 생긴다.
- 따라서 두 티어 모두 `base_cols`(= 프레임에 있으나 후보목록에 없는 컬럼)에
  DSX가 섞이지 않는다.
- 주의: `sel.CANDIDATE_COLUMNS`는 import 시점에 한 번 조립된 별도 리스트라
  이 패치로 바뀌지 않는다. 위 두 경로가 그걸 쓰지 않아 문제없지만,
  나중에 `CANDIDATE_COLUMNS`를 직접 읽는 코드를 이 실험에 끌어들이면
  그때는 그 리스트도 함께 걸러야 한다.

## 2026-08-26 Codex 실행 후 발견한 적용범위 주의
- 이 스크립트가 호출하는 v1 `run_ultra()`의 +4h는 구 공식인
  `청천지수 기본`이다. 현재 공식 +4h `청천지수+날씨군집화`는 별도 v2
  패치 파일에 있으므로 이 스크립트의 +4h 결과는 최신 공식모델
  채택판단에 사용하면 안 된다.
- 최신 +4h를 다시 비교할 때는 후보특성 목록뿐 아니라 날씨군집 원천 목록
  `model_improvement_round2_v1_2026-08-21.CLUSTER_SOURCE`에서도 DSX를
  제거해야 진짜 DSWRFLX 제외 비교가 된다.
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
OUT = ROOT / "outputs" / "DSWRFLX_제외_재검증_v1_2026-08-26"
DSX = "DSWRFLX_bsrn정제"
DIF = "DIFSWRF_bsrn정제"
LABEL_BEFORE = "DSX_포함(기존)"
LABEL_AFTER = f"{DSX}_제외"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("dswrflx_excl_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
daily_audit = _load("dswrflx_excl_daily", "daily_direct_final_audit_v1_2026-08-25.py")
harness = e2e.harness
# 기존 공식 E2E fold_models를 비교 실험이 덮어쓰지 않도록 저장 위치만
# 실험 전용 폴더로 분리한다. 학습·선택·예측 로직에는 영향이 없다.
e2e.MODEL_DIR = OUT / "fold_models"


# ─────────────────────────── 초단기·단기 ───────────────────────────

def _remove_dsx_from_candidates() -> dict:
    """단기·초단기 후보목록에서 DSX를 빼고, 복원용 원본 스냅샷을 반환한다.

    어디에도 없으면 가정이 틀린 것이므로 조용히 넘어가지 않고 예외를
    던진다(가짜로 "제외 성공"했다고 결과를 낼 위험 방지).
    """
    before = {
        "전체후보": list(harness.FEATURE_SETS["전체후보"]),
        "FORECAST_COLUMNS": list(harness.sel.FORECAST_COLUMNS),
        "OBSERVED_COLUMNS": list(harness.sel.OBSERVED_COLUMNS),
    }
    found_where = []
    if DSX in harness.FEATURE_SETS["전체후보"]:
        harness.FEATURE_SETS["전체후보"] = [c for c in harness.FEATURE_SETS["전체후보"] if c != DSX]
        found_where.append("FEATURE_SETS[전체후보]")
    if DSX in harness.sel.FORECAST_COLUMNS:
        harness.sel.FORECAST_COLUMNS = [c for c in harness.sel.FORECAST_COLUMNS if c != DSX]
        found_where.append("sel.FORECAST_COLUMNS")
    if DSX in harness.sel.OBSERVED_COLUMNS:
        harness.sel.OBSERVED_COLUMNS = [c for c in harness.sel.OBSERVED_COLUMNS if c != DSX]
        found_where.append("sel.OBSERVED_COLUMNS")
    if not found_where:
        raise RuntimeError(
            f"{DSX}를 후보특성 목록(FEATURE_SETS[전체후보]·sel.OBSERVED_COLUMNS·"
            "sel.FORECAST_COLUMNS) 어디서도 못 찾았다 — 컬럼명이 바뀌었거나 "
            "가정이 틀렸을 수 있다. 중단."
        )
    print(f"[몽키패치] {DSX} 제거됨: {found_where}")
    return before


def _restore_candidates(before: dict) -> None:
    harness.FEATURE_SETS["전체후보"] = before["전체후보"]
    harness.sel.FORECAST_COLUMNS = before["FORECAST_COLUMNS"]
    harness.sel.OBSERVED_COLUMNS = before["OBSERVED_COLUMNS"]


def run_ultra_short_both(capacity_kw: float, seed: int):
    """e2e.run_ultra()/run_short()를 "포함"→"제외" 순서로 그대로 두 번
    호출한다. 함수 내부 로직은 전혀 건드리지 않는다(재구현 아님)."""
    print(f"\n=== [1/2] {LABEL_BEFORE} — 초단기·단기(기존 공식 그대로 재현) ===")
    ultra_before, ultra_audit_before = e2e.run_ultra(capacity_kw, seed)
    short_before, short_audit_before = e2e.run_short(capacity_kw, seed)
    ultra_before["구성"] = LABEL_BEFORE
    short_before["구성"] = LABEL_BEFORE

    print(f"\n=== [2/2] {LABEL_AFTER} — 초단기·단기 ===")
    saved = _remove_dsx_from_candidates()
    try:
        ultra_after, ultra_audit_after = e2e.run_ultra(capacity_kw, seed)
        short_after, short_audit_after = e2e.run_short(capacity_kw, seed)
    finally:
        _restore_candidates(saved)
    ultra_after["구성"] = LABEL_AFTER
    short_after["구성"] = LABEL_AFTER

    hourly = pd.concat([ultra_before, short_before, ultra_after, short_after], ignore_index=True)
    audits = {
        "초단기_기존": ultra_audit_before, "단기_기존": short_audit_before,
        "초단기_제외": ultra_audit_after, "단기_제외": short_audit_after,
    }
    return hourly, audits


KEYS = ["티어", "수평_h", "폴드", "발행시각"]


def align_same_rows(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """포함/제외 두 실행을 **동일행**으로 맞춘다(공정 비교의 전제).

    ★왜 필요한가★: `run_ultra`/`run_short`는 폴드마다 특성을 새로 선택하고
    `required = [c for c in features if c not in native_ok]`로 `dropna`한다.
    DSX를 빼면 선택특성이 달라져 `required`가 달라지고, 그 결과 **살아남는
    시험행 집합 자체가 달라질 수 있다**. 각자 행으로 지표를 내면 "성능이
    변했다"가 특성 효과인지 표본 구성 차이인지 구분할 수 없다.
    일간은 `daily_audit.compare()`가 이미 `validate="one_to_one"` 병합으로
    동일행을 강제하므로, 여기서 초단기·단기에 같은 원칙을 맞춰준다.

    교집합 밖으로 밀려난 행 수도 함께 반환해 조용히 표본이 줄지 않게 한다.
    """
    before = hourly[hourly["구성"] == LABEL_BEFORE]
    after = hourly[hourly["구성"] == LABEL_AFTER]
    merged = before[KEYS + ["실제_kW", "예측_kW"]].merge(
        after[KEYS + ["실제_kW", "예측_kW"]], on=KEYS,
        suffixes=("_기존", "_제외"), validate="one_to_one")
    # 같은 (티어·수평·폴드·발행시각)이면 실측 타깃도 당연히 같아야 한다.
    # 다르면 표본 정렬이 깨진 것이므로 조용히 넘어가지 않고 중단한다.
    mismatch = (merged["실제_kW_기존"] - merged["실제_kW_제외"]).abs() > 1e-9
    if bool(mismatch.any()):
        raise RuntimeError(
            f"동일 키인데 실측값이 다른 행 {int(mismatch.sum())}건 — 표본 정렬 이상. 중단.")
    coverage = pd.DataFrame([{
        "구성": LABEL_BEFORE, "자체_행수": len(before), "교집합_행수": len(merged),
        "교집합밖_행수": len(before) - len(merged),
    }, {
        "구성": LABEL_AFTER, "자체_행수": len(after), "교집합_행수": len(merged),
        "교집합밖_행수": len(after) - len(merged),
    }])
    return merged, coverage


def hourly_performance_delta(hourly: pd.DataFrame, capacity_kw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """동일행으로 맞춘 뒤 티어·수평별 nMAE/nRMSE와 그 변화량을 낸다."""
    merged, coverage = align_same_rows(hourly)
    rows = []
    for (tier, h), g in merged.groupby(["티어", "수평_h"], sort=False):
        actual = g["실제_kW_기존"].to_numpy()
        rec = {"티어": tier, "수평_h": h, "동일행_n": len(g)}
        for label, col in ((LABEL_BEFORE, "예측_kW_기존"), (LABEL_AFTER, "예측_kW_제외")):
            e = actual - g[col].to_numpy()
            mae, rmse = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
            rec[f"MAE_kW_{label}"] = round(mae, 3)
            rec[f"RMSE_kW_{label}"] = round(rmse, 3)
            rec[f"nMAE_pct_{label}"] = round(mae / capacity_kw * 100, 3)
            rec[f"nRMSE_pct_{label}"] = round(rmse / capacity_kw * 100, 3)
        rec["MAE_kW_변화"] = round(rec[f"MAE_kW_{LABEL_AFTER}"] - rec[f"MAE_kW_{LABEL_BEFORE}"], 3)
        rec["RMSE_kW_변화"] = round(rec[f"RMSE_kW_{LABEL_AFTER}"] - rec[f"RMSE_kW_{LABEL_BEFORE}"], 3)
        rec["nMAE_pct_변화"] = round(rec[f"nMAE_pct_{LABEL_AFTER}"] - rec[f"nMAE_pct_{LABEL_BEFORE}"], 3)
        rec["nRMSE_pct_변화"] = round(rec[f"nRMSE_pct_{LABEL_AFTER}"] - rec[f"nRMSE_pct_{LABEL_BEFORE}"], 3)
        rows.append(rec)
    return pd.DataFrame(rows), coverage


# ────────────────────────────── 일간 ──────────────────────────────

def run_daily_both(capacity_kw: float, seed: int):
    """corrected_dataset()으로 동일 특성공학을 만들고, DSX 파생 컬럼만
    빼서 predict_oof()·compare()를 그대로 재사용한다(재구현 아님)."""
    data, features, n_partial = daily_audit.corrected_dataset(capacity_kw)
    dsx_cols = [c for c in features if DSX in c]
    if not dsx_cols:
        raise RuntimeError(
            f"일간 특성목록에서 {DSX} 관련 컬럼을 못 찾았다("
            f"기대: 목표일예보_{DSX}_sum/_mean류) — 가정이 틀렸을 수 있다. 중단."
        )
    print(f"\n[일간] 부분가용일 {n_partial}일, 전체특성 {len(features)}개, "
          f"{DSX} 관련 제외 대상: {dsx_cols}")
    features_after = [c for c in features if c not in dsx_cols]

    before_rows, before_audit = daily_audit.predict_oof(
        data, features, capacity_kw, seed, LABEL_BEFORE)
    after_rows, after_audit = daily_audit.predict_oof(
        data, features_after, capacity_kw, seed, LABEL_AFTER)
    # daily_audit.compare()는 (요약dict, 폴드별DataFrame, 병합원본DataFrame)
    # 3-튜플을 반환한다(단일 DataFrame 아님 — 08-26 mean_communication_ok
    # 스크립트를 실제로 돌려보고 나서 AttributeError로 발견한 오류를 여기도
    # 똑같이 갖고 있어서 같이 고쳤다).
    compare_verdict, compare_folds, compare_merged = daily_audit.compare(
        before_rows, after_rows, LABEL_AFTER)
    return (before_rows, after_rows, compare_verdict, compare_folds,
            before_audit, after_audit, dsx_cols)


def daily_performance(before_rows: pd.DataFrame, after_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in (("기존", before_rows), ("제외", after_rows)):
        e = g["실제_kWh"] - g["예측_kWh"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        wape = float(e.abs().sum() / g["실제_kWh"].abs().sum() * 100)
        rows.append({"티어": "일간", "수평_h": "D+1", "n": len(g), "구성": label,
                    "MAE_kWh": round(mae, 2), "RMSE_kWh": round(rmse, 2), "WAPE_pct": round(wape, 2)})
    return pd.DataFrame(rows)


# ──────────────────────────────── main ────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e2e.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) / "
          f"제외대상: {DSX} 단독(DIF는 기존 C전략 유지) / 폴드: 공식 5폴드\n")

    hourly, hourly_audits = run_ultra_short_both(capacity_kw, seed)
    hourly.to_csv(OUT / "초단기_단기_동일행_예측정답.csv", index=False, encoding="utf-8-sig")
    hourly_perf, hourly_coverage = hourly_performance_delta(hourly, capacity_kw)
    hourly_perf.to_csv(OUT / "초단기_단기_성능비교.csv", index=False, encoding="utf-8-sig")
    hourly_coverage.to_csv(OUT / "초단기_단기_표본커버리지.csv", index=False, encoding="utf-8-sig")

    (before_daily, after_daily, compare_verdict, compare_folds, daily_audit_before,
     daily_audit_after, dsx_cols) = run_daily_both(capacity_kw, seed)
    daily_rows = pd.concat([
        before_daily.assign(구성=LABEL_BEFORE), after_daily.assign(구성=LABEL_AFTER),
    ], ignore_index=True)
    daily_rows.to_csv(OUT / "일간_동일행_예측정답.csv", index=False, encoding="utf-8-sig")
    compare_folds.to_csv(OUT / "일간_폴드별_비교.csv", index=False, encoding="utf-8-sig")
    daily_perf = daily_performance(before_daily, after_daily)
    daily_perf.to_csv(OUT / "일간_성능비교.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        hourly_perf.assign(결과구분="초단기·단기"),
        daily_perf.assign(결과구분="일간"),
    ], ignore_index=True, sort=False).to_csv(
        OUT / "성능비교_포함vs제외.csv", index=False, encoding="utf-8-sig")

    (OUT / "감사로그.json").write_text(
        json.dumps({
            "일간_제외컬럼": dsx_cols,
            "비교전용_모델저장폴더": str(e2e.MODEL_DIR),
            "초단기_단기_감사": hourly_audits,
            "일간_compare_요약": compare_verdict,
            "일간_감사_기존": daily_audit_before.to_dict("records"),
            "일간_감사_제외": daily_audit_after.to_dict("records"),
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n=== 초단기·단기 표본 커버리지(동일행 정렬 결과) ===")
    print(hourly_coverage.to_string(index=False))
    print("\n=== 초단기·단기 성능비교(동일행 기준, 포함 vs 제외) ===")
    print(hourly_perf.to_string(index=False))
    print("\n=== 일간 성능비교(포함 vs 제외) ===")
    print(daily_perf.to_string(index=False))
    print("\n=== 일간 폴드별 비교(daily_audit.compare 원본) ===")
    print(compare_folds.to_string(index=False))
    print("\n=== 일간 compare() 요약 판정(주의: 개선목적 임계값 — retrain_exclude_")
    print("mean_communication_ok_v1_2026-08-26.py의 동일 주석 참고) ===")
    print(json.dumps(compare_verdict, ensure_ascii=False, indent=2, default=str))
    print(f"\n저장 완료: {OUT}")
    print(
        "\n★주의★ 이 결과는 채택 여부를 자동 결정하지 않는다 — "
        "nMAE_변화/일간 비교표를 Claude에게 그대로 전달해 판단받은 뒤 "
        "실제 운영모델(운영모델_v2) 재생성 여부를 정할 것."
    )


if __name__ == "__main__":
    main()
