# -*- coding: utf-8 -*-
"""일간 모델 특성군 재판정 — v5·공식B·219kW·정정 공식5폴드 기준(A→B→C→D 누적).

## 배경
`04_평가검증/일간모델_특성_학술근거_감사_v1_2026-08-24.md`(Codex 감사)가
현재 "전체특성" 일간모델의 각 특성군에 학술근거 등급을 매기고 비교군을
동결해뒀지만, **실제 v5 기준 ablation 비교는 아직 한 번도 실행되지
않았다.** 이 스크립트가 그 감사문서의 "v5 재학습 사전동결 규칙" 1~5번을
그대로 실행한다(6번 계층조정은 별도 이슈로 분리 — 아래 "범위 밖" 참고).

## 특성군 정의(감사문서 표 그대로 매핑)
- **A(핵심)**: 목표일 NWP GHI(DSWRF/DSWRFLX/DIFSWRF)·운량(TCDC/LCDC/MCDC/
  HCDC/SKY)·기온·습도·풍속, 과거발전량(7일전/2일전 lag+이동평균/표준편차),
  연주기 sin/cos·월, 2일전 관측(기온·습도·풍속·전운량·일사량), 해발고도·
  설비용량(상수, 무해). = 감사문서 "강함/강함~중간" 등급 전부.
- **B(+보조기상)**: 2일전 관측 기압(현지·해면)·지면온도·풍향·일조시간.
  = "조건부" 등급.
- **C(+강수·적설)**: 목표일예보 강수확률(POP), 2일전 관측 강수량·적설.
  = "조건부" 등급.
- **D(+설비건강)**: 2일전 평균 입력전력·입력전압·주파수·역률·통신상태·
  가용인버터수. = "간접·약함" 등급. **D = 지금의 "전체특성"과 동일.**
- **E(+요일)**: 감사문서가 "기본 제외 후보(증분효과 명확할 때만 유지)"로
  판정한 항목이라 A~D엔 안 넣고, D 다음에 별도 4단계로 증분효과만
  확인한다(요일=근무일정 대리변수 위험, 채택 안 되면 최종구성에서 뺀다).

## 방법(감사문서 사전동결 규칙 1~5 그대로)
1. A를 기준모델로 둔다.
2. B→C→D를 그룹 단위로 누적 추가(라운드2와 동일하게, 채택된 것만
   다음 단계 기준으로 이어받음 — 재구현 안 하고 `model_improvement_
   round2_v1_2026-08-21.verdict` 그대로 재사용).
3. 중앙값(fillna)·튜닝은 폴드 학습구간에서만(`hyperparameter_tuning_v1.
   tune_fold` 재사용).
4. 동일 시험일(같은 `data` 행, 특성만 다름)에서 MAE·RMSE 비교.
5. 개선율 1% 미만이면 단순모델 우선, 어느 폴드든 5% 이상 악화하면 보류.

## 범위 밖(별도 이슈)
사전동결 규칙 6번(시간모델합계와 계층조정 후 최종판정)은 여기 포함하지
않는다 — AGENTS.md "9. 일간 계층조정의 붕괴"가 이미 별도로 추적 중인
이슈(가중치가 직접모델100%/시간모델0%로 붕괴, 데이터 더 쌓이기 전까지
확정 보류)라 이 ablation과 결합하면 두 미해결 이슈가 뒤섞인다. 이
스크립트는 "어떤 특성군을 쓸지"만 확정한다.

## 산출물 (`outputs/일간_특성군_재판정_v1_2026-08-24/`)
- `판정_전체.csv`, `{단계}_{개선안}/판정표.csv`, `계절안정성.csv`,
  `동일행_예측정답.csv`, `채택상태.json`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "일간_특성군_재판정_v1_2026-08-24"
STATE_PATH = OUT / "채택상태.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("dfa_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
improvement = _load("dfa_round2", "model_improvement_round2_v1_2026-08-21.py")
tuning = _load("dfa_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
fhv = _load("dfa_fhv", "fold_hour_prevalidation_v1_2026-08-24.py")
OFFICIAL_WINDOWS = fhv.OFFICIAL_B_WINDOWS
ACTUAL = "실제_일간발전량_kWh"

STEP_NAMES = {1: "B_보조기상", 2: "C_강수적설", 3: "D_설비건강", 4: "E_요일"}

B_SUFFIXES = {"기상청관측_현지기압_hPa", "기상청관측_해면기압_hPa", "기상청관측_지면온도_C",
             "기상청관측_풍향_deg", "기상청관측_일조시간_hr"}
C_SUFFIXES = {"기상청관측_강수량_mm", "기상청관측_적설_cm"}
D_SUFFIXES = {"plant_input_power_kw", "mean_input_voltage_v", "mean_frequency_hz",
             "mean_power_factor", "mean_communication_ok", "inverters_available"}
A_SUFFIXES = {"기상청관측_기온_C", "기상청관측_상대습도_pct", "기상청관측_전운량_pct",
             "기상청관측_일사량_W_m2", "기상청관측_풍속_m_s"}


def classify(col: str) -> str:
    if col in ("해발고도_m", "설비용량_kW"):
        return "A"
    if col in ("7일전_일간발전량_kWh", "7일전_결측여부", "2일전_일간발전량_kWh", "2일전_결측여부",
              "2일전기준_7일이동평균_kWh", "2일전기준_30일이동평균_kWh", "2일전기준_30일표준편차_kWh",
              "목표일_연주기_sin", "목표일_연주기_cos", "목표일_월", "목표일_DIFSWRF_유효개수"):
        return "A"
    if col.startswith("목표일예보_POP_"):
        return "C"
    if col.startswith("목표일예보_"):
        return "A"
    if col.startswith("2일전평균_"):
        suf = col[len("2일전평균_"):]
        if suf in B_SUFFIXES:
            return "B"
        if suf in C_SUFFIXES:
            return "C"
        if suf in D_SUFFIXES:
            return "D"
        if suf in A_SUFFIXES:
            return "A"
        raise ValueError(f"미분류 2일전평균_ 특성: {col}")
    if col == "목표일_요일":
        return "E"
    raise ValueError(f"미분류 특성(감사문서 매핑에 없음): {col}")


def _cast(params, integer_names):
    if params is None:
        return None
    out = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        if k in integer_names and v is not None:
            v = int(v)
        out[k] = v
    return out


def predict_daily(data: pd.DataFrame, features: list[str], capacity_kw: float, seed: int) -> pd.DataFrame:
    daily_capacity = capacity_kw * 24
    rows = []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index < end)]
        if len(train) < 60 or len(test) < 10:
            print(f"  [일간 {fold}] 표본부족(train={len(train)}, test={len(test)}) — 건너뜀")
            continue
        med = train[features].median(numeric_only=True)
        x_train, x_test = train[features].fillna(med), test[features].fillna(med)
        params, _ = tuning.tune_fold("LightGBM", train.fillna(med), features, ACTUAL, daily_capacity, seed)
        params = _cast(params, ["n_estimators", "num_leaves", "max_depth", "min_child_samples"])
        model = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **(params or {}))
        model.fit(x_train, train[ACTUAL])
        pred = np.clip(model.predict(x_test), 0, daily_capacity)
        for ts, y, p in zip(test.index, test[ACTUAL], pred):
            rows.append({"날짜": ts, "실제": y, "예측": p, "폴드": fold})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"용량프로필: {config['site'].get('capacity_profile')} ({capacity_kw}kW) / "
          f"폴드: 공식 5폴드(가을 11-17 정정) / 일간 특성군 재판정(A→B→C→D)\n")

    data, features_all = e2e.build_daily_dataset_v5(capacity_kw)
    groups: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for col in features_all:
        groups[classify(col)].append(col)
    for g in "ABCDE":
        print(f"  그룹 {g}: {len(groups[g])}개 특성")
    assert sorted(sum(groups.values(), [])) == sorted(features_all), "분류 누락/중복 특성 있음"

    current = list(groups["A"])
    print(f"\n[기준모델 A] {len(current)}개 특성으로 시작\n")
    all_rows, verdict_rows, fold_rows = [], [], []

    for step in (1, 2, 3, 4):
        add_group = STEP_NAMES[step][0]  # "B"|"C"|"D"|"E"
        candidate = current + groups[add_group]
        print(f"=== 단계{step}: {STEP_NAMES[step]} — 현재 {len(current)}개 → 후보 {len(candidate)}개 ===")
        print("[현재구성 예측]")
        rows_cur = predict_daily(data, current, capacity_kw, seed)
        print("[후보구성 예측]")
        rows_cand = predict_daily(data, candidate, capacity_kw, seed)

        merged = rows_cur.merge(rows_cand, on=["날짜", "폴드"], suffixes=("_현재", "_후보"))
        merged = merged.rename(columns={"실제_현재": "실제", "예측_현재": "현행예측", "예측_후보": "후보예측"})
        merged = merged.drop(columns=["실제_후보"])
        if merged.empty:
            raise RuntimeError(f"시험행 없음: 단계{step}")

        result, folds = improvement.verdict(merged)
        result.update({"단계": step, "개선안": STEP_NAMES[step], "현재특성수": len(current),
                       "후보특성수": len(candidate), "시험행수": len(merged)})
        verdict_rows.append(result)
        for fr in folds:
            fr.update({"단계": step, "개선안": STEP_NAMES[step]})
            fold_rows.append(fr)
        merged["단계"], merged["개선안"] = step, STEP_NAMES[step]
        all_rows.append(merged)

        print(f"  -> {result['판정사유']} / MAE {result['MAE개선율_pct']:.2f}% / "
              f"RMSE {result['RMSE개선율_pct']:.2f}% / 최대계절악화 {result['최대계절_MAE악화율_pct']:.2f}%")
        if result["채택"]:
            current = candidate
            print(f"  → {add_group} 채택, 현재 구성 {len(current)}개로 갱신")
        else:
            print(f"  → {add_group} 미채택, 현재 구성 {len(current)}개 유지")

        step_dir = OUT / f"{step}_{STEP_NAMES[step]}"
        step_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(step_dir / "동일행_예측정답.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([result]).to_csv(step_dir / "판정표.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(folds).to_csv(step_dir / "계절안정성.csv", index=False, encoding="utf-8-sig")

    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_csv(OUT / "판정_전체.csv", index=False, encoding="utf-8-sig")

    final_groups = [g for g in "ABCDE" if g == "A" or any(c in current for c in groups[g])]
    payload = {
        "완료": True,
        "최종채택특성군": final_groups,
        "최종특성수": len(current),
        "최종특성목록": current,
        "전체특성수_현재공식(D)": len(features_all),
        "동결규칙": "동일 시험일, MAE·RMSE 모두 1% 이상 개선, 계절 MAE 5% 악화 없음",
        "범위밖": "사전동결 규칙6(시간모델합계 계층조정)은 별도 이슈(AGENTS.md '9. 일간 계층조정의 붕괴')로 분리",
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n=== 최종 판정표 ===")
    print(verdict_df.to_string(index=False))
    print(f"\n최종 채택 특성군: {final_groups} ({len(current)}개 특성, 현재 공식 D={len(features_all)}개 대비)")
    print(f"저장 완료: {OUT}")


if __name__ == "__main__":
    main()
