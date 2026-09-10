# -*- coding: utf-8 -*-
"""⑤v2→v3 패치 — 일간 특성군 재판정 채택안(B그룹만 제외, 53개 특성) 반영.

## ★08-24 수정: 최초 순방향(A→B→C→D→E 누적) 결과는 실제 재학습에서
## 기각됨 — LOGO(각 그룹을 실제 배포 전체특성에서 하나씩 빼보는) 재검증으로
## 대체★
`daily_feature_group_ablation_v5_공식B_v1_2026-08-24.py`(순방향 누적,
바닥(A) 기준에서 B→C→D→E를 하나씩 추가)는 **A+C(46특성) 채택**을
보고했다(A 대비 MAE 5.71%/RMSE 4.41% 개선). 그런데 이걸 실제로
재학습해서 **실제 배포판(전체특성 58개, MAE 110.41kWh)과 직접 비교**
하니 개선폭이 0.24%로 쪼그라들고 **2_가을_결함종료후 폴드가 10.71%
악화**해 동결규칙(5%p)을 위반했다 — **⑥ round2에서 이미 한 번 겪은
것과 정확히 같은 버그**: 순차 누적판정이 "바닥부터 하나씩 얹는" 기준을
쓰다 보니, 실제 배포판이 이미 갖고 있는 특성(B/D/E)들과의 상호작용을
못 보고 A 단독 대비 개선율만 본 것.

**수정**: 바닥이 아니라 **실제 배포 전체특성(58개)에서 한 그룹씩 빼보는
LOGO(leave-one-group-out)**로 재검증했다(`outputs/일간_특성군_재판정_
v1_2026-08-24/LOGO_채택상태.json`). 결과:
- **B(보조기상: 기압·지면온도·풍향·일조시간) 제외 → 채택**(MAE 2.43%/
  RMSE 1.97% 개선, 전 폴드 개선 또는 무변화, 5%p 위반 0건)
- C(강수·적설) 제외 → 기각(제외하면 오히려 악화, 가을 폴드 14.34%↓)
- D(설비건강) 제외 → 기각(근소 개선이나 RMSE 1%미만이라 문턱 미달)
- E(요일) 제외 → 기각(가을 폴드 10.63% 악화 — "약함" 등급이었지만
  실제로는 제거하면 손해)

**최종 채택: 전체 58개 특성에서 B그룹(5개)만 빼고 53개.**

## 범위(초단기·단기·일간 중 일간만)
초단기·단기는 이번 판정과 무관하므로 재학습하지 않고 v2 산출물을 그대로
재사용한다(재구현 금지 원칙).

## 사전동결 규칙 6번(계층조정)은 범위 밖
AGENTS.md "9. 일간 계층조정의 붕괴"(가중치 직접모델100%/시간모델0%로
붕괴, 데이터 축적 전까지 확정 보류)가 이미 별도로 추적 중인 미해결
이슈라 이 패치와 섞지 않는다 — 이 패치는 "어떤 특성을 쓸지"만 확정한다.

## 산출물 (`outputs/E2E_v5_공식B_v3_일간특성군_2026-08-24/`)
v2와 동일한 파일 세트, 일간만 53특성(B그룹 제외)으로 교체.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC_V2 = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24"
ABLATION_STATE = ROOT / "outputs" / "일간_특성군_재판정_v1_2026-08-24" / "LOGO_채택상태.json"
OUT = ROOT / "outputs" / "E2E_v5_공식B_v3_일간특성군_2026-08-24"
MODEL_DIR = OUT / "fold_models"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("v3_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
improvement = e2e.improvement
tuning = _load("v3_tuning", "hyperparameter_tuning_v1_2026-08-21.py")
OFFICIAL_WINDOWS = e2e.OFFICIAL_WINDOWS
ACTUAL = "실제_일간발전량_kWh"


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


def run_daily_v3(capacity_kw: float, seed: int, features: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    data, _all_features = e2e.build_daily_dataset_v5(capacity_kw)
    daily_capacity = capacity_kw * 24
    rows, audits = [], []
    for fold, s, e in OFFICIAL_WINDOWS:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train = data[data.index < start]
        test = data[(data.index >= start) & (data.index < end)]
        if len(train) < 60 or len(test) < 10:
            print(f"  [일간 {fold}] 표본부족(train={len(train)}, test={len(test)}) — 건너뜀")
            continue
        medians = train[features].median(numeric_only=True)
        x_train, x_test = train[features].fillna(medians), test[features].fillna(medians)

        params, _ = tuning.tune_fold("LightGBM", train.fillna(medians), features, ACTUAL, daily_capacity, seed)
        params = _cast(params, ["n_estimators", "num_leaves", "max_depth", "min_child_samples"])
        model = LGBMRegressor(objective="regression_l1", random_state=seed, verbosity=-1, n_jobs=4, **(params or {}))
        model.fit(x_train, train[ACTUAL])

        variant = "전체특성-보조기상제외(53특성, v3)"
        artifact = MODEL_DIR / e2e._safe(f"일간_{fold}_{variant}.joblib")
        raw, diff = e2e._roundtrip(model, {
            "tier": "일간", "horizon": "D+1", "variant": variant, "features": features,
            "medians": medians.to_dict(), "params": params, "target_transform": "daily_kWh",
            "정책": "B_구간제외", "용량프로필": "inverter_registered_sum_219",
            "특성군_근거": "일간모델_특성_학술근거_감사_v1_2026-08-24.md, LOGO 재검증: B(보조기상)만 제외",
            "train_end": str(train.index.max()), "test_start": str(test.index.min()),
        }, x_test, artifact)
        pred = np.clip(raw, 0, daily_capacity)
        actual = test[ACTUAL].to_numpy()
        issued_at = test.index - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
        for issued, target_date, y, p in zip(issued_at, test.index, actual, pred):
            rows.append({"plant_id": e2e.PLANT_ID, "plant_name": e2e.PLANT_NAME, "티어": "일간",
                        "수평_h": "D+1", "공식구성": variant, "폴드": fold,
                        "발행시각": issued, "대상일": target_date, "실제_kWh": y, "예측_kWh": p,
                        "모델파일": str(artifact)})
        audits.append({"티어": "일간", "수평_h": "D+1", "폴드": fold,
                       "학습최종시각": train.index.max(), "시험최초시각": test.index.min(),
                       "학습_시험_분리통과": bool(train.index.max() < test.index.min()),
                       "추정타깃포함": False, "정책": "B_구간제외",
                       "모델재적재_동일": bool(diff <= 1e-12), "재적재차이": diff,
                       "특성수": len(features), "선택구조": "튜닝(B제외,53특성)",
                       "학습행수": len(train), "시험행수": len(test)})
        print(f"[일간 {fold}] {variant} 완료 (학습{len(train):,}/시험{len(test):,})")
    return pd.DataFrame(rows), audits


def make_performance(ac: pd.DataFrame, daily: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    rows = []
    for (tier, h), g in ac.groupby(["티어", "수평_h"]):
        e = g["실제_kW"] - g["예측_kW"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        rows.append({"티어": tier, "수평_h": h, "n": len(g), "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
                    "nMAE_pct": round(mae / capacity_kw * 100, 3), "nRMSE_pct": round(rmse / capacity_kw * 100, 3)})
    if len(daily):
        e = daily["실제_kWh"] - daily["예측_kWh"]
        mae, rmse = float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))
        wape = float(e.abs().sum() / daily["실제_kWh"].abs().sum() * 100)
        rows.append({"티어": "일간", "수평_h": "D+1", "n": len(daily),
                    "MAE_kWh": round(mae, 2), "RMSE_kWh": round(rmse, 2), "WAPE_pct": round(wape, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    if not SRC_V2.exists():
        raise FileNotFoundError(f"v2 산출물이 없다: {SRC_V2}")
    features = json.loads(ABLATION_STATE.read_text(encoding="utf-8"))["최종특성목록"]
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"v3 패치 범위: 일간만 B그룹 제외({len(features)}개 특성)로 재학습. 초단기·단기는 v2 그대로 재사용.\n")

    ac_v2 = pd.read_csv(SRC_V2 / "행단위_ac_power_예측정답.csv", encoding="utf-8-sig",
                        parse_dates=["발행시각", "대상시각"])
    daily_v2 = pd.read_csv(SRC_V2 / "행단위_daily_예측정답.csv", encoding="utf-8-sig",
                           parse_dates=["발행시각", "대상일"])
    audits_v2 = pd.read_csv(SRC_V2 / "누출_및_재적재_감사.csv", encoding="utf-8-sig")

    print("=== 일간 재학습(B그룹 제외, 53특성) ===")
    daily_v3, aud_daily_v3 = run_daily_v3(capacity_kw, seed, features)

    ac = ac_v2.copy()  # 초단기·단기 변경 없음
    aud_unchanged_mask = ~(audits_v2["티어"] == "일간")
    audits = pd.concat([audits_v2[aud_unchanged_mask], pd.DataFrame(aud_daily_v3)], ignore_index=True)

    referenced = set(Path(p).name for p in
                     pd.concat([ac["모델파일"]]).dropna().unique())
    for src in (SRC_V2 / "fold_models").glob("*.joblib"):
        if src.name in referenced:
            shutil.copy2(src, MODEL_DIR / src.name)

    perf = make_performance(ac, daily_v3, capacity_kw)
    ac.to_csv(OUT / "행단위_ac_power_예측정답.csv", index=False, encoding="utf-8-sig")
    daily_v3.to_csv(OUT / "행단위_daily_예측정답.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(OUT / "성능_전체.csv", index=False, encoding="utf-8-sig")
    audits.to_csv(OUT / "누출_및_재적재_감사.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 성능(v3·일간 B그룹제외반영) ===")
    print(perf.to_string(index=False))
    print("\n=== 감사 요약 ===")
    print(f"조합 수: {len(audits)} / 학습시험분리 전부통과: {bool(audits['학습_시험_분리통과'].all())} "
          f"/ 재적재동일 전부통과: {bool(audits['모델재적재_동일'].all())} "
          f"/ 추정타깃포함: {int(audits['추정타깃포함'].sum())}건 "
          f"/ fold_models 파일수: {len(list(MODEL_DIR.glob('*.joblib')))}")

    # v2 대비 일간 폴드별(계절별) 악화율 — 동결규칙(5%p) 재확인
    daily_v2_e = daily_v2.assign(e=lambda d: (d["실제_kWh"] - d["예측_kWh"]).abs())
    fold_v2 = daily_v2_e.groupby("폴드")["e"].mean().rename("MAE_kWh_v2")
    daily_v3_e = daily_v3.assign(e=lambda d: (d["실제_kWh"] - d["예측_kWh"]).abs())
    fold_v3 = daily_v3_e.groupby("폴드")["e"].mean().rename("MAE_kWh_v3")
    cmp_df = pd.concat([fold_v2, fold_v3], axis=1)
    cmp_df["MAE_변화율_pct"] = (cmp_df["MAE_kWh_v3"] - cmp_df["MAE_kWh_v2"]) / cmp_df["MAE_kWh_v2"] * 100
    cmp_df["동결규칙_5pt_위반"] = cmp_df["MAE_변화율_pct"] >= 5
    cmp_df.to_csv(OUT / "v2대비_폴드별_비교.csv", encoding="utf-8-sig")
    print("\n=== v2(전체특성58) vs v3(B제외 53특성) — 폴드별(계절별) ===")
    print(cmp_df.to_string())
    violations = int(cmp_df["동결규칙_5pt_위반"].sum())
    print(f"\n계절 악화 동결규칙 위반: {violations}건")

    daily_perf_v2 = daily_v2_e["e"].mean()
    daily_perf_v3 = daily_v3_e["e"].mean()
    print(f"\n전체 MAE: v2(58특성) {daily_perf_v2:.2f}kWh → v3(53특성) {daily_perf_v3:.2f}kWh "
          f"({(daily_perf_v2 - daily_perf_v3) / daily_perf_v2 * 100:+.2f}% 개선)")

    summary = {
        "실행유형": "⑤v2→v3 패치: 일간 특성군 LOGO 재판정 채택안(B그룹 제외, 53특성) 반영",
        "패치범위": ["일간_D+1"],
        "변경없음": "초단기·단기 전체(v2와 완전 동일 — 재학습 안 하고 v2 산출물 재사용)",
        "일간_특성수": {"이전(전체특성)": 58, "신규(B제외)": len(features)},
        "일간_MAE_kWh": {"v2": round(float(daily_perf_v2), 2), "v3": round(float(daily_perf_v3), 2)},
        "폴드": [w[0] for w in OFFICIAL_WINDOWS],
        "조합수": int(len(audits)),
        "학습시험분리_전부통과": bool(audits["학습_시험_분리통과"].all()),
        "재적재_전부통과": bool(audits["모델재적재_동일"].all()),
        "계절악화_동결규칙_위반건수": violations,
        "라이브_API_호출": False,
        "근거": "outputs/일간_특성군_재판정_v1_2026-08-24/채택상태.json",
        "범위밖": "사전동결 규칙6(계층조정)은 AGENTS.md '9. 일간 계층조정의 붕괴'로 별도 추적, 이 패치와 무관",
        "KPX_영향": "없음 — KPX는 단기(day-ahead) 모델만 쓰고 일간 총량 모델은 안 씀",
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
