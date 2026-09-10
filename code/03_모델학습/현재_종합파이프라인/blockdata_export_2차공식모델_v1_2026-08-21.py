# -*- coding: utf-8 -*-
"""⑥ Blockdata 출력 규격화 — 2차 공식모델(구조개선 5건 반영) 재적용.

## 왜 필요한가
`blockdata_export_v1_2026-08-21.py`(이하 v1)는 **1차 결과**(⑤ 최종통합재검증
판정표) 기준으로 만들었다. 그 뒤 성능개선 ①~④와 엄격 E2E 리플레이가
진행되며 채택 구성이 바뀌었다(초단기 +1h·+2h → raw+폴드내부구조선택,
단기 전 수평 → 기본+폴드내부구조선택, 청천지수+튜닝 계열 철회 등 —
AGENTS.md "2차 공식 결과 확정" 절 참고). **v1의 산출물은 이제 stale하다.**

한편 `end_to_end_replay_backtest_v1_2026-08-21.py --round2`가 자체적으로
`Blockdata형식_ac_power_리플레이.csv`/`_energy_리플레이.csv`를 만들긴
했지만, 그건 **모델 성능·누출 감사**(학습/시험 시점분리, 재적재 일치)만
확인했을 뿐 v1이 만든 **Blockdata 필드 규격 검증**(자릿수 2자리, 물리범위,
시각정합성, daily_energy 두 경로 비음수·단조성 등 22건)은 거치지 않았고,
값도 소수점 6자리 그대로다. 이 스크립트는 **v1의 검증된 로직(규격 정의·
경로B 누적 계산·total_energy 구조·payload 형식·22건 검증)을 그대로
재사용**하면서, 입력만 2차 공식모델의 행단위 산출물로 교체한다
(재사용, 중복구현 안 함).

## 입력
`outputs/E2E_리플레이백테스트_2차결과_v1_2026-08-21/`
(`행단위_ac_power_예측정답.csv`, `행단위_daily_final_예측정답.csv`).
이미 티어·수평별로 **단일 공식구성만** 들어있어(판정표 조회 불필요),
v1의 `resolve_adopted()`는 이 스크립트에서 쓰지 않는다.

## 스모크테스트 방식이 v1과 다르다
v1은 스모크테스트용으로 모델을 **새로 재학습**했다. 이번엔 2차 E2E
리플레이가 이미 저장해둔 **실제 배포후보 joblib**(`fold_models/`, 40개
중 가장 최근 폴드인 5_초여름의 초단기 +1h raw+구조선택)을 그대로
재적재해 추론한다 — 재학습 없이 "저장→적재→추론→규격출력"을 실제
공식 산출물로 검증하는 게 더 강한 증거이기 때문이다.

## 출력 (`outputs/Blockdata_규격화_2차공식모델_v1_2026-08-21/`)
v1과 동일한 산출물 세트(규격정의·ac_power·daily_energy 두 경로·
total_energy·payload 예시·스모크테스트 결과·검증표·요약) + `요약.txt`
안에 1차 대비 무엇이 바뀌었는지 명시.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC2 = ROOT / "outputs" / "E2E_리플레이백테스트_2차결과_v1_2026-08-21"
OUT2 = ROOT / "outputs" / "Blockdata_규격화_2차공식모델_v1_2026-08-21"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# v1의 검증된 로직(규격정의·경로B 누적·total_energy·payload·22건 검증)을
# 그대로 재사용한다 — 새로 베끼지 않는다.
bd1 = _load("bd_v1", "blockdata_export_v1_2026-08-21.py")

PLANT_ID = bd1.PLANT_ID
PLANT_NAME = bd1.PLANT_NAME


# ── ac_power / daily_final 변환(2차 행단위 산출물 → v1과 같은 컬럼 구조) ──
def build_ac_power_v2(ac_raw: pd.DataFrame) -> pd.DataFrame:
    a = ac_raw.copy()
    a["발행시각"] = pd.to_datetime(a["발행시각"])
    a["대상시각"] = pd.to_datetime(a["대상시각"])
    return pd.DataFrame({
        "plant_id": a["plant_id"],
        "티어": a["티어"],
        "수평_h": a["수평_h"],
        "채택모델": a["공식구성"],
        "발행시각": a["발행시각"],
        "대상시각": a["대상시각"],
        "집계기준": a["티어"].map(bd1.BUCKET),
        "예측_ac_power_kw": a["예측_kW"].round(2),
        "실측_ac_power_kw": a["실제_kW"].round(2),
        "폴드": a["폴드"],
    })


def build_daily_final_v2(daily_raw: pd.DataFrame) -> pd.DataFrame:
    d = daily_raw.copy()
    d["날짜"] = pd.to_datetime(d["대상일"])
    return pd.DataFrame({
        "plant_id": d["plant_id"],
        "날짜": d["날짜"],
        "경로": "A_일간모델(직접모델, 2차 공식구성)",
        "채택모델": d["공식구성"],
        "예측_일간최종총량_kwh": d["예측_kWh"].round(2),
        "실측_일간최종총량_kwh": d["실제_kWh"].round(2),
        "폴드": d["폴드"],
    })


# ── 스모크테스트: 재학습 없이 2차가 저장한 실제 폴드모델을 재적재 ──
def run_smoke_test_v2(capacity_kw: float) -> dict:
    fold_dir = SRC2 / "fold_models"
    model_path = fold_dir / "초단기_h1_5_초여름_raw+폴드내부구조선택.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"2차 폴드모델을 찾을 수 없음: {model_path}")
    bundle = joblib.load(model_path)
    if "raw" not in bundle["variant"]:
        raise RuntimeError(f"예상과 다른 변형: {bundle['variant']} (raw 전제 위반 — 역변환 로직 없음)")

    ultra_mod = bd1._load("ultra_smoke2", "train_ultra_short_official_v1_2026-08-21.py")
    harness_mod = bd1._load("harness_smoke2", "backtest_harness_v1_2026-08-20밤.py")
    quarter = ultra_mod.load_15min_base()
    hourly_df = pd.read_csv(harness_mod.DATASETS["v3_고정tm"]["path"],
                             parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    frame = ultra_mod.build_ultra_short_frame(quarter, hourly_df, bundle["horizon_h"])

    features = bundle["features"]
    test_start = pd.Timestamp(bundle["test_start"])
    candidates = frame.loc[frame.index >= test_start]
    valid = candidates.dropna(subset=features)
    if not len(valid):
        raise RuntimeError(f"저장된 폴드({model_path.name})의 시험구간에서 결측없는 샘플을 못 찾음")
    sample_time = valid.index[-1]  # 그 폴드 시험구간의 가장 최근 시각(=이 배포후보가 마지막으로 본 시각과 가장 가까움)
    x = frame.loc[[sample_time], features]

    pred_kw = float(np.clip(bundle["model"].predict(x)[0], 0, capacity_kw))
    issued_at = sample_time
    target_at = issued_at + pd.Timedelta(hours=bundle["horizon_h"])

    payload = {
        "plant_id": PLANT_ID, "plant_name": PLANT_NAME, "plant_capacity": capacity_kw,
        "forecast": {
            "issued_at": issued_at.strftime("%Y-%m-%d %H:%M:%S"),
            "target_at": target_at.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": bundle["tier"], "horizon_h": bundle["horizon_h"],
            "aggregation": bd1.BUCKET[bundle["tier"]],
            "model": f"{bundle['variant']}(2차 공식모델, 실제 저장분 재적재)",
            "plant_ac_power_kw": round(pred_kw, 2),
        },
    }
    return {
        "통과": True,
        "model_path": str(model_path),
        "재사용_방식": "2차 E2E 리플레이가 저장한 실제 배포후보 joblib을 그대로 재적재(재학습 안 함)",
        "폴드": model_path.stem.split("_", 2)[-1].rsplit("_", 1)[0],
        "구조": bundle.get("structure_name"),
        "feature_count": len(features),
        "sample_issued_at": issued_at.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_target_at": target_at.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_ac_power_kw": round(pred_kw, 2),
        "payload": payload,
    }


def main() -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])

    summary2 = json.loads((SRC2 / "요약.json").read_text(encoding="utf-8"))
    print("=== 2차 공식구성(요약.json에서 확인) ===")
    for tier, v in summary2["공식구성"].items():
        print(f"  {tier}: {v}")

    ac_raw = pd.read_csv(SRC2 / "행단위_ac_power_예측정답.csv", encoding="utf-8-sig")
    daily_raw = pd.read_csv(SRC2 / "행단위_daily_final_예측정답.csv", encoding="utf-8-sig")

    spec = bd1.build_spec(capacity_kw)
    (OUT2 / "규격정의.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ac_power 변환(2차) ===")
    ac = build_ac_power_v2(ac_raw)
    ac.to_csv(OUT2 / "예측_ac_power.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(ac):,}행")

    print("\n=== daily_energy 변환 — 경로A(일간총량, 2차) ===")
    daily_final = build_daily_final_v2(daily_raw)
    daily_final.to_csv(OUT2 / "예측_daily_energy_일간총량.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(daily_final):,}행")

    print("\n=== daily_energy 변환 — 경로B(대상시각별 누적시계열, v1 로직 재사용) ===")
    daily_cum = bd1.build_daily_cumulative(ac)
    daily_cum.to_csv(OUT2 / "예측_daily_energy_누적시계열.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(daily_cum):,}행")

    print("\n=== total_energy 구조(v1 로직 재사용, 앵커 null) ===")
    total = bd1.build_total_energy(daily_final)
    total.to_csv(OUT2 / "예측_total_energy.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(total):,}행")

    payload = bd1.build_payload_sample(ac, daily_final, capacity_kw)
    (OUT2 / "blockdata_payload_예시.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 엔드투엔드 스모크테스트(2차 실제 저장 폴드모델 재적재) ===")
    try:
        smoke = run_smoke_test_v2(capacity_kw)
        print(f"  통과 — {smoke['재사용_방식']}")
        print(f"  모델: {smoke['model_path']}")
        print(f"  샘플 발행시각 {smoke['sample_issued_at']} → 대상시각 {smoke['sample_target_at']}, "
              f"예측 {smoke['predicted_ac_power_kw']}kW (구조: {smoke['구조']}, 특성 {smoke['feature_count']}개)")
    except Exception as e:  # noqa: BLE001
        smoke = {"통과": False, "오류": f"{type(e).__name__}: {e}"}
        print(f"  ★실패★ {smoke['오류']}")
    (OUT2 / "스모크테스트_결과.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== 자동 규격 점검(v1의 verify() 재사용) ===")
    harness_mod = bd1._load("harness_verify2", "backtest_harness_v1_2026-08-20밤.py")
    checks = bd1.verify(ac, daily_final, daily_cum, capacity_kw, harness_mod)
    checks.to_csv(OUT2 / "검증_규격점검.csv", index=False, encoding="utf-8-sig")
    print(checks.to_string(index=False))

    fails = checks[checks["결과"] != "통과"]
    fail_count = len(fails) + (0 if smoke.get("통과") else 1)

    lines = [
        "⑥ Blockdata 출력 규격화 — 2차 공식모델 재적용 요약",
        f"발전소: {PLANT_NAME}(plant_id={PLANT_ID}), 용량 {capacity_kw}kW",
        "",
        "★1차(v1) 대비 바뀐 것★",
        "- 초단기 +1h·+2h: raw/청천지수+튜닝 → raw+폴드내부구조선택(리프규제 등)",
        "- 단기 +1h·+24h·+48h: 기본/튜닝 혼재 → 전부 기본+폴드내부구조선택",
        "- 초단기 +3h·+4h(청천지수)·일간(전체특성+튜닝파라미터)은 변경 없음",
        "- 스모크테스트: 재학습 대신 2차가 저장한 실제 배포후보 joblib을 재적재",
        "",
        f"ac_power 행수: {len(ac):,} / daily_energy 경로A: {len(daily_final):,}행 "
        f"/ 경로B: {len(daily_cum):,}행",
        f"규격점검 {len(checks)}건 중 실패 {len(fails)}건, "
        f"스모크테스트 {'통과' if smoke.get('통과') else '실패'}",
        f"총 실패 {fail_count}건",
    ]
    if len(fails):
        lines += ["", "★규격점검 실패 항목★"] + [f"- {r['항목']} / {r['대상']}: {r['상세']}"
                                              for _, r in fails.iterrows()]
    if not smoke.get("통과"):
        lines += ["", "★스모크테스트 실패★", f"- {smoke.get('오류')}"]
    (OUT2 / "요약.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n저장 완료: {OUT2}")
    if fail_count:
        print(f"\n★규격점검/스모크테스트 총 {fail_count}건 실패 — 요약.txt 확인★")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
