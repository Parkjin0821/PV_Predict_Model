# -*- coding: utf-8 -*-
"""⑦ Blockdata 출력 규격화 — 3차 공식모델(v5·공식B·219kW) 재적용.

## 왜 필요한가
`blockdata_export_2차공식모델_v1_2026-08-21.py`(이하 v2)는 2차 공식모델
(v3 데이터·240kW·구경계 5폴드) 산출물 기준으로 만들었다. 그 뒤 v5 발견
(시간단위 데이터에 인버터5 결함 미반영) → 전면 재작업 → ⑤ v5·공식
정책B·219kW·정정 공식5폴드로 전 모델 재학습이 끝났다(AGENTS.md
"⑤ v5·공식B·219kW 전 모델 재학습 완료" 절). **v2의 산출물은 이제
stale하다.** 이 스크립트는 v1/v2가 이미 검증한 로직(규격정의·경로B
누적계산·total_energy 구조·payload 형식·22건 검증)을 **그대로
재사용**하면서 입력만 3차(⑤) E2E 재학습의 행단위 산출물로 교체한다.

## 입력
`outputs/E2E_v5_공식B_v1_2026-08-24/`
(`행단위_ac_power_예측정답.csv`, `행단위_daily_예측정답.csv`) — 파일명이
2차(`_daily_final_...`)와 다르므로 그대로 읽는다. 이미 티어·수평별로
단일 공식구성만 들어있다(①~④ 후보 재판정은 ⑥에서 별도 — 여기서 구성을
새로 고르지 않음).

## 스모크테스트가 v2와 다른 이유
v2 스모크테스트는 순수 v3 15분자료(`ultra_mod.load_15min_base()`)로
프레임을 만들어 재적재했다. **3차 모델은 v5 데이터(`DIFSWRF_bsrn정제_
결측여부` 플래그 특성 포함)로 학습됐으므로 그 로더를 그대로 써야
특성이 일치한다** — 그래서 v3 로더 대신 `defect_policy_comparison_v1_
2026-08-21.load_ultra_frame`(+DIFSWRF 플래그, e2e_retrain_v5와 동일)을
쓴다. 재학습 없이 3차 E2E가 저장한 실제 배포후보 joblib
(`fold_models/`, 가장 최근 폴드 5_초여름의 초단기 +1h raw+구조선택
[B_구간제외])을 그대로 재적재해 추론한다.

## 출력 (`outputs/Blockdata_규격화_3차공식모델_v1_2026-08-24/`)
v1/v2와 동일한 산출물 세트 + `요약.txt` 안에 2차 대비 무엇이 바뀌었는지
명시. 실제 Blockdata API 호출·전송은 0건(전부 오프라인 변환·검증).
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
SRC3 = ROOT / "outputs" / "E2E_v5_공식B_v1_2026-08-24"
OUT3 = ROOT / "outputs" / "Blockdata_규격화_3차공식모델_v1_2026-08-24"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# v1의 검증된 로직(규격정의·경로B 누적·total_energy·payload·22건 검증)을
# 그대로 재사용한다 — 새로 베끼지 않는다.
bd1 = _load("bd3_v1", "blockdata_export_v1_2026-08-21.py")
dpc = _load("bd3_dpc", "defect_policy_comparison_v1_2026-08-21.py")

PLANT_ID = bd1.PLANT_ID
PLANT_NAME = bd1.PLANT_NAME
DIF = "DIFSWRF_bsrn정제"


# ── ac_power / daily_final 변환(3차 행단위 산출물 → v1과 같은 컬럼 구조) ──
def build_ac_power_v3(ac_raw: pd.DataFrame) -> pd.DataFrame:
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


def build_daily_final_v3(daily_raw: pd.DataFrame) -> pd.DataFrame:
    d = daily_raw.copy()
    d["날짜"] = pd.to_datetime(d["대상일"])
    return pd.DataFrame({
        "plant_id": d["plant_id"],
        "날짜": d["날짜"],
        "경로": "A_일간모델(직접모델, 3차 공식구성·v5패치)",
        "채택모델": d["공식구성"],
        "예측_일간최종총량_kwh": d["예측_kWh"].round(2),
        "실측_일간최종총량_kwh": d["실제_kWh"].round(2),
        "폴드": d["폴드"],
    })


# ── 스모크테스트: 재학습 없이 3차가 저장한 실제 폴드모델을 재적재 ──
def run_smoke_test_v3(capacity_kw: float) -> dict:
    fold_dir = SRC3 / "fold_models"
    model_path = fold_dir / "초단기_h1_5_초여름_raw+구조선택[B_구간제외].joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"3차 폴드모델을 찾을 수 없음: {model_path}")
    bundle = joblib.load(model_path)
    if "raw" not in bundle["variant"]:
        raise RuntimeError(f"예상과 다른 변형: {bundle['variant']} (raw 전제 위반 — 역변환 로직 없음)")

    # ★v2와 다른 점★: v5 로더(DIFSWRF 결측여부 플래그 포함)를 그대로 써야
    # 3차 모델의 특성과 일치한다(순수 v3 로더는 이 플래그가 없음).
    frame = dpc.load_ultra_frame(bundle["horizon_h"])
    if DIF in frame.columns:
        frame[f"{DIF}_결측여부"] = frame[DIF].isna().astype(float)

    features = bundle["features"]
    test_start = pd.Timestamp(bundle["test_start"])
    candidates = frame.loc[frame.index >= test_start]
    valid = candidates.dropna(subset=features)
    if not len(valid):
        raise RuntimeError(f"저장된 폴드({model_path.name})의 시험구간에서 결측없는 샘플을 못 찾음")
    sample_time = valid.index[-1]  # 그 폴드 시험구간의 가장 최근 시각
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
            "model": f"{bundle['variant']}(3차 공식모델, 실제 저장분 재적재)",
            "plant_ac_power_kw": round(pred_kw, 2),
        },
    }
    return {
        "통과": True,
        "model_path": str(model_path),
        "재사용_방식": "3차 E2E 재학습이 저장한 실제 배포후보 joblib을 그대로 재적재(재학습 안 함)",
        "폴드": model_path.stem.split("_", 2)[-1].rsplit("_", 1)[0],
        "구조": bundle.get("structure_name"),
        "정책": bundle.get("정책"),
        "용량프로필": bundle.get("용량프로필"),
        "feature_count": len(features),
        "sample_issued_at": issued_at.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_target_at": target_at.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_ac_power_kw": round(pred_kw, 2),
        "payload": payload,
    }


# ── v1.verify()의 "시각정합성" 체크는 참조자료가 v3(집계_15분_자료.
# parquet, v3_고정tm의 plant_output_kw)로 고정돼 있다. 3차는 v5로
# 재학습했으므로 그 참조가 이제 stale하다 — 그대로 쓰면 인버터5 결함이
# 있던 구간이 아닌 곳까지 v3/v5 값 차이로 "거짓 실패"가 난다(실제로
# 최초 실행에서 5건 실패 확인, 아래 참고). 참조자료만 v5로 바꾸고 나머지
# 체크(물리범위·자릿수·경로불일치·비음수·단조성·중복·결측)는 v1.verify를
# 그대로 재사용한다.
def verify_v5(ac: pd.DataFrame, daily_final: pd.DataFrame, daily_cum: pd.DataFrame,
              capacity_kw: float) -> pd.DataFrame:
    checks = []

    def add(항목, 대상, 통과, 상세):
        checks.append({"항목": 항목, "대상": 대상,
                       "결과": "통과" if 통과 else "★실패★", "상세": 상세})

    q15_v5 = pd.read_parquet(dpc.V5_DIR / "집계_15분_자료_v5.parquet")["발전출력_kW"]
    v5_1h = pd.read_parquet(dpc.V5_DIR / "집계_1시간_자료_v5.parquet")["발전출력_kW"]
    for tier, ref, ref_name in [("초단기", q15_v5, "집계_15분_자료_v5.parquet"),
                                ("단기", v5_1h, "집계_1시간_자료_v5.parquet")]:
        sub = ac[ac["티어"] == tier]
        for H, s in sub.groupby("수평_h"):
            joined = s.set_index("대상시각")["실측_ac_power_kw"].to_frame("csv실측")
            joined["집계실측"] = ref.reindex(joined.index).round(2)
            valid = joined.dropna()
            if not len(valid):
                add("시각정합성(v5)", f"{tier} +{H}h", False, "대상시각이 참조자료와 하나도 안 맞음")
                continue
            diff = (valid["csv실측"] - valid["집계실측"]).abs()
            ok = bool(np.all(diff.to_numpy() <= bd1.TOL_KW)) and len(valid) / len(joined) > 0.99
            add("시각정합성(v5)", f"{tier} +{H}h", ok,
                f"매칭 {len(valid)}/{len(joined)}행, 최대 절대차 {diff.max():.4f}kW "
                f"(허용오차 {bd1.TOL_KW}kW — 대상시각 = 발행시각 + {bd1.target_offset_hours(tier, H)}h, "
                f"참조자료: {ref_name})")

    for tier, s in ac.groupby("티어"):
        bad = ((s["예측_ac_power_kw"] < 0) | (s["예측_ac_power_kw"] > capacity_kw)).sum()
        add("ac_power 유효범위", tier, bad == 0, f"0~{capacity_kw}kW 벗어난 예측 {bad}행")

    for col, df, name in [
        ("예측_ac_power_kw", ac, "ac_power"),
        ("예측_일간최종총량_kwh", daily_final, "daily_energy_경로A"),
        ("예측_daily_energy_누적_kwh", daily_cum, "daily_energy_경로B"),
    ]:
        if not len(df):
            continue
        v = df[col].dropna()
        bad = int((v.round(2) != v).sum())
        add("자릿수(소수점 2자리)", name, bad == 0, f"규격 위반 {bad}행")

    if len(daily_final) and len(daily_cum):
        a = daily_final.set_index("날짜")["예측_일간최종총량_kwh"]
        for path, b in daily_cum.groupby("경로"):
            b = b.sort_values("대상시각")
            counts = b.groupby("날짜").size()
            full_dates = counts[counts >= counts.max() * 0.95].index
            day_end = (b[b["날짜"].isin(full_dates)]
                       .groupby("날짜")["예측_daily_energy_누적_kwh"].last())
            common = a.index.intersection(day_end.index)
            if len(common) < 5:
                add("daily_energy 경로간 불일치", path, True,
                    f"공통 완전일 {len(common)}일 — 비교 표본 부족(판정 보류)")
                continue
            rel = ((day_end.loc[common] - a.loc[common]).abs()
                   / a.loc[common].replace(0, np.nan) * 100).dropna()
            add("daily_energy 경로간 불일치", path, True,
                f"공통 {len(common)}일, 중앙 불일치 {rel.median():.1f}%, 최대 {rel.max():.1f}% "
                "(정보용 — 두 경로는 서로 다른 모델이므로 불일치 자체가 실패는 아님)")

    if len(daily_cum):
        bad_neg = int((daily_cum["예측_daily_energy_누적_kwh"] < 0).sum())
        add("daily_energy 비음수", "경로B 누적", bad_neg == 0, f"음수 {bad_neg}행")

        non_monotonic = 0
        for _, g in daily_cum.groupby(["경로", "날짜"]):
            g = g.sort_values("대상시각")
            if (g["예측_daily_energy_누적_kwh"].diff().dropna() < -1e-6).any():
                non_monotonic += 1
        add("daily_energy 누적 단조성", "경로B", non_monotonic == 0,
            f"하루 안에서 누적값이 감소하는 (경로,날짜) 조합 {non_monotonic}개")

    if len(ac):
        dup = int(ac.duplicated(subset=["티어", "수평_h", "발행시각"]).sum())
        add("중복행", "ac_power", dup == 0, f"중복 {dup}행")
        na = int(ac["예측_ac_power_kw"].isna().sum())
        add("예측 결측", "ac_power", na == 0, f"결측 {na}행")

    return pd.DataFrame(checks)


def main() -> None:
    OUT3.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])

    print("=== 3차 공식구성(2차와 동일 — AGENTS.md '⑤' 절 참고, 데이터=v5·정책=공식B·용량=219kW) ===")
    print("  초단기 +1h·+2h: raw+폴드내부구조선택 / +3h·+4h: 청천지수 기본")
    print("  단기 전체: 기본+폴드내부구조선택 / 일간: 전체특성+폴드내부튜닝(v5패치)")

    ac_raw = pd.read_csv(SRC3 / "행단위_ac_power_예측정답.csv", encoding="utf-8-sig")
    daily_raw = pd.read_csv(SRC3 / "행단위_daily_예측정답.csv", encoding="utf-8-sig")

    spec = bd1.build_spec(capacity_kw)
    (OUT3 / "규격정의.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ac_power 변환(3차) ===")
    ac = build_ac_power_v3(ac_raw)
    ac.to_csv(OUT3 / "예측_ac_power.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(ac):,}행")

    print("\n=== daily_energy 변환 — 경로A(일간총량, 3차) ===")
    daily_final = build_daily_final_v3(daily_raw)
    daily_final.to_csv(OUT3 / "예측_daily_energy_일간총량.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(daily_final):,}행")

    print("\n=== daily_energy 변환 — 경로B(대상시각별 누적시계열, v1 로직 재사용) ===")
    daily_cum = bd1.build_daily_cumulative(ac)
    daily_cum.to_csv(OUT3 / "예측_daily_energy_누적시계열.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(daily_cum):,}행")

    print("\n=== total_energy 구조(v1 로직 재사용, 앵커 null) ===")
    total = bd1.build_total_energy(daily_final)
    total.to_csv(OUT3 / "예측_total_energy.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(total):,}행")

    payload = bd1.build_payload_sample(ac, daily_final, capacity_kw)
    (OUT3 / "blockdata_payload_예시.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 엔드투엔드 스모크테스트(3차 실제 저장 폴드모델 재적재, v5 로더) ===")
    try:
        smoke = run_smoke_test_v3(capacity_kw)
        print(f"  통과 — {smoke['재사용_방식']}")
        print(f"  모델: {smoke['model_path']}")
        print(f"  정책: {smoke['정책']} / 용량프로필: {smoke['용량프로필']}")
        print(f"  샘플 발행시각 {smoke['sample_issued_at']} → 대상시각 {smoke['sample_target_at']}, "
              f"예측 {smoke['predicted_ac_power_kw']}kW (구조: {smoke['구조']}, 특성 {smoke['feature_count']}개)")
    except Exception as e:  # noqa: BLE001
        smoke = {"통과": False, "오류": f"{type(e).__name__}: {e}"}
        print(f"  ★실패★ {smoke['오류']}")
    (OUT3 / "스모크테스트_결과.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== 자동 규격 점검(시각정합성만 v5 참조로 교체, 나머지는 v1.verify와 동일 로직) ===")
    checks = verify_v5(ac, daily_final, daily_cum, capacity_kw)
    checks.to_csv(OUT3 / "검증_규격점검.csv", index=False, encoding="utf-8-sig")
    print(checks.to_string(index=False))

    fails = checks[checks["결과"] != "통과"]
    fail_count = len(fails) + (0 if smoke.get("통과") else 1)

    lines = [
        "⑦ Blockdata 출력 규격화 — 3차 공식모델(v5·공식B·219kW) 재적용 요약",
        f"발전소: {PLANT_NAME}(plant_id={PLANT_ID}), 용량 {capacity_kw}kW",
        "",
        "★2차(v2) 대비 바뀐 것★",
        "- 데이터: v3(인버터5 결함 시간단위 미반영) → v5(복구완료)",
        "- 정책: (2차는 정책 미분리) → 공식 정책B(결함구간 학습·시험 제외)",
        "- 설비용량: 240kW(발전소 등록값) → 219kW(인버터 등록정격합, 신 기본 프로필)",
        "- 폴드: 구경계(가을 10-21 시작) → 정정 공식5폴드(가을 11-17 시작)",
        "- 채택 공식구성 자체는 2차와 동일(①~④ 후보 재판정은 ⑥에서 별도 진행)",
        "- 스모크테스트: v5 로더(DIFSWRF 결측여부 플래그 포함)로 프레임 재구성 후 3차 실제 저장 폴드모델 재적재",
        "",
        f"ac_power 행수: {len(ac):,} / daily_energy 경로A: {len(daily_final):,}행 "
        f"/ 경로B: {len(daily_cum):,}행",
        f"규격점검 {len(checks)}건 중 실패 {len(fails)}건, "
        f"스모크테스트 {'통과' if smoke.get('통과') else '실패'}",
        f"총 실패 {fail_count}건",
        "",
        "실제 Blockdata API 호출·전송 0건(전부 오프라인 변환·검증).",
    ]
    if len(fails):
        lines += ["", "★규격점검 실패 항목★"] + [f"- {r['항목']} / {r['대상']}: {r['상세']}"
                                              for _, r in fails.iterrows()]
    if not smoke.get("통과"):
        lines += ["", "★스모크테스트 실패★", f"- {smoke.get('오류')}"]
    (OUT3 / "요약.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n저장 완료: {OUT3}")
    if fail_count:
        print(f"\n★규격점검/스모크테스트 총 {fail_count}건 실패 — 요약.txt 확인★")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
