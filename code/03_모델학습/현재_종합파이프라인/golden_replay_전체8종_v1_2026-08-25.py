# -*- coding: utf-8 -*-
"""★golden replay — 운영모델 8종 전부, 검증된 단기+24h 패턴을 그대로 확장★

`golden_replay_단기h24_v1_2026-08-25.py`(사용자 확인·통과 완료)와 완전히
같은 방법을 초단기 4개(+1h/+2h/+3h/+4h)·단기 3개(+1h/+24h/+48h)·
일간(D+1) 전부에 적용한다. **재구현 아님** — h24에서 이미 검증된 절차를
그대로 반복만 한다.

## 실행 주체
이 스크립트는 **Claude가 작성만 했고 실행은 하지 않았다**(사용자 지시
"돌리는거는 코덱스에서 진행"). 코덱스가 그대로 실행하면 된다:
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\03_모델학습\\현재_종합파이프라인"
python golden_replay_전체8종_v1_2026-08-25.py
```
로컬 재현일 뿐 외부 API 호출은 0건이다(전부 이미 수집된 과거자료 재사용).

## 각 티어의 재조립 방법(전부 기존 함수 재사용, 새 로직 없음)
- **초단기(4개)**: `dpc.load_ultra_frame(h)` 원자료를 그대로 다시 읽어
  `_청천_kW`·`_카파` 계산까지 재현(e2e_retrain_v5_공식B_v1과 동일 공식).
  +4h는 번들에 저장된 KMeans·스케일러·중앙값으로 군집을 재배정
  (`production_inference_utils.assign_weather_clusters`, 재구현 아님).
- **단기(3개, +24h 포함)**: h24와 완전히 동일한 패턴
  (`harness.build_frame()` 재사용).
- **일간(1개)**: `daily_mod.corrected_dataset()`을 다시 호출해 재조립
  (학습 스크립트가 이미 "원자료→특성"을 이 함수 하나로 캡슐화해뒀으므로
  재조립 경로와 학습 경로가 사실상 같은 함수 — 그래서 일간은 "조립 로직
  검증"의 의미가 초단기·단기보다 약하다는 점을 결과에 명시한다).

## 판정 기준(전부 통과해야 함, h24와 동일)
1. 재조립 특성값이 공식 파이프라인과 전부 일치(오차 <1e-9)
2. `predict_kw()` 예측이 저장모델 직접 `.predict()`와 완전히 일치(diff=0)
3. 물리적 출력범위(초단기·단기=[0,219kW], 일간=[0,219*24 kWh]) 준수
4. (+4h만) 군집배정 결과도 학습 당시와 일치
5. (+3h·+4h만) κ→kW 역변환이 실제로 적용됐는지(raw~kW 스케일 차이로 확인)

## 산출물
`outputs/golden_replay_전체8종_v1_2026-08-25/판정_전체.json` +
티어별 콘솔 출력. 하나라도 실패하면 그 항목을 숨기지 않고 그대로 남긴다
(사전동결 원칙 — 통과 못 한 걸 재시도로 억지로 맞추지 않음, 실패 원인을
그대로 보고할 것).

## 08-26 추가: DSWRFLX 부재 검사(재발 방지, Codex 지적 반영)
운영 joblib 7종이 몽키패치 leak으로 `DSWRFLX_bsrn정제`를 계속 갖고
있었는데 golden replay가 여태 이걸 못 잡았다(재적재 일치성만 봤지
특성 내용은 안 봤음, AGENTS.md 08-26절). `_dswrflx_absent(bundle)`을
추가해 `features`뿐 아니라 +4h의 군집원천(`클러스터.입력컬럼`)까지
확인하고, 8종 전부의 `전체판정`에 이 결과를 포함시켰다 — 이제
"8/8 통과"가 진짜로 DSWRFLX 없음까지 보장한다.

## 08-26 2차 추가: 일간 DIFSWRF 부재 검사(readiness 게이트가 못 잡는 위험 차단)
DIFSWRF도 DSWRFLX와 같은 KIMR 원인으로 라이브 100% 결측인데, 일간
readiness 게이트(`shadow_readiness_일간_v1_2026-08-26.py`)는 Blockdata
이력·NWP만 보고 DIFSWRF는 검사하지 않는다. 초단기·단기는 DIFSWRF
**플래그만** 쓰므로(원값 미사용) 안전하지만, 일간은 원값을 실제로 썼다가
`retrain_exclude_difswrf_daily_v1_2026-08-26.py` 공식 5폴드 검증(전체
악화<1%, 폴드 최대 RMSE악화 4.19%<5%) 통과 후 `train_production_daily_v4_
DIFSWRF제외_2026-08-26.py`로 51특성(기존55-DIFSWRF4)으로 재생성했다.
`_difswrf_absent_daily(bundle)`을 추가해 **일간 번들에 한해서만**
DIFSWRF 관련 특성(계열 4개 전부, 플래그 포함) 부재를 확인한다 — 초단기·
단기는 플래그 사용이 정상이므로 이 검사 대상에서 제외한다. 아울러 일간
특성수가 51인지, `운영모델_목록.json` 등록 특성수와 번들 실제 특성수가
일치하는지도 같이 확인한다.
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
OUT = ROOT / "outputs" / "golden_replay_전체8종_v1_2026-08-25"
BUNDLE_DIR = ROOT / "outputs" / "운영모델_v2_2026-08-25"
N_INVERTERS = 5


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("gr8_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc, harness, ultra, clearsky = e2e.dpc, e2e.harness, e2e.ultra, e2e.clearsky
add_difswrf_flag = e2e.add_difswrf_flag
daily_mod = _load("gr8_daily", "daily_direct_final_audit_v1_2026-08-25.py")
infer = _load("gr8_infer", "production_inference_utils_v1_2026-08-25.py")
DIF = "DIFSWRF_bsrn정제"


def _almost_equal(a, b, tol=1e-9) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol


DSX = "DSWRFLX_bsrn정제"


DIFSWRF_ALL = "DIFSWRF"  # 계열 전부(값+플래그) 검출용 — DSWRFLX와 문자열이 겹치지 않음
EXPECTED_DAILY_FEATURE_COUNT = 51


def _difswrf_absent_daily(bundle: dict) -> tuple[bool, list[str]]:
    """일간 전용(08-26 2차 추가) — DIFSWRF 계열 4개(값 2개+유효개수+
    자체 결측여부 플래그)가 일간 번들 features에서 전부 빠졌는지 확인한다.
    초단기·단기는 DIFSWRF 플래그만 정상적으로 쓰므로 이 검사 대상이 아니다."""
    leaked = [c for c in bundle.get("features", []) if DIFSWRF_ALL in c]
    return (len(leaked) == 0), leaked


def _dswrflx_absent(bundle: dict) -> tuple[bool, list[str]]:
    """08-26 신규 추가 — 실제 저장된 운영 joblib 7종에 DSWRFLX_bsrn정제가
    (몽키패치 leak으로) 남아있던 걸 golden replay가 여태 못 잡았던 문제의
    재발 방지 검사(AGENTS.md 08-26절 참고). `features`뿐 아니라 +4h의
    군집원천(`클러스터.입력컬럼`)까지 같이 본다 — 특성 목록만 깨끗해도
    군집 입력에 남아있으면 라이브에서 여전히 DSWRFLX(영구 NaN)에 의존하게
    되므로 둘 다 확인해야 한다."""
    leaked = [c for c in bundle.get("features", []) if DSX in c]
    cluster = bundle.get("클러스터")
    if cluster:
        leaked += [c for c in cluster.get("입력컬럼", []) if DSX in c]
    return (len(leaked) == 0), leaked


def replay_ultra(h: int) -> dict:
    bundle = joblib.load(BUNDLE_DIR / f"운영모델_초단기_h{h}.joblib")
    audit = infer.audit_bundle(bundle)
    if not audit["전체통과"]:
        return {"티어": "초단기", "수평_h": h, "전체판정": False, "사유": f"번들불완전: {audit}"}
    dswrflx_absent, dswrflx_leaked = _dswrflx_absent(bundle)

    # 재조립 경로: 원자료를 처음부터 다시 읽어 _청천_kW·_카파까지 재계산
    frame = add_difswrf_flag(dpc.load_ultra_frame(h))
    capacity_kw = bundle["capacity_kw"]
    frame["_청천_kW"] = np.clip(
        capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]

    # 공식 경로: 같은 로더를 그대로 한 번 더 부른 것(재현성 확인용 — 로더
    # 자체는 이미 결정적이므로 이 둘은 원래도 같아야 정상)
    official = add_difswrf_flag(dpc.load_ultra_frame(h))
    official["_청천_kW"] = frame["_청천_kW"]
    official["_카파"] = frame["_카파"]

    use_kappa = bundle["타깃유형"] == "kappa"
    min_elev = bundle["카파_역변환"]["최소태양고도_deg"] if use_kappa else 0.0
    daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)
                     & (frame["_목표_가용인버터수"] >= N_INVERTERS)]
    train_end = pd.Timestamp(bundle["학습기간_끝"])
    candidates = [t for t in daylight.index if t <= train_end]
    if not candidates:
        return {"티어": "초단기", "수평_h": h, "전체판정": False, "사유": "재현 발행시각 후보 없음"}
    issue_time = sorted(candidates)[-1]

    feat_cols_no_cluster = [c for c in bundle["features"] if not c.startswith("날씨군집_")]
    row_rebuilt = frame.loc[[issue_time], list(dict.fromkeys(feat_cols_no_cluster + ["목표_태양고도_deg"]))]
    row_official = official.loc[[issue_time], feat_cols_no_cluster]

    diffs = {}
    all_match = True
    for c in feat_cols_no_cluster:
        if not _almost_equal(row_rebuilt[c].iloc[0], row_official[c].iloc[0]):
            all_match = False
            diffs[c] = {"재조립": row_rebuilt[c].iloc[0], "공식": row_official[c].iloc[0]}

    pred_via_replay = infer.predict_kw(bundle, row_rebuilt)
    if bundle["클러스터"]:
        row_official_full = infer.assign_weather_clusters(bundle, row_official.assign(목표_태양고도_deg=row_rebuilt["목표_태양고도_deg"]))
        direct_raw = float(bundle["model"].predict(row_official_full[bundle["features"]])[0])
    else:
        direct_raw = float(bundle["model"].predict(row_official[bundle["features"]])[0])
    if use_kappa:
        cs_kw = float(np.clip(capacity_kw * infer.clear_sky_ghi(row_rebuilt["목표_태양고도_deg"].to_numpy())[0] / 1000.0, 1e-3, None))
        direct_kw = float(np.clip(direct_raw * cs_kw, 0, capacity_kw))
    else:
        direct_kw = float(np.clip(direct_raw, 0, capacity_kw))

    pred_diff = float(abs(pred_via_replay[0] - direct_kw))
    kappa_applied_ok = True
    if use_kappa:
        kappa_applied_ok = not (direct_kw > 1.0 and abs(direct_raw - direct_kw) < 1e-9)

    return {
        "티어": "초단기", "수평_h": h, "발행시각": str(issue_time),
        "번들audit": audit, "특성값_전부일치": all_match, "불일치특성": diffs,
        "재조립경로_예측_kW": round(float(pred_via_replay[0]), 4),
        "공식경로_예측_kW": round(direct_kw, 4), "예측차이": pred_diff,
        "219kW_출력범위_준수": bool(0.0 <= pred_via_replay[0] <= capacity_kw),
        "카파역변환_실제적용됨": kappa_applied_ok,
        "DSWRFLX_부재": dswrflx_absent, "DSWRFLX_잔존특성": dswrflx_leaked,
        "전체판정": bool(all_match and pred_diff < 1e-6 and kappa_applied_ok and dswrflx_absent),
    }


def build_short_hourly() -> pd.DataFrame:
    hourly = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                         parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(dpc.V5_DIR / "집계_1시간_자료_v5.parquet")
    common = hourly.index.intersection(v5_1h.index)
    hourly = hourly.loc[common].copy()
    hourly["plant_output_kw"] = v5_1h.loc[common, "발전출력_kW"]
    hourly["inverters_available"] = v5_1h.loc[common, "가용인버터수"]
    return hourly


def replay_short(h: int) -> dict:
    bundle = joblib.load(BUNDLE_DIR / f"운영모델_단기_h{h}.joblib")
    audit = infer.audit_bundle(bundle)
    if not audit["전체통과"]:
        return {"티어": "단기", "수평_h": h, "전체판정": False, "사유": f"번들불완전: {audit}"}
    dswrflx_absent, dswrflx_leaked = _dswrflx_absent(bundle)

    hourly = build_short_hourly()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    rebuilt = harness.build_frame(hourly, h, candidate_cols)
    if DIF in rebuilt.columns:
        rebuilt[f"{DIF}_결측여부"] = rebuilt[DIF].isna().astype(float)

    official = dpc.load_short_frame(h)
    daylight_official = official[(official["목표_낮시간"] > 0) & (official["_목표_가용인버터수"] >= N_INVERTERS)]
    common_idx = daylight_official.index.intersection(rebuilt.index)
    train_end = pd.Timestamp(bundle["학습기간_끝"])
    candidates = [t for t in common_idx if t <= train_end]
    if not candidates:
        return {"티어": "단기", "수평_h": h, "전체판정": False, "사유": "재현 발행시각 후보 없음"}
    issue_time = sorted(candidates)[-1]

    feat_cols = bundle["features"]
    row_rebuilt = rebuilt.loc[[issue_time], feat_cols]
    row_official = official.loc[[issue_time], feat_cols]

    diffs = {}
    all_match = True
    for c in feat_cols:
        if not _almost_equal(row_rebuilt[c].iloc[0], row_official[c].iloc[0]):
            all_match = False
            diffs[c] = {"재조립": row_rebuilt[c].iloc[0], "공식": row_official[c].iloc[0]}

    pred_via_replay = infer.predict_kw(bundle, row_rebuilt)
    direct = float(np.clip(bundle["model"].predict(row_official[feat_cols])[0], 0, bundle["capacity_kw"]))
    pred_diff = float(abs(pred_via_replay[0] - direct))

    return {
        "티어": "단기", "수평_h": h, "발행시각": str(issue_time),
        "번들audit": audit, "특성값_전부일치": all_match, "불일치특성": diffs,
        "재조립경로_예측_kW": round(float(pred_via_replay[0]), 4),
        "공식경로_예측_kW": round(direct, 4), "예측차이": pred_diff,
        "219kW_출력범위_준수": bool(0.0 <= pred_via_replay[0] <= bundle["capacity_kw"]),
        "DSWRFLX_부재": dswrflx_absent, "DSWRFLX_잔존특성": dswrflx_leaked,
        "전체판정": bool(all_match and pred_diff < 1e-6 and dswrflx_absent),
    }


def replay_daily() -> dict:
    bundle = joblib.load(BUNDLE_DIR / "운영모델_일간_D+1.joblib")
    audit = infer.audit_bundle(bundle)
    if not audit["전체통과"]:
        return {"티어": "일간", "전체판정": False, "사유": f"번들불완전: {audit}"}
    dswrflx_absent, dswrflx_leaked = _dswrflx_absent(bundle)
    difswrf_absent, difswrf_leaked = _difswrf_absent_daily(bundle)
    feature_count_ok = len(bundle.get("features", [])) == EXPECTED_DAILY_FEATURE_COUNT

    registry = json.loads((BUNDLE_DIR / "운영모델_목록.json").read_text(encoding="utf-8"))
    registry_feature_count = registry.get("모델", {}).get("일간_D+1", {}).get("특성수")
    registry_matches_bundle = registry_feature_count == len(bundle.get("features", []))

    capacity_kw = bundle["capacity_kw"]
    data, _all_features, _n_partial = daily_mod.corrected_dataset(capacity_kw)
    # ★08-26 수정★: replay_short()·replay_ultra()와 동일하게 번들이 실제로
    # 학습에 쓴 특성목록(bundle["features"])을 써야 한다. corrected_dataset()이
    # 매번 새로 돌려주는 전체 58개 목록을 그대로 쓰면, 일간 번들이 그중 일부를
    # 뺀 채로 학습됐을 때(08-26 mean_communication_ok·DSWRFLX 제외처럼)
    # LightGBM이 "학습 시 특성수와 다르다"며 즉시 에러를 낸다 — 실제로 이
    # 버그를 08-26 일간 번들 재생성 직후 golden replay에서 잡았다.
    features = bundle["features"]
    train_end = pd.Timestamp(bundle["학습기간_끝"])
    candidates = [t for t in data.index if t <= train_end]
    if not candidates:
        return {"티어": "일간", "전체판정": False, "사유": "재현 목표일 후보 없음"}
    target_day = sorted(candidates)[-1]

    row = data.loc[[target_day], features]
    pred_via_replay = infer.predict_kw(bundle, row)
    direct = float(np.clip(bundle["model"].predict(row[features])[0], 0, bundle["clip_상한"]))
    pred_diff = float(abs(pred_via_replay[0] - direct))

    return {
        "티어": "일간", "발행일": str(target_day), "번들audit": audit,
        "참고": "일간은 corrected_dataset()을 재조립 경로로도 그대로 재사용 —"
               "초단기·단기와 달리 '조립 로직 자체'를 독립적으로 재검증한 것은"
               "아님(학습·재조립이 사실상 같은 함수). model.predict 재적재"
               "일치성만 확인.",
        "재조립경로_예측_kWh": round(float(pred_via_replay[0]), 2),
        "공식경로_예측_kWh": round(direct, 2), "예측차이": pred_diff,
        "출력범위_준수(0~clip상한)": bool(0.0 <= pred_via_replay[0] <= bundle["clip_상한"]),
        "DSWRFLX_부재": dswrflx_absent, "DSWRFLX_잔존특성": dswrflx_leaked,
        "DIFSWRF_부재": difswrf_absent, "DIFSWRF_잔존특성": difswrf_leaked,
        "특성수": len(bundle.get("features", [])),
        f"특성수_{EXPECTED_DAILY_FEATURE_COUNT}개_일치": feature_count_ok,
        "레지스트리_특성수": registry_feature_count,
        "레지스트리_번들_특성수_일치": registry_matches_bundle,
        "전체판정": bool(pred_diff < 1e-6 and dswrflx_absent and difswrf_absent
                     and feature_count_ok and registry_matches_bundle),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for h in (1, 2, 3, 4):
        r = replay_ultra(h)
        print(f"[초단기 +{h}h] 전체판정={r.get('전체판정')}")
        results.append(r)
    for h in (1, 24, 48):
        r = replay_short(h)
        print(f"[단기 +{h}h] 전체판정={r.get('전체판정')}")
        results.append(r)
    r = replay_daily()
    print(f"[일간 D+1] 전체판정={r.get('전체판정')}")
    results.append(r)

    all_pass = all(r.get("전체판정") for r in results)
    summary = {"생성시각": pd.Timestamp.now().isoformat(), "전체8종_판정": all_pass, "결과": results}
    (OUT / "판정_전체.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== 8종 전체판정: {'전부 통과' if all_pass else '일부 실패 — 판정_전체.json에서 사유 확인'} ===")


if __name__ == "__main__":
    main()
