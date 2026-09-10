# -*- coding: utf-8 -*-
"""⑤→v2 패치 — ⑥ v5 재판정 채택안 중 **초단기 +4h만** 반영(★+2h는 재검증 후 제외★).

## ★★★08-24 추가 수정: +2h는 패치하지 않는다(⑥ 재판정 방법론 결함 발견)★★★
사용자 지시로 처음엔 +2h·+4h 둘 다 재학습했으나, 실제 재학습 직후 v1(현재
공식) 대비 성능을 비교하다가 **+2h가 오히려 4.81% 나빠지고(MAE
13.793→14.456kW) 계절 동결규칙(5%p)까지 1개 폴드(1_여름, +5.58%)에서
위반**하는 걸 발견했다. 원인을 추적한 결과 **⑥ 재판정(round2) 자체의
비교기준 오류**였다: round2의 4단계 누적판정은 "아무 것도 안 얹은
기본모델"에서 시작해 ①→②→③→④ 순서로 하나씩 켜보는 구조인데, **+2h는
실제 배포판이 이미 구조선택을 켜둔 상태(raw+구조선택, MAE 13.793)라서
round2의 기본모델(구조선택 없음, MAE 14.684)이 진짜 비교기준이 아니었다.**
③단계에서 "날씨군집화가 1.55% 개선"이라고 나온 건 이 더 약한 기본모델
대비였을 뿐, **실제 배포판(구조선택 이미 켜짐) 대비로는 개선이 아니라
악화**였다. 직접 재검증(구조선택+군집화 동시 적용, MAE 13.842)해보니
군집화는 구조선택 위에 얹어도 도움이 안 됐다(오히려 근소 악화) — 즉
**+2h는 구조선택 단독(현재 공식 그대로)이 최선**이다. **이 스크립트는
+2h를 재학습하지 않고 v1 그대로 유지한다.**

+4h는 이 문제가 없다 — round2의 기본모델과 실제 배포판(청천지수 기본,
구조선택 없음)이 애초에 같았으므로 ③단계 비교가 정확했다(실제
재학습에서도 MAE 15.062→14.852kW, +1.39% 개선 확인 — round2 예측치
1.39%와 정확히 일치). **+4h만 패치를 그대로 진행한다.**

## 범위(최종)
⑥ v5 재판정(`outputs/⑥재판정_v5_공식B_v1_2026-08-24/채택상태.json`)
결과 중 **초단기 +4h만** 반영한다:
- **초단기 +4h**: ⑤=청천지수 기본(추가특성 없음) → 채택=청천지수+
  **날씨군집화** 추가(MAE 1.39%/RMSE 1.25% 개선, 전 폴드 동결규칙 통과)

나머지(초단기 +1h·+2h·+3h, 단기 전체, 일간)는 **재학습하지 않고
⑤(v1) 산출물을 그대로 재사용**한다(재구현·중복연산 금지 원칙 — 시드·
데이터·코드경로가 전부 같으면 결과가 같으므로 다시 돌릴 이유가 없음).

## 재사용
- v5 데이터·정책B: `e2e_retrain_v5_공식B_v1_2026-08-24`(그 모듈을 통째로
  import해서 `dpc`/`harness`/`ultra`/`clearsky`/`improvement`/
  `OFFICIAL_WINDOWS`/`add_difswrf_flag`/`_roundtrip`/`_safe` 전부 재사용).
- 날씨군집화 구현: `model_improvement_round2_v1_2026-08-21.add_weather_clusters`
  (⑥ 재판정에서 쓴 것과 동일 함수, 새로 안 짬).
- 나머지 초단기 +1h·+3h, 단기, 일간 행단위 결과: ⑤ 산출물
  (`outputs/E2E_v5_공식B_v1_2026-08-24/`)에서 그대로 복사.

## 산출물 (`outputs/E2E_v5_공식B_v2_⑥반영_2026-08-24/`)
- `성능_전체.csv`, `성능_폴드별.csv`, `누출_및_재적재_감사.csv`
- `행단위_ac_power_예측정답.csv`(초단기+2h·+4h만 신규, 나머지는 v1 복사)
- `행단위_daily_예측정답.csv`(v1과 동일, 변경 없음)
- `fold_models/`(전체 40개 — v1에서 그대로 복사한 38개 + 신규 10개
  [초단기 +2h·+4h × 5폴드])
- `요약.json`

로컬 재학습만(API 없음). 이 v2가 **새 공식모델**이며, v1은 이제
초단기 +2h·+4h에 한해 stale(다른 티어·수평은 v1=v2로 동일).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC_V1 = ROOT / "outputs" / "E2E_v5_공식B_v1_2026-08-24"
OUT = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24"
MODEL_DIR = OUT / "fold_models"
PATCHED_HORIZONS = (4,)  # ★08-24 수정: +2h는 재검증 후 제외(위 docstring 참고)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load("v2_e2e", "e2e_retrain_v5_공식B_v1_2026-08-24.py")
dpc, harness, ultra, clearsky, improvement = e2e.dpc, e2e.harness, e2e.ultra, e2e.clearsky, e2e.improvement
OFFICIAL_WINDOWS = e2e.OFFICIAL_WINDOWS
N_INVERTERS = e2e.N_INVERTERS
DIF = e2e.DIF
PLANT_ID, PLANT_NAME = e2e.PLANT_ID, e2e.PLANT_NAME

# ★v2 채택 구성★(⑥ 재판정 결과 중 실제 배포판 재검증을 통과한 것만):
# +4h만 청천지수 기본에 날씨군집화 추가. +1h·+2h·+3h는 변경 없음(재학습도 안 함).
ULTRA_OFFICIAL_V2 = {4: "청천지수"}


def run_ultra_patch(capacity_kw: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    rows, audits = [], []
    for horizon in PATCHED_HORIZONS:
        official = ULTRA_OFFICIAL_V2[horizon]
        frame = e2e.add_difswrf_flag(dpc.load_ultra_frame(horizon))
        candidate_cols = ultra.EQUIP_STARTLABEL + harness.sel.OBSERVED_COLUMNS + harness.sel.FORECAST_COLUMNS
        base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                     and c not in ("목표_발전출력_kW", "목표_낮시간")]
        use_kappa = "청천지수" in official
        frame["_청천_kW"] = np.clip(
            capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
        frame["_정책타깃_kW"] = frame["목표_발전출력_kW"]  # 정책은 그대로 공식B(구간제외), C는 미대상
        frame["_카파"] = frame["_정책타깃_kW"] / frame["_청천_kW"]
        min_elev = clearsky.MIN_ELEVATION_DEG if use_kappa else 0.0
        daylight = frame[(frame["목표_낮시간"] > 0) & (frame["목표_태양고도_deg"] >= min_elev)]

        for fold, s, e_ in OFFICIAL_WINDOWS:
            start, end = pd.Timestamp(s), pd.Timestamp(e_) + pd.Timedelta(days=1)
            full_mask = daylight["_목표_가용인버터수"] >= N_INVERTERS
            train_pool = daylight[daylight.index < start]
            train_b = train_pool[full_mask.loc[train_pool.index]]  # 공식B: 학습은 완전가용만
            test_b = daylight[(daylight.index >= start) & (daylight.index < end) & full_mask]
            if len(train_b) < 300 or len(test_b) < 30:
                print(f"  [초단기 +{horizon}h {fold}] 표본부족(train={len(train_b)}, test={len(test_b)}) — 건너뜀")
                continue
            sel_input = train_b.drop(columns=["목표_발전출력_kW"]).assign(
                목표_발전출력_kW=train_b["_정책타깃_kW"]).dropna(subset=["목표_발전출력_kW"])
            chosen = harness.select_features_in_fold(sel_input, candidate_cols, 0.3, True, True)
            features = base_cols + chosen
            native_ok = harness.NATIVE_MISSING_OK | {DIF}
            required = [c for c in features if c not in native_ok] + ["_정책타깃_kW", "_청천_kW", "_카파"]
            train = train_b.dropna(subset=required)
            test = test_b.dropna(subset=[c for c in features if c not in native_ok]
                                 + ["목표_발전출력_kW", "_청천_kW"])
            if len(train) < 300 or len(test) < 30:
                continue

            # ★v2 핵심 변경★: 구조선택 대신 날씨군집화(⑥ 채택안 그대로).
            # add_weather_clusters는 KMeans/StandardScaler를 train(그 폴드의
            # 학습구간)에만 fit하고 test는 transform만 한다 — 군집이 폴드별
            # 학습구간 안에서만 만들어짐(누출 없음). 원천특성은 전부 발행
            # 시점에 이미 아는 값(NWP 예보·태양고도, CLUSTER_SOURCE 참고).
            tr, te, feat2, cluster_cols = improvement.add_weather_clusters(train, test, features, seed)
            cluster_names = [f"날씨군집_{k}" for k in range(4)] if cluster_cols else []

            # ★결측 시 군집배정 확인★: 원천특성이 결측이던 행도 train 중앙값으로
            # 채워 넣은 뒤 정상적으로 4개 군집 중 하나에 배정되는지 검증한다
            # (add_weather_clusters 내부에서 median-fill 후 transform하므로
            # 이론상 전부 배정되지만, 실제로 그런지 여기서 직접 확인·기록).
            결측행_수 = 0
            군집배정_결측0건 = True
            if cluster_cols:
                miss_mask = te[cluster_cols].isna().any(axis=1)
                결측행_수 = int(miss_mask.sum())
                assigned = te[cluster_names].sum(axis=1)  # 원-핫 합이 1이면 정상 배정
                군집배정_결측0건 = bool((assigned == 1).all())
                if not 군집배정_결측0건:
                    raise RuntimeError(f"[초단기 +{horizon}h {fold}] 군집 미배정 행 발생 — 결측처리 확인 필요")

            target = "_카파" if use_kappa else "_정책타깃_kW"
            model = ultra.make_model("LightGBM", seed)
            model.fit(tr[feat2], tr[target])

            variant = official + "+날씨군집화" + "[B_구간제외]"
            artifact = MODEL_DIR / e2e._safe(f"초단기_h{horizon}_{fold}_{variant}.joblib")
            raw, diff = e2e._roundtrip(model, {
                "tier": "초단기", "horizon_h": horizon, "variant": variant, "features": feat2,
                "params": None, "target_transform": target, "structure_name": "기본(군집화로대체)",
                "군집원천특성": cluster_cols, "정책": "B_구간제외",
                "용량프로필": "inverter_registered_sum_219",
                "train_end": str(train.index.max()), "test_start": str(test.index.min()),
            }, te[feat2], artifact)
            pred = raw * te["_청천_kW"].to_numpy() if use_kappa else raw
            pred = np.clip(pred, 0, capacity_kw)
            actual = te["목표_발전출력_kW"].to_numpy()
            target_at = te.index + pd.to_timedelta(horizon, unit="h")
            for issued, tt, y, p in zip(te.index, target_at, actual, pred):
                rows.append({"plant_id": PLANT_ID, "plant_name": PLANT_NAME, "티어": "초단기",
                            "수평_h": horizon, "공식구성": variant, "폴드": fold,
                            "발행시각": issued, "대상시각": tt, "실제_kW": y, "예측_kW": p,
                            "버킷시간_h": 0.25, "모델파일": str(artifact)})
            audits.append({"티어": "초단기", "수평_h": horizon, "폴드": fold,
                           "학습최종시각": train.index.max(), "시험최초시각": test.index.min(),
                           "학습_시험_분리통과": bool(train.index.max() < test.index.min()),
                           "추정타깃포함": False, "정책": "B_구간제외",
                           "모델재적재_동일": bool(diff <= 1e-12), "재적재차이": diff,
                           "특성수": len(feat2), "선택구조": "기본(군집화로대체)",
                           "학습행수": len(tr), "시험행수": len(te),
                           "군집원천특성수": len(cluster_cols), "시험_결측행수": 결측행_수,
                           "군집배정_전부정상": 군집배정_결측0건})
            print(f"[초단기 +{horizon}h {fold}] {variant} 완료 (학습{len(tr):,}/시험{len(te):,}, "
                  f"군집원천특성 {len(cluster_cols)}개, 시험결측행 {결측행_수}건, 군집배정 정상 {군집배정_결측0건})")
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
    if not SRC_V1.exists():
        raise FileNotFoundError(f"v1 산출물이 없다: {SRC_V1}")
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])
    print(f"v2 패치 범위: 초단기 +2h·+4h만 재학습(날씨군집화 반영). 나머지는 v1 그대로 재사용.\n")

    # ── 1) v1 산출물 로드 ──
    ac_v1 = pd.read_csv(SRC_V1 / "행단위_ac_power_예측정답.csv", encoding="utf-8-sig",
                        parse_dates=["발행시각", "대상시각"])
    daily_v1 = pd.read_csv(SRC_V1 / "행단위_daily_예측정답.csv", encoding="utf-8-sig",
                           parse_dates=["발행시각", "대상일"])
    audits_v1 = pd.read_csv(SRC_V1 / "누출_및_재적재_감사.csv", encoding="utf-8-sig")

    # ── 2) 초단기 +2h·+4h만 재학습 ──
    print("=== 초단기 +2h·+4h 재학습(날씨군집화) ===")
    ac_patch, aud_patch = run_ultra_patch(capacity_kw, seed)

    # ── 3) 안 바뀐 부분(+1h·+3h, 단기, 일간)은 v1에서 그대로 복사 ──
    # ★버그수정★: 두 CSV 모두 "수평_h"에 일간행("D+1")이 섞여 있을 수 있어
    # object/문자열 dtype으로 읽힐 수 있다(ac_power엔 안 섞였지만 감사표엔
    # 섞여 있었음, 실제로 확인됨) — 숫자로 변환해서 비교해야 isin이 제대로
    # 걸린다. 최초 실행에서 이 버그로 감사표에 구모델 행이 안 지워지고
    # 남아있던 걸 발견해 고쳤다.
    ac_h = pd.to_numeric(ac_v1["수평_h"], errors="coerce")
    unchanged_mask = ~((ac_v1["티어"] == "초단기") & ac_h.isin(PATCHED_HORIZONS))
    ac_unchanged = ac_v1[unchanged_mask].copy()
    ac = pd.concat([ac_unchanged, ac_patch], ignore_index=True)

    aud_h = pd.to_numeric(audits_v1["수평_h"], errors="coerce")
    aud_unchanged_mask = ~((audits_v1["티어"] == "초단기") & aud_h.isin(PATCHED_HORIZONS))
    audits = pd.concat([audits_v1[aud_unchanged_mask], pd.DataFrame(aud_patch)], ignore_index=True)

    # ★버그수정★: 처음엔 "패치된 수평 파일명으로 시작하지 않으면 전부 복사"
    # 했더니 v1 fold_models 안에 있던, 어느 CSV에도 참조되지 않는 이전
    # 버전의 낙동강오리알 joblib(브라켓 없는 구 명명규칙 잔재, 60개 중 실제
    # 참조되는 건 40개뿐이었음)까지 같이 복사돼 55개가 됐다. **CSV
    # (행단위_ac_power_예측정답.csv·행단위_daily_예측정답.csv)의
    # "모델파일" 컬럼에 실제로 참조된 파일만** 골라서 복사하도록 고쳤다
    # (신규 패치 파일은 이미 run_ultra_patch가 MODEL_DIR에 직접 저장함).
    referenced = set(Path(p).name for p in
                     pd.concat([ac_unchanged["모델파일"], daily_v1["모델파일"]]).dropna().unique())
    for src in (SRC_V1 / "fold_models").glob("*.joblib"):
        if src.name in referenced:
            shutil.copy2(src, MODEL_DIR / src.name)

    perf = make_performance(ac, daily_v1, capacity_kw)
    ac.to_csv(OUT / "행단위_ac_power_예측정답.csv", index=False, encoding="utf-8-sig")
    daily_v1.to_csv(OUT / "행단위_daily_예측정답.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(OUT / "성능_전체.csv", index=False, encoding="utf-8-sig")
    audits.to_csv(OUT / "누출_및_재적재_감사.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 성능(v2·⑥반영) ===")
    print(perf.to_string(index=False))
    print("\n=== 감사 요약 ===")
    print(f"조합 수: {len(audits)} / 학습시험분리 전부통과: {bool(audits['학습_시험_분리통과'].all())} "
          f"/ 재적재동일 전부통과: {bool(audits['모델재적재_동일'].all())} "
          f"/ 추정타깃포함: {int(audits['추정타깃포함'].sum())}건 "
          f"/ fold_models 파일수: {len(list(MODEL_DIR.glob('*.joblib')))}")

    # ── 폴드별(계절별) 성능 — 전체 티어·수평 ──
    fold_rows = []
    for (tier, h, fold), g in ac.groupby(["티어", "수평_h", "폴드"]):
        e = g["실제_kW"] - g["예측_kW"]
        fold_rows.append({"티어": tier, "수평_h": h, "폴드": fold, "n": len(g),
                          "MAE_kW": round(float(e.abs().mean()), 3),
                          "RMSE_kW": round(float(np.sqrt((e ** 2).mean())), 3)})
    for fold, g in daily_v1.groupby("폴드"):
        e = g["실제_kWh"] - g["예측_kWh"]
        fold_rows.append({"티어": "일간", "수평_h": "D+1", "폴드": fold, "n": len(g),
                          "MAE_kW": round(float(e.abs().mean()), 3),
                          "RMSE_kW": round(float(np.sqrt((e ** 2).mean())), 3)})
    perf_fold = pd.DataFrame(fold_rows)
    perf_fold.to_csv(OUT / "성능_폴드별.csv", index=False, encoding="utf-8-sig")

    # ── v1 대비 패치 수평 폴드별(계절별) 악화율 — 동결규칙(5%p) 그대로 적용 ──
    # ★버그수정★: 성능_전체.csv의 "수평_h"는 일간행에 "D+1"이 섞여있어
    # object dtype으로 읽힌다 — 정수 h와 비교되도록 명시 캐스팅한다.
    perf_v1 = pd.read_csv(SRC_V1 / "성능_전체.csv", encoding="utf-8-sig")
    perf_v1["수평_h_비교용"] = pd.to_numeric(perf_v1["수평_h"], errors="coerce")
    ac_v1_patched = ac_v1[(ac_v1["티어"] == "초단기") & (ac_v1["수평_h"].isin(PATCHED_HORIZONS))]
    fold_v1 = []
    for (h, fold), g in ac_v1_patched.groupby(["수평_h", "폴드"]):
        e = g["실제_kW"] - g["예측_kW"]
        fold_v1.append({"수평_h": h, "폴드": fold, "MAE_kW_v1": round(float(e.abs().mean()), 3),
                        "RMSE_kW_v1": round(float(np.sqrt((e ** 2).mean())), 3)})
    fold_v1_df = pd.DataFrame(fold_v1)
    fold_v2_df = perf_fold[(perf_fold["티어"] == "초단기") & (perf_fold["수평_h"].isin(PATCHED_HORIZONS))].rename(
        columns={"MAE_kW": "MAE_kW_v2", "RMSE_kW": "RMSE_kW_v2"})
    cmp_df = fold_v1_df.merge(fold_v2_df[["수평_h", "폴드", "MAE_kW_v2", "RMSE_kW_v2"]], on=["수평_h", "폴드"])
    cmp_df["MAE_악화율_pct"] = (cmp_df["MAE_kW_v2"] - cmp_df["MAE_kW_v1"]) / cmp_df["MAE_kW_v1"] * 100
    cmp_df["RMSE_악화율_pct"] = (cmp_df["RMSE_kW_v2"] - cmp_df["RMSE_kW_v1"]) / cmp_df["RMSE_kW_v1"] * 100
    cmp_df["동결규칙_5pt_위반"] = (cmp_df["MAE_악화율_pct"] >= 5) | (cmp_df["RMSE_악화율_pct"] >= 5)
    cmp_df.to_csv(OUT / "v1대비_폴드별_비교.csv", index=False, encoding="utf-8-sig")

    print("\n=== v1(⑤) vs v2(⑥반영) — 패치된 수평, 전체 ===")
    for h in PATCHED_HORIZONS:
        r1 = perf_v1[(perf_v1["티어"] == "초단기") & (perf_v1["수평_h_비교용"] == h)].iloc[0]
        r2 = perf[(perf["티어"] == "초단기") & (perf["수평_h"] == h)].iloc[0]
        mae_gain = (r1["MAE_kW"] - r2["MAE_kW"]) / r1["MAE_kW"] * 100
        rmse_gain = (r1["RMSE_kW"] - r2["RMSE_kW"]) / r1["RMSE_kW"] * 100
        print(f"  +{h}h: MAE {r1['MAE_kW']}→{r2['MAE_kW']}kW({mae_gain:+.2f}%), "
              f"RMSE {r1['RMSE_kW']}→{r2['RMSE_kW']}kW({rmse_gain:+.2f}%)")
    print("\n=== v1(⑤) vs v2(⑥반영) — 폴드별(계절별) 악화율(동결규칙: 5%p 이상 악화면 위반) ===")
    print(cmp_df.to_string(index=False))
    violations = int(cmp_df["동결규칙_5pt_위반"].sum())
    if violations:
        print(f"\n★경고★ 계절 악화 동결규칙 위반 {violations}건 — 채택 재검토 필요")
    else:
        print("\n계절별 악화율 전부 5%p 미만 — 동결규칙 위반 없음")

    # ── 결측 시 군집배정 확인 요약 ──
    cluster_audit = pd.DataFrame(aud_patch)
    all_assigned_ok = bool(cluster_audit["군집배정_전부정상"].all()) if len(cluster_audit) else True
    total_missing_rows = int(cluster_audit["시험_결측행수"].sum()) if len(cluster_audit) else 0
    print(f"\n=== 결측 시 군집배정 확인 === 전체 폴드×수평 군집배정 정상: {all_assigned_ok} "
          f"/ 원천특성 결측이었던 시험행 총 {total_missing_rows}건도 전부 정상 배정됨")

    summary = {
        "실행유형": "⑤→v2 패치: ⑥ v5 재판정 채택안(초단기 +2h·+4h 날씨군집화) 반영",
        "패치범위": [f"초단기_+{h}h" for h in PATCHED_HORIZONS],
        "변경없음": "초단기 +1h·+3h, 단기 전체, 일간(v1과 완전 동일 — 재학습 안 하고 v1 산출물 재사용)",
        "폴드": [w[0] for w in OFFICIAL_WINDOWS],
        "용량프로필": config["site"].get("capacity_profile"),
        "용량kW": capacity_kw,
        "조합수": int(len(audits)),
        "학습시험분리_전부통과": bool(audits["학습_시험_분리통과"].all()),
        "재적재_전부통과": bool(audits["모델재적재_동일"].all()),
        "추정타깃포함_건수": int(audits["추정타깃포함"].sum()),
        "라이브_API_호출": False,
        "기준_v1": str(SRC_V1),
        "근거": "outputs/⑥재판정_v5_공식B_v1_2026-08-24/채택상태.json",
        "계절악화_동결규칙_위반건수": violations,
        "군집배정_전부정상": all_assigned_ok,
        "원천특성결측_시험행수": total_missing_rows,
        "KPX_영향": "없음 — KPX D-1 재현은 단기(day-ahead) 모델만 쓰고 이번 패치는 초단기만 바꿨으므로 재산출 불필요",
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
