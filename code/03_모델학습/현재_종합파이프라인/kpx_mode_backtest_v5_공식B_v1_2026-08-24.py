# -*- coding: utf-8 -*-
"""⑦ KPX D-1 재현 백테스트 — 3차 공식모델 기준(v5·공식 정책B·219kW) 재생성.

## 왜 다시 하나
`kpx_mode_backtest_v1_2026-08-21.py`(이하 v1)는 v3 데이터(인버터5 결함이
시간단위 자료에 미반영된 상태)로 만들었다. 그 뒤 v5 발견 → 전면 재작업 →
⑤ v5·공식 정책B·219kW 전 모델 재학습이 끝났으므로 KPX 재현도 그 기준으로
다시 만들어야 한다(AGENTS.md "⑦ Blockdata/KPX 재생성" 항목). **v1의
산출물은 이제 stale하다.**

## 재사용(재구현 금지 원칙) — v1과 동일한 로직을 그대로 가져다 쓴다
- 발행구조·누출방지·정산공식·기준선 정의: v1의 `asof_values`,
  `build_kpx_frame`(아래서 target 가용인버터수 컬럼 하나만 추가),
  `settle`. KPX 규격(제출시각·10%필터·오차식·정산단가)은 전부 v1 문서화
  그대로 유효(제도 자체는 안 바뀜).
- 구조선택·모델: `model_improvement_round2_v1_2026-08-21`
  (`choose_structure_kfold` — 08-24 표준화 결정에 따라 v1이 쓰던
  `choose_structure`(단일 80/20분할) 대신 이걸 쓴다, ⑤·⑥과 동일 원칙).

## v1과 다른 점
1. **데이터**: `plant_output_kw`/`inverters_available`를 v5
   (`집계_1시간_자료_v5.parquet`)로 치환(defect_policy_comparison의
   `load_short_frame`과 동일한 치환 방식).
2. **정책 B 적용**: 학습행은 **대상시각 가용인버터수=5**인 행만 사용
   (공식 B, 다른 모든 3차 산출물과 동일 정의). **시험행도 항상 완전가용만**
   (정책과 무관, 고정 — e2e_retrain_v5·defect_policy_comparison과 동일
   원칙). 이 때문에 "이상구간(결함기간)" 폴드는 시험행 커버리지가
   급감할 수 있다 — **은폐하지 않고 폴드별 커버리지를 그대로 보고한다**
   (2026-08-21 KPX 1차 재현에서도 이미 "이상구간 평가는 실패"라고
   명시했던 것과 같은 원칙, 이번엔 정책B로 그 실패를 정량화).
3. **폴드 경계 정정**: 이상구간을 08-24 확정판 경계(v5 자체진단 기준
   실제 인버터5 결함 08-19~11-16)에 맞춰 `2025-08-15~11-16`으로,
   가을(결함종료후)을 `11-17` 시작으로 고쳤다(v1은 08-21 시점의
   잠정경계 `08-15~10-20`을 썼음 — defect_policy_comparison의 08-24
   정정과 동일한 수정).
4. **설비용량**: 219kW(신 기본 프로필)만 공식으로 산출하되, 240kW는
   민감도 비교로 계속 병기(v1과 동일하게 두 값 다 낸다).
5. **구조선택**: `choose_structure`→`choose_structure_kfold`.

## 산출물 (`outputs/KPX재현_백테스트_3차_v1_2026-08-24/`)
v1과 동일한 산출물 세트 + `커버리지_폴드별.csv`(신규, 정책B 적용 후
폴드별 전체대상행수·완전가용시험행수·커버리지%).

로컬 학습·평가만 수행한다(외부 API 호출 0건).
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
V5_DIR = ROOT / "outputs" / "v5_복구_2026-08-21"

# ★08-24 공식 반영★: 청천지수(카파) 변환 + 날씨군집화를 공식 채택했다.
# 검증 근거는 `kpx_dayahead_final_verify_v1_2026-08-24.py`
# (재현성 정확일치, 시드 4종 전부 채택, NMAE 7.865%→7.360%).
# `raw`(구 공식, raw 타깃 + 구조선택만)도 재현 가능하게 모드로 남겨둔다
# — 구 산출물과 대조하거나 개선효과를 다시 확인할 때 쓴다.
OFFICIAL_MODE = "kappa_cluster"
MODE_OUT = {
    "kappa_cluster": "KPX재현_백테스트_4차_카파군집_2026-08-24",
    "raw": "KPX재현_백테스트_3차_v1_2026-08-24",
}

PLANT_ID = 6715
PLANT_NAME = "광주 광주시청"
N_INVERTERS = 5

ISSUE_HOUR = 10          # D-1 10:00 제출(1차)
FEATURE_REF_HOUR = 9     # D-1 09:00까지의 관측만 사용(10시는 미완결이라 제외)
MAX_TARGET_HOUR = 21     # NWP 리드 15~36h 범위 안(D 00~21시)
# ★08-24 정정★: v1은 이상구간을 08-15~10-20으로 잠정 설정했으나, v5
# 자체진단(결측구간_목록.csv)이 확인한 실제 인버터5 결함종료일은 11-16.
ANOMALY = (pd.Timestamp("2025-08-15"), pd.Timestamp("2025-11-16"))

# ★08-24 정정 6폴드★(이상구간 08-15~11-16, 가을 11-17 시작) — 다른
# 스크립트가 이 KPX day-ahead 모델을 기준으로 재판정할 때 재사용하도록
# 모듈 상수로 뺐다(원래 main() 내부 지역변수였음, 로직 변경 없음).
WINDOWS = [
    ("1_여름", "2025-06-01", "2025-08-14"),
    ("2_이상구간(결함구간포함)", "2025-08-15", "2025-11-16"),
    ("3_가을_결함종료후", "2025-11-17", "2025-12-14"),
    ("4_겨울", "2025-12-15", "2026-02-14"),
    ("5_봄", "2026-02-15", "2026-04-14"),
    ("6_초여름", "2026-04-15", "2026-08-04"),
]

SETTLE_RULES = {
    "신기준(2025-06-01~, 태양광)": [(4.0, 4), (6.0, 3)],
    "구기준(참고, 폐지됨)": [(6.0, 4), (8.0, 3)],
}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("kpx3_harness", "backtest_harness_v1_2026-08-20밤.py")
clearsky = _load("kpx3_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
improvement = _load("kpx3_round2", "model_improvement_round2_v1_2026-08-21.py")


def _safe(text: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in text)


def asof_values(s: pd.Series, at: pd.DatetimeIndex) -> np.ndarray:
    """`at`의 각 시각 이하에서 가장 최근의 non-NaN 값(asof). v1과 동일."""
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    uniq = pd.DatetimeIndex(pd.unique(at)).sort_values()
    merged = s.reindex(s.index.union(uniq)).ffill().reindex(uniq)
    return merged.reindex(at).to_numpy()


def build_kpx_frame(src: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    """D-1 10:00 발행 → D일 00~21시 프로필(v1과 동일 로직 + 목표 가용인버터수 1컬럼 추가)."""
    idx = src.index
    targets = idx[(idx.hour <= MAX_TARGET_HOUR)]
    target_day = targets.normalize()
    issue_at = target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
    ref_at = target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=FEATURE_REF_HOUR)

    out = pd.DataFrame(index=targets)
    out["_발행시각"] = issue_at
    out["_발행기준시각"] = ref_at
    out["리드타임_h"] = (targets - issue_at).total_seconds() / 3600.0

    obs_cols = [c for c in candidate_cols
                if c in harness.sel.STARTLABEL_COLUMNS or c in harness.sel.OBSERVED_COLUMNS]
    for c in obs_cols:
        out[c] = asof_values(src[c], ref_at)

    power = src["plant_output_kw"].dropna()
    stale = asof_values(pd.Series(power.index.astype("int64"), index=power.index), ref_at)
    out["_관측지연_h"] = (ref_at.astype("int64").to_numpy() - stale) / 3.6e12

    for hours in [0, 1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = asof_values(power, ref_at - pd.Timedelta(hours=hours))
    for hours in [6, 24]:
        for tag, r in (("이동평균", power.rolling(f"{hours}h").mean()),
                       ("이동표준편차", power.rolling(f"{hours}h").std())):
            out[f"발전출력_{hours}시간{tag}_kW"] = asof_values(r, ref_at)

    for c in candidate_cols:
        if c in harness.sel.FORECAST_COLUMNS:
            out[c] = src[c].reindex(targets).to_numpy()

    elev = src["solar_elevation_deg"].reindex(targets)
    out["목표_태양고도_deg"] = elev.to_numpy()
    minute = targets.hour * 60 + targets.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * targets.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * targets.dayofyear / 365.25)

    out["목표_발전출력_kW"] = src["plant_output_kw"].reindex(targets).to_numpy()
    out["목표_낮시간"] = (out["목표_태양고도_deg"] > 0).astype(float)
    # ★3차 신규★: 공식 정책B 적용을 위한 대상시각 가용인버터수.
    out["_목표_가용인버터수"] = src["inverters_available"].reindex(targets).to_numpy()

    same_hour_prev = []
    for t, r in zip(targets, ref_at):
        cand = t - pd.Timedelta(days=1)
        if cand > r:
            cand = t - pd.Timedelta(days=2)
        same_hour_prev.append(cand)
    same_hour_prev = pd.DatetimeIndex(same_hour_prev)
    out["_지속성_기준시각"] = same_hour_prev
    out["_지속성_단순_kW"] = asof_values(power, same_hour_prev)

    cs_all = pd.Series(
        np.clip(clearsky.clear_sky_ghi(src["solar_elevation_deg"].to_numpy()), 1e-6, None),
        index=src.index)
    cs_prev = cs_all.reindex(same_hour_prev).to_numpy()
    cs_tgt = cs_all.reindex(targets).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa_prev = out["_지속성_단순_kW"].to_numpy() / np.where(cs_prev > 1e-3, cs_prev, np.nan)
    out["_지속성_청천_kW"] = kappa_prev * cs_tgt
    return out


def settle(err_pct: np.ndarray, actual_kwh: np.ndarray, rule: list) -> tuple[np.ndarray, np.ndarray]:
    (t1, p1), (t2, p2) = rule
    unit = np.where(err_pct <= t1, p1, np.where(err_pct <= t2, p2, 0.0))
    return unit, unit * actual_kwh


def main(mode: str | None = None) -> None:
    mode = mode or OFFICIAL_MODE
    if mode not in MODE_OUT:
        raise SystemExit(f"알 수 없는 모드: {mode} (가능: {list(MODE_OUT)})")
    OUT = ROOT / "outputs" / MODE_OUT[mode]
    MODEL_DIR = OUT / "fold_models"
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    use_kappa = mode == "kappa_cluster"
    print(f"모드: {mode}"
          f"{' (공식 — 청천지수 카파 변환 + 날씨군집화)' if use_kappa else ' (구 공식 — raw 타깃 + 구조선택)'}\n")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])  # 219(공식)
    seed = int(config["random_seed"])

    src = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                      parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    v5_1h = pd.read_parquet(V5_DIR / "집계_1시간_자료_v5.parquet")
    common = src.index.intersection(v5_1h.index)
    src = src.loc[common].copy()
    src["plant_output_kw"] = v5_1h.loc[common, "발전출력_kW"]
    src["inverters_available"] = v5_1h.loc[common, "가용인버터수"]

    candidate_cols = harness.FEATURE_SETS["전체후보"]
    frame = build_kpx_frame(src, candidate_cols)
    # ★공식(kappa_cluster) 모드★: 청천지수(카파) 타깃을 쓰기 위한 파생컬럼.
    # 태양고도 기반 청천 GHI는 `ultra_short_clearsky_v1`(초단기 +3h·+4h가
    # 이미 공식으로 쓰는 것)을 그대로 재사용한다.
    frame["_청천_kW"] = np.clip(
        capacity_kw * clearsky.clear_sky_ghi(frame["목표_태양고도_deg"].to_numpy()) / 1000.0, 1e-3, None)
    frame["_카파"] = frame["목표_발전출력_kW"] / frame["_청천_kW"]
    windows = WINDOWS

    base_cols = [c for c in frame.columns
                 if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    rows, audits, coverage = [], [], []
    daylight = frame[frame["목표_낮시간"] > 0]

    for fold, s, e in windows:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        all_target = daylight[(daylight.index >= start) & (daylight.index < end)]
        full_mask_all = daylight["_목표_가용인버터수"] >= N_INVERTERS
        train_all = daylight[(daylight.index < start) & full_mask_all]  # 공식B: 학습은 완전가용만
        test_all = all_target[full_mask_all.loc[all_target.index]]      # 평가도 완전가용만(정책무관 고정)
        cov = len(test_all) / len(all_target) * 100 if len(all_target) else 0.0
        coverage.append({"폴드": fold, "전체대상행수": len(all_target), "완전가용시험행수": len(test_all),
                         "커버리지_pct": round(cov, 2)})
        if len(train_all) < 200 or len(test_all) < 30:
            print(f"[{fold}] 표본부족(train={len(train_all)}, test={len(test_all)}, 커버리지={cov:.1f}%) — 건너뜀")
            continue
        chosen = harness.select_features_in_fold(
            train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
        if use_kappa:
            required += ["_청천_kW", "_카파"]
        train = train_all.dropna(subset=required)
        # 시험행은 카파 정답이 없어도 되지만(예측은 청천으로 복원), 청천값은 필요
        test_required = ([c for c in features if c not in harness.NATIVE_MISSING_OK]
                         + ["목표_발전출력_kW"] + (["_청천_kW"] if use_kappa else []))
        test = test_all.dropna(subset=test_required)
        if len(train) < 200 or len(test) < 30:
            print(f"[{fold}] 결측제거 후 표본부족 — 건너뜀")
            continue

        target = "_카파" if use_kappa else "목표_발전출력_kW"
        target_mode = "kappa" if use_kappa else "raw"
        params, structure_name = improvement.choose_structure_kfold(
            "단기", train, features, target, target_mode, capacity_kw, seed)

        # ★공식(kappa_cluster)★: 날씨군집화 특성을 추가한다. 경계·KMeans는
        # 학습폴드에서만 적합하고 시험폴드는 transform만 한다(누출 없음).
        if use_kappa:
            train_fit, test_fit, fit_features, cluster_cols = improvement.add_weather_clusters(
                train, test, features, seed)
        else:
            train_fit, test_fit, fit_features, cluster_cols = train, test, features, []

        model = improvement.make_model("단기", seed, params)
        model.fit(train_fit[fit_features], train_fit[target])

        artifact = MODEL_DIR / _safe(f"KPX3_{fold}.joblib")
        before = np.asarray(model.predict(test_fit[fit_features]), float)
        joblib.dump({"mode": f"KPX_D-1_10시_1회제출(v5·공식B·{mode})", "fold": fold,
                     "features": fit_features, "target": target, "target_mode": target_mode,
                     "군집원천특성": cluster_cols,
                     "params": params, "structure_name": structure_name, "정책": "B_구간제외",
                     "용량프로필": "inverter_registered_sum_219",
                     "train_end": str(train.index.max()), "test_start": str(test.index.min()),
                     "model": model}, artifact)
        after = np.asarray(joblib.load(artifact)["model"].predict(test_fit[fit_features]), float)
        diff = float(np.max(np.abs(before - after))) if len(before) else 0.0

        # 카파 예측은 청천 GHI를 곱해 kW로 되돌린다
        pred = after * test["_청천_kW"].to_numpy() if use_kappa else after
        pred = np.clip(pred, 0, capacity_kw)
        for i, t in enumerate(test.index):
            r = test.iloc[i]
            rows.append({
                "plant_id": PLANT_ID, "폴드": fold,
                "발행시각": r["_발행시각"], "대상시각": t,
                "리드타임_h": r["리드타임_h"], "관측지연_h": r["_관측지연_h"],
                "실제_kW": r["목표_발전출력_kW"], "예측_kW": pred[i],
                "지속성_단순_kW": r["_지속성_단순_kW"], "지속성_청천_kW": r["_지속성_청천_kW"],
                "이상구간": bool(ANOMALY[0] <= t.normalize() <= ANOMALY[1]),
                "모델파일": str(artifact),
            })
        audits.append({
            "폴드": fold, "학습최종대상시각": train.index.max(), "시험최초대상시각": test.index.min(),
            "학습_시험_분리통과": bool(train.index.max() < test.index.min()),
            "발행시각_모두_대상일전날10시": bool((test["_발행시각"] < test.index).all()),
            "최대리드타임_h": float(test["리드타임_h"].max()),
            "최소리드타임_h": float(test["리드타임_h"].min()),
            "시험정답_특성포함없음": ("목표_발전출력_kW" not in fit_features
                                and "_카파" not in fit_features),
            "모델재적재_최대차이": diff, "모델재적재_동일": bool(diff <= 1e-12),
            "선택구조": structure_name, "특성수": len(fit_features),
            "타깃": target, "군집특성수": len(cluster_cols), "모델파일": str(artifact),
        })
        print(f"[{fold}] 학습 {len(train):,} / 시험 {len(test):,}(커버리지 {cov:.1f}%) / 구조 {structure_name} / 재적재차이 {diff:.2e}")

    df = pd.DataFrame(rows)
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "커버리지_폴드별.csv", index=False, encoding="utf-8-sig")
    if not len(df):
        raise SystemExit("평가 행이 없다 — 입력·폴드 정의 확인 필요")

    recs = []
    for cap in (219.0, 240.0):
        thr = 0.1 * cap
        d = df.copy()
        d["설비용량_kW"] = cap
        d["이용률_pct"] = d["실제_kW"] / cap * 100
        d["정산대상"] = d["실제_kW"] >= thr
        d["오차율_pct"] = (d["실제_kW"] - d["예측_kW"]).abs() / cap * 100
        for rname, rule in SETTLE_RULES.items():
            sub = d[d["정산대상"]]
            unit, amt = settle(sub["오차율_pct"].to_numpy(), sub["실제_kW"].to_numpy(), rule)
            for anom_label, mask in (("이상구간포함", np.ones(len(sub), bool)),
                                     ("이상구간제외", ~sub["이상구간"].to_numpy())):
                if mask.sum() == 0:
                    continue
                e = sub["오차율_pct"].to_numpy()[mask]
                recs.append({
                    "설비용량_kW": cap, "정산기준": rname, "이상구간": anom_label,
                    "정산대상_시각수": int(mask.sum()),
                    "전체_시각수": int(len(d) if anom_label == "이상구간포함" else (~d["이상구간"]).sum()),
                    "정산대상_비율_pct": round(mask.sum() / (len(d) if anom_label == "이상구간포함"
                                                       else (~d["이상구간"]).sum()) * 100, 2),
                    "평균NMAE_pct": round(float(e.mean()), 3),
                    f"통과_{rule[0][0]:.0f}pct이하_비율": round(float((e <= rule[0][0]).mean() * 100), 1),
                    f"통과_{rule[0][0]:.0f}~{rule[1][0]:.0f}pct_비율": round(
                        float(((e > rule[0][0]) & (e <= rule[1][0])).mean() * 100), 1),
                    "정산탈락_비율_pct": round(float((e > rule[1][0]).mean() * 100), 1),
                    "예상정산액_원": round(float(amt[mask].sum()), 0),
                })
    settle_df = pd.DataFrame(recs)

    perf = []
    for anom_label in ("이상구간포함", "이상구간제외"):
        base = df if anom_label == "이상구간포함" else df[~df["이상구간"]]
        for scope in ("낮시간만", "24시간환산"):
            y = base["실제_kW"].to_numpy(); p = base["예측_kW"].to_numpy()
            rmse = float(np.sqrt(np.mean((y - p) ** 2)))
            mae = float(np.mean(np.abs(y - p)))
            if scope == "24시간환산":
                day_slots = len(base) / max(base["대상시각"].dt.normalize().nunique(), 1)
                f = np.sqrt(day_slots / 22.0)
                rmse *= f; mae *= f
            for cap in (219.0, 240.0):
                perf.append({"이상구간": anom_label, "평가범위": scope, "설비용량_kW": cap,
                             "n": len(base), "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
                             "nMAE_pct": round(mae / cap * 100, 3),
                             "nRMSE_pct": round(rmse / cap * 100, 3)})
    perf_df = pd.DataFrame(perf)

    sk = []
    for anom_label in ("이상구간포함", "이상구간제외"):
        base = df if anom_label == "이상구간포함" else df[~df["이상구간"]]
        y = base["실제_kW"].to_numpy(); p = base["예측_kW"].to_numpy()
        for bname, bcol in (("단순지속성(직전 동일시각 실측)", "지속성_단순_kW"),
                            ("청천지수지속성(smart)", "지속성_청천_kW")):
            b = base[bcol].to_numpy()
            ok = ~np.isnan(b)
            if ok.sum() < 30:
                continue
            f = lambda a, c: float(np.sqrt(np.mean((a - c) ** 2)))
            rm, rb = f(y[ok], p[ok]), f(y[ok], np.clip(b[ok], 0, capacity_kw))
            ma, mb = float(np.mean(np.abs(y[ok] - p[ok]))), float(np.mean(np.abs(y[ok] - np.clip(b[ok], 0, capacity_kw))))
            sk.append({"이상구간": anom_label, "기준선": bname, "n": int(ok.sum()),
                       "모델_RMSE_kW": round(rm, 3), "기준선_RMSE_kW": round(rb, 3),
                       "Skill_RMSE": round(1 - rm / rb, 4),
                       "모델_MAE_kW": round(ma, 3), "기준선_MAE_kW": round(mb, 3),
                       "Skill_MAE": round(1 - ma / mb, 4)})
    skill_df = pd.DataFrame(sk)

    df.to_csv(OUT / "행단위_예측정답_정산.csv", index=False, encoding="utf-8-sig")
    perf_df.to_csv(OUT / "요약_수평별.csv", index=False, encoding="utf-8-sig")
    settle_df.to_csv(OUT / "요약_정산시뮬레이션.csv", index=False, encoding="utf-8-sig")
    skill_df.to_csv(OUT / "요약_기준선비교.csv", index=False, encoding="utf-8-sig")
    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(OUT / "누출_및_재적재_감사.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("\n=== 커버리지(정책B 적용 후, 폴드별) ===")
    print(coverage_df.to_string(index=False))
    print("\n=== 감사 ===")
    print(audit_df[["폴드", "학습_시험_분리통과", "발행시각_모두_대상일전날10시",
                    "최소리드타임_h", "최대리드타임_h", "모델재적재_동일", "선택구조"]].to_string(index=False))
    print("\n=== 성능(평가범위·용량·이상구간별) ===")
    print(perf_df.to_string(index=False))
    print("\n=== 기준선 대비 스킬 ===")
    print(skill_df.to_string(index=False))
    print("\n=== 정산 시뮬레이션(시간대별 판정) ===")
    print(settle_df.to_string(index=False))

    summary = {
        "실행유형": f"KPX 예측제도 재현(v5·공식B·219kW, D-1 10:00 1회 제출 → 익일 00~21시 프로필) — 모드={mode}",
        "모델구성": ("청천지수(카파) 타깃 + 폴드내부 구조선택 + 날씨군집화 특성(★08-24 공식 채택★, "
                  "근거: kpx_dayahead_final_verify_v1_2026-08-24 — 재현성 정확일치·시드4종 전부 채택)"
                  if use_kappa else "raw 타깃 + 폴드내부 구조선택(구 공식)"),
        "라이브_API_호출": False,
        "정책": "B_구간제외(학습=완전가용만, 시험=항상 완전가용만 고정)",
        "발행시각": "D-1 10:00 KST(1차 제출만, 보수적)",
        "대상시각": "D 00:00~21:00(22~23시는 NWP 리드범위 밖·항상 야간이라 제외)",
        "리드타임_h": [float(df["리드타임_h"].min()), float(df["리드타임_h"].max())],
        "폴드": [w[0] for w in windows],
        "이상구간_경계_정정": "08-15~10-20(v1 잠정) → 08-15~11-16(v5 확정 결함종료일 기준)",
        "이상구간_포함": True,
        "평가행수": int(len(df)),
        "감사_전부통과": bool(audit_df["학습_시험_분리통과"].all()
                          and audit_df["모델재적재_동일"].all()
                          and audit_df["발행시각_모두_대상일전날10시"].all()),
        "적용정산기준": "2025-06-01 개정 신기준(≤4%→4원, 4~6%→3원, >6%→0). 구기준은 참고용만.",
        "한계": "1차(10시) 제출만 재현했고 2차(17시) 제출은 반영 안 함. "
                "설비용량은 219kW(공식)·240kW(민감도) 양쪽 산출. "
                "구조는 동일 6계절을 보며 선택된 것이라 완전 독립검증은 아니다. "
                "정책B 적용으로 이상구간(결함기간) 폴드의 시험 커버리지가 급감할 수 있음(커버리지_폴드별.csv 참고, 은폐 안 함).",
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    if not summary["감사_전부통과"]:
        print("★감사 실패 — 확인 필요★")
        raise SystemExit(1)


if __name__ == "__main__":
    # 인자 없으면 공식 모드(kappa_cluster). `raw`를 주면 구 공식을 재현한다.
    main(sys.argv[1] if len(sys.argv) > 1 else None)
