# -*- coding: utf-8 -*-
"""KPX 재생에너지 발전량 예측제도 재현 백테스트 (⑦ 비판적 재검증 후속).

## 왜 이 스크립트가 필요한가
08-21 비판적 재검증에서 **기존 `단기 +24h` 수치를 KPX와 나란히 놓은 것이
잘못**임이 드러났다. KPX는 D-1 고정시각에 **1회 제출로 익일 24시간
프로필 전체**를 내는데, 우리 `+24h`는 발행시각이 07~20시로 흩어진
**롤링 단일지점 예측**이라 과제 구조 자체가 다르다. 이 스크립트는
KPX 방식을 그대로 재현해 처음으로 **같은 과제**에서 평가한다.

## 확인된 KPX 규격(원문 근거)
- **제출**: "10시(1차 제출), 17시(2차 제출)에 익일 예측발전량을 전력거래소에
  제출" — 한국스마트그리드협회 SG표준 `PM-GAP-10` 관련 자료
  (https://www.ksga.org/sgstandard/framework/03/09/04.do).
  **이 스크립트는 보수적으로 1차 제출(D-1 10:00)만 재현한다**(2차 17시
  제출은 정보가 더 많아 유리하므로, 1차만 쓰는 게 하한 추정).
- **10% 필터**: "시간대별 설비이용률이 10% 이상인 시각만 해당시각의
  예측오차율을 측정"(같은 출처). 즉 A_t ≥ 0.1×S 인 시각만 평가·정산.
- **판정 단위**: 등록시험은 "1개월간의 평균 예측오차율 10% 이하",
  운영 중 정산은 **시간대별** 예측오차율로 판정(같은 출처).
  ★기존 분석이 평균 NMAE만 보고 "인센티브 등급"이라 한 것이 오류였다★
- **오차식**: NMAE(%) = (100/n)·Σ|A_t − F_t|/S, A_t ≥ 0.1×S
  (김상진 외, 한국태양에너지학회지 42(6), 173-183, 2022,
  https://doi.org/10.7836/kses.2022.42.6.173 — 국내 15개 발전소 실증).
  시간대별 오차율은 같은 식의 항 하나: |A_t − F_t|/S × 100.
- **★정산단가(2025-06-01 개정, 태양광 즉시 적용)★**:
  오차율 ≤4% → 4원/kWh, 4~6% → 3원/kWh, >6% → 정산 없음.
  **구 기준(≤6% → 4원, 6~8% → 3원)은 폐지됐다** — 전기신문
  (https://www.electimes.com/news/articleView.html?idxno=350873),
  해줌 정리(https://blog.haezoom.com/notice_05/).
  **우리 시험구간(2025-06-01~2026-08-04)은 전부 신 기준 적용 대상이다**
  — 기존 분석이 구 기준(6/8%)으로 판단한 것도 오류였다.
  두 기준 모두 산출해 비교한다.
- 정산액 = 단가 × 해당 시각 실제 발전량(kWh). 시간평균 kW × 1h = kWh.

## 발행 구조와 누출 방지
- **발행시각 = (D-1) 10:00 KST**, 대상 = D일 00~21시.
  D 22~23시는 제외한다 — v3 고정-tm NWP가 D-1 09:00 발행·리드 15~36h라
  D 21시까지만 덮는다(22~23시 값은 같은 날 09시 발행분일 수 있어
  **누출 위험**). 두 시각 모두 광주에서 항상 일몰 이후라 발전량 0이므로
  제외해도 정산·오차에 영향이 없다(일몰 최대 ~19:40 KST).
- **리드타임 14~31시간**(D 00:00 ~ D 21:00 기준).
- 발행시점 가용정보만 사용:
  - 장비 텔레메트리·ASOS 관측: **(D-1) 09:00 이하의 최신값**(asof).
    10:00 시각 자체는 그 시간이 끝나지 않아 쓰지 않는다(보수적).
  - NWP: 대상시각의 값(= D-1 09:00 발행분).
  - 지속성 기준선: **(D-1) 09:00 이하에서 같은 시(hour)의 최신 실측**
    (h≥10이면 자연히 D-2, h<10이면 D-1) — 발행시점에 실제로 알 수 있는
    값만 쓴다.
- 학습행은 항상 `대상일 < 폴드 시작일`. 특성선택도 폴드 학습구간 안에서만.

## 이상구간 포함(2025-08-15~10-20)
기존 5폴드는 이 구간을 통째로 건너뛰었다. 여기서는 **연속 6폴드**로
바꿔 포함하고, 결과를 "포함/제외" 둘 다 보고한다.
(참고: 이 구간은 발전량 평균 2.60kW·결측 42.8%라 10% 필터에서 대부분
탈락한다 — 오차율보다 **정산 대상 시간 수 급감**으로 나타난다.)

## 산출물 (`outputs/KPX재현_백테스트_v1_2026-08-21/`)
- `행단위_예측정답_정산.csv` — 대상시각별 예측·실측·오차율·정산 판정·정산액
- `요약_수평별.csv` — 낮시간/24시간, 240/219kW, 이상구간 포함/제외 전 조합
- `요약_정산시뮬레이션.csv` — 신·구 기준 시간대별 통과율과 예상 정산액
- `요약_기준선비교.csv` — 단순/청천지수 지속성 대비 스킬
- `누출_및_재적재_감사.csv`, `fold_models/`, `요약.json`

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
OUT = ROOT / "outputs" / "KPX재현_백테스트_v1_2026-08-21"
MODEL_DIR = OUT / "fold_models"

PLANT_ID = 6715
PLANT_NAME = "광주 광주시청"

ISSUE_HOUR = 10          # D-1 10:00 제출(1차)
FEATURE_REF_HOUR = 9     # D-1 09:00까지의 관측만 사용(10시는 미완결이라 제외)
MAX_TARGET_HOUR = 21     # NWP 리드 15~36h 범위 안(D 00~21시)
ANOMALY = (pd.Timestamp("2025-08-15"), pd.Timestamp("2025-10-20"))

# 정산 기준. 2025-06-01부터 태양광은 신 기준만 유효하나, 구 기준도 비교용 산출.
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


harness = _load("kpx_harness", "backtest_harness_v1_2026-08-20밤.py")
clearsky = _load("kpx_clearsky", "ultra_short_clearsky_v1_2026-08-21.py")
improvement = _load("kpx_round2", "model_improvement_round2_v1_2026-08-21.py")


def _safe(text: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in text)


def asof_values(s: pd.Series, at: pd.DatetimeIndex) -> np.ndarray:
    """`at`의 각 시각 이하에서 가장 최근의 non-NaN 값(asof).

    `at`은 중복을 포함한다(하루 22개 대상시각이 같은 발행기준시각을
    공유하므로) — 그래서 유니크 시각으로 한 번만 계산한 뒤 되돌려
    매핑한다(중복 인덱스로 reindex하면 pandas가 거부한다).
    """
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    uniq = pd.DatetimeIndex(pd.unique(at)).sort_values()
    merged = s.reindex(s.index.union(uniq)).ffill().reindex(uniq)
    return merged.reindex(at).to_numpy()


def build_kpx_frame(src: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    """D-1 10:00 발행 → D일 00~21시 프로필. 한 행 = (대상시각) 하나.

    같은 날의 22개 행은 **모두 같은 발행시각·같은 발행시점 특성**을
    공유하고, NWP·태양위치·리드타임만 대상시각별로 다르다 —
    이게 KPX 방식(1회 제출로 익일 전체)의 핵심이다.
    """
    idx = src.index
    targets = idx[(idx.hour <= MAX_TARGET_HOUR)]
    target_day = targets.normalize()
    issue_at = target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
    ref_at = target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=FEATURE_REF_HOUR)

    out = pd.DataFrame(index=targets)
    out["_발행시각"] = issue_at
    out["_발행기준시각"] = ref_at
    out["리드타임_h"] = (targets - issue_at).total_seconds() / 3600.0

    # ── 발행시점 관측(장비·ASOS): ref_at 이하의 최신값(asof) ──
    # 실제 운영과 동일하게 "그 시점까지 들어와 있는 가장 최근 값"을 쓴다.
    # 이상구간처럼 결측이 많은 구간도 버리지 않고 평가하기 위함이며,
    # 값이 얼마나 오래된 것인지(_관측지연_h)를 함께 기록해 은폐하지 않는다.
    obs_cols = [c for c in candidate_cols
                if c in harness.sel.STARTLABEL_COLUMNS or c in harness.sel.OBSERVED_COLUMNS]
    for c in obs_cols:
        out[c] = asof_values(src[c], ref_at)

    power = src["plant_output_kw"].dropna()
    # 실측이 얼마나 오래된 값인지(이상구간은 결측 42.8%라 은폐하면 안 됨)
    stale = asof_values(pd.Series(power.index.astype("int64"), index=power.index), ref_at)
    out["_관측지연_h"] = (ref_at.astype("int64").to_numpy() - stale) / 3.6e12

    # 발행기준시각 기준 과거 발전량 래그·이동통계(전부 발행 이전 값)
    for hours in [0, 1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = asof_values(power, ref_at - pd.Timedelta(hours=hours))
    for hours in [6, 24]:
        for tag, r in (("이동평균", power.rolling(f"{hours}h").mean()),
                       ("이동표준편차", power.rolling(f"{hours}h").std())):
            out[f"발전출력_{hours}시간{tag}_kW"] = asof_values(r, ref_at)

    # ── 대상시각 정보: NWP 예보 + 태양위치 + 달력 ──
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

    # ── 지속성 기준선 2종(발행시점에 알 수 있는 값만) ──
    # 같은 시(hour)의 최신 실측: h>=10이면 D-2, h<10이면 D-1이 자연히 선택된다.
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
    """시간대별 정산 판정 → (단가원, 정산액원). 구간 밖이면 0."""
    unit = np.zeros_like(err_pct, dtype=float)
    (t1, p1), (t2, p2) = rule
    unit = np.where(err_pct <= t1, p1, np.where(err_pct <= t2, p2, 0.0))
    return unit, unit * actual_kwh


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    src = pd.read_csv(harness.DATASETS["v3_고정tm"]["path"],
                      parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    candidate_cols = harness.FEATURE_SETS["전체후보"]
    frame = build_kpx_frame(src, candidate_cols)

    # ★이상구간을 포함하는 연속 6폴드★(기존 5폴드는 이 구간을 건너뛰었다)
    windows = [
        ("1_여름", "2025-06-01", "2025-08-14"),
        ("2_이상구간(신규포함)", "2025-08-15", "2025-10-20"),
        ("3_가을", "2025-10-21", "2025-12-14"),
        ("4_겨울", "2025-12-15", "2026-02-14"),
        ("5_봄", "2026-02-15", "2026-04-14"),
        ("6_초여름", "2026-04-15", "2026-08-04"),
    ]

    base_cols = [c for c in frame.columns
                 if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    rows, audits = [], []
    daylight = frame[frame["목표_낮시간"] > 0]

    for fold, s, e in windows:
        start, end = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index < end)]
        if len(train_all) < 200 or len(test_all) < 30:
            print(f"[{fold}] 표본부족 — 건너뜀")
            continue
        chosen = harness.select_features_in_fold(
            train_all.dropna(subset=["목표_발전출력_kW"]), candidate_cols, 0.3, True, True)
        features = base_cols + chosen
        required = [c for c in features if c not in harness.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=required)
        if len(train) < 200 or len(test) < 30:
            print(f"[{fold}] 결측제거 후 표본부족 — 건너뜀")
            continue

        params, structure_name = improvement.choose_structure(
            "단기", train, features, "목표_발전출력_kW", "raw", capacity_kw, seed)
        model = improvement.make_model("단기", seed, params)
        model.fit(train[features], train["목표_발전출력_kW"])

        artifact = MODEL_DIR / _safe(f"KPX_{fold}.joblib")
        before = np.asarray(model.predict(test[features]), float)
        joblib.dump({"mode": "KPX_D-1_10시_1회제출", "fold": fold, "features": features,
                     "params": params, "structure_name": structure_name,
                     "train_end": str(train.index.max()), "test_start": str(test.index.min()),
                     "model": model}, artifact)
        after = np.asarray(joblib.load(artifact)["model"].predict(test[features]), float)
        diff = float(np.max(np.abs(before - after))) if len(before) else 0.0

        pred = np.clip(after, 0, capacity_kw)
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
            "시험정답_특성포함없음": "목표_발전출력_kW" not in features,
            "모델재적재_최대차이": diff, "모델재적재_동일": bool(diff <= 1e-12),
            "선택구조": structure_name, "특성수": len(features), "모델파일": str(artifact),
        })
        print(f"[{fold}] 학습 {len(train):,} / 시험 {len(test):,} / 구조 {structure_name} / 재적재차이 {diff:.2e}")

    df = pd.DataFrame(rows)
    if not len(df):
        raise SystemExit("평가 행이 없다 — 입력·폴드 정의 확인 필요")

    # ── 정산 시뮬레이션(시간대별 판정) ──
    recs = []
    for cap in (240.0, 219.0):
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

    # ── 성능 요약(낮시간 / 24시간 전체, 용량 2종, 이상구간 2종) ──
    perf = []
    for anom_label in ("이상구간포함", "이상구간제외"):
        base = df if anom_label == "이상구간포함" else df[~df["이상구간"]]
        for scope in ("낮시간만", "24시간환산"):
            y = base["실제_kW"].to_numpy(); p = base["예측_kW"].to_numpy()
            rmse = float(np.sqrt(np.mean((y - p) ** 2)))
            mae = float(np.mean(np.abs(y - p)))
            # 24시간 환산: 야간(태양고도<=0)은 실측·예측 모두 0이라 오차 0.
            # 하루 22시각(00~21시) 중 낮시간이 차지하는 비율로 환산한다.
            if scope == "24시간환산":
                day_slots = len(base) / max(base["대상시각"].dt.normalize().nunique(), 1)
                f = np.sqrt(day_slots / 22.0)
                rmse *= f; mae *= f
            for cap in (240.0, 219.0):
                perf.append({"이상구간": anom_label, "평가범위": scope, "설비용량_kW": cap,
                             "n": len(base), "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
                             "nMAE_pct": round(mae / cap * 100, 3),
                             "nRMSE_pct": round(rmse / cap * 100, 3)})
    perf_df = pd.DataFrame(perf)

    # ── 기준선 대비 스킬 ──
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
            rm, rb = f(y[ok], p[ok]), f(y[ok], np.clip(b[ok], 0, 240.0))
            ma, mb = float(np.mean(np.abs(y[ok] - p[ok]))), float(np.mean(np.abs(y[ok] - np.clip(b[ok], 0, 240.0))))
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
        "실행유형": "KPX 예측제도 재현(D-1 10:00 1회 제출 → 익일 00~21시 프로필)",
        "라이브_API_호출": False,
        "발행시각": "D-1 10:00 KST(1차 제출만, 보수적)",
        "대상시각": "D 00:00~21:00(22~23시는 NWP 리드범위 밖·항상 야간이라 제외)",
        "리드타임_h": [float(df["리드타임_h"].min()), float(df["리드타임_h"].max())],
        "폴드": [w[0] for w in windows],
        "이상구간_포함": True,
        "평가행수": int(len(df)),
        "감사_전부통과": bool(audit_df["학습_시험_분리통과"].all()
                          and audit_df["모델재적재_동일"].all()
                          and audit_df["발행시각_모두_대상일전날10시"].all()),
        "적용정산기준": "2025-06-01 개정 신기준(≤4%→4원, 4~6%→3원, >6%→0). 구기준은 참고용만.",
        "한계": "1차(10시) 제출만 재현했고 2차(17시) 제출은 반영 안 함. "
                "설비용량은 240kW(공칭)·219kW(Blockdata 등록합) 양쪽 산출 — 명판 확인 전까지 확정 불가. "
                "구조는 동일 5계절을 보며 선택된 것이라 완전 독립검증은 아니다.",
    }
    (OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    if not summary["감사_전부통과"]:
        print("★감사 실패 — 확인 필요★")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
