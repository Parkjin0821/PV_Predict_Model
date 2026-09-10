# -*- coding: utf-8 -*-
"""백테스트 하네스 — 데이터셋·특성세트·모델을 갈아끼우며 비교하는 공용 틀.

## 왜 만드는가 (08-20 밤, 사용자 확정)
NWP 고정-tm 재수집(Codex 진행 중)이 끝나면 익일예측 계열을 전부 다시
검증해야 한다. 그때 매번 스크립트를 새로 짜지 않도록, **바꿔 끼울 축을
인자로 빼둔 공용 하네스**를 미리 만들어 둔다.

바꿔 끼울 수 있는 축 3개:
  1. 데이터셋 경로        (누출 NWP → 고정-tm NWP)
  2. 특성세트             (폴드 안에서 재선택 or 고정 리스트)
  3. 모델 후보 + 앙상블   (LightGBM / XGBoost / 조합)

## 누출 방지 설계 (이 프로젝트가 08-20에 두 번 데인 부분)
- **특성선택을 폴드 안에서 수행**: `feature_mode="refit"`이면 각 폴드의
  학습구간만으로 상관계수를 다시 계산해 임계값 필터를 적용한다. 시험구간을
  본 적 없는 상태로 특성이 정해진다.
- **앙상블 가중치도 OOF에서만 최적화**: 시험구간 성능을 보고 비율을
  정하면 그 자체가 누출이다. `model_common.optimize_nonnegative_weights`를
  각 폴드의 학습구간 내부 홀드아웃(마지막 20%)에서 구한 뒤 시험구간에
  적용한다.
- **누출 플래그**: 데이터셋이 아직 누출 NWP를 쓰는 상태면 `leakage_note`를
  결과 JSON·CSV에 그대로 박아 넣어, 나중에 이 숫자를 실수로 인용하지
  못하게 한다.

## 시간정렬 규약 (08-20 밤 v3에서 확정, 그대로 재사용)
- 장비텔레메트리(시작시각 라벨, 타깃과 동일 규약): shift(1)
- ASOS 관측(종료시각 라벨): shift(0)
- NWP·격자 예보(종료시각 라벨이지만 사전발표): shift(-horizon)
- 타깃: `plant_output_kw.shift(-(horizon-1))`

## 사용 예
    python backtest_harness_v1_2026-08-20밤.py --horizon 24 --feature-mode refit
    python backtest_harness_v1_2026-08-20밤.py --horizon 1 --models LightGBM XGBoost --ensemble
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_common import optimize_nonnegative_weights  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sel_v3", ROOT / "select_features_by_correlation_threshold_v3_2026-08-20밤.py"
)
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

# ── 갈아끼우는 지점 1: 데이터셋 ────────────────────────────────────────────
DATASETS = {
    "v2_누출NWP": {
        "path": Path(
            r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
            r"\gwangju_1hour_model_dataset_official_v2_2026-08-20밤.csv"
        ),
        "leakage_note": (
            "★무효★ NWP가 목표시각 직전 런(리드타임 3~6h)으로 수집돼 "
            "익일예측(+24h/+48h)에서 100% 미래정보 누출. +1h만 유효."
        ),
    },
    "v3_고정tm": {
        "path": Path(
            r"C:\Users\u-cube\JIN\태양광 발전\환경요인_기상청_ASOS\processed"
            r"\gwangju_1hour_model_dataset_official_v3_fixed_tm_2026-08-21.csv"
        ),
        "leakage_note": "",  # 08-21: 고정 tm(D 09:00 KST, 리드타임 15~36h) 재수집 완료. 누출 없음.
    },
}

# ── 갈아끼우는 지점 2: 특성세트 ────────────────────────────────────────────
FINAL_14 = [
    "plant_input_power_kw", "DSWRF", "기상청관측_일조시간_hr", "mean_power_factor", "REH",
    "mean_input_voltage_v", "추정_출력온도", "기상청관측_상대습도_pct", "기상청관측_전운량_pct",
    "DSWRFLX_bsrn정제", "LCDC", "TCDC", "POP", "SKY",
]
LEAKY_NWP = ["DSWRF", "DSWRFLX_bsrn정제", "LCDC", "TCDC", "추정_출력온도"]
NATIVE_MISSING_OK = {"DSWRFLX_bsrn정제"}

# ── Blockdata 실배포 가능성(08-20 밤 확정, 데이터로 재검증되는 게 아니라
#    "실시간 API에 필드가 아예 없다"는 하드 제약이라 상관분석과 별개로 고정 제외) ──
BLOCKDATA_EXCLUDED = {"mean_inverter_temperature_c"}

# ── 다중공선성 가지치기 기준(08-20 밤 확정, O'Brien 2007 근거) ──
VIF_SEVERE = 10.0
PAIR_CORR_HIGH = 0.8

FEATURE_SETS = {
    "final14": FINAL_14,
    "final14_누출제외": [c for c in FINAL_14 if c not in LEAKY_NWP],
    "전체후보": sel.CANDIDATE_COLUMNS,
}


# ── 갈아끼우는 지점 3: 모델 ────────────────────────────────────────────────
def make_model(name: str, seed: int):
    if name == "LightGBM":
        return LGBMRegressor(
            n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=30,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=0.3,
            random_state=seed, n_jobs=4, verbosity=-1,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=220, learning_rate=0.04, max_depth=6, min_child_weight=5,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=4, verbosity=0,
        )
    raise ValueError(f"알 수 없는 모델: {name}")


@dataclass
class BacktestConfig:
    dataset_key: str = "v2_누출NWP"
    feature_set: str = "final14"
    feature_mode: str = "fixed"      # "fixed" | "refit"(폴드 내 상관분석 재수행)
    refit_threshold: float = 0.3
    prune_multicollinearity: bool = False   # 08-21: VIF+쌍상관 가지치기(폴드 내)
    deploy_filter: bool = False              # 08-21: Blockdata 배포불가 특성 제외
    horizon: int = 1
    models: list[str] = field(default_factory=lambda: ["LightGBM", "XGBoost"])
    ensemble: bool = False


def build_frame(df: pd.DataFrame, horizon: int, candidate_cols: list[str]) -> pd.DataFrame:
    """v3 정렬 규약을 수평 H로 일반화한 특성·타깃 프레임."""
    out = pd.DataFrame(index=df.index)
    for c in candidate_cols:
        if c in sel.STARTLABEL_COLUMNS:
            out[c] = df[c].shift(1)
        elif c in sel.OBSERVED_COLUMNS:
            out[c] = df[c]
        elif c in sel.FORECAST_COLUMNS:
            out[c] = df[c].shift(-horizon)
        else:
            raise ValueError(f"분류되지 않은 특성: {c}")

    power = df["plant_output_kw"].shift(1)
    for hours in [1, 2, 3, 6, 24]:
        out[f"발전출력_{hours}시간전_kW"] = power.shift(hours - 1)
    for hours in [6, 24]:
        out[f"발전출력_{hours}시간이동평균_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).mean()
        out[f"발전출력_{hours}시간이동표준편차_kW"] = power.rolling(hours, min_periods=max(3, hours // 2)).std()
    out["목표_태양고도_deg"] = df["solar_elevation_deg"].shift(-(horizon - 1))
    target_time = df.index + pd.to_timedelta(horizon - 1, unit="h")
    minute = target_time.hour * 60 + target_time.minute
    out["일주기_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["일주기_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["연주기_sin"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    out["연주기_cos"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)

    out["목표_발전출력_kW"] = df["plant_output_kw"].shift(-(horizon - 1))
    out["목표_낮시간"] = (df["solar_elevation_deg"].shift(-(horizon - 1)) > 0).astype(float)
    out["_지속성_직전출력_kW"] = df["plant_output_kw"].shift(1)
    return out


def compute_vif(x: pd.DataFrame) -> pd.Series:
    x = (x - x.mean()) / x.std(ddof=0)
    vifs = {}
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        if others.shape[1] == 0:
            vifs[col] = 1.0
            continue
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs)


def prune_multicollinearity(train: pd.DataFrame, cols: list[str], target_corr: dict[str, float]) -> list[str]:
    """★08-21 신규★ 폴드 학습구간 안에서 VIF+쌍상관(둘 다 걸리는 것만,
    O'Brien 2007 원칙) 가지치기를 수행한다. 상관분석과 마찬가지로 시험구간을
    보지 않고 학습구간만으로 계산한다."""
    if len(cols) < 2:
        return cols
    x = train[cols].dropna()
    if len(x) < 30:
        return cols
    try:
        vif = compute_vif(x)
    except Exception:
        return cols
    corr_matrix = x.corr(method="pearson")
    drop_candidates = set()
    severe_vif = set(vif[vif >= VIF_SEVERE].index)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr_matrix.loc[a, b]
            if pd.notna(r) and abs(r) >= PAIR_CORR_HIGH:
                weaker = a if abs(target_corr.get(a, 0)) < abs(target_corr.get(b, 0)) else b
                if weaker in severe_vif:
                    drop_candidates.add(weaker)
    return [c for c in cols if c not in drop_candidates]


def select_features_in_fold(
    train: pd.DataFrame, candidate_cols: list[str], threshold: float,
    apply_multicollinearity: bool = False, apply_deploy_filter: bool = False,
) -> list[str]:
    """★누출 방지 핵심★ 학습구간만으로 상관계수를 계산해 특성을 고른다.

    08-21: 다중공선성 가지치기·Blockdata 배포가능성 필터를 이 함수 안에서
    같이 처리하도록 확장(전날 밤 참고치가 이 두 단계를 빠뜨렸던 문제 수정)."""
    y = train["목표_발전출력_kW"]
    target_corr: dict[str, float] = {}
    keep = []
    for c in candidate_cols:
        sub = pd.concat([train[c], y], axis=1).dropna()
        if len(sub) < 30:
            continue
        r = np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1]
        if np.isfinite(r):
            target_corr[c] = r
            if abs(r) >= threshold:
                keep.append(c)

    if apply_multicollinearity:
        keep = prune_multicollinearity(train, keep, target_corr)

    if apply_deploy_filter:
        keep = [c for c in keep if c not in BLOCKDATA_EXCLUDED]

    return keep


def metrics(y: np.ndarray, p: np.ndarray, capacity_kw: float, ref_rmse: float | None = None) -> dict:
    e = y - p
    mae = float(np.abs(e).mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    denom = float(np.abs(y).sum())
    out = {
        "n": int(len(y)),
        "MAE_kW": round(mae, 3), "RMSE_kW": round(rmse, 3),
        "nMAE_pct": round(mae / capacity_kw * 100, 3),
        "nRMSE_pct": round(rmse / capacity_kw * 100, 3),
        "WAPE_pct": round(float(np.abs(e).sum() / denom * 100), 2) if denom > 0 else None,
    }
    if ref_rmse and ref_rmse > 0:
        out["SkillScore"] = round(1 - rmse / ref_rmse, 4)
    return out


def run_backtest(cfg: BacktestConfig) -> dict:
    app_config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(app_config["site"]["capacity_kw"])
    seed = int(app_config["random_seed"])
    windows = app_config["cross_validation_windows"]

    ds = DATASETS[cfg.dataset_key]
    df = pd.read_csv(ds["path"], parse_dates=["time"], low_memory=False).set_index("time").sort_index()

    candidate_cols = FEATURE_SETS[cfg.feature_set]
    frame = build_frame(df, cfg.horizon, candidate_cols)
    daylight = frame[frame["목표_낮시간"] > 0]

    base_cols = [c for c in frame.columns
                 if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]

    fold_results = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fold_name = f"{i}_{w.get('_계절', '')}"

        train_all = daylight[daylight.index < start]
        test_all = daylight[(daylight.index >= start) & (daylight.index <= end)]
        if len(train_all) < 200 or len(test_all) < 30:
            continue

        # ── 특성선택: 폴드 학습구간 안에서만 ──
        if cfg.feature_mode == "refit":
            tr_for_sel = train_all.dropna(subset=["목표_발전출력_kW"])
            chosen = select_features_in_fold(
                tr_for_sel, candidate_cols, cfg.refit_threshold,
                apply_multicollinearity=cfg.prune_multicollinearity,
                apply_deploy_filter=cfg.deploy_filter,
            )
        else:
            chosen = list(candidate_cols)
            if cfg.deploy_filter:
                chosen = [c for c in chosen if c not in BLOCKDATA_EXCLUDED]
        feature_cols = base_cols + chosen

        required = [c for c in feature_cols if c not in NATIVE_MISSING_OK] + [
            "목표_발전출력_kW", "_지속성_직전출력_kW"
        ]
        train = train_all.dropna(subset=required)
        test = test_all.dropna(subset=required)
        if len(train) < 200 or len(test) < 30:
            continue

        y_train = train["목표_발전출력_kW"].to_numpy()
        y_test = test["목표_발전출력_kW"].to_numpy()
        pers = np.clip(test["_지속성_직전출력_kW"].to_numpy(), 0, capacity_kw)
        ref = metrics(y_test, pers, capacity_kw)
        ref_rmse = ref["RMSE_kW"]

        row = {"폴드": fold_name, "학습표본": len(train), "시험표본": len(test),
               "선택특성수": len(chosen), "선택특성": chosen, "지속성": ref}

        # ── 앙상블 가중치용 내부 홀드아웃(학습구간 마지막 20%) ──
        cut = int(len(train) * 0.8)
        inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]

        test_preds, inner_preds = {}, {}
        for name in cfg.models:
            m = make_model(name, seed)
            m.fit(train[feature_cols], y_train)
            p = np.clip(m.predict(test[feature_cols]), 0, capacity_kw)
            test_preds[name] = p
            row[name] = metrics(y_test, p, capacity_kw, ref_rmse)

            if cfg.ensemble and len(inner_va) > 30:
                m2 = make_model(name, seed)
                m2.fit(inner_tr[feature_cols], inner_tr["목표_발전출력_kW"])
                inner_preds[name] = np.clip(m2.predict(inner_va[feature_cols]), 0, capacity_kw)

        if cfg.ensemble and len(inner_preds) >= 2:
            names = list(inner_preds.keys())
            matrix = np.column_stack([inner_preds[n] for n in names])
            res = optimize_nonnegative_weights(inner_va["목표_발전출력_kW"].to_numpy(), matrix)
            weights = np.asarray(res["가중치"], float)
            ens = np.clip(np.column_stack([test_preds[n] for n in names]) @ weights, 0, capacity_kw)
            row["앙상블"] = metrics(y_test, ens, capacity_kw, ref_rmse)
            row["앙상블_가중치"] = {n: round(float(v), 4) for n, v in zip(names, weights)}

        fold_results.append(row)

    keys = ["지속성"] + list(cfg.models) + (["앙상블"] if cfg.ensemble else [])
    summary = []
    for k in keys:
        vals = [r[k] for r in fold_results if k in r]
        if not vals:
            continue
        summary.append({
            "구성": k, "폴드수": len(vals),
            "평균MAE_kW": round(float(np.mean([v["MAE_kW"] for v in vals])), 3),
            "평균RMSE_kW": round(float(np.mean([v["RMSE_kW"] for v in vals])), 3),
            "평균nMAE_pct": round(float(np.mean([v["nMAE_pct"] for v in vals])), 3),
            "평균nRMSE_pct": round(float(np.mean([v["nRMSE_pct"] for v in vals])), 3),
            "평균WAPE_pct": round(float(np.mean([v["WAPE_pct"] for v in vals if v.get("WAPE_pct")])), 2),
            "평균SkillScore": round(float(np.mean([v["SkillScore"] for v in vals if v.get("SkillScore") is not None])), 4)
            if any(v.get("SkillScore") is not None for v in vals) else None,
        })

    return {
        "설정": {**cfg.__dict__, "데이터셋경로": str(ds["path"])},
        "누출경고": ds["leakage_note"],
        "폴드별": fold_results,
        "요약": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v2_누출NWP", choices=list(DATASETS))
    ap.add_argument("--feature-set", default="final14", choices=list(FEATURE_SETS))
    ap.add_argument("--feature-mode", default="fixed", choices=["fixed", "refit"])
    ap.add_argument("--refit-threshold", type=float, default=0.3)
    ap.add_argument("--prune-multicollinearity", action="store_true")
    ap.add_argument("--deploy-filter", action="store_true")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=["LightGBM", "XGBoost"])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = BacktestConfig(
        dataset_key=args.dataset, feature_set=args.feature_set,
        feature_mode=args.feature_mode, refit_threshold=args.refit_threshold,
        prune_multicollinearity=args.prune_multicollinearity, deploy_filter=args.deploy_filter,
        horizon=args.horizon, models=args.models, ensemble=args.ensemble,
    )
    result = run_backtest(cfg)

    if result["누출경고"]:
        print("!" * 70)
        print(f"[누출경고] {result['누출경고']}")
        print("!" * 70)
    print(f"\n데이터셋={cfg.dataset_key}  특성={cfg.feature_set}({cfg.feature_mode})  "
          f"다중공선성가지치기={cfg.prune_multicollinearity}  배포필터={cfg.deploy_filter}  "
          f"수평=+{cfg.horizon}h  모델={cfg.models}  앙상블={cfg.ensemble}\n")
    for r in result["폴드별"]:
        print(f"  [{r['폴드']}] 학습{r['학습표본']} 시험{r['시험표본']} 선택특성{r['선택특성수']}개: {sorted(r['선택특성'])}")
    print()
    print(pd.DataFrame(result["요약"]).to_string(index=False))

    tag = args.tag or f"{cfg.dataset_key}_{cfg.feature_set}_{cfg.feature_mode}_h{cfg.horizon}"
    out_dir = ROOT / "outputs" / "백테스트하네스"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(result["요약"]).to_csv(out_dir / f"{tag}_요약.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
