# -*- coding: utf-8 -*-
"""김제·영광 D+1을 실제 목표시각 기준 4계절로 재평가한다.

기존 30일 expanding walk-forward와 채택 모델/특성을 그대로 재현하고,
각 시험행을 대표 폴드 계절이 아닌 target_time_kst의 실제 월로 분류한다.
계절 climatology는 매 폴드 학습구간의 같은 계절·같은 시각 평균만 사용한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "김제_영광_D1_4계절_climatology_동일행비교_v1_2026-09-14"
TARGET = "plant_ac_power_kw"
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
CONFIGS = {
    "김제": {
        "dir": ROOT / "김제_준비_2026-09-01",
        "phase": "factor_reverify_v6_hourly_gimje_2026-09-03.py",
        "official": "요인재검증_v6_phase2_2026-09-03/phase2_요약.json",
        "capacity": 1100.0,
    },
    "영광": {
        "dir": ROOT / "영광_준비_2026-09-03",
        "phase": "factor_build_v6_hourly_yeonggwang_2026-09-03.py",
        "official": "요인구축_v6_phase2_2026-09-03/phase2_요약.json",
        "capacity": 634.0,
    },
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def season4(ts: pd.Series) -> pd.Series:
    month = pd.to_datetime(ts).dt.month
    return month.map({1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                      6: "여름", 7: "여름", 8: "여름", 9: "가을",
                      10: "가을", 11: "가을", 12: "겨울"})


def score(g: pd.DataFrame, pred_col: str) -> dict:
    err = g[TARGET] - g[pred_col]
    return {
        "n": int(len(g)),
        "독립일수": int(g["target_day"].nunique()),
        "MAE_kW": float(err.abs().mean()),
        "RMSE_kW": float(np.sqrt(np.mean(err ** 2))),
        "WAPE_pct": float(100 * err.abs().sum() / g[TARGET].abs().sum()),
    }


def run_region(region: str, cfg: dict, common) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    phase = load(f"phase_{region}", cfg["dir"] / cfg["phase"])
    official_path = cfg["dir"] / "outputs" / cfg["official"]
    official = json.loads(official_path.read_text(encoding="utf-8"))
    features = official["최종특성"]
    model_name = official["선정모델(MAE기준)"]
    harness = common._load_harness()
    frame, _ = phase.build_hourly_frame()
    frame = frame.copy()
    frame["target_time_kst"] = pd.to_datetime(frame["target_time_kst"])
    if "target_day" in frame.columns:
        frame["target_day"] = pd.to_datetime(frame["target_day"])
    else:
        frame["target_day"] = frame["target_time_kst"].dt.normalize()
    frame["계절4"] = season4(frame["target_time_kst"])
    frame["target_hour"] = frame["target_time_kst"].dt.hour
    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    fold_owner = phase
    if not hasattr(fold_owner, "expanding_folds_full_coverage"):
        fold_owner = load(f"folds_{region}", phase.V1_SCRIPT)
    folds = fold_owner.expanding_folds_full_coverage(days, 90, 30)
    required = [c for c in features if c not in phase.NATIVE_MISSING_OK]
    rows = []

    for fold_no, (train_days, test_days) in enumerate(folds, start=1):
        if train_days.max() >= test_days.min():
            raise RuntimeError(f"{region} fold {fold_no}: 시간누출")
        train = frame[frame["issue_day"].isin(train_days)].dropna(subset=required + [TARGET]).copy()
        test = frame[frame["issue_day"].isin(test_days)].dropna(subset=required + [TARGET]).copy()
        if len(train) < 100 or len(test) < 5:
            continue
        model = harness.make_model(model_name, 42)
        model.fit(train[features], train[TARGET])
        test["모델예측_kW"] = np.clip(model.predict(test[features]), 0, cfg["capacity"])

        same = train.groupby(["계절4", "target_hour"])[TARGET].mean()
        hour = train.groupby("target_hour")[TARGET].mean()
        global_mean = float(train[TARGET].mean())
        clim = []
        for s, h in zip(test["계절4"], test["target_hour"]):
            v = same.get((s, h), np.nan)
            if pd.isna(v):
                v = hour.get(h, global_mean)
            clim.append(float(v))
        test["climatology예측_kW"] = np.clip(clim, 0, cfg["capacity"])
        test["지역"] = region
        test["폴드"] = fold_no
        test["학습최종일"] = pd.Timestamp(train_days.max())
        test["시험최초일"] = pd.Timestamp(test_days.min())
        rows.append(test[["지역", "폴드", "issue_day", "target_time_kst", "target_day", "계절4",
                          TARGET, "모델예측_kW", "climatology예측_kW", "학습최종일", "시험최초일"]])

    oof = pd.concat(rows, ignore_index=True)
    summaries = []
    for season in SEASON_ORDER:
        g = oof[oof["계절4"] == season]
        if g.empty:
            continue
        model_s = score(g, "모델예측_kW")
        clim_s = score(g, "climatology예측_kW")
        summaries.append({
            "지역": region, "계절": season, **{f"모델_{k}": v for k, v in model_s.items()},
            **{f"climatology_{k}": v for k, v in clim_s.items()},
            "MAE_개선율_pct": float(100 * (clim_s["MAE_kW"] - model_s["MAE_kW"]) / clim_s["MAE_kW"]),
            "RMSE_개선율_pct": float(100 * (clim_s["RMSE_kW"] - model_s["RMSE_kW"]) / clim_s["RMSE_kW"]),
            "판정": "표본충분" if model_s["독립일수"] >= 30 else "표본부족_판정보류",
        })

    old_json = json.loads((cfg["dir"] / "outputs" / "백테스트_v1_2026-09-07" / "백테스트_결과.json").read_text(encoding="utf-8"))
    old_folds = pd.DataFrame(old_json["폴드별"])[["폴드", "계절"]].rename(columns={"계절": "기존대표계절"})
    actual = (oof.groupby(["폴드", "계절4"]).size().unstack(fill_value=0)
              .reindex(columns=SEASON_ORDER, fill_value=0).reset_index())
    cross = old_folds.merge(actual, on="폴드", how="left")
    cross.insert(0, "지역", region)
    cross["경계혼합"] = cross[SEASON_ORDER].gt(0).sum(axis=1) > 1
    manifest = {
        "지역": region, "모델": model_name, "특성수": len(features), "OOF행수": len(oof),
        "폴드수": int(oof["폴드"].nunique()), "경계혼합폴드수": int(cross["경계혼합"].sum()),
        "leakage_audit": "통과: 모든 폴드 학습최종일 < 시험최초일; climatology는 해당 폴드 학습행만 사용",
        "season_rule": "target_time_kst 실제 월 기준: 봄3~5/여름6~8/가을9~11/겨울12~2",
        "status": "분류표준_후보_운영연결보류",
    }
    return oof, pd.DataFrame(summaries), cross, manifest


def main() -> None:
    common = load("regional_phase2_common", ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py")
    OUT.mkdir(parents=True, exist_ok=True)
    all_oof, all_summary, all_cross, manifests = [], [], [], {}
    for region, cfg in CONFIGS.items():
        oof, summary, cross, manifest = run_region(region, cfg, common)
        all_oof.append(oof); all_summary.append(summary); all_cross.append(cross)
        manifests[region] = manifest
        print(region, json.dumps(manifest, ensure_ascii=False))
        print(summary.to_string(index=False))
    pd.concat(all_oof, ignore_index=True).to_csv(OUT / "동일행_OOF_4계절_climatology.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_summary, ignore_index=True).to_csv(OUT / "4계절_모델대_climatology.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_cross, ignore_index=True).to_csv(OUT / "기존대표계절_실제월경계_교차표.csv", index=False, encoding="utf-8-sig")
    (OUT / "manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
