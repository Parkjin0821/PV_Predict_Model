# -*- coding: utf-8 -*-
"""부안 D+1 총출력 공식모델(phase2, 09-03) 정식 백테스트(09-07).

영광 백테스트(`backtest_yeonggwang_v1_2026-09-07.py`)와 동일 원리 -
phase2가 이미 확정한 특성·모델을 재선정 없이 그대로 다시 검증하고,
광주 `backtest_harness_v1_2026-08-20밤.py`의 `metrics()`로 용량정규화
지표(nMAE%·nRMSE%·WAPE%·SkillScore)와 계절별 breakdown을 추가한다.
부안은 옛날(08-31) 백테스트가 있지만 v6 이전 특성구성 기준이라 지금
phase2 공식모델(09-03) 검증이 아니다 - 그래서 새로 한다.

capacity_kw=1000(인버터 등록용량 합계 125kW×8대 - 발전소 명판값
998.715kW가 아니라 광주 219kW 인버터합 관례와 동일하게 인버터합 사용).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

for font_path in [r"C:\Windows\Fonts\malgun.ttf"]:
    if Path(font_path).is_file():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
PHASE1_SCRIPT = HERE / "factor_reverify_v6_hourly_buan_2026-09-03.py"
PHASE2_MODULE = PIPELINE_ROOT / "factor_phase2_multicollinearity_modelselect_v1_2026-09-03.py"
HARNESS_SCRIPT = PIPELINE_ROOT / "backtest_harness_v1_2026-08-20밤.py"
OFFICIAL_PHASE2 = HERE / "outputs" / "요인재검증_v6_phase2_2026-09-03" / "phase2_요약.json"
OUT_DIR = HERE / "outputs" / "백테스트_v1_2026-09-07"
SEED = 42
TARGET = "plant_ac_power_kw"
INITIAL_TRAIN_DAYS = 60
TEST_BLOCK_DAYS = 20
CAPACITY_KW = 1000.0  # 인버터 등록용량 합계(125kW*8대)

KOREAN_SEASON = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official = json.loads(OFFICIAL_PHASE2.read_text(encoding="utf-8"))
    features = official["최종특성"]
    model_name = official["선정모델(MAE기준)"]
    print(f"[부안] 공식채택 모델 재검증: {model_name}, 특성 {len(features)}개")

    phase1_mod = _load_module("buan_phase1_bt", PHASE1_SCRIPT)
    phase2_mod = _load_module("phase2_common_buan_bt", PHASE2_MODULE)
    harness = phase2_mod._load_harness()
    bt_harness = _load_module("gwangju_harness_for_metrics_buan", HARNESS_SCRIPT)
    v5 = phase1_mod._load_module("buan_v5_for_folds_bt", phase1_mod.V5_SCRIPT)

    frame = phase1_mod.build_hourly_frame(v5)
    if isinstance(frame, tuple):
        frame = frame[0]
    days = pd.DatetimeIndex(np.sort(frame["issue_day"].unique()))
    folds = v5.expanding_folds_full_coverage(days, INITIAL_TRAIN_DAYS, TEST_BLOCK_DAYS)

    required = [c for c in features if c not in phase1_mod.NATIVE_MISSING_OK]
    fold_rows = []
    all_true, all_pred, all_season = [], [], []

    for i, (train_days, test_days) in enumerate(folds, start=1):
        assert train_days.max() < test_days.min(), "시간누출: 학습일이 시험일보다 뒤"
        train = frame[frame["issue_day"].isin(train_days)].dropna(subset=required + [TARGET])
        test = frame[frame["issue_day"].isin(test_days)].dropna(subset=required + [TARGET])
        if len(train) < 100 or len(test) < 5:
            continue

        model = harness.make_model(model_name, SEED)
        model.fit(train[features], train[TARGET])
        pred = np.clip(model.predict(test[features]), 0, CAPACITY_KW)
        y_true = test[TARGET].to_numpy()
        pers = test["lag_1day_same_slot_kw"].to_numpy() if "lag_1day_same_slot_kw" in test.columns else None

        months = pd.DatetimeIndex(test_days).month
        season_counts = pd.Series([KOREAN_SEASON[m] for m in months]).value_counts()
        season_label = season_counts.idxmax()

        ref_rmse = bt_harness.metrics(y_true, pers, CAPACITY_KW)["RMSE_kW"] if pers is not None else None
        m = bt_harness.metrics(y_true, pred, CAPACITY_KW, ref_rmse=ref_rmse)
        fold_rows.append({"폴드": i, "계절": season_label, "시험발행일수": len(test_days),
                          "시험행수": len(test), **m})
        all_true.append(y_true)
        all_pred.append(pred)
        all_season.extend([season_label] * len(y_true))

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    season_all = np.array(all_season)

    pooled = bt_harness.metrics(true_all, pred_all, CAPACITY_KW)
    print(f"[부안] pooled 전체: MAE {pooled['MAE_kW']}kW, nMAE {pooled['nMAE_pct']}%, "
          f"RMSE {pooled['RMSE_kW']}kW, WAPE {pooled['WAPE_pct']}%")

    seasonal = []
    for season in ["봄", "여름", "가을", "겨울"]:
        mask = season_all == season
        if mask.sum() < 10:
            seasonal.append({"계절": season, "표본수": int(mask.sum()), "_비고": "표본 부족(10행 미만)"})
            continue
        seasonal.append({"계절": season, **bt_harness.metrics(true_all[mask], pred_all[mask], CAPACITY_KW)})
    print("[부안] 계절별:", json.dumps(seasonal, ensure_ascii=False))

    fdf = pd.DataFrame(fold_rows)
    season_color = {"봄": "#57a06b", "여름": "#e0576f", "가을": "#e07a3f", "겨울": "#5b8def"}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for season, group in fdf.groupby("계절"):
        ax.scatter(group["폴드"], group["nMAE_pct"], label=season, color=season_color.get(season), s=60, zorder=3)
    ax.plot(fdf["폴드"], fdf["nMAE_pct"], color="#ccc", linewidth=1, zorder=1)
    ax.set_xlabel("폴드 번호(walk-forward, 시험블록 20일씩)")
    ax.set_ylabel("정규화 MAE, nMAE (%, 인버터등록용량 1,000kW 기준)")
    ax.set_title(f"부안 D+1 총출력({model_name}): 폴드별·계절별 nMAE")
    ax.legend(title="시험블록 대표 계절")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_계절별_nMAE_추이.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for season in ["봄", "여름", "가을", "겨울"]:
        mask = season_all == season
        if mask.sum() == 0:
            continue
        ax.scatter(true_all[mask], pred_all[mask], s=5, alpha=0.3, color=season_color.get(season), label=season)
    lim = max(true_all.max(), pred_all.max()) * 1.02
    ax.plot([0, lim], [0, lim], color="#888", linestyle="--", linewidth=1, label="완전일치선")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("실측 발전량 (kW)"); ax.set_ylabel("예측 발전량 (kW)")
    ax.set_title(f"부안 D+1 총출력({model_name}): 실측 대 예측(계절별)\n(pooled {len(true_all)}행, MAE {pooled['MAE_kW']}kW, nMAE {pooled['nMAE_pct']}%)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_실측vs예측_산점도.png", dpi=150)
    plt.close(fig)

    resid = pred_all - true_all
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid, bins=60, color="#5b8def", alpha=0.85)
    ax.axvline(0, color="#333", linewidth=1)
    ax.axvline(float(np.mean(resid)), color="#e0576f", linewidth=1.5, linestyle="--",
              label=f"평균오차(bias) {np.mean(resid):.2f}kW")
    ax.set_xlabel("예측 - 실측 (kW, 양수=과대예측)")
    ax.set_ylabel("빈도")
    ax.set_title("부안 D+1 총출력: 잔차 분포(계통적 편향 확인)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_잔차_히스토그램.png", dpi=150)
    plt.close(fig)

    result = {
        "지역": "부안", "모델": model_name, "특성수": len(features),
        "capacity_kw_기준": CAPACITY_KW,
        "pooled_전체": pooled, "계절별": seasonal, "폴드별": fold_rows,
        "잔차_평균bias_kW": round(float(np.mean(resid)), 3),
        "잔차_표준편차_kW": round(float(np.std(resid)), 3),
    }
    (OUT_DIR / "백테스트_결과.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[성공] 백테스트 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
