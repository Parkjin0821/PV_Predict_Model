"""광주 5분 발전량의 결측 보간 민감도 비교.

정상 관측 구간에서 1/2/3/6 슬롯을 인위적으로 가려 실제값과 복원값을
비교한다. 국내 태양광 결측 연구는 보간법 비교 필요성의 근거로만 쓰며,
특정 2슬롯 기준을 논문이 직접 제시한 것으로 해석하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

DATA = Path(r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주\v3_multihorizon_2026-08-14\03_분석데이터\gwangju_5min_model_dataset.csv")
OUT = Path(__file__).resolve().parent / "outputs" / "광주_결측보간_국내근거비교_v1_2026-09-09"
SEED = 42
GAPS = [1, 2, 3, 6]
N_PER_GAP = 500
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def estimates(y: np.ndarray, start: int, length: int) -> dict[str, np.ndarray]:
    stop = start + length
    xq = np.arange(start, stop)
    left, right = start - 1, stop
    linear = np.interp(xq, [left, right], [y[left], y[right]])
    persistence = np.repeat(y[left], length)
    side = np.r_[y[max(0, start - 3):start], y[stop:min(len(y), stop + 3)]]
    sma = np.repeat(np.nanmean(side), length)
    xs = np.array([start - 2, start - 1, stop, stop + 1])
    spline = CubicSpline(xs, y[xs], bc_type="natural")(xq)
    return {"직전값": persistence, "선형": linear, "이동평균": sma, "스플라인": spline}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(DATA, parse_dates=["time"], low_memory=False).sort_values("time").reset_index(drop=True)
    y = pd.to_numeric(d["plant_output_kw"], errors="coerce").to_numpy(float)
    clean = d["inverters_raw_observed"].eq(d["reported_inverter_count"]) & np.isfinite(y)
    daylight = pd.to_numeric(d["solar_elevation_deg"], errors="coerce").gt(0).to_numpy()
    dt_ok = d["time"].diff().eq(pd.Timedelta(minutes=5)).to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    examples = {}
    for length in GAPS:
        valid = []
        for s in range(2, len(d) - length - 2):
            ix = np.arange(s - 2, s + length + 2)
            if clean.iloc[ix].all() and daylight[s:s + length].all() and dt_ok[ix[1:]].all():
                valid.append(s)
        chosen = rng.choice(valid, size=min(N_PER_GAP, len(valid)), replace=False)
        for s in chosen:
            truth = y[s:s + length]
            for method, pred in estimates(y, int(s), length).items():
                err = pred - truth
                boundary_ramp = abs(y[s - 1] - y[s - 2]) + abs(y[s + length + 1] - y[s + length])
                rows.append({"gap_slots": length, "gap_minutes": 5 * length, "method": method,
                             "start_time": d.loc[s, "time"], "n_points": length,
                             "boundary_ramp_kw": float(boundary_ramp),
                             "mae_kw": float(np.mean(np.abs(err))), "mse_kw2": float(np.mean(err ** 2)),
                             "max_abs_error_kw": float(np.max(np.abs(err))),
                             "physical_violation_count": int(((pred < 0) | (pred > 240)).sum())})
        examples[length] = int(chosen[0]) if len(chosen) else None
    raw = pd.DataFrame(rows)
    sample_ramp = raw[["gap_slots", "start_time", "boundary_ramp_kw"]].drop_duplicates()
    sample_ramp["ramp_regime"] = sample_ramp.groupby("gap_slots")["boundary_ramp_kw"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1_안정", "Q2", "Q3", "Q4_급변"])
    )
    raw = raw.merge(sample_ramp[["gap_slots", "start_time", "ramp_regime"]],
                    on=["gap_slots", "start_time"], how="left")
    summary = (raw.groupby(["gap_slots", "gap_minutes", "method"], as_index=False)
               .agg(samples=("start_time", "count"), points=("n_points", "sum"),
                    MAE_kW=("mae_kw", "mean"), RMSE_kW=("mse_kw2", lambda x: float(np.sqrt(np.mean(x)))),
                    P95_max_abs_error_kW=("max_abs_error_kw", lambda x: float(np.quantile(x, .95))),
                    physical_violations=("physical_violation_count", "sum")))
    raw.to_csv(OUT / "마스킹복원_개별결과.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "보간법_공백길이별_비교.csv", index=False, encoding="utf-8-sig")
    ramp_summary = (raw.groupby(["gap_minutes", "ramp_regime", "method"], observed=True, as_index=False)
                    .agg(samples=("start_time", "count"), MAE_kW=("mae_kw", "mean"),
                         RMSE_kW=("mse_kw2", lambda x: float(np.sqrt(np.mean(x)))),
                         P95_max_abs_error_kW=("max_abs_error_kw", lambda x: float(np.quantile(x, .95))),
                         physical_violations=("physical_violation_count", "sum")))
    ramp_summary.to_csv(OUT / "보간법_급변도별_비교.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method, g in summary.groupby("method"):
        ax.plot(g["gap_minutes"], g["MAE_kW"], marker="o", linewidth=2, label=method)
    ax.set(xlabel="Artificial gap length (minutes)", ylabel="Reconstruction MAE (kW)",
           title="Gwangju PV: gap-imputation reconstruction error")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "01_공백길이별_MAE.png", dpi=180); plt.close(fig)

    s = examples[2]
    if s is not None:
        lo, hi = s - 8, s + 2 + 8; fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(d.loc[lo:hi-1, "time"], y[lo:hi], "o-", color="#9ca3af", label="observed truth")
        xq = d.loc[s:s+1, "time"]
        for method, pred in estimates(y, s, 2).items(): ax.plot(xq, pred, "o--", label=method)
        ax.axvspan(xq.iloc[0], xq.iloc[-1], color="#f59e0b", alpha=.12)
        ax.set(ylabel="Plant output (kW)", title="Example: reconstruction of a masked 10-minute gap")
        ax.grid(alpha=.25); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(OUT / "02_10분공백_방법별_예시.png", dpi=180); plt.close(fig)

    ten = ramp_summary[ramp_summary["gap_minutes"].eq(10)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method, g in ten.groupby("method", observed=True):
        ax.plot(g["ramp_regime"].astype(str), g["MAE_kW"], marker="o", linewidth=2, label=method)
    ax.set(xlabel="Boundary ramp quartile", ylabel="Reconstruction MAE (kW)",
           title="Gwangju PV: 10-minute gap error by ramp regime")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "03_10분공백_급변도별_MAE.png", dpi=180); plt.close(fig)

    best = summary.loc[summary.groupby("gap_slots")["MAE_kW"].idxmin()].to_dict("records")
    report = {"status":"실증_비교완료_공식채택전", "source":str(DATA), "seed":SEED,
              "selection_rule":"국내 문헌은 결측처리와 방법 비교 필요성만 지지; 허용 슬롯 수는 본 마스킹 복원시험과 후속 모델성능 비교로 결정",
              "domestic_reference":{"title":"태양광 발전량 데이터의 시계열 모델 적용을 위한 결측치 보간 방법 연구","url":"https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10612135","scope":"SMA와 Kalman 비교; 본 프로젝트의 2슬롯·선형보간을 직접 규정하지 않음"},
              "best_by_gap_mae":best,
              "ramp_stratification":"경계 전후 5분 출력변화 절대값 합을 공백길이별 사분위로 구분",
              "next_gate":"보간 전후 예측모델 walk-forward 성능 비교 후 최종 채택"}
    (OUT / "결론.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False)); print("OUT", OUT)


if __name__ == "__main__": main()
