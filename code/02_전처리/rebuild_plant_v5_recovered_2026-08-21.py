# -*- coding: utf-8 -*-
"""★v5★ v4 정합성 검증 실패 원인 규명 후 원본 집계 로직을 정확히 복제.

## v4가 실패한 이유(사용자 재검사로 발견, 원본 빌더 대조로 원인 확정)
v4는 원본 인버터 엑셀을 직접 5분 평균만 내서 합쳤는데, **실제 v3
`gwangju_5min_model_dataset.csv`를 만든 원본 스크립트**
(`코덱스/결과물/.../v3_multihorizon_2026-08-14/05_재현코드/
build_train_gwangju_multihorizon.py`의 `build_regular()`)를 대조해보니
v4가 빠뜨린 전처리가 3개 더 있었다:

1. **인버터별 물리범위 필터 누락**: 원본은 `출력전력`을 **인버터 개별
   단위로 0~80kW 클리핑**한 뒤에 합산한다(원본 코드 88행:
   `raw.loc[~raw["출력전력"].between(0, 80), "출력전력"] = np.nan`).
   v4는 발전소 총합(0~240kW)만 걸러서, 인버터 1대의 이상치가 그대로
   합산에 들어갔다 — v4 정합성검증에서 나온 **최대오차 61.78kW**를
   설명한다.
2. **야간 결측 0-채움 누락**: 원본은 야간(원본은 관측 단파복사
   ≤3W/m²를 기준으로 썼다 — 이 스크립트는 그 원자료가 없어 **태양고도
   ≤0°를 대체 기준으로 씀, 아래서 명시**)이고 결측이면 출력전력·
   입력전력·입력전류를 0으로 채운다.
3. **단기 결측 시간보간 누락**: 원본은 내부(가장자리 제외) 2스텝(10분)
   이내 결측을 시간보간(`interpolate(method="time", limit=2,
   limit_area="inside")`)으로 채운다.

**`min_count=5` → `min_count=1`이 유일하게 의도된 변경**이어야 하는데,
v4는 그 김에 위 3개를 함께 놓쳤다 — 그래서 비결측 구간에서도 v3와
달라졌다(정합성 실패는 스크립트가 설계대로 "신뢹 안 함"을 올바르게
보고한 것).

## v5에서 고친 것
- 위 1~3번을 원본과 동일하게(태양고도 대체만 예외) 재현.
- **가용인버터수 집계 방식 수정**: 15분/1시간 집계에서 `max()` 대신
  **최소·평균·완전가용비율(5대 모두인 5분버킷 비율)** 을 함께 기록한다
  (`max()`는 그 시간 중 한 순간만 5대여도 "5대 가용"으로 보이는 착시를
  만든다 — 사용자 지적으로 발견).
- **`신규복구_시각수` 계산 버그 수정**: 기존 v4는 24시간 전체를 비교해
  v3의 야간 0값과 v4의 야간 NaN이 섞여 음수가 나왔다. **낮시간
  (물리적낮==1)만, 공통 인덱스에서만** "기존 NaN → 신규 유효"를 센다.

## ★태양고도 대체에 대한 명시적 주의★
원본의 야간판정 기준(관측 단파복사 ≤3W/m²)은 이 스크립트가 접근하지
못하는 별도 기상원자료에 있다. **태양고도 ≤0°는 근사치**이며, 새벽·
저녁 박명 구간에서 원본과 약간 다르게 판정될 수 있다. 이 차이는
사소(수 분 단위)해서 정합성 검증 결과로 영향의 크기를 직접 확인한다.

## 실행법
```
cd "C:\\Users\\u-cube\\JIN\\태양광 발전\\광주_PV_예측모델_통합_v1_2026-08-19\\02_전처리"
python rebuild_plant_v5_recovered_2026-08-21.py
```
API 호출 없음. v4와 마찬가지로 엑셀 5개 파싱이라 수 분 소요.

## 산출물 (`outputs/v5_복구_2026-08-21/`)
v4와 동일한 파일 세트 + `정합성검증_v3대비.csv`가 **이번엔 통과해야
정상**이다. 통과 기준: 비결측 공통구간에서 평균절대차 <0.5kW,
최대절대차 <5kW(측정오차·반올림 수준). 통과 못 하면 이 스크립트도
**신뢰하지 않는다**(v4와 같은 원칙 유지).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PIPE_DIR = Path(__file__).resolve().parents[1] / "03_모델학습" / "현재_종합파이프라인"
_spec = importlib.util.spec_from_file_location("pv_pipeline", PIPE_DIR / "pv_pipeline.py")
pv_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["pv_pipeline"] = pv_pipeline
_spec.loader.exec_module(pv_pipeline)

INVERTER_DIR = Path(r"C:\Users\u-cube\JIN\태양광 발전\배만수 부장님 엑셀 파일")
INVERTER_FILES = {
    1: "광주 광주시청 _1번 인버터 로그_계산됨.xlsx",
    2: "광주 광주시청_2번 인버터 로그_계산됨.xlsx",
    3: "광주 광주시청_3번 인버터 로그_계산됨.xlsx",
    4: "광주 광주시청_4번 인버터 로그_계산됨.xlsx",
    5: "광주 광주시청_5번 인버터 로그_계산됨.xlsx",
}
OUT = PIPE_DIR / "outputs" / "v5_복구_2026-08-21"

POWER_COLS = {"출력전력": "발전출력_kW", "입력전력": "입력전력_kW", "입력전류": "입력전류_A"}
MEAN_COLS = {"입력전압": "입력전압평균_V", "주파수": "주파수평균_Hz",
             "역률": "역률평균", "온도": "인버터평균온도_C"}
N_INVERTERS = 5
PER_INVERTER_POWER_CLIP = (0.0, 80.0)   # 원본 88행과 동일(발전소 240kW 필터와 별개)
PER_INVERTER_TEMP_CLIP = (-50.0, 80.0)  # 원본과 동일


def load_inverter_raw(n: int) -> pd.DataFrame:
    usecols = ["생성일"] + list(POWER_COLS) + list(MEAN_COLS)
    df = pd.read_excel(INVERTER_DIR / INVERTER_FILES[n], usecols=usecols)
    df["생성일"] = pd.to_datetime(df["생성일"], errors="coerce")
    df = df.dropna(subset=["생성일"]).set_index("생성일").sort_index()
    for c in list(POWER_COLS) + list(MEAN_COLS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ★원본 88행과 동일: 인버터 개별 단위 물리범위 클리핑(합산 전)★
    lo, hi = PER_INVERTER_POWER_CLIP
    df.loc[~df["출력전력"].between(lo, hi), "출력전력"] = np.nan
    tlo, thi = PER_INVERTER_TEMP_CLIP
    df.loc[~df["온도"].between(tlo, thi), "온도"] = np.nan
    return df


def process_inverter_5min(n: int, grid: pd.DatetimeIndex, night: pd.Series) -> pd.DataFrame:
    """5분 격자 정렬 + 야간 0채움 + 단기결측 시간보간(전부 원본 로직 재현)."""
    raw = load_inverter_raw(n)
    agg = raw[list(POWER_COLS) + list(MEAN_COLS)].resample("5min").mean().reindex(grid)
    agg["_raw_observed"] = agg["출력전력"].notna().astype("int8")  # 보간·0채움 전 원상태(진단용)

    # ★원본 96~99행과 동일: 야간+결측이면 전력계열을 0으로★
    # (태양고도 대체 — docstring 참고)
    for c in POWER_COLS:
        agg.loc[night & agg[c].isna(), c] = 0.0

    # ★원본 100~102행과 동일: 내부 2스텝(10분) 이내 결측만 시간보간★
    cols = list(POWER_COLS) + list(MEAN_COLS)
    agg[cols] = agg[cols].interpolate(method="time", limit=2, limit_area="inside")
    agg.columns = [f"{c}__inv{n}" if c in cols else c for c in agg.columns]
    return agg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = pv_pipeline.load_config()
    capacity = float(config["site"]["capacity_kw"])

    print("=== 1) 그리드·야간 판정(태양고도 대체) 준비 ===")
    probe = load_inverter_raw(1)
    for n in range(2, N_INVERTERS + 1):
        probe = probe.combine_first(load_inverter_raw(n))
    grid = pd.date_range(probe.index.min().floor("5min"), probe.index.max().ceil("5min"), freq="5min")
    elev, azi = pv_pipeline.solar_position(
        grid, float(config["site"]["latitude"]), float(config["site"]["longitude"]))
    night = pd.Series(elev <= 0.0, index=grid)
    print(f"  격자 {len(grid):,}개(5분), 야간비율 {night.mean()*100:.1f}%")

    print("\n=== 2) 인버터별 5분 처리(개별클리핑→야간0채움→단기보간) ===")
    per_inv = {}
    for n in range(1, N_INVERTERS + 1):
        per_inv[n] = process_inverter_5min(n, grid, night)
        raw_valid = int(per_inv[n][f"출력전력__inv{n}"].notna().sum())
        print(f"  인버터{n}: 유효 {raw_valid:,}/{len(grid):,} "
              f"(클리핑·보간·야간채움 후, 원시결측률 은닉없이 raw_observed 별도 보존)")

    print("\n=== 3) 발전소 합산(min_count=1 — 유일하게 의도된 변경) ===")
    five = pd.DataFrame(index=grid)
    for src, dst in POWER_COLS.items():
        stacked = pd.concat([per_inv[n][f"{src}__inv{n}"] for n in range(1, N_INVERTERS + 1)], axis=1)
        five[dst] = stacked.sum(axis=1, skipna=True, min_count=1)
    for src, dst in MEAN_COLS.items():
        stacked = pd.concat([per_inv[n][f"{src}__inv{n}"] for n in range(1, N_INVERTERS + 1)], axis=1)
        five[dst] = stacked.mean(axis=1, skipna=True)
    raw_obs_stack = pd.concat([per_inv[n]["_raw_observed"] for n in range(1, N_INVERTERS + 1)], axis=1)
    avail_stack = pd.concat(
        [per_inv[n][f"출력전력__inv{n}"].notna() for n in range(1, N_INVERTERS + 1)], axis=1)
    five["가용인버터수"] = avail_stack.sum(axis=1).astype("int8")          # 야간채움·보간 포함(원본 의미와 동일)
    five["원시관측인버터수"] = raw_obs_stack.sum(axis=1).astype("int8")     # 채움·보간 전(투명성용 추가지표)
    five["부분가용여부"] = ((five["가용인버터수"] > 0) & (five["가용인버터수"] < N_INVERTERS)).astype("int8")
    five["완전가용"] = (five["가용인버터수"] == N_INVERTERS).astype("int8")

    five.loc[~five["발전출력_kW"].between(0, capacity), "발전출력_kW"] = np.nan  # 발전소 단위 필터(원본과 동일하게 별도 유지)
    five["태양고도_deg"] = elev
    five["태양방위각_deg"] = azi
    five["물리적낮"] = (elev > 0).astype("int8")
    five["핵심낮시간"] = (elev > 10).astype("int8")
    print(f"  5분 {len(five):,}행 / 발전출력 결측 {five['발전출력_kW'].isna().mean()*100:.2f}%"
          f" / 부분가용 {five['부분가용여부'].mean()*100:.2f}%")

    print("\n=== 4) 15분·1시간·일간 재집계(기존 충족률 규칙 동일: 15분=1.0, 1시간=0.75) ===")
    f15 = pv_pipeline.aggregate_power(five, "15min", required_fraction=1.0)
    f1h = pv_pipeline.aggregate_power(five, "1h", required_fraction=0.75)
    for f, freq in ((f15, "15min"), (f1h, "1h")):
        g = five["가용인버터수"].resample(freq)
        f["가용인버터수_최소"] = g.min()
        f["가용인버터수_평균"] = g.mean().round(2)
        f["완전가용비율"] = five["완전가용"].resample(freq).mean().round(3)
        f["부분가용여부"] = (five["부분가용여부"].resample(freq).max()).astype("int8")
    daily = pv_pipeline.build_daily_actual(five)
    day_only = five[five["물리적낮"] == 1]
    daily["가용인버터수_낮시간최소"] = day_only.groupby(day_only.index.normalize())["가용인버터수"].min()
    daily["가용인버터수_낮시간평균"] = day_only.groupby(day_only.index.normalize())["가용인버터수"].mean().round(2)
    daily["부분가용일"] = (daily["가용인버터수_낮시간평균"] < N_INVERTERS).astype("int8")

    f15.to_parquet(OUT / "집계_15분_자료_v5.parquet")
    f1h.to_parquet(OUT / "집계_1시간_자료_v5.parquet")
    daily.to_parquet(OUT / "집계_일간_실제발전량_v5.parquet")
    print(f"  15분 {len(f15):,}행(결측 {f15['발전출력_kW'].isna().mean()*100:.2f}%)"
          f" / 1시간 {len(f1h):,}행(결측 {f1h['발전출력_kW'].isna().mean()*100:.2f}%)"
          f" / 일간 유효 {daily['일간발전량_kWh'].notna().sum()}/{len(daily)}일")

    print("\n=== 5) 완결성 감사표(날짜×시간대) ===")
    a = f1h.copy()
    a["날짜"] = a.index.normalize(); a["시"] = a.index.hour
    audit = a.reset_index()[["날짜", "시", "발전출력_kW", "가용인버터수_최소", "가용인버터수_평균",
                             "완전가용비율", "물리적낮"]]
    audit["타깃결측"] = audit["발전출력_kW"].isna().astype("int8")
    audit.to_csv(OUT / "완결성_감사표_날짜시간대.csv", index=False, encoding="utf-8-sig")
    day_audit = audit[audit["물리적낮"] == 1]
    print("  낮시간 타깃결측률(시간대별):")
    print("   ", dict(day_audit.groupby("시")["타깃결측"].apply(lambda x: round(x.mean() * 100, 2))))
    print("  낮시간 정오대(10~15시) 결측률:",
          round(day_audit[day_audit["시"].between(10, 15)]["타깃결측"].mean() * 100, 2), "%")

    print("\n=== 6) v3 대비 정합성 검증(★표본구성 동일 vs 신규확장을 구분해서 판정★) ===")
    # ★설계상 중요★: "양쪽 유효"인 시각도 v5가 v3보다 5분표본을 더 많이
    # 가질 수 있다(그게 바로 복구 효과다 — 예: 1시간 집계에서 v3는
    # 9/12개 5분버킷만 있었는데 v5는 12/12개라 평균 자체가 달라진다).
    # 이건 오류가 아니라 의도한 차이이므로, **표본구성(발전자료개수)이
    # 완전히 같은 시각만** 재현성 기준으로 삼는다. 표본이 늘어난
    # 시각은 별도로 세되 값 차이를 재현성 실패로 계산하지 않는다.
    checks = []
    for label, oldp, new in (("15분", PIPE_DIR / "outputs" / "집계_15분_자료.parquet", f15),
                             ("1시간", PIPE_DIR / "outputs" / "집계_1시간_자료.parquet", f1h)):
        if not oldp.exists():
            continue
        old = pd.read_parquet(oldp)
        common = old.index.intersection(new.index)
        old_c, new_c = old["발전출력_kW"].reindex(common), new["발전출력_kW"].reindex(common)
        old_n, new_n = old["발전자료개수"].reindex(common), new["발전자료개수"].reindex(common)
        both = old_c.notna() & new_c.notna()
        same_comp = both & (old_n == new_n)          # 재현성 판정 대상
        expanded = both & (new_n > old_n)              # 표본 확장(복구 효과) — 값이 달라도 정상
        d_same = (old_c[same_comp] - new_c[same_comp]).abs()
        physically_day = new["물리적낮"].reindex(common) == 1
        newly = int((old_c.isna() & new_c.notna() & physically_day).sum())
        checks.append({
            "대상": label, "공통시각": len(common), "양쪽유효": int(both.sum()),
            "표본동일_재현성대상": int(same_comp.sum()),
            "표본동일_평균절대차_kW": round(float(d_same.mean()) if len(d_same) else 0.0, 5),
            "표본동일_최대절대차_kW": round(float(d_same.max()) if len(d_same) else 0.0, 5),
            "표본확장_시각수(복구효과,정상)": int(expanded.sum()),
            "낮시간_신규복구_시각수(기존NaN→유효)": newly,
        })
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "정합성검증_v3대비.csv", index=False, encoding="utf-8-sig")
    print(chk.to_string(index=False))

    passed = bool(len(chk) and (chk["표본동일_평균절대차_kW"] < 0.01).all()
                  and (chk["표본동일_최대절대차_kW"] < 1.0).all())
    lines = [
        "v5 복구 재생성 요약(v4 실패원인 3개 수정: 인버터별 클리핑·야간0채움·단기보간)",
        f"정합성검증 통과 여부: {'통과' if passed else '★실패 — 아직 신뢰 불가★'}",
        chk.to_string(index=False),
    ]
    (OUT / "요약.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장 완료: {OUT}")
    print(f"\n{'★★★정합성 통과 — v5를 다음 단계에 사용 가능★★★' if passed else '★★★정합성 실패 — v5도 아직 신뢰하지 말 것★★★'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
