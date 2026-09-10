# -*- coding: utf-8 -*-
"""인버터별 가용여부(15분·1시간) + 용량가중 보정계수 산출.

## 목적
C_용량가중추정보정 정책(단순 5/가용대수가 아니라 **가용 인버터들의
정격용량 합** 기준 보정)을 계산하려면 "몇 대가 가용했는지"뿐 아니라
"**어느** 인버터가 가용했는지"가 필요하다. v5(`rebuild_plant_v5_
recovered_2026-08-21.py`)는 발전소 총합계열만 저장하고 인버터별
가용여부는 저장하지 않았으므로, **v5와 동일한 인버터별 전처리 함수를
그대로 재사용**(재구현 안 함)해 이 표만 별도로 만든다.

## ★정격용량 출처 — 단일 소스만 존재함(명시적 한계)★
사용자가 "Blockdata 등록값과 원본 설비자료 용량을 혼합하지 말라"고
지시했으나, AGENTS.md를 확인한 결과 **원본 설비자료(인버터 로그
엑셀)에는 정격용량 열 자체가 없다.** 유일한 출처는 **Blockdata
실시간 API 등록값**(`GET /data/6715`, 08-19 스냅샷)이다:

| 인버터 | 1 | 2 | 3 | 4 | 5 | 합계 |
|---|---|---|---|---|---|---|
| 정격용량(kW) | 50 | 50 | 30 | 39 | 50 | 219 |

**이 값은 제조사 명판·설비대장과 교차검증되지 않았다**(AGENTS.md
"한계·아쉬운 점" 2번). 혼합할 두 번째 소스가 없으므로, 이 스크립트는
**Blockdata 등록값 단일 출처만** 쓰고 모든 산출물에 그 사실을 명시한다.
명판이 확보되면 이 상수(`INVERTER_RATED_KW`)만 교체하면 된다.

## 산출물 (`outputs/v5_복구_2026-08-21/`)
- `인버터별_가용_15분.parquet` — 15분 슬롯 × 인버터1~5 가용여부(그 슬롯의
  5분버킷 전부에서 유효해야 가용=1, 15분 충족률 규칙(1.0)과 동일 기준)
- `인버터별_가용_1시간.parquet` — 1시간 슬롯 × 인버터1~5 가용여부(그
  시간의 5분버킷 중 75% 이상 유효해야 가용=1, 1시간 충족률 규칙과 동일)
- `용량가중_보정계수_15분.parquet` / `_1시간.parquet` —
  `보정계수 = 전체정격합(219) / 가용인버터 정격합`(가용 0대인 슬롯은 NaN)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PIPE_DIR = Path(__file__).resolve().parent
OUT = PIPE_DIR / "outputs" / "v5_복구_2026-08-21"

_spec = importlib.util.spec_from_file_location(
    "rebuild_v5", Path(__file__).resolve().parents[2] / "02_전처리" / "rebuild_plant_v5_recovered_2026-08-21.py")
rebuild_v5 = importlib.util.module_from_spec(_spec)
sys.modules["rebuild_v5"] = rebuild_v5
_spec.loader.exec_module(rebuild_v5)

# ★출처: Blockdata 실시간 API 등록값(08-19 스냅샷, GET /data/6715).
#   원본 설비자료(인버터 엑셀)에는 정격용량 열이 없어 대조군이 없다.
#   명판·설비대장 미검증 — AGENTS.md "한계·아쉬운 점" 2번 참고.★
INVERTER_RATED_KW = {1: 50.0, 2: 50.0, 3: 30.0, 4: 39.0, 5: 50.0}
TOTAL_RATED_KW = sum(INVERTER_RATED_KW.values())  # 219.0
CAPACITY_SOURCE = "Blockdata 등록값(GET /data/6715, 08-19 스냅샷, 명판·설비대장 미검증)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = rebuild_v5.pv_pipeline.load_config()

    print("=== 인버터별 5분 처리(v5와 동일 함수 재사용: 클리핑→야간0채움→단기보간) ===")
    probe = rebuild_v5.load_inverter_raw(1)
    for n in range(2, rebuild_v5.N_INVERTERS + 1):
        probe = probe.combine_first(rebuild_v5.load_inverter_raw(n))
    grid = pd.date_range(probe.index.min().floor("5min"), probe.index.max().ceil("5min"), freq="5min")
    elev, _ = rebuild_v5.pv_pipeline.solar_position(
        grid, float(config["site"]["latitude"]), float(config["site"]["longitude"]))
    night = pd.Series(elev <= 0.0, index=grid)

    avail_5min = {}
    for n in range(1, rebuild_v5.N_INVERTERS + 1):
        agg = rebuild_v5.process_inverter_5min(n, grid, night)
        avail_5min[n] = agg[f"출력전력__inv{n}"].notna()
        print(f"  인버터{n}(정격{INVERTER_RATED_KW[n]}kW): 5분 가용률 {avail_5min[n].mean()*100:.1f}%")

    avail_df = pd.DataFrame(avail_5min)
    avail_df.columns = [f"inv{n}" for n in avail_df.columns]

    print(f"\n출처: {CAPACITY_SOURCE}")
    print(f"정격용량: {INVERTER_RATED_KW} / 합계 {TOTAL_RATED_KW}kW")

    for freq, rule, label in (("15min", 1.0, "15분"), ("1h", 0.75, "1시간")):
        # 원 파이프라인의 충족률 규칙과 동일 기준으로 "그 구간에 이 인버터가
        # 가용했다"를 판정한다(15분=전부 유효, 1시간=75% 이상 유효).
        frac = avail_df.resample(freq).mean()
        avail_period = (frac >= rule).astype("int8")
        avail_period.to_parquet(OUT / f"인버터별_가용_{label}.parquet")

        rated = pd.Series(INVERTER_RATED_KW).reindex([int(c[3:]) for c in avail_period.columns])
        rated.index = avail_period.columns
        avail_rated_sum = avail_period.mul(rated, axis=1).sum(axis=1)
        factor = (TOTAL_RATED_KW / avail_rated_sum.replace(0, pd.NA)).rename("용량가중_보정계수")
        n_avail = avail_period.sum(axis=1).rename("가용인버터수")
        out = pd.concat([n_avail, avail_rated_sum.rename("가용정격합_kW"), factor], axis=1)
        out.to_parquet(OUT / f"용량가중_보정계수_{label}.parquet")
        print(f"  {label}: {len(out):,}슬롯, 보정계수 결측(가용0대) {factor.isna().sum()}건, "
              f"보정계수 평균(가용<5만) {factor[n_avail < 5].mean():.4f}, "
              f"최대 {factor.max():.4f}")

    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()
