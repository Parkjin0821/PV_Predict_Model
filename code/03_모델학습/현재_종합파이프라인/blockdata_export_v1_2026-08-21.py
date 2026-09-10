# -*- coding: utf-8 -*-
"""⑥ Blockdata 출력 규격화 — 로드맵 08-21 6단계.

⑤에서 확정된 **최종 공식모델의 예측값**을 Blockdata 실시간 발전데이터 API
(`GET https://api.blockdata.kr/data/{발전소 ID}`, 광주시청=6715)의
필드명·단위·자릿수에 맞춰 변환하고, 그 변환이 실제로 규격에 맞는지
자동 점검한다.

## ★08-21 재검사로 발견된 결함 5건 — 전부 수정 완료(아래 각 항목 참고)★
1차 실행은 완료됐으나 시각정합성 7건 실패로 "완료" 처리하지 않고
재검사했다. 수정 내역:
1. **초단기 시각정합성 실패는 허용오차 버그였다** — 실제 최대 차이는
   0.01kW(두 독립적인 `.round(2)` 호출의 부동소수점 경계 문제)뿐이고
   시간 오프셋 자체는 맞았다. `diff.max() <= 0.01`처럼 부동소수점을
   정확히 비교하면 부동소수점 표현 오차로 거짓 실패가 난다 →
   `np.isclose(..., atol=0.011)`로 교체.
2. **단기 시각정합성 실패는 잘못된 참조자료 때문이었다** — 검증기가
   `집계_1시간_자료.parquet`(수집 파이프라인의 중간 산출물)와 비교했는데,
   실제 단기 모델이 학습·시험에 쓴 자료는 `harness.DATASETS["v3_고정tm"]`
   (`gwangju_1hour_model_dataset_official_v3_fixed_tm_2026-08-21.csv`)의
   `plant_output_kw`다. 두 자료는 17,033개 공통 시각 중 5,109개에서
   0.02kW 넘게 다르다(최대 4.3kW) — **모델이 실제로 본 적 없는 자료와
   비교해 거짓 실패를 낸 것**. 참조를 v3로 교체 → 최대 차이 0.00kW로
   정확히 일치 확인.
3. **`daily_energy` 경로B는 설명과 실제 구현이 달랐다** — docstring은
   "시각별 누적곡선"이라 해놓고 실제로는 **일별 최종 합계**(하루에 한
   행)만 냈다. Blockdata 실시간 `daily_energy`(자정 리셋 후 매 갱신마다
   커지는 러닝 합계)에 대응하려면 **대상시각마다** 그 시각까지의
   `cumsum(예측_kW × 버킷시간)`을 내야 한다 → `build_daily_cumulative()`로
   분리 신설(행 단위 시계열), 경로A(일간모델 최종총량)는
   `build_daily_final()`로 별도 분리해 의미 혼동을 없앴다.
4. 현재 스크립트는 ⑤가 저장한 OOF 예측을 변환할 뿐, 발전소 정적정보+
   실시간 입력 샘플 → 특성생성 → **저장된 모델 파일**로 추론 → Blockdata
   출력까지 실제로 이어지는지 확인하는 end-to-end 테스트는 아니었다
   (그리고 애초에 이 프로젝트에는 "배포용으로 저장된 모델 파일" 자체가
   지금까지 없었다 — 전부 CV 폴드 안에서 학습·평가만 했음). →
   `run_smoke_test()` 신설: 가장 단순한 채택 구성(초단기 +1h raw)을
   전체 가용 데이터로 재학습해 joblib으로 저장하고, **그 파일을 다시
   불러와** 최신 시각 1건을 추론해 Blockdata payload 형태까지 만든다.
   이 재학습은 ⑤의 최종성능 판단을 대체하지 않는다 — 목적은 오직
   "저장→적재→추론→규격출력" 경로가 실제로 안 끊기는지 확인하는 것.
5. **검증 실패가 있어도 프로세스가 exit 0으로 끝나 자동화가 못
   잡아냈다** → 점검·스모크테스트 중 하나라도 실패하면 산출물을 전부
   저장한 뒤 `SystemExit(1)`로 비정상 종료하도록 수정.

## 왜 이 단계가 필요한가(08-19 확정 설계)
"예측값 vs Blockdata DB 실측값"을 나중에 바로 맞대볼 수 있는 형태로
최종 산출물을 내보내야 한다. 학습 중간값은 정규화·내부 컬럼명이지만
**최종 산출물은 반드시 실제 단위로 역변환해 Blockdata 스키마로** 낸다.

## ★규격 대조에서 확인된 4가지 구조적 차이(은폐하지 않고 명시)★
API 명세서(`배만수 부장님 엑셀 파일/Blockdata_API_제공정보.xlsx`,
시트 "발전데이터(실시간)")를 다시 대조한 결과:

1. **인버터별 vs 발전소 총합**
   명세의 응답은 `plant_id/plant_name/plant_address/plant_capacity` 뒤에
   `inverter[]` 배열이 오고, `dc_volt`~`total_energy`(항목 8~21)는 그
   배열 원소의 필드다. 즉 `ac_power`는 **인버터 1대의 출력**이다.
   우리 모델이 예측하는 건 **발전소 총출력**(인버터 1~5번 합)이므로,
   대응 관계는 `우리 예측 = Σ inverter[].ac_power`이다. 인버터별로
   쪼개서 내보내지 않는다(모델이 인버터 단위를 예측하지 않으므로 —
   임의 배분은 근거 없는 수치를 만드는 것이라 하지 않는다).
   따라서 payload는 `plant_ac_power_kw`(발전소 합계) 레벨로만 낸다.

2. **순간값 vs 구간평균**
   Blockdata `ac_power`는 5분마다 갱신되는 **그 순간의 스냅샷**이다.
   우리 초단기 예측은 **15분 버킷 평균**, 단기는 **1시간 버킷 평균**
   (원자료 5분 순간값을 버킷 평균으로 집계해 학습했으므로).
   숫자를 같은 필드에 담더라도 **집계기준이 다르다** — 각 행에
   `집계기준` 컬럼으로 명시하고, 나중에 실측과 대조할 때 반드시 같은
   기준으로 집계한 실측과 비교해야 한다.

3. **`daily_energy`는 자정 리셋 누적, 우리 일간모델은 하루 최종 총량**
   `daily_energy`는 하루 동안 계속 증가하는 러닝 누적이다. 우리는 두
   경로로 만들 수 있다: ①일간모델(계층조정)의 **하루 최종 총량**(경로A,
   `build_daily_final`) ②초단기·단기 kW 예측을 시간적분한 **대상시각별
   누적곡선**(경로B, `build_daily_cumulative`). 둘은 서로 다른 모델이라
   값이 어긋날 수 있다 — 특히 ⑤에서 일간 계층조정 가중치가 직접모델
   100%/시간모델합계 0%로 붕괴해 두 경로가 사실상 독립이 됐다.
   **두 경로를 모두 내보내고 불일치율을 검증표에 기록한다**(한쪽을
   임의로 정답으로 삼지 않는다).

4. **`total_energy`는 모델이 예측할 수 없다**
   설치 이후 절대 누적량이라 예측 대상이 아니라 **앵커(기준 실측값) +
   예측 증분**으로만 정의된다. 과거 xlsx 로그로는 앵커를 신뢰할 수
   없다(2025-08-15~10-20 원인미확보 저출력 구간, 인버터 5번 부분복구
   구간 존재 — AGENTS.md 참고). 그래서 여기서는 **앵커 자리만 만들고
   값은 null**로 두고, 운영 배포 시 API 최신 `total_energy`를 주입하는
   구조로 규격만 고정한다.

## ★시각 규약(티어별로 다름 — 여기서 틀리면 전부 어긋난다)★
- **초단기**: `build_ultra_short_frame`이 `shift(-H*4)` → 대상시각 =
  **발행시각 + H시간**
- **단기**: `build_frame`이 `shift(-(H-1))` (시작라벨 (H-1) 트릭) →
  대상시각 = **발행시각 + (H-1)시간**
이 규약이 맞는지 추측으로 두지 않고, 내보낸 대상시각의 **실측 집계값이
⑤ 행단위 CSV의 `실제_kW`와 일치하는지** 자동 대조해 검증한다
(`검증_규격점검.csv`의 `시각정합성` 항목). 어긋나면 즉시 드러난다.
**참조자료는 반드시 각 모델이 실제로 학습·시험에 쓴 자료와 같아야
한다**(위 결함 2번 참고 — 다른 자료와 비교하면 거짓 실패가 난다).

## 입력
`outputs/최종통합재검증_v1_2026-08-21/` 의 `판정표.csv`·`{초단기,단기,
일간}_행단위.csv`. **최종 채택 후보를 하드코딩하지 않고 판정표의
`최종채택` 컬럼에서 읽는다** — ⑤를 다시 돌려 판정이 바뀌면 이 스크립트도
자동으로 따라간다(문서와 코드가 어긋나는 걸 방지).

## 출력 (`outputs/Blockdata_규격화_v1_2026-08-21/`)
- `규격정의.json` — 필드·단위·타입·자릿수·집계기준 확정 정의
- `예측_ac_power.csv` — 티어·수평별 시각별 예측 출력(kW)
- `예측_daily_energy_일간총량.csv` — 경로A(일간모델 계층조정 최종총량, 1행/일)
- `예측_daily_energy_누적시계열.csv` — 경로B(대상시각별 누적, Blockdata
  `daily_energy`와 동일한 "자정 리셋 러닝 합계" 의미)
- `예측_total_energy.csv` — 앵커 구조(증분 누적, 앵커 null)
- `blockdata_payload_예시.json` — 실제 API 응답 모양 그대로의 예측 payload
- `smoke_초단기_1h_raw_deploy.joblib` — 스모크테스트용 재학습 모델 파일
- `스모크테스트_결과.json` — end-to-end(정적정보+샘플입력→추론→출력) 점검 결과
- `검증_규격점검.csv` — 자동 점검 결과(통과/실패)
- `요약.txt`

로컬 변환·재학습·검증만 수행한다(외부 API 호출 없음). 점검 또는
스모크테스트가 하나라도 실패하면 **산출물을 전부 저장한 뒤 exit code 1**로
종료한다(자동화가 실패를 놓치지 않도록).
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
SRC = ROOT / "outputs" / "최종통합재검증_v1_2026-08-21"
OUT = ROOT / "outputs" / "Blockdata_규격화_v1_2026-08-21"

PLANT_ID = 6715
PLANT_NAME = "광주 광주시청"

# 시각정합성 비교 허용오차(kW). 우리 값·참조값 둘 다 독립적으로
# round(2)를 거치므로 이론상 최대 0.01kW까지 벌어질 수 있고, 부동소수점
# 이진표현 오차까지 감안해 살짝 여유를 둔다(수정 1번).
TOL_KW = 0.011

# 티어별 기준선 후보명(판정표에 '최종채택' 행이 없으면 = 기존 기본모델 유지)
BASELINE = {"초단기": "raw", "단기": "기본", "일간": "전체특성+기본파라미터"}


def _load(name: str, filename: str):
    """다른 검증된 스크립트의 함수를 그대로 재사용하기 위한 동적 로더
    (⑤ `final_consolidated_verification_v1`과 동일한 패턴)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 대상시각 = 발행시각 + OFFSET_HOURS(티어, H). 위 docstring의 시각 규약 참고.
def target_offset_hours(tier: str, horizon: int) -> int:
    if tier == "초단기":
        return horizon           # shift(-H*4), 15분 해상도
    if tier == "단기":
        return horizon - 1       # shift(-(H-1)), 시작라벨 트릭
    raise ValueError(f"시각 규약이 정의되지 않은 티어: {tier}")


BUCKET = {"초단기": "15분 버킷평균", "단기": "1시간 버킷평균"}
BUCKET_HOURS = {"초단기": 0.25, "단기": 1.0}


# ── 규격 정의 ───────────────────────────────────────────────────────
def build_spec(capacity_kw: float) -> dict:
    """Blockdata 필드 ↔ 우리 예측 필드의 확정 대응표(⑥의 핵심 산출물)."""
    return {
        "출처": "Blockdata_API_제공정보.xlsx / 시트 '발전데이터(실시간)' "
                "(배만수 부장님 제공, GET https://api.blockdata.kr/data/{발전소 ID})",
        "발전소": {"plant_id": PLANT_ID, "plant_name": PLANT_NAME,
                   "plant_capacity_kw": capacity_kw},
        "갱신주기_명세": "약 5분마다 수집·갱신(명세서 참고사항 4번)",
        "구조_주의": (
            "명세상 dc_volt~total_energy는 inverter[] 배열 원소의 필드다. "
            "우리 모델은 발전소 총합만 예측하므로 인버터별로 쪼개지 않고 "
            "Σ inverter[] 수준(plant_*)으로만 대응시킨다."
        ),
        "필드": [
            {
                "blockdata_필드": "ac_power",
                "blockdata_단위": "kW",
                "blockdata_자릿수": "소수점 둘째자리",
                "blockdata_집계": "5분 간격 순간값(인버터별)",
                "우리_필드": "예측_ac_power_kw",
                "우리_집계": "초단기=15분 버킷평균 / 단기=1시간 버킷평균(발전소 총합)",
                "대응": "우리 예측 ≈ Σ inverter[].ac_power",
                "차이_비고": "순간값 vs 구간평균 — 실측 대조 시 실측도 같은 버킷으로 평균낼 것",
                "유효범위": [0.0, capacity_kw],
                "산출_파일": "예측_ac_power.csv",
            },
            {
                "blockdata_필드": "daily_energy",
                "blockdata_단위": "kWh",
                "blockdata_자릿수": "소수점 둘째자리",
                "blockdata_집계": "당일 자정 리셋 러닝 누적(인버터별, 갱신마다 값이 커짐)",
                "우리_필드": "경로A=예측_일간최종총량_kwh / 경로B=예측_daily_energy_누적_kwh",
                "우리_집계": "경로A=일간 계층조정모델의 하루 최종 총량(1행/일) / "
                             "경로B=초단기·단기 kW 예측을 대상시각마다 그날 자정부터 "
                             "누적적분한 시계열(Blockdata daily_energy와 동일한 의미)",
                "대응": "우리 예측 ≈ Σ inverter[].daily_energy",
                "차이_비고": "두 경로가 서로 다른 모델이라 불일치 가능 — 둘 다 내보내고 불일치율 기록",
                "유효범위": [0.0, capacity_kw * 24],
                "산출_파일": {"경로A": "예측_daily_energy_일간총량.csv",
                            "경로B": "예측_daily_energy_누적시계열.csv"},
            },
            {
                "blockdata_필드": "total_energy",
                "blockdata_단위": "kWh",
                "blockdata_자릿수": "소수점 둘째자리",
                "blockdata_집계": "설치 이후 절대 누적(인버터별)",
                "우리_필드": "예측_total_energy_kwh",
                "우리_집계": "앵커(배포시점 API 실측 total_energy) + 예측 daily_energy 누적증분",
                "대응": "우리 예측 ≈ Σ inverter[].total_energy",
                "차이_비고": (
                    "★모델 예측 대상이 아님★ 절대 누적값이라 앵커 없이는 산출 불가. "
                    "과거 로그는 이상구간·인버터5 부분복구로 앵커 신뢰 불가 → "
                    "여기서는 앵커=null, 증분만 산출. 운영 시 API 최신값 주입."
                ),
                "유효범위": None,
                "산출_파일": "예측_total_energy.csv",
            },
        ],
        "미대응_blockdata_필드": {
            "목록": ["dc_volt", "dc_current", "dc_power", "ac_volt_r/s/t",
                     "ac_current_r/s/t", "pf", "freq",
                     "inverter[].number", "inverter[].capacity"],
            "이유": "예측 대상이 아니라 모델 입력(또는 미사용) 필드다. "
                    "예측 payload에는 포함하지 않는다(빈 값을 채워 넣어 "
                    "예측한 것처럼 보이게 하지 않는다).",
        },
    }


# ── 최종 채택 모델 확정(판정표에서 읽음) ────────────────────────────
def resolve_adopted(verdict: pd.DataFrame, rows: dict) -> list[dict]:
    """(티어, 수평)별 최종 채택 후보명을 판정표에서 결정한다."""
    adopted = []
    for tier, df in rows.items():
        if not len(df):
            continue
        groups = [(None,)] if tier == "일간" else sorted(df["수평_h"].unique())
        for H in groups:
            if tier == "일간":
                sel = verdict[(verdict["티어"] == "일간") & (verdict["최종채택"])]
                label = sel.iloc[0]["후보"] if len(sel) else BASELINE[tier]
                adopted.append({"티어": tier, "수평_h": None, "후보": label})
            else:
                sel = verdict[(verdict["티어"] == tier)
                              & (verdict["수평_h"] == H)
                              & (verdict["최종채택"])]
                label = sel.iloc[0]["후보"] if len(sel) else BASELINE[tier]
                adopted.append({"티어": tier, "수평_h": int(H), "후보": label})
    return adopted


# ── ac_power 변환 ───────────────────────────────────────────────────
def build_ac_power(rows: dict, adopted: list[dict]) -> pd.DataFrame:
    out = []
    for a in adopted:
        tier, H, label = a["티어"], a["수평_h"], a["후보"]
        if tier == "일간":
            continue
        df = rows[tier]
        sub = df[(df["수평_h"] == H) & (df["후보"] == label)].copy()
        if not len(sub):
            print(f"  ! [{tier} +{H}h] 후보 '{label}' 행이 없다 — 건너뜀")
            continue
        sub["발행시각"] = pd.to_datetime(sub["발행시각"])
        sub["대상시각"] = sub["발행시각"] + pd.Timedelta(hours=target_offset_hours(tier, H))
        out.append(pd.DataFrame({
            "plant_id": PLANT_ID,
            "티어": tier,
            "수평_h": H,
            "채택모델": label,
            "발행시각": sub["발행시각"],
            "대상시각": sub["대상시각"],
            "집계기준": BUCKET[tier],
            "예측_ac_power_kw": sub["예측_kW"].round(2),
            "실측_ac_power_kw": sub["실제_kW"].round(2),
            "폴드": sub["폴드"],
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── daily_energy 경로A: 일간모델(계층조정) 하루 최종 총량 ───────────
def build_daily_final(rows: dict, adopted: list[dict]) -> pd.DataFrame:
    """경로A. 한 행 = 하루. Blockdata의 '그날 마지막으로 관측된
    daily_energy'에 대응하는 값이지, 시계열이 아니다(수정 3번 — 경로B와
    분리)."""
    daily_pick = next((a for a in adopted if a["티어"] == "일간"), None)
    if daily_pick is None or not len(rows["일간"]):
        return pd.DataFrame()
    d = rows["일간"]
    sub = d[d["후보"] == daily_pick["후보"]].copy()
    if not len(sub):
        return pd.DataFrame()
    sub["날짜"] = pd.to_datetime(sub["날짜"])
    return pd.DataFrame({
        "plant_id": PLANT_ID,
        "날짜": sub["날짜"],
        "경로": "A_일간모델(계층조정)",
        "채택모델": daily_pick["후보"],
        "예측_일간최종총량_kwh": sub["계층조정_kWh"].round(2),
        "실측_일간최종총량_kwh": sub["실제_일간총량_kWh"].round(2),
        "폴드": sub["폴드"],
    })


# ── daily_energy 경로B: 대상시각별 누적(자정 리셋 러닝 합계) ────────
def build_daily_cumulative(ac: pd.DataFrame) -> pd.DataFrame:
    """경로B. 한 행 = 한 대상시각(=ac_power와 같은 행 단위). 그 날짜의
    첫 버킷부터 이 시각까지 `예측_ac_power_kw × 버킷시간`을 누적한 값 —
    Blockdata `daily_energy`가 매 5분 갱신마다 보여주는 "지금까지의
    당일 누적"과 동일한 의미다(수정 3번 — 이전 버전은 일별 합계 1행만
    냈는데, 그건 이 경로가 아니라 경로A의 정의였다)."""
    frames = []
    for tier in ["초단기", "단기"]:
        sub = ac[ac["티어"] == tier]
        if not len(sub):
            continue
        hours = BUCKET_HOURS[tier]
        for H, s in sub.groupby("수평_h"):
            s = s.sort_values("대상시각").copy()
            s["날짜"] = s["대상시각"].dt.normalize()
            s["예측_daily_energy_누적_kwh"] = (
                s.groupby("날짜")["예측_ac_power_kw"].cumsum() * hours
            ).round(2)
            s["실측_daily_energy_누적_kwh"] = (
                s.groupby("날짜")["실측_ac_power_kw"].cumsum() * hours
            ).round(2)
            s["일자내_버킷순번"] = s.groupby("날짜").cumcount() + 1
            frames.append(pd.DataFrame({
                "plant_id": PLANT_ID,
                "경로": f"B_시간적분({tier} +{H}h)",
                "채택모델": s["채택모델"].values,
                "날짜": s["날짜"].values,
                "대상시각": s["대상시각"].values,
                "일자내_버킷순번": s["일자내_버킷순번"].values,
                "예측_daily_energy_누적_kwh": s["예측_daily_energy_누적_kwh"].values,
                "실측_daily_energy_누적_kwh": s["실측_daily_energy_누적_kwh"].values,
                "폴드": s["폴드"].values,
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── total_energy 구조(앵커 + 증분) ──────────────────────────────────
def build_total_energy(daily_final: pd.DataFrame) -> pd.DataFrame:
    """앵커는 null. 운영 시 API 최신 total_energy를 앵커에 넣으면 절대값이 된다."""
    if not len(daily_final):
        return pd.DataFrame()
    a = daily_final.sort_values("날짜")
    out = pd.DataFrame({
        "plant_id": PLANT_ID,
        "날짜": a["날짜"],
        "예측_일간최종총량_kwh": a["예측_일간최종총량_kwh"].values,
        "누적증분_kwh": np.round(np.cumsum(a["예측_일간최종총량_kwh"].values), 2),
        "앵커_total_energy_kwh": np.nan,   # 운영 배포 시 API 실측값 주입
        "예측_total_energy_kwh": np.nan,   # = 앵커 + 누적증분 (앵커 없으면 산출 불가)
    })
    out["앵커_주입방법"] = "GET /data/6715 의 Σ inverter[].total_energy 최신값"
    return out


# ── payload 예시 ────────────────────────────────────────────────────
def build_payload_sample(ac: pd.DataFrame, daily_final: pd.DataFrame, capacity_kw: float) -> dict:
    """실제 API 응답 모양에 맞춘 예측 payload 1건(형식 고정용 샘플)."""
    if not len(ac):
        return {}
    row = ac.sort_values("대상시각").iloc[-1]
    day = pd.Timestamp(row["대상시각"]).normalize()
    dsel = daily_final[daily_final["날짜"] == day] if len(daily_final) else pd.DataFrame()
    daily_val = float(dsel.iloc[0]["예측_일간최종총량_kwh"]) if len(dsel) else None
    return {
        "_설명": "Blockdata /data/{발전소 ID} 응답 구조에 맞춘 '예측' payload 형식 샘플. "
                 "실측 응답과 달리 inverter[] 개별 값은 채우지 않는다(모델이 발전소 총합만 예측).",
        "plant_id": PLANT_ID,
        "plant_name": PLANT_NAME,
        "plant_capacity": capacity_kw,
        "forecast": {
            "issued_at": pd.Timestamp(row["발행시각"]).strftime("%Y-%m-%d %H:%M:%S"),
            "target_at": pd.Timestamp(row["대상시각"]).strftime("%Y-%m-%d %H:%M:%S"),
            "tier": row["티어"],
            "horizon_h": int(row["수평_h"]),
            "aggregation": row["집계기준"],
            "model": row["채택모델"],
            "plant_ac_power_kw": float(row["예측_ac_power_kw"]),
            "plant_daily_energy_kwh(경로A, 일간총량)": daily_val,
            "plant_total_energy_kwh": None,
            "_total_energy_비고": "앵커(API 최신 total_energy) 주입 전에는 산출 불가",
        },
    }


# ── 엔드투엔드 스모크테스트 ──────────────────────────────────────────
def run_smoke_test(config: dict, capacity_kw: float, seed: int) -> dict:
    """발전소 정적정보 + 실시간 입력 샘플 → 특성생성 → (재학습해 저장한)
    모델 파일 → 추론 → Blockdata 출력 변환까지 실제로 이어지는지 확인
    (수정 4번). 지금까지 이 프로젝트에 "배포용으로 저장된 모델 파일"이
    없었으므로, 이 스모크테스트가 그 첫 산출물이기도 하다.

    가장 단순한 채택 구성(초단기 +1h, raw kW, LightGBM 기본파라미터 —
    ⑤ 판정표상 대안후보 없이 기존 기본모델 유지)을 골라 **전체 가용
    데이터**로 재학습한다. 이 재학습은 ⑤의 최종성능 판단(폴드별
    rolling-origin 검증)을 대체하지 않는다 — 목적은 오직 "이 경로가
    끝까지 안 끊기는가"이지 성능 재평가가 아니다.
    """
    harness_mod = _load("harness_smoke", "backtest_harness_v1_2026-08-20밤.py")
    ultra_mod = _load("ultra_smoke", "train_ultra_short_official_v1_2026-08-21.py")

    quarter = ultra_mod.load_15min_base()
    hourly_df = pd.read_csv(harness_mod.DATASETS["v3_고정tm"]["path"],
                             parse_dates=["time"], low_memory=False).set_index("time").sort_index()
    H = 1
    frame = ultra_mod.build_ultra_short_frame(quarter, hourly_df, H)
    candidate_cols = (ultra_mod.EQUIP_STARTLABEL + harness_mod.sel.OBSERVED_COLUMNS
                       + harness_mod.sel.FORECAST_COLUMNS)
    base_cols = [c for c in frame.columns if c not in candidate_cols and not c.startswith("_")
                 and c not in ("목표_발전출력_kW", "목표_낮시간")]
    daylight = frame[frame["목표_낮시간"] > 0]

    for_sel = daylight.dropna(subset=["목표_발전출력_kW"])
    chosen = harness_mod.select_features_in_fold(for_sel, candidate_cols, 0.3, True, True)
    feature_cols = base_cols + chosen
    required = [c for c in feature_cols if c not in harness_mod.NATIVE_MISSING_OK] + ["목표_발전출력_kW"]
    train_full = daylight.dropna(subset=required)
    if len(train_full) < 500:
        raise RuntimeError(f"스모크테스트 학습표본이 너무 적다({len(train_full)}행)")

    model = ultra_mod.make_model("LightGBM", seed)
    model.fit(train_full[feature_cols], train_full["목표_발전출력_kW"])

    model_path = OUT / "smoke_초단기_1h_raw_deploy.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols, "horizon_h": H,
                 "capacity_kw": capacity_kw, "trained_rows": len(train_full)}, model_path)

    # ★핵심★: 저장된 파일을 "다시 불러와서" 추론한다(메모리의 model
    # 변수를 그대로 쓰지 않음) — 이래야 "저장→적재→추론" 경로를 실제로
    # 검증하는 것이다.
    loaded = joblib.load(model_path)

    # "실시간 입력 샘플" — Blockdata 실시간 API 연동은 아직 없으므로,
    # 특성이 전부 갖춰진 가장 최근 시각을 대리(proxy)로 쓴다. 실제 배포
    # 시엔 이 자리에 API 응답(+KMA 입력)이 같은 컬럼 구조로 들어간다.
    sample_time = train_full.index[-1]
    x = frame.loc[[sample_time], loaded["feature_cols"]]
    # NATIVE_MISSING_OK 특성(예: DSWRFLX_bsrn정제)은 원래도 결측이 허용되는
    # 항목이라(하네스가 애초에 dropna 대상에서 뺌) 여기서까지 결측 없음을
    # 요구하면 안 된다 — required(학습 시 실제로 dropna에 쓴 목록)만 검사.
    required_features = [c for c in loaded["feature_cols"] if c not in harness_mod.NATIVE_MISSING_OK]
    if x[required_features].isna().any(axis=None):
        raise RuntimeError(f"스모크테스트 샘플({sample_time})에 결측 필수 특성이 있음")

    pred_kw = float(np.clip(loaded["model"].predict(x)[0], 0, capacity_kw))
    issued_at = sample_time
    target_at = issued_at + pd.Timedelta(hours=loaded["horizon_h"])

    payload = {
        "plant_id": PLANT_ID, "plant_name": PLANT_NAME, "plant_capacity": capacity_kw,
        "forecast": {
            "issued_at": issued_at.strftime("%Y-%m-%d %H:%M:%S"),
            "target_at": target_at.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": "초단기", "horizon_h": loaded["horizon_h"],
            "aggregation": BUCKET["초단기"],
            "model": "raw(스모크테스트 전체재학습, LightGBM 기본파라미터)",
            "plant_ac_power_kw": round(pred_kw, 2),
        },
    }
    return {
        "통과": True,
        "model_path": str(model_path),
        "feature_count": len(loaded["feature_cols"]),
        "train_rows": int(loaded["trained_rows"]),
        "sample_issued_at": issued_at.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_target_at": target_at.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_ac_power_kw": round(pred_kw, 2),
        "payload": payload,
    }


# ── 자동 점검 ───────────────────────────────────────────────────────
def verify(ac: pd.DataFrame, daily_final: pd.DataFrame, daily_cum: pd.DataFrame,
           capacity_kw: float, harness_mod) -> pd.DataFrame:
    checks = []

    def add(항목, 대상, 통과, 상세):
        checks.append({"항목": 항목, "대상": 대상,
                       "결과": "통과" if 통과 else "★실패★", "상세": 상세})

    # 1) 시각정합성 — 내보낸 대상시각의 실측 집계값이, 그 모델이 실제로
    #    학습·시험에 쓴 자료와 같은가(티어별 shift 규약이 맞는지 검증).
    #    ★참조자료는 반드시 각 모델이 실제로 본 자료여야 한다★(수정 2번
    #    — 단기를 엉뚱한 집계자료와 비교해 거짓 실패가 났던 원인).
    q15 = pd.read_parquet(ROOT / "outputs" / "집계_15분_자료.parquet")["발전출력_kW"]
    short_ref = pd.read_csv(harness_mod.DATASETS["v3_고정tm"]["path"],
                             usecols=["time", "plant_output_kw"], parse_dates=["time"],
                             low_memory=False).set_index("time").sort_index()["plant_output_kw"]
    for tier, ref in [("초단기", q15), ("단기", short_ref)]:
        sub = ac[ac["티어"] == tier]
        for H, s in sub.groupby("수평_h"):
            joined = s.set_index("대상시각")["실측_ac_power_kw"].to_frame("csv실측")
            joined["집계실측"] = ref.reindex(joined.index).round(2)
            valid = joined.dropna()
            if not len(valid):
                add("시각정합성", f"{tier} +{H}h", False, "대상시각이 참조자료와 하나도 안 맞음")
                continue
            diff = (valid["csv실측"] - valid["집계실측"]).abs()
            # 부동소수점 정확 비교(<=) 대신 허용오차 비교(수정 1번) —
            # 우리 값·참조값 둘 다 독립적으로 round(2)되므로 최대 0.01kW
            # 차이는 정상이다.
            ok = bool(np.all(diff.to_numpy() <= TOL_KW)) and len(valid) / len(joined) > 0.99
            add("시각정합성", f"{tier} +{H}h", ok,
                f"매칭 {len(valid)}/{len(joined)}행, 최대 절대차 {diff.max():.4f}kW "
                f"(허용오차 {TOL_KW}kW — 대상시각 = 발행시각 + {target_offset_hours(tier, H)}h, "
                f"참조자료: {'집계_15분_자료.parquet' if tier == '초단기' else 'v3_고정tm(plant_output_kw)'})")

    # 2) 물리 범위
    for tier, s in ac.groupby("티어"):
        bad = ((s["예측_ac_power_kw"] < 0) | (s["예측_ac_power_kw"] > capacity_kw)).sum()
        add("ac_power 유효범위", tier, bad == 0,
            f"0~{capacity_kw}kW 벗어난 예측 {bad}행")

    # 3) 자릿수(소수점 둘째자리) — ac_power·daily_energy 경로A·경로B 전부
    for col, df, name in [
        ("예측_ac_power_kw", ac, "ac_power"),
        ("예측_일간최종총량_kwh", daily_final, "daily_energy_경로A"),
        ("예측_daily_energy_누적_kwh", daily_cum, "daily_energy_경로B"),
    ]:
        if not len(df):
            continue
        v = df[col].dropna()
        bad = int((v.round(2) != v).sum())
        add("자릿수(소수점 2자리)", name, bad == 0, f"규격 위반 {bad}행")

    # 4) daily_energy 두 경로 불일치율 — 경로A(일간 최종총량)와 경로B의
    #    "그 날 마지막 누적값"이 얼마나 다른가.
    if len(daily_final) and len(daily_cum):
        a = daily_final.set_index("날짜")["예측_일간최종총량_kwh"]
        for path, b in daily_cum.groupby("경로"):
            b = b.sort_values("대상시각")
            counts = b.groupby("날짜").size()
            full_dates = counts[counts >= counts.max() * 0.95].index
            # 하루가 통째로 채워진 날만(부분 누적을 총량과 비교하면 무의미)
            day_end = (b[b["날짜"].isin(full_dates)]
                       .groupby("날짜")["예측_daily_energy_누적_kwh"].last())
            common = a.index.intersection(day_end.index)
            if len(common) < 5:
                add("daily_energy 경로간 불일치", path, True,
                    f"공통 완전일 {len(common)}일 — 비교 표본 부족(판정 보류)")
                continue
            rel = ((day_end.loc[common] - a.loc[common]).abs()
                   / a.loc[common].replace(0, np.nan) * 100).dropna()
            add("daily_energy 경로간 불일치", path, True,
                f"공통 {len(common)}일, 중앙 불일치 {rel.median():.1f}%, 최대 {rel.max():.1f}% "
                "(정보용 — 두 경로는 서로 다른 모델이므로 불일치 자체가 실패는 아님)")

    # 5) daily_energy 경로B: 비음수 + 하루 안에서 단조 비감소(누적이므로
    #    감소하면 그 자체로 규격 위반)
    if len(daily_cum):
        bad_neg = int((daily_cum["예측_daily_energy_누적_kwh"] < 0).sum())
        add("daily_energy 비음수", "경로B 누적", bad_neg == 0, f"음수 {bad_neg}행")

        non_monotonic = 0
        for _, g in daily_cum.groupby(["경로", "날짜"]):
            g = g.sort_values("대상시각")
            if (g["예측_daily_energy_누적_kwh"].diff().dropna() < -1e-6).any():
                non_monotonic += 1
        add("daily_energy 누적 단조성", "경로B", non_monotonic == 0,
            f"하루 안에서 누적값이 감소하는 (경로,날짜) 조합 {non_monotonic}개")

    # 6) 중복·결측
    if len(ac):
        dup = int(ac.duplicated(subset=["티어", "수평_h", "발행시각"]).sum())
        add("중복행", "ac_power", dup == 0, f"중복 {dup}행")
        na = int(ac["예측_ac_power_kw"].isna().sum())
        add("예측 결측", "ac_power", na == 0, f"결측 {na}행")

    return pd.DataFrame(checks)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    capacity_kw = float(config["site"]["capacity_kw"])
    seed = int(config["random_seed"])

    verdict = pd.read_csv(SRC / "판정표.csv", encoding="utf-8-sig")
    rows = {
        "초단기": pd.read_csv(SRC / "초단기_행단위.csv", encoding="utf-8-sig"),
        "단기": pd.read_csv(SRC / "단기_행단위.csv", encoding="utf-8-sig"),
        "일간": pd.read_csv(SRC / "일간_행단위.csv", encoding="utf-8-sig"),
    }

    print("=== 최종 채택 모델(판정표에서 자동 확정) ===")
    adopted = resolve_adopted(verdict, rows)
    for a in adopted:
        h = "-" if a["수평_h"] is None else f"+{a['수평_h']}h"
        print(f"  {a['티어']:4s} {h:6s} → {a['후보']}")

    spec = build_spec(capacity_kw)
    (OUT / "규격정의.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ac_power 변환 ===")
    ac = build_ac_power(rows, adopted)
    ac.to_csv(OUT / "예측_ac_power.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(ac):,}행")

    print("\n=== daily_energy 변환 — 경로A(일간총량) ===")
    daily_final = build_daily_final(rows, adopted)
    daily_final.to_csv(OUT / "예측_daily_energy_일간총량.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(daily_final):,}행")

    print("\n=== daily_energy 변환 — 경로B(대상시각별 누적시계열) ===")
    daily_cum = build_daily_cumulative(ac)
    daily_cum.to_csv(OUT / "예측_daily_energy_누적시계열.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(daily_cum):,}행 / 경로: "
          f"{sorted(daily_cum['경로'].unique()) if len(daily_cum) else []}")

    print("\n=== total_energy 구조(앵커 null) ===")
    total = build_total_energy(daily_final)
    total.to_csv(OUT / "예측_total_energy.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(total):,}행 — 앵커 미주입 상태(운영 시 API 실측값 필요)")

    payload = build_payload_sample(ac, daily_final, capacity_kw)
    (OUT / "blockdata_payload_예시.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 엔드투엔드 스모크테스트(정적정보+샘플입력 → 저장모델 재적재 → 추론 → 규격출력) ===")
    try:
        smoke = run_smoke_test(config, capacity_kw, seed)
        print(f"  통과 — 저장모델: {smoke['model_path']}")
        print(f"  샘플 발행시각 {smoke['sample_issued_at']} → 대상시각 {smoke['sample_target_at']}, "
              f"예측 {smoke['predicted_ac_power_kw']}kW (학습표본 {smoke['train_rows']:,}행, "
              f"특성 {smoke['feature_count']}개)")
    except Exception as e:  # noqa: BLE001 — 스모크테스트는 실패해도 나머지 산출물은 저장해야 함
        smoke = {"통과": False, "오류": f"{type(e).__name__}: {e}"}
        print(f"  ★실패★ {smoke['오류']}")
    (OUT / "스모크테스트_결과.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== 자동 규격 점검 ===")
    harness_mod = _load("harness_verify", "backtest_harness_v1_2026-08-20밤.py")
    checks = verify(ac, daily_final, daily_cum, capacity_kw, harness_mod)
    checks.to_csv(OUT / "검증_규격점검.csv", index=False, encoding="utf-8-sig")
    print(checks.to_string(index=False))

    fails = checks[checks["결과"] != "통과"]
    fail_count = len(fails) + (0 if smoke.get("통과") else 1)

    lines = [
        "⑥ Blockdata 출력 규격화 — 요약",
        f"발전소: {PLANT_NAME}(plant_id={PLANT_ID}), 용량 {capacity_kw}kW",
        f"ac_power 행수: {len(ac):,} / daily_energy 경로A: {len(daily_final):,}행 "
        f"/ 경로B: {len(daily_cum):,}행",
        f"규격점검 {len(checks)}건 중 실패 {len(fails)}건, "
        f"스모크테스트 {'통과' if smoke.get('통과') else '실패'}",
        f"총 실패 {fail_count}건",
        "",
        "★규격 대조에서 확정된 구조적 차이 4가지(보고서에 그대로 옮길 것)★",
        "1. Blockdata ac_power/daily_energy/total_energy는 inverter[] 원소 필드다 —",
        "   우리 예측(발전소 총합)은 Σ inverter[]에 대응한다. 인버터별로 쪼개지 않는다.",
        "2. Blockdata는 5분 순간값, 우리 예측은 15분/1시간 버킷평균 — 집계기준이 다르다.",
        "3. daily_energy는 두 경로(경로A=일간모델 최종총량 / 경로B=대상시각별 누적시계열)가",
        "   있고 서로 어긋날 수 있다 — ⑤에서 일간 계층조정 가중치가 직접100%/시간0%로",
        "   붕괴해 두 경로가 사실상 독립이다.",
        "4. total_energy는 절대 누적값이라 모델 예측 대상이 아니다 — 앵커+증분 구조로만 정의.",
        "",
        "★남은 판단(사용자 확인 필요)★",
        "- daily_energy 운영 기본경로를 A(일간모델)로 할지 B(시간적분)로 할지 미확정.",
        "- total_energy 앵커 주입은 실제 API 연동 시점에만 가능(지금은 형식만 고정).",
        "- 스모크테스트는 초단기 +1h(가장 단순한 구성)만 검증했다 — 나머지 티어·수평의",
        "  실제 배포모델 저장·재적재는 아직 확장 안 함.",
    ]
    if len(fails):
        lines += ["", "★규격점검 실패 항목★"] + [f"- {r['항목']} / {r['대상']}: {r['상세']}"
                                              for _, r in fails.iterrows()]
    if not smoke.get("통과"):
        lines += ["", "★스모크테스트 실패★", f"- {smoke.get('오류')}"]
    (OUT / "요약.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n저장 완료: {OUT}")
    if fail_count:
        print(f"\n★규격점검/스모크테스트 총 {fail_count}건 실패 — 요약.txt 확인, "
              f"⑥ 완료 처리하지 말 것★")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
