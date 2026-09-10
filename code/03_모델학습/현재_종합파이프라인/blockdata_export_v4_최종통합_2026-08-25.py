# -*- coding: utf-8 -*-
"""⑦ Blockdata 출력 규격화 재적용 — 08-25 일간 58특성 최종모델 반영(오프라인 백테스트, API 미연동).

## 왜 필요한가
`blockdata_export_3차공식모델_v3_2026-08-24.py`(이하 v3)는 일간 53특성
모델(`E2E_v5_공식B_v3_일간특성군_2026-08-24`) 기준으로 규격점검·스모크
테스트를 통과시켰다. 그런데 08-25에 일간모델이 **전체58특성 직접모델**로
재확정되면서(부분가용 lag/native missing 정정) v3의 daily_energy
경로A(직접모델) 산출물이 stale해졌다. ac_power·경로B(누적시계열)는
초단기·단기 산출물 그대로라 안 바뀐다(v2와 동일, 08-25 단기 DIFSWRF
코드수정도 결과에 영향 없음을 이미 확인함).

## 하는 일 — Blockdata 스키마로의 "오프라인 백테스트"
사용자가 확인하고 싶은 것: 실제 Blockdata **API에 연결(전송/폴링)**한 게
아니라, **Blockdata가 기대하는 출력 스키마로 우리 백테스트 예측·실측을
변환해 22건 규격점검 + 재적재 스모크테스트를 다시 통과시키는 것**이다.
외부 API 호출·Blockdata 실전송은 0건(v1/v2/v3와 동일 원칙).

## 재사용(재구현 금지 원칙)
`blockdata_export_3차공식모델_v1_2026-08-24.py`의 검증된 로직(규격정의·
ac_power 변환·경로B 누적·total_energy·payload·스모크테스트·22건 verify)을
그대로 재사용한다. 새로 짜는 건 "일간 OOF 파일을 옛 스키마로 맞추는
어댑터"뿐이다.

## 입력
- ac_power: `outputs/E2E_v5_공식B_v2_⑥반영_2026-08-24/행단위_ac_power_
  예측정답.csv`(현재 공식 초단기·단기, 안 바뀜)
- daily: `outputs/일간_직접모델_최종감사_v1_2026-08-25/최종_직접모델_
  OOF.csv`(현재 공식 일간, 58특성) — 컬럼(날짜·폴드·실제_kWh·예측_kWh·
  후보)을 v1이 기대하는 스키마(plant_id·대상일·공식구성·폴드·예측_kWh·
  실제_kWh)로 어댑터 변환만 한다(값 자체는 안 건드림).
- fold_models: 스모크테스트에 필요한 초단기 +1h 5_초여름 모델 1개만
  v2에서 복사(그 모델은 v1 이후 안 바뀜).

## 출력 (`outputs/Blockdata_규격화_v4_최종통합_2026-08-25/`)
v1과 동일한 산출물 세트.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
AC_SRC = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24"
DAILY_SRC = ROOT / "outputs" / "일간_직접모델_최종감사_v1_2026-08-25"
STAGING = ROOT / "outputs" / "E2E_v5_공식B_v4_최종통합_2026-08-25"
PLANT_ID = 6715
SMOKE_MODEL_NAME = "초단기_h1_5_초여름_raw+구조선택[B_구간제외].joblib"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_staging() -> None:
    """v1이 기대하는 SRC3 폴더 구조(ac+daily+fold_models 한 곳)를 어댑터로 구성.
    원본 파일은 전혀 수정하지 않고 STAGING에만 복사/변환해 쓴다."""
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "fold_models").mkdir(exist_ok=True)

    # ac_power: v2 그대로 복사(내용 무변경 — 08-25 단기 코드수정이 결과에
    # 영향 없음을 이미 별도 재현으로 확인했으므로 그대로 신뢰)
    shutil.copy2(AC_SRC / "행단위_ac_power_예측정답.csv", STAGING / "행단위_ac_power_예측정답.csv")

    # daily: 08-25 OOF를 v1이 기대하는 컬럼명으로 어댑터 변환
    oof = pd.read_csv(DAILY_SRC / "최종_직접모델_OOF.csv", encoding="utf-8-sig")
    adapted = pd.DataFrame({
        "plant_id": PLANT_ID,
        "대상일": oof["날짜"],
        "공식구성": "전체58특성-직접모델(08-25 최종감사, 부분가용lag+native missing 정정)",
        "폴드": oof["폴드"],
        "예측_kWh": oof["예측_kWh"],
        "실제_kWh": oof["실제_kWh"],
    })
    adapted.to_csv(STAGING / "행단위_daily_예측정답.csv", index=False, encoding="utf-8-sig")

    # 스모크테스트용 초단기 모델 1개만 복사(그 모델은 v1 이후 안 바뀜)
    src_model = AC_SRC / "fold_models" / SMOKE_MODEL_NAME
    if not src_model.exists():
        raise FileNotFoundError(f"스모크테스트용 모델을 못 찾음: {src_model}")
    shutil.copy2(src_model, STAGING / "fold_models" / SMOKE_MODEL_NAME)

    print(f"스테이징 구성 완료: ac {len(pd.read_csv(STAGING/'행단위_ac_power_예측정답.csv')):,}행, "
          f"daily {len(adapted):,}행")


def main() -> None:
    build_staging()
    bd = _load("bd_v4_base", "blockdata_export_3차공식모델_v1_2026-08-24.py")
    bd.SRC3 = STAGING
    bd.OUT3 = ROOT / "outputs" / "Blockdata_규격화_v4_최종통합_2026-08-25"
    print(f"\n입력(어댑터 스테이징): {bd.SRC3}\n출력: {bd.OUT3}\n")
    bd.main()


if __name__ == "__main__":
    main()
