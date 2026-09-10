# -*- coding: utf-8 -*-
"""⑦ Blockdata 출력 규격화 — 3차 공식모델 v2(⑥ 초단기 +4h 패치 반영) 재적용.

## 왜 필요한가
`blockdata_export_3차공식모델_v1_2026-08-24.py`(이하 bd-v1)는 ⑤(v1)
E2E 재학습 산출물 기준이었다. 그 뒤 ⑥ v5 재판정 결과 중 검증을 통과한
초단기 +4h(날씨군집화 추가)만 반영한 v2 재학습
(`e2e_retrain_v5_공식B_v2_⑥반영_v1_2026-08-24.py`)이 새 공식모델이
됐으므로, Blockdata 산출물도 그 기준으로 다시 만든다. **bd-v1의
산출물(`Blockdata_규격화_3차공식모델_v1_2026-08-24/`)은 초단기 +4h에
한해 이제 stale.**

## 재사용(재구현 금지 원칙)
bd-v1을 모듈로 불러와 입력·출력 경로만 v2로 바꿔 그대로 재사용한다
(규격정의·ac_power/daily_energy 변환·경로B 누적·total_energy·payload·
v5 참조 verify·스모크테스트 로직 전부 동일 — 새로 안 짬). 스모크테스트는
초단기 +1h(이번 패치로 안 바뀐 수평) 모델을 그대로 재사용한다 — bd-v1의
스모크테스트 하네스가 raw 변형만 지원해서(kappa 역변환 로직 없음),
kappa인 +4h 대신 이번에도 변경 없는 +1h로 저장→재적재 검증한다(그래도
+4h 데이터는 ac_power CSV 안에 정상적으로 포함돼 22건 규격점검 전부를
받는다 — 스모크테스트는 파이프라인 자체의 무결성 확인용).

## 입력
`outputs/E2E_v5_공식B_v2_⑥반영_2026-08-24/`

## 출력
`outputs/Blockdata_규격화_3차공식모델_v2_2026-08-24/`
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bd3v1 = _load("bd3v2_base", "blockdata_export_3차공식모델_v1_2026-08-24.py")

# ★핵심★: bd-v1 모듈의 경로 상수만 v2로 바꿔치기하고 그 main()을 그대로 호출한다.
bd3v1.SRC3 = ROOT / "outputs" / "E2E_v5_공식B_v2_⑥반영_2026-08-24"
bd3v1.OUT3 = ROOT / "outputs" / "Blockdata_규격화_3차공식모델_v2_2026-08-24"

if __name__ == "__main__":
    print(f"입력: {bd3v1.SRC3}\n출력: {bd3v1.OUT3}\n")
    bd3v1.main()
