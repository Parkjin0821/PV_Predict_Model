# -*- coding: utf-8 -*-
"""⑦ Blockdata 출력 규격화 — 3차 공식모델 v3(일간 B그룹 제외 53특성 반영) 재적용.

## 왜 필요한가
`e2e_retrain_v5_공식B_v3_일간특성군_v1_2026-08-24.py`가 일간모델만 새로
바꿨다(초단기·단기는 v2와 완전 동일). Blockdata의 daily_energy 경로A
(일간총량, 직접모델)가 이 변경의 영향을 받으므로 다시 만든다. ac_power·
경로B(누적시계열)는 초단기·단기 산출물 그대로라 바뀔 게 없지만, 입력
파일 자체가 v3 디렉터리로 바뀌었으니 전체를 다시 돌려 일관성을
보장한다.

## 재사용
`blockdata_export_3차공식모델_v1_2026-08-24.py`를 모듈로 불러와 경로만
v3로 바꾼다(v2와 동일한 패턴).
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


bd3v1 = _load("bd3v3_base", "blockdata_export_3차공식모델_v1_2026-08-24.py")

bd3v1.SRC3 = ROOT / "outputs" / "E2E_v5_공식B_v3_일간특성군_2026-08-24"
bd3v1.OUT3 = ROOT / "outputs" / "Blockdata_규격화_3차공식모델_v3_2026-08-24"

if __name__ == "__main__":
    print(f"입력: {bd3v1.SRC3}\n출력: {bd3v1.OUT3}\n")
    bd3v1.main()
