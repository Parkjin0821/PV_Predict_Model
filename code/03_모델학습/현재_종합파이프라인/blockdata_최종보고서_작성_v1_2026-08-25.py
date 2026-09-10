# -*- coding: utf-8 -*-
"""12항목 명세를 실제로 다 채운 완결 보고서(기획서 포맷) 작성.

`blockdata_backtest_최종보고_v1_2026-08-25.py`가 만든 CSV들을 새로
계산하지 않고 그대로 읽어 markdown 표로 전부 박아넣는다(요약본처럼
"CSV 참고"로 넘기지 않음). 데이터 재생성 없음 — 순수 보고서 조판.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs" / "Blockdata_오프라인백테스트_최종보고_v1_2026-08-25"
OUT = SRC / "Blockdata_오프라인백테스트_완결보고서_v1_2026-08-25.md"


def to_markdown_table(d: pd.DataFrame) -> str:
    """tabulate 미설치 환경 대응 — 직접 GFM 표 조판."""
    headers = [str(c) for c in d.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in d.itertuples(index=False)]
    line0 = "| " + " | ".join(headers) + " |"
    line1 = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = [line0, line1] + ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def df_md(path: str, cols: list[str] | None = None, round_map: dict | None = None) -> str:
    d = pd.read_csv(SRC / path, encoding="utf-8-sig")
    if cols:
        d = d[cols]
    if round_map:
        for c, n in round_map.items():
            if c in d.columns:
                d[c] = d[c].round(n)
    return to_markdown_table(d)


def main():
    meta = json.loads((SRC / "Blockdata_오프라인백테스트_최종요약.json").read_text(encoding="utf-8"))
    daily_a = json.loads((SRC / "daily_energy_A_성능.json").read_text(encoding="utf-8"))

    parts = []
    parts.append("# Blockdata 오프라인 백테스트 — 완결 보고서 (2026-08-25)\n")
    parts.append("사용자 12항목 명세를 전부 실제 수치로 채운 문서. 재학습 없음 — "
                  "08-25 최종 공식모델(초단기·단기 v2 + 일간 58특성 직접모델)의 저장된 "
                  "예측·정답(OOF)과 Blockdata v4 변환본을 그대로 집계했다.\n")

    # 1. 기준정보
    parts.append("## 1. 검증 기준정보\n")
    for k, v in meta["메타"].items():
        parts.append(f"- **{k}**: {v}")
    parts.append("")

    # 2. ac_power 수평별
    parts.append("## 2. ac_power 수평별 정확도\n")
    parts.append("초단기 +1/+2/+3/+4h, 단기 +1/+24/+48h — 절대 합치지 않고 각각 표기.\n")
    parts.append(df_md("ac_power_수평별_성능.csv"))
    parts.append("\n### 2-1. 폴드(계절)별\n")
    parts.append(df_md("ac_power_폴드별_성능.csv"))
    parts.append("\n### 2-2. 시간대별(대상시각 KST 기준)\n")
    parts.append(df_md("ac_power_시간대별_성능.csv"))
    parts.append("")

    # 3. Skill
    parts.append("## 3. Skill 기준선 상세\n")
    parts.append("계산식: Skill = 1 − 모델_RMSE(또는 MAE) / 기준선_RMSE(또는 MAE). "
                  "두 기준선(단순지속성, 청천지수지속성) 모두 유효한 **공통 시험행**만 써서 "
                  "같은 티어·수평 안에서 모델_MAE가 기준선 표기와 무관하게 동일함을 보장했다 "
                  "(최초 실행에서 이 문제를 발견해 수정한 이력은 AGENTS.md 08-25절 참고). "
                  "낮시간 필터는 기준선에도 모델과 동일하게 적용됨.\n")
    parts.append(df_md("Skill_기준선별_비교.csv"))
    parts.append("")

    # 4. 변환전후
    parts.append("## 4. Blockdata 변환 전후 성능 동일성\n")
    parts.append(f"키검증(티어·수평·발행시각·대상시각 기준, 74,743행 전체): "
                  f"{meta['핵심수치']['Blockdata_변환전후_키검증']}\n")
    parts.append(df_md("ac_power_변환전후_감사.csv"))
    parts.append("")

    # 5. daily A
    parts.append("## 5. daily_energy 경로A(직접모델) 정확도\n")
    for k, v in daily_a.items():
        parts.append(f"- **{k}**: {v}")
    parts.append("\n### 5-1. 계절(폴드)별\n")
    parts.append(df_md("daily_energy_A_계절별_성능.csv"))
    parts.append("")

    # 6. daily B
    parts.append("## 6. daily_energy 경로B(시간적분 누적) 정확도\n")
    parts.append("### 6-1. 누적시점별(모든 버킷)\n")
    parts.append(df_md("daily_energy_B_누적시점별_성능.csv"))
    parts.append("\n### 6-2. 일마감 총량(완전일만 — 그 (경로,날짜)의 버킷수가 "
                  "해당 경로 최빈 버킷수의 90% 이상인 날짜만 인정)\n")
    parts.append(df_md("daily_energy_B_일마감_성능.csv"))
    parts.append("")

    # 7. A vs B
    parts.append("## 7. 경로A vs 경로B 비교(동일 날짜만)\n")
    parts.append(df_md("daily_energy_A_B_동일날짜비교.csv"))
    parts.append("\n**운영 권고**: 경로A(D+1 일간 총량 공식예측)와 경로B(시간대별 ac_power "
                 "적분 누적 참고값)는 서로 다른 모델이다. **어느 쪽도 임의로 정답 취급하지 "
                 "않는다.** 일간 공식모델이 직접모델(경로A)로 확정됐고, 일마감 기준 경로A가 "
                 "경로B보다 대체로 정확하므로(6-2 표 대비 5번 daily A 요약) **일마감 공식값은 "
                 "경로A를 쓰고, 경로B는 발전소가 하루 중 실시간으로 갱신되는 운영 참고치로만 "
                 "쓸 것을 권고한다.**\n")

    # 8. 계절·시간대·출력구간·구름전이
    parts.append("## 8. 계절·시간대별 취약구간\n")
    parts.append("(2-1, 2-2절과 데이터는 동일 — 여기서는 출력구간·구름전이 관점만 추가)\n")
    parts.append("### 8-1. 출력구간별(티어 전체 수평 합산)\n")
    parts.append(df_md("ac_power_출력구간별_성능.csv"))
    parts.append("\n### 8-2. 구름전이 구간별(전운량 변화량 3분위, 수평별)\n")
    parts.append(df_md("ac_power_구름전이별_성능.csv"))
    parts.append("\n**가을 결함구간 vs 겨울 물리적 난이도**: 가을 결함구간(2025-08-19~11-16, "
                 "인버터5 통신두절)은 정책B로 **학습·시험 자체에서 제외**돼 있어 위 2-1 "
                 "폴드별 표의 `2_가을_결함종료후`는 결함이 끝난 뒤(11-17~)만 담고 있다. "
                 "반면 겨울의 상대적 저성능은 결측·정책 문제가 아니라 **낮은 태양고도·짧은 "
                 "일조시간이라는 순수한 물리적 난이도**다 — 둘을 같은 원인으로 섞지 않는다.\n")

    # 9. 규격점검
    parts.append("## 9. Blockdata 규격 감사 결과(23건 개별)\n")
    parts.append(df_md("Blockdata_규격점검_23건.csv"))
    parts.append(f"\n통과: {meta['핵심수치']['규격점검_통과']}. "
                 "**규격점검 통과는 '형식이 맞는가'만 보장하며 예측 정확도 통과와 다른 "
                 "의미다** — 정확도 답은 2·3·5번 절 참고.\n")

    # 10. total_energy
    parts.append("## 10. total_energy 한계\n")
    for x in meta["total_energy_한계"]:
        parts.append(f"- {x}")
    parts.append("")

    # 11. 이번 검증의 한계
    parts.append("## 11. 이번 검증의 한계\n")
    for x in meta["이번검증의_한계"]:
        parts.append(f"- {x}")
    parts.append("- **알려진 gap**: 규격점검·스모크테스트 스크립트에 \"실패 시 exit code 1\" "
                 "처리가 없다(실패건수는 요약.txt에 기록하지만 프로세스 종료코드는 항상 0).\n")

    # 12. 최종 결론
    parts.append("## 12. 최종 결론(3문장, 분리)\n")
    for k, v in meta["결론_3문장_분리"].items():
        parts.append(f"**{k}**: {v}\n")

    parts.append("\n---\n")
    parts.append(f"산출물 원본 CSV/JSON 경로: `{SRC}`")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"저장: {OUT}")
    print(f"길이: {len(OUT.read_text(encoding='utf-8')):,}자")


if __name__ == "__main__":
    main()
