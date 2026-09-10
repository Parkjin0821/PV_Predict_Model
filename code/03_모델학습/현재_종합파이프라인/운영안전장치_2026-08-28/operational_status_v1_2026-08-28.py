# -*- coding: utf-8 -*-
"""공통 운영 상태코드·예측레코드 스키마(사용자 지시 4번).

모든 예측기·대시보드가 같은 어휘를 쓰도록 상태코드를 한 곳에 정의한다.
이 파일은 **정의와 검증만** 담당하고 수집·예측·저장은 하지 않는다.

## 상태코드(사용자 지정 그대로)
- normal          : 최신 정상 입력으로 예측
- fallback_03run  : 09시런 실패로 정상 03시런 사용
- degraded        : 사전 허용된 대체입력 또는 일부 품질저하
- blocked         : 필수 입력 부족으로 예측 미생성
- error           : 실행·저장·모델 오류

## 설계 원칙
- `normal`/`fallback_03run`만 "예측값이 존재"할 수 있다. `degraded`는
  사전 허용된 경우에만 값을 가질 수 있고, `blocked`/`error`는 예측값을
  가지면 안 된다(가짜 성공 방지 — AGENTS.md 08-26~27 전반의 원칙).
- 상태코드만으로는 부족하므로 `사유`·`사용런`·`결측특성`·`보간내역`을
  항상 함께 저장하도록 레코드 스키마를 강제한다.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

NORMAL = "normal"
FALLBACK_03RUN = "fallback_03run"
DEGRADED = "degraded"
BLOCKED = "blocked"
ERROR = "error"

ALL_STATUSES = (NORMAL, FALLBACK_03RUN, DEGRADED, BLOCKED, ERROR)

# 예측값(숫자)을 가질 수 있는 상태. degraded는 "사전 허용된 대체입력"일
# 때만이므로 별도 플래그(allow_value_when_degraded)로 호출부가 명시한다.
VALUE_ALLOWED = (NORMAL, FALLBACK_03RUN)
VALUE_FORBIDDEN = (BLOCKED, ERROR)

# NWP 런 식별자
RUN_03 = "03run_prev18utc"   # 전날 18UTC = 당일 03시 KST
RUN_09 = "09run_day00utc"    # 당일 00UTC = 당일 09시 KST
RUN_NONE = "none"


class StatusError(ValueError):
    """상태코드 계약 위반(예측값과 상태가 모순되는 경우 등)."""


@dataclass
class PredictionRecord:
    """예측 1건의 필수 저장 항목(사용자 지시 4번의 '각 예측 레코드에
    반드시 저장' 목록을 그대로 필드화)."""
    status: str
    status_reason: str
    nwp_run_used: str
    input_reference_time: str      # 입력 기준시각(발행시각)
    target_time: str               # 대상시각
    missing_features: list = field(default_factory=list)
    interpolation_log: dict = field(default_factory=dict)   # 보간·fallback 내역
    bundle_version: str = ""
    created_at: str = ""
    tier: str = ""
    horizon_h: Any = None
    predicted_value: Any = None    # kW 또는 kWh, 상태가 허용할 때만

    def validate(self, allow_value_when_degraded: bool = False) -> None:
        if self.status not in ALL_STATUSES:
            raise StatusError(f"정의되지 않은 상태코드: {self.status}")
        if not self.status_reason:
            raise StatusError("status_reason은 비워둘 수 없다(원인 없는 상태 금지)")
        if self.nwp_run_used not in (RUN_03, RUN_09, RUN_NONE):
            raise StatusError(f"정의되지 않은 NWP 런 식별자: {self.nwp_run_used}")

        has_value = self.predicted_value is not None
        if self.status in VALUE_FORBIDDEN and has_value:
            raise StatusError(
                f"{self.status} 상태인데 예측값({self.predicted_value})이 있다 — "
                "가짜 성공 금지 원칙 위반")
        if self.status in VALUE_ALLOWED and not has_value:
            raise StatusError(f"{self.status} 상태인데 예측값이 없다")
        if self.status == DEGRADED:
            if has_value and not allow_value_when_degraded:
                raise StatusError(
                    "degraded 상태에서 값을 저장하려면 호출부가 "
                    "allow_value_when_degraded=True로 사전 허용을 명시해야 한다")
        # blocked인데 결측특성이 비어있으면 '왜 막혔는지'를 알 수 없다.
        if self.status == BLOCKED and not self.missing_features and not self.status_reason:
            raise StatusError("blocked인데 결측특성·사유가 둘 다 비어있다")

    def to_dict(self) -> dict:
        return asdict(self)


def classify_nwp_outcome(run09_ok: bool, run03_ok: bool) -> tuple[str, str, str]:
    """NWP 가용성 → (상태코드, 사용런, 사유). 사용자 지시 1번의 분기 그대로.

    ★자동 fallback 금지★: 03시·09시 모두 실패해도 관측값·과거평균·0·
    지속성모델로 조용히 대체하지 않는다(별도 백테스트 통과 전까지).
    """
    if run09_ok:
        return NORMAL, RUN_09, "09시런 완결성 검사 통과 — 최신 런 승격"
    if run03_ok:
        return (FALLBACK_03RUN, RUN_03,
                "09시런 미가용/불완전 — 사전검증된 동일 목표일 03시런 유지")
    return (BLOCKED, RUN_NONE,
            "03시런·09시런 모두 미가용/불완전 — 2차 fallback은 백테스트 "
            "미검증이므로 자동대체 금지(AGENTS.md 08-27 잔여리스크 1)")


# ============================================================
# 08-28 사용자 외부검토("클로드 코드에 대한 구멍 및 보완점 4가지") 반영.
# 아래 4개 유틸은 이 안전장치 패키지의 다른 모든 모듈이 공유해서 쓴다
# (재구현 금지 원칙 - 한 곳에만 정의).
# ============================================================

# --- 보완 1: 운영 DB 쓰기 접근의 물리적 강제 차단 -----------------------
# 지적: "클로드 코드가 스크립트 작성 중 실수로 운영 DB 경로를 참조하거나
# 테스트용 DB 생성 시 실수로 운영 DB 파일을 덮어쓸 가능성" - 코드 리뷰로
# 막는 데는 한계가 있으므로 OS/SQLite 레벨에서 물리적으로 차단한다.
KNOWN_OPERATIONAL_DB_NAMES = frozenset({
    "blockdata_history.sqlite3", "kma_live_inputs.sqlite3", "shadow_predictions.sqlite3",
})


class OperationalDBWriteBlocked(Exception):
    """운영 DB에 쓰기를 시도했거나, 테스트 DB 경로가 운영 DB 파일명과
    같아 안전하지 않다고 판단된 경우."""


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """운영 DB를 조회할 때는 항상 이 함수를 쓴다. SQLite URI `mode=ro`로
    접속해 이 커넥션으로는 애초에 INSERT/UPDATE/DELETE가 SQLite 자체에서
    거부된다(파이썬 로직 실수 여부와 무관한 물리적 강제) - 코드에서
    실수로 쓰기 쿼리를 넣어도 `sqlite3.OperationalError: attempt to write
    a readonly database`로 즉시 실패한다."""
    db_path = Path(db_path)
    uri = "file:%s?mode=ro" % db_path.resolve().as_posix()
    return sqlite3.connect(uri, uri=True)


def assert_safe_test_db_path(path: str | Path) -> None:
    """오프라인 테스트가 만드는 임시 SQLite 경로가 안전한지 코드 차원에서
    확인한다: (1) 파일명이 운영 DB 파일명과 같으면 무조건 차단,
    (2) 시스템 임시디렉터리(tempfile.gettempdir()) 하위가 아니면 차단.
    테스트 픽스처가 실수로 프로젝트 폴더나 운영 DB 경로를 가리키는 사고를
    막는다 - `tests/`의 모든 sqlite3.connect() 호출 전에 이 검사를 거치는
    걸 권장한다."""
    p = Path(path).resolve()
    if p.name in KNOWN_OPERATIONAL_DB_NAMES:
        raise OperationalDBWriteBlocked(
            "테스트 DB 경로의 파일명이 운영 DB와 동일하다: %s - 절대 금지" % p)
    tmp_root = Path(tempfile.gettempdir()).resolve()
    try:
        p.relative_to(tmp_root)
    except ValueError:
        raise OperationalDBWriteBlocked(
            "테스트 DB 경로가 시스템 임시디렉터리(%s) 밖이다: %s - "
            "오프라인 테스트는 반드시 tempfile 기반 경로만 써야 한다"
            % (tmp_root, p))


# --- 보완 2: Windows os.replace()의 파일 락 대응 ------------------------
# 지적: "Windows에서는 프로세스가 파일을 열고 있는 동안 os.replace()를
# 수행하면 PermissionError가 발생" - POSIX rename과 달리 Windows는 대상
# 파일이 열려 있으면(백신 스캔, 다른 프로세스의 읽기 등) 원자적 교체가
# 실패할 수 있다. 짧은 재시도로 일시적 락을 흡수하고, 그래도 실패하면
# staging 잔여물을 정리한 뒤 명확히 예외를 낸다(조용히 무시하지 않음).
def atomic_replace_with_retry(src: str | Path, dst: str | Path,
                              retries: int = 5, delay_sec: float = 0.2) -> None:
    """`os.replace(src, dst)`를 재시도와 함께 수행. 모든 재시도가
    PermissionError로 실패하면 staging 파일(src)을 정리하고 예외를 올린다
    - 잔여물을 남기지 않는다(기존 nwp_dual_run_manager의 "잔여물 없음"
    원칙과 동일한 기준을 Windows 파일락 상황에도 적용)."""
    src, dst = Path(src), Path(dst)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay_sec * (attempt + 1))  # 점증 backoff
    # 전부 실패 - staging 잔여물 정리 시도(정리 자체가 실패해도 원래
    # 예외를 감추지 않는다)
    try:
        if src.exists():
            src.unlink()
    except OSError:
        pass
    raise OperationalDBWriteBlocked(
        "os.replace(%s -> %s)가 %d회 재시도 후에도 PermissionError로 실패 "
        "(Windows 파일 락 의심) - staging 정리 시도함. 원본 예외: %s"
        % (src, dst, retries, last_exc)) from last_exc


# --- 보완 4: 출력 파일이 실제로 존재/의미있는 내용인지 실행로그에 증명 --
# 지적: "구문검사 통과만 보고 성공으로 착각하고 실제 end-to-end 흐름을
# 검증하지 않은 채 끝날 수 있음" - write_outputs() 등이 경로 문자열만
# 반환하고 끝내지 않고, 실제 파일 크기·상위 5줄을 실행 로그에 직접
# 출력하도록 이 함수를 표준 후처리로 쓴다.
def verify_file_output(path: str | Path, n_lines: int = 5) -> dict:
    """파일이 실제로 존재하는지, 크기가 0이 아닌지, 상위 n_lines가 어떤
    내용인지 확인해 dict로 반환하고 그대로 print해서 쓴다. 파일이 없거나
    비어있으면 "성공"이라 주장하지 않고 그 사실을 그대로 담는다."""
    p = Path(path)
    if not p.is_file():
        return {"경로": str(p), "존재": False, "크기_bytes": 0, "상위%d줄" % n_lines: []}
    size = p.stat().st_size
    lines: list[str] = []
    try:
        with p.open("r", encoding="utf-8-sig", errors="replace") as f:
            for _ in range(n_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    except OSError as exc:
        lines = ["<읽기 실패: %s>" % exc]
    return {
        "경로": str(p), "존재": True, "크기_bytes": size,
        "상위%d줄" % n_lines: lines,
        "비어있음_주의": size == 0,
    }


def verify_and_print_outputs(paths: dict[str, str], n_lines: int = 5) -> dict:
    """write_outputs() 등이 반환한 {이름: 경로} 딕셔너리 전체를
    verify_file_output()으로 검증하고 즉시 print한다 - 호출부가 "산출물
    경로를 반환했으니 성공"이라고 넘어가지 않고, 이 함수를 거쳐야 실제
    파일 존재·크기·내용이 실행 로그에 남는다."""
    import json
    report = {}
    for name, p in paths.items():
        info = verify_file_output(p, n_lines=n_lines)
        report[name] = info
        print("[산출물검증] %s: 존재=%s 크기=%dB" % (name, info["존재"], info["크기_bytes"]))
        for line in info.get("상위%d줄" % n_lines, []):
            print("   | %s" % line)
    return report
