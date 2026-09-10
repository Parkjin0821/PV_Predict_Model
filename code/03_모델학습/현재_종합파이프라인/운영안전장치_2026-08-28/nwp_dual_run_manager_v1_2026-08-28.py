# -*- coding: utf-8 -*-
"""NWP 이중런 완결성 검사·원자적 승격 관리자(사용자 지시 1번).

## 이 모듈이 하는 일 / 하지 않는 일
- **한다**: 이미 수집·저장된 `kma_live_inputs.sqlite3`의 NWP를 목표일별로
  읽어 03시런/09시런 각각의 완결성을 검사하고, 규칙에 따라 어느 런을
  쓸지 결정해 매니페스트를 **원자적으로** 승격한다.
- **안 한다**: 외부 API 호출(사용자 지시로 금지). 수집 자체는 기존
  `collect_kma_nwp_live_v1_2026-08-25.py`(09시런)와 아래 "미구현" 항목의
  03시런 수집기가 담당한다.

## 현재 확인된 공백(숨기지 않고 기록)
기존 운영 수집기는 fixed_run_for_issue_day()가 당일 00UTC(=09시런)로
고정돼 있어 03시런(전날 18UTC)을 운영 DB에 저장하는 경로가 아직 없다.
measure_nwp_availability_v1_2026-08-27.py는 진단 전용이라 운영 DB에
쓰지 않는다. 따라서 이 관리자는 지금 실행하면 03시런을 "미수집"으로
정직하게 보고한다. 03시런 수집기는 run_offset_hours 인자를 받는 형태로
기존 수집기를 확장하면 되며, API 호출이 필요하므로 이번 작업 범위
밖이다(Codex가 08-30 이후 수행).

## 승격 규칙(config/nwp_dual_run.json에서 읽음, 결과 보기 전 동결)
1. promotion_enabled=false(현재)면 09시런 승격을 시도하지 않는다.
2. 09시런이 필수변수x필수시각 완결성을 통과하고 승격허용시각 이후면 승격.
3. 실패/불완전이면 기존 정상 03시런을 그대로 유지(덮어쓰지 않음).
4. 둘 다 불가면 blocked - 관측값/평균/0/지속성으로 조용히 대체 금지.

## 원자성/멱등성
- staging 파일에 먼저 쓰고 -> 검증 -> os.replace로 승격(POSIX/NTFS 모두
  원자적). 예외가 나도 기존 매니페스트는 손상되지 않는다.
- 같은 입력으로 여러 번 실행해도 결과가 같다(멱등). 이미 같은 내용이면
  재기록하지 않고 unchanged로 보고한다.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config" / "nwp_dual_run.json"
MANIFEST_DIR = HERE / "outputs" / "nwp_manifest"
LOG_PATH = HERE / "outputs" / "nwp_이중런_로그.csv"
KST = ZoneInfo("Asia/Seoul")
DEFAULT_KMA_DB = Path(
    r"C:\Users\u-cube\JIN\코덱스\결과물\예측모델\광주"
    r"\kma_live_inputs_v1_2026-08-25\kma_live_inputs.sqlite3")

sys.path.insert(0, str(HERE))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("opstatus", HERE / "operational_status_v1_2026-08-28.py")
opstatus = _ilu.module_from_spec(_spec)
sys.modules["opstatus"] = opstatus
_spec.loader.exec_module(opstatus)


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tm_utc(target_day: datetime, which: str) -> datetime:
    """목표일(D+1) 기준으로 각 런의 모델 발표시각(UTC naive)을 계산한다.

    09시런: 목표일 전날(D) 00UTC = D 09 KST
    03시런: 목표일 전전날(D-1) 18UTC = D 03 KST
    (기존 fixed_run_for_issue_day()가 issue_day D -> D 00UTC를 쓰고
    목표가 D+1인 것과 정확히 같은 규약)
    """
    issue_day = target_day - timedelta(days=1)
    base = datetime(issue_day.year, issue_day.month, issue_day.day)
    if which == opstatus.RUN_09:
        return base
    if which == opstatus.RUN_03:
        return base - timedelta(hours=6)
    raise ValueError("알 수 없는 런: " + str(which))


def check_run_completeness(db_path: Path, target_day: datetime, which: str,
                           cfg: dict, as_of: datetime | None = None) -> dict:
    """한 런의 완결성 검사. DB를 읽기만 하며 어떤 것도 쓰지 않는다."""
    run_tm = _run_tm_utc(target_day, which)
    run_tm_str = run_tm.strftime("%Y-%m-%dT%H:%M:%SZ")
    req_vars = cfg["required_vars"]
    req_hours = cfg["required_target_hours"]
    day_str = target_day.date().isoformat()

    result = {
        "런": which, "런_tm_utc": run_tm_str, "목표일": day_str,
        "수집됨": False, "완결": False, "필수값_기대": len(req_vars) * len(req_hours),
        "필수값_보유": 0, "결측수": 0, "결측률": 1.0,
        "부족변수": [], "부족시각": [], "최초수신": None, "사유": "",
    }
    if not Path(db_path).is_file():
        result["사유"] = "KMA DB 없음: " + str(db_path)
        return result

    # 08-28 외부검토 보완 1: 운영 DB(kma_live_inputs.sqlite3)를 조회할 수
    # 있는 경로라 읽기전용 접속을 강제한다(코드 실수로 쓰기 쿼리가 섞여도
    # SQLite 자체가 거부하도록 물리적으로 차단).
    with opstatus.connect_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT variable, target_time_kst, value, is_missing, first_received_at "
            "FROM nwp_values WHERE requested_tm_utc=? AND substr(target_time_kst,1,10)=?",
            (run_tm_str, day_str),
        ).fetchall()

    if not rows:
        result["사유"] = "해당 런의 저장자료 없음(미수집)"
        return result
    result["수집됨"] = True

    if as_of is not None:
        cutoff = as_of.isoformat()
        rows = [r for r in rows if (r[4] or "") <= cutoff]
        if not rows:
            result["사유"] = "as_of(" + cutoff + ") 이전 수신분 없음"
            return result

    have = {}
    received = []
    for var, tkst, val, is_missing, first_recv in rows:
        try:
            hour = int(str(tkst)[11:13])
        except (ValueError, IndexError):
            continue
        if var not in req_vars or hour not in req_hours:
            continue
        if not is_missing and val is not None:
            have[(var, hour)] = float(val)
        if first_recv:
            received.append(first_recv)

    result["필수값_보유"] = len(have)
    expected = result["필수값_기대"]
    result["결측수"] = expected - len(have)
    result["결측률"] = (result["결측수"] / expected) if expected else 1.0
    result["최초수신"] = min(received) if received else None
    result["부족변수"] = sorted(set(v for v in req_vars
                              if any((v, h) not in have for h in req_hours)))
    result["부족시각"] = sorted(set(h for h in req_hours
                              if any((v, h) not in have for v in req_vars)))

    if result["결측률"] <= float(cfg["max_missing_ratio"]):
        result["완결"] = True
        result["사유"] = "필수변수x필수시각 완결"
    else:
        result["사유"] = ("완결성 미달(결측률 %.3f > 허용 %s) 부족변수=%s" % (
            result["결측률"], cfg["max_missing_ratio"], result["부족변수"]))
    return result


def decide(target_day: datetime, cfg: dict, db_path: Path,
           now_kst: datetime | None = None) -> dict:
    """03/09 검사 -> 상태/사용런 결정. 순수함수(파일 안 씀)."""
    now_kst = now_kst or datetime.now(tz=KST)
    now_naive = now_kst.replace(tzinfo=None)

    c03 = check_run_completeness(db_path, target_day, opstatus.RUN_03, cfg, now_naive)
    c09 = check_run_completeness(db_path, target_day, opstatus.RUN_09, cfg, now_naive)

    promo_blocked_reason = ""
    run09_usable = c09["완결"]
    if run09_usable and not cfg.get("promotion_enabled", False):
        run09_usable = False
        promo_blocked_reason = ("promotion_enabled=false - 08-30 가용시각 실측 "
                                "확정 전까지 09시런 승격 비활성(설정파일 통제)")
    if run09_usable:
        deadline = cfg.get("run09_promotion_deadline_kst")
        if deadline:
            hh, mm = (int(x) for x in deadline.split(":"))
            limit = now_naive.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if now_naive > limit:
                sec = cfg.get("run09_secondary_deadline_kst")
                shh, smm = (int(x) for x in sec.split(":")) if sec else (23, 59)
                slimit = now_naive.replace(hour=shh, minute=smm, second=0, microsecond=0)
                if now_naive > slimit:
                    run09_usable = False
                    promo_blocked_reason = (
                        "승격 허용시각 경과(1차 " + str(deadline) + ", 2차 " + str(sec) +
                        ") - 마감 후 승격은 해당 제출에 의미 없음")

    status, run_used, reason = opstatus.classify_nwp_outcome(run09_usable, c03["완결"])
    if promo_blocked_reason and c03["완결"]:
        reason = reason + " / 09시런 미승격 사유: " + promo_blocked_reason
    elif promo_blocked_reason:
        reason = reason + " / " + promo_blocked_reason

    return {
        "목표일": target_day.date().isoformat(),
        "판정시각_kst": now_kst.isoformat(timespec="seconds"),
        "상태": status, "사용런": run_used, "사유": reason,
        "run03_검사": c03, "run09_검사": c09,
        "설정_promotion_enabled": cfg.get("promotion_enabled", False),
        "설정_2차fallback허용": cfg.get("allow_secondary_fallback", False),
    }


def promote(decision: dict, manifest_dir: Path = MANIFEST_DIR) -> dict:
    """staging -> 검증 -> os.replace 원자적 승격.

    불완전 자료가 기존 정상자료를 덮지 않도록: 새 판정이 blocked인데
    기존 매니페스트가 normal/fallback이면 기존 것을 유지하고 새 판정은
    .rejected.json으로만 남긴다.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    day = decision["목표일"]
    final = manifest_dir / ("nwp_" + day + ".json")
    staging = manifest_dir / ("nwp_" + day + ".staging.json")
    rejected = manifest_dir / ("nwp_" + day + ".rejected.json")

    prev = None
    if final.is_file():
        try:
            prev = json.loads(final.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = None

    outcome = {"승격": False, "결과": "", "경로": str(final)}

    if prev and prev.get("상태") in opstatus.VALUE_ALLOWED \
            and decision["상태"] in opstatus.VALUE_FORBIDDEN:
        rejected.write_text(json.dumps(decision, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        outcome["결과"] = ("기존 " + prev["상태"] + " 유지 - 신규 " + decision["상태"] +
                          "는 불완전이라 덮어쓰지 않음(rejected에 보존)")
        return outcome

    if prev:
        same = all(prev.get(k) == decision.get(k) for k in ("상태", "사용런", "목표일"))
        if same:
            outcome["결과"] = "unchanged(동일 판정 - 멱등)"
            return outcome

    try:
        staging.write_text(json.dumps(decision, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        chk = json.loads(staging.read_text(encoding="utf-8"))
        for key in ("목표일", "상태", "사용런", "사유"):
            if key not in chk:
                raise ValueError("staging 검증 실패 - 필수키 누락: " + key)
        if chk["상태"] not in opstatus.ALL_STATUSES:
            raise ValueError("staging 검증 실패 - 알 수 없는 상태: " + chk["상태"])
        # 08-28 외부검토 보완 2: Windows는 대상 파일이 다른 프로세스에
        # 열려있으면(백신 스캔 등) os.replace()가 PermissionError를 낼 수
        # 있다 - 재시도+backoff로 일시적 락을 흡수한다.
        opstatus.atomic_replace_with_retry(staging, final)
        outcome["승격"] = True
        outcome["결과"] = decision["상태"] + " / " + decision["사용런"]
    except Exception as exc:
        if staging.exists():
            staging.unlink()
        outcome["결과"] = ("승격 실패(" + type(exc).__name__ + ": " + str(exc) +
                          ") - 기존 매니페스트 보존")
    return outcome


def append_log(decision: dict, outcome: dict, log_path: Path = LOG_PATH) -> None:
    """필수 로그 항목(사용자 지시 1번 목록)을 CSV로 누적."""
    import csv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "기록시각": datetime.now(tz=KST).isoformat(timespec="seconds"),
        "목표일": decision["목표일"],
        "요청런_03_수집됨": decision["run03_검사"]["수집됨"],
        "요청런_09_수집됨": decision["run09_검사"]["수집됨"],
        "실제사용런": decision["사용런"],
        "run03_최초수신": decision["run03_검사"]["최초수신"],
        "run09_최초수신": decision["run09_검사"]["최초수신"],
        "판정시각": decision["판정시각_kst"],
        "run03_완결": decision["run03_검사"]["완결"],
        "run09_완결": decision["run09_검사"]["완결"],
        "run03_결측률": round(decision["run03_검사"]["결측률"], 4),
        "run09_결측률": round(decision["run09_검사"]["결측률"], 4),
        "run09_부족변수": ";".join(decision["run09_검사"]["부족변수"]),
        "실패원인": decision["사유"],
        "최종품질상태": decision["상태"],
        "승격여부": outcome["승격"],
        "승격결과": outcome["결과"],
        "매니페스트": outcome["경로"],
    }
    exists = log_path.is_file()
    with log_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run(target_day: datetime | None = None, db_path: Path | None = None,
        cfg: dict | None = None, now_kst: datetime | None = None) -> dict:
    cfg = cfg or load_config()
    db_path = Path(db_path or DEFAULT_KMA_DB)
    now_kst = now_kst or datetime.now(tz=KST)
    if target_day is None:
        target_day = (now_kst.replace(tzinfo=None) + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
    decision = decide(target_day, cfg, db_path, now_kst)
    outcome = promote(decision)
    append_log(decision, outcome)
    return {"decision": decision, "outcome": outcome}


if __name__ == "__main__":
    res = run()
    d, o = res["decision"], res["outcome"]
    print(json.dumps({
        "목표일": d["목표일"], "상태": d["상태"], "사용런": d["사용런"],
        "사유": d["사유"],
        "run03": {k: d["run03_검사"][k] for k in ("수집됨", "완결", "결측률", "사유")},
        "run09": {k: d["run09_검사"][k] for k in ("수집됨", "완결", "결측률", "사유")},
        "승격": o,
    }, ensure_ascii=False, indent=2))
    # 08-28 외부검토 보완 4: "승격됨"이라고 문자열로만 주장하지 않고
    # 실제 매니페스트·로그 파일이 존재/크기/내용을 갖는지 검증해 출력.
    if o.get("승격"):
        opstatus.verify_and_print_outputs({"매니페스트": o["경로"], "로그": str(LOG_PATH)})
