"""광주 PV 용량 기준 프로필을 안전하게 조회·전환한다.

기존 스크립트는 site.capacity_kw를 읽으므로, 전환 시 active_profile과
capacity_kw를 함께 원자적으로 갱신한다. 219/240은 단순 환산값이 아니라
서로 다른 Blockdata 필드의 정의다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def capacity_context(config: dict) -> dict:
    site = config["site"]
    policy = site["capacity_policy"]
    active = policy["active_profile"]
    profile = policy["profiles"][active]
    active_kw = float(profile["capacity_kw"])
    legacy_kw = float(site["capacity_kw"])
    if abs(active_kw - legacy_kw) > 1e-9:
        raise ValueError(
            f"용량 설정 불일치: active_profile={active}({active_kw}kW), "
            f"site.capacity_kw={legacy_kw}kW"
        )
    return {"profile": active, **profile}


def set_profile(name: str, path: Path = CONFIG_PATH) -> dict:
    config = load_config(path)
    policy = config["site"]["capacity_policy"]
    if name not in policy["profiles"]:
        raise KeyError(f"알 수 없는 용량 프로필: {name}")
    policy["active_profile"] = name
    config["site"]["capacity_profile"] = name
    config["site"]["capacity_kw"] = float(policy["profiles"][name]["capacity_kw"])

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)
    return capacity_context(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="광주 PV 용량 기준 조회·전환")
    parser.add_argument("--profile", help="전환할 capacity_policy 프로필")
    parser.add_argument("--show", action="store_true", help="현재값과 전체 프로필 표시")
    args = parser.parse_args()

    if args.profile:
        current = set_profile(args.profile)
        print(f"전환 완료: {current['profile']} = {current['capacity_kw']}kW")
        print(f"출처: {current['source']}")
        print(f"상태: {current['status']}")

    config = load_config()
    current = capacity_context(config)
    if args.show or not args.profile:
        print(f"현재 활성 프로필: {current['profile']} = {current['capacity_kw']}kW")
        for name, profile in config["site"]["capacity_policy"]["profiles"].items():
            marker = "*" if name == current["profile"] else " "
            print(f"{marker} {name}: {profile['capacity_kw']}kW | {profile['label']}")


if __name__ == "__main__":
    main()
