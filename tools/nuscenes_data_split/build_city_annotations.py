"""Build per-sample city annotations from nuScenes metadata.

The output schema is a JSON list:
[
  {"token": "<sample_token>", "city": "Boston|Singapore|Unknown"}
]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _common import (
    dump_json,
    index_by_token,
    load_nuscenes_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nuScenes sample-token to city annotations."
    )
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output intermediate city annotation JSON.",
    )
    return parser.parse_args()


def city_from_location(location: Optional[str]) -> str:
    if not location:
        return "Unknown"

    location = location.lower()
    if "boston" in location:
        return "Boston"
    if "singapore" in location:
        return "Singapore"
    return "Unknown"


def build_city_annotations(dataroot: Path, version: str) -> Tuple[List[Dict[str, str]], Counter]:
    samples = load_nuscenes_table(dataroot, version, "sample")
    scenes = index_by_token(load_nuscenes_table(dataroot, version, "scene"))
    logs = index_by_token(load_nuscenes_table(dataroot, version, "log"))

    annotations: List[Dict[str, str]] = []
    counts = Counter()

    for sample in samples:
        sample_token = sample["token"]
        scene = scenes.get(sample["scene_token"])
        log = logs.get(scene["log_token"]) if scene else None
        city = city_from_location(log.get("location") if log else None)

        annotations.append({"token": sample_token, "city": city})
        counts[city] += 1

    return annotations, counts


def main() -> None:
    args = parse_args()
    annotations, counts = build_city_annotations(args.dataroot, args.version)
    dump_json(annotations, args.output_json)

    print(f"Saved {len(annotations)} city annotations to {args.output_json}")
    for city, count in sorted(counts.items()):
        print(f"{city}: {count}")


if __name__ == "__main__":
    main()
