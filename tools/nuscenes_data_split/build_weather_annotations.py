"""Build per-sample rain/night annotations from nuScenes scene descriptions.

The output schema is a JSON list:
[
  {
    "target_token": "<sample_token>",
    "scene_token": "<scene_token>",
    "description": "<scene_description>"
  }
]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from _common import (
    dump_json,
    index_by_token,
    load_nuscenes_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nuScenes rain/night sample annotations."
    )
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output intermediate weather annotation JSON.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["rain", "night"],
        help="Case-insensitive scene-description keywords to keep.",
    )
    return parser.parse_args()


def description_matches(description: str, keywords: List[str]) -> bool:
    description = description.lower()
    return any(keyword.lower() in description for keyword in keywords)


def build_weather_annotations(
    dataroot: Path, version: str, keywords: List[str]
) -> Tuple[List[Dict[str, str]], Counter]:
    samples = load_nuscenes_table(dataroot, version, "sample")
    scenes = index_by_token(load_nuscenes_table(dataroot, version, "scene"))

    annotations: List[Dict[str, str]] = []
    counts = Counter()

    for sample in samples:
        sample_token = sample["token"]
        scene = scenes.get(sample["scene_token"])
        if not scene:
            continue

        description = scene.get("description", "")
        if not description_matches(description, keywords):
            continue

        annotations.append(
            {
                "target_token": sample_token,
                "scene_token": scene["token"],
                "description": description,
            }
        )
        for keyword in keywords:
            if keyword.lower() in description.lower():
                counts[keyword.lower()] += 1

    return annotations, counts


def main() -> None:
    args = parse_args()
    annotations, counts = build_weather_annotations(
        args.dataroot, args.version, args.keywords
    )
    dump_json(annotations, args.output_json)

    print(f"Saved {len(annotations)} weather annotations to {args.output_json}")
    for keyword, count in sorted(counts.items()):
        print(f"{keyword}: {count}")


if __name__ == "__main__":
    main()
