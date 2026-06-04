"""Filter nuScenes info pkl files using city and weather annotations."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from _common import (
    dump_pickle,
    get_info_records,
    get_sample_token,
    load_json,
    load_pickle,
    normalize_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create source/target nuScenes info pkl splits."
    )
    parser.add_argument("--info", type=Path, required=True, help="Input nuScenes info pkl.")
    parser.add_argument("--output", type=Path, required=True, help="Output filtered pkl.")
    parser.add_argument("--city-annotations", type=Path, help="City annotation JSON.")
    parser.add_argument("--weather-annotations", type=Path, help="Rain/night annotation JSON.")
    parser.add_argument("--include-city", nargs="+", help="Keep only these cities.")
    parser.add_argument("--exclude-city", nargs="+", help="Drop these cities.")
    parser.add_argument(
        "--include-weather",
        nargs="+",
        help="Keep samples whose weather description contains one of these keywords.",
    )
    parser.add_argument(
        "--exclude-weather",
        nargs="+",
        help="Drop samples whose weather description contains one of these keywords.",
    )
    return parser.parse_args()


def load_city_lookup(path: Any) -> Dict[str, str]:
    if path is None:
        return {}
    return {
        row["token"]: row["city"].lower()
        for row in load_json(path)
        if row.get("token") and row.get("city")
    }


def load_weather_lookup(path: Any) -> Dict[str, str]:
    if path is None:
        return {}
    lookup = {}
    for row in load_json(path):
        token = row.get("target_token") or row.get("token")
        if token:
            lookup[token] = row.get("description", "").lower()
    return lookup


def contains_any(text: str, keywords: Set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def should_keep(
    sample_token: str,
    city_lookup: Dict[str, str],
    weather_lookup: Dict[str, str],
    include_city: Set[str],
    exclude_city: Set[str],
    include_weather: Set[str],
    exclude_weather: Set[str],
) -> bool:
    city = city_lookup.get(sample_token, "")
    weather = weather_lookup.get(sample_token, "")

    if include_city and city not in include_city:
        return False
    if exclude_city and city in exclude_city:
        return False
    if include_weather and not contains_any(weather, include_weather):
        return False
    if exclude_weather and contains_any(weather, exclude_weather):
        return False
    return True


def filter_infos(
    nuscenes_info: Dict[str, Any],
    city_lookup: Dict[str, str],
    weather_lookup: Dict[str, str],
    include_city: Set[str],
    exclude_city: Set[str],
    include_weather: Set[str],
    exclude_weather: Set[str],
) -> Tuple[Dict[str, Any], Counter]:
    info_key, records = get_info_records(nuscenes_info)

    filtered_records = []
    stats = Counter()
    for record in records:
        sample_token = get_sample_token(record)
        stats["input"] += 1
        if should_keep(
            sample_token,
            city_lookup,
            weather_lookup,
            include_city,
            exclude_city,
            include_weather,
            exclude_weather,
        ):
            filtered_records.append(record)
            stats["kept"] += 1

    output_info = copy(nuscenes_info)
    output_info[info_key] = filtered_records
    return output_info, stats


def main() -> None:
    args = parse_args()

    if (args.include_city or args.exclude_city) and not args.city_annotations:
        raise ValueError("--city-annotations is required for city filters.")
    if (args.include_weather or args.exclude_weather) and not args.weather_annotations:
        raise ValueError("--weather-annotations is required for weather filters.")

    nuscenes_info = load_pickle(args.info)
    filtered_info, stats = filter_infos(
        nuscenes_info=nuscenes_info,
        city_lookup=load_city_lookup(args.city_annotations),
        weather_lookup=load_weather_lookup(args.weather_annotations),
        include_city=normalize_values(args.include_city),
        exclude_city=normalize_values(args.exclude_city),
        include_weather=normalize_values(args.include_weather),
        exclude_weather=normalize_values(args.exclude_weather),
    )
    dump_pickle(filtered_info, args.output)

    print(f"Input samples: {stats['input']}")
    print(f"Kept samples: {stats['kept']}")
    print(f"Saved filtered info to {args.output}")


if __name__ == "__main__":
    main()
