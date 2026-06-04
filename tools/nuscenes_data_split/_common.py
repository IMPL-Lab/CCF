"""Shared helpers for nuScenes split generation tools."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


INFO_LIST_KEYS = ("infos", "data_list")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def dump_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file)


def get_info_records(nuscenes_info: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return the sample list from known MMDet3D-style nuScenes info schemas."""
    for key in INFO_LIST_KEYS:
        records = nuscenes_info.get(key)
        if isinstance(records, list):
            return key, records
    expected = ", ".join(INFO_LIST_KEYS)
    raise KeyError(f"Cannot find a nuScenes info list. Expected one of: {expected}")


def get_sample_token(record: Dict[str, Any]) -> str:
    for key in ("token", "sample_token"):
        token = record.get(key)
        if token:
            return token
    raise KeyError(f"Info record does not contain a sample token: {record.keys()}")


def load_nuscenes_table(dataroot: Path, version: str, table_name: str) -> List[Dict[str, Any]]:
    return load_json(dataroot / version / f"{table_name}.json")


def index_by_token(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {record["token"]: record for record in records}


def normalize_values(values: Optional[Iterable[str]]) -> Set[str]:
    return {value.lower() for value in values or []}
