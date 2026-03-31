from __future__ import annotations

import json
import os
from pathlib import Path

from tonno.config import Config

_CACHE_DIR_ENV = "TONNO_CACHE_DIR"
_DEFAULT_CACHE_DIR = ".tonno-cache"


def _cache_dir() -> Path:
    return Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR))


def _cache_path(fn_name: str) -> Path:
    return _cache_dir() / f"{fn_name}.json"


def _make_key(key_values: dict[str, int | float | str]) -> str:
    """Deterministic string key from key-value pairs."""
    return json.dumps(key_values, sort_keys=True, separators=(",", ":"))


def load_best(
    fn_name: str,
    device_name: str,
    key_values: dict[str, int | float | str],
) -> Config | None:
    """Load the cached best config, or None if not found."""
    path = _cache_path(fn_name)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    device_data = data.get(device_name)
    if device_data is None:
        return None

    entry = device_data.get(_make_key(key_values))
    if entry is None:
        return None

    return Config.from_dict(entry["config"])


def save_best(
    fn_name: str,
    device_name: str,
    key_values: dict[str, int | float | str],
    config: Config,
    time_ms: float,
) -> None:
    """Save the best config to the cache."""
    path = _cache_path(fn_name)

    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text())

    if device_name not in data:
        data[device_name] = {}

    data[device_name][_make_key(key_values)] = {
        "config": config.to_dict(),
        "time_ms": round(time_ms, 4),
        "key_values": key_values,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
