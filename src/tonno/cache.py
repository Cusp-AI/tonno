from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CACHE_DIR_ENV = "TONNO_CACHE_DIR"
_DEFAULT_CACHE_DIR = ".tonno-cache"


def _cache_dir() -> Path:
    return Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR))


def _cache_path(fn_name: str) -> Path:
    return _cache_dir() / f"{fn_name}.json"


def _make_key(key_values: dict[str, Any]) -> str:
    """Deterministic string key from key-value pairs.

    Numpy scalar values (e.g. from x.shape[0]) are coerced to Python
    native types so json.dumps does not raise TypeError.
    """
    native = {k: v.item() if hasattr(v, "item") else v for k, v in key_values.items()}
    return json.dumps(native, sort_keys=True, separators=(",", ":"))


def load_best(
    fn_name: str,
    device_name: str,
    key_values: dict[str, Any],
) -> Any | None:
    """Return the raw cached config data, or None if not found.

    The caller is responsible for decoding the returned value back into
    the original config type.
    """
    path = _cache_path(fn_name)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    device_data = data.get(device_name)
    if device_data is None:
        return None

    entry = device_data.get(_make_key(key_values))
    if entry is None:
        return None

    return entry["config"]


def save_best(
    fn_name: str,
    device_name: str,
    key_values: dict[str, Any],
    config_data: Any,
    time_ms: float,
) -> None:
    """Save the encoded config data to the cache.

    ``config_data`` must be JSON-serialisable.  The caller is responsible
    for encoding the config before calling this function.
    """
    path = _cache_path(fn_name)

    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}

    if device_name not in data:
        data[device_name] = {}

    native_key_values = {k: v.item() if hasattr(v, "item") else v for k, v in key_values.items()}
    data[device_name][_make_key(key_values)] = {
        "config": config_data,
        "time_ms": round(time_ms, 4),
        "key_values": native_key_values,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
