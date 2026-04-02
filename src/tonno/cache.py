from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_CACHE_DIR_ENV = "TONNO_CACHE_DIR"
_DEFAULT_CACHE_DIR = ".tonno-cache"

# Sentinel returned by load_best when no cache entry is found.
# Distinguishes "not found" from "found with a null/None config value".
_MISSING: Any = object()

_write_lock = threading.Lock()


def _cache_dir() -> Path:
    return Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR))


def _cache_path(fn_name: str) -> Path:
    # Guard against path traversal (e.g. fn_name="../../evil") by checking the
    # lexical parts — no resolve() so symlinks within the cache dir are allowed.
    if ".." in Path(fn_name).parts:
        raise ValueError(
            f"fn_name {fn_name!r} contains '..'; path traversal is not allowed."
        )
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
) -> Any:
    """Return the raw cached config data, or ``_MISSING`` if not found.

    Returns ``_MISSING`` (not ``None``) so that callers can distinguish
    "entry not present" from "entry present with a null config value".
    The caller is responsible for decoding the returned value back into
    the original config type.
    """
    path = _cache_path(fn_name)
    if not path.exists():
        return _MISSING

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return _MISSING
    if not isinstance(data, dict):
        return _MISSING

    device_data = data.get(device_name)
    if not isinstance(device_data, dict):
        return _MISSING

    entry = device_data.get(_make_key(key_values))
    if not isinstance(entry, dict) or "config" not in entry:
        return _MISSING

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

    Thread-safe within a single process: a lock prevents concurrent
    read-modify-write races when multiple threads write different keys
    at the same time.  Cross-process safety (e.g. two separate Python
    interpreters sharing the same cache directory) is not guaranteed;
    use a dedicated cache directory per process if that matters.
    """
    path = _cache_path(fn_name)

    with _write_lock:
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = {}

        if device_name not in data:
            data[device_name] = {}

        native_key_values = {
            k: v.item() if hasattr(v, "item") else v for k, v in key_values.items()
        }
        data[device_name][_make_key(key_values)] = {
            "config": config_data,
            "time_ms": round(time_ms, 4),
            "key_values": native_key_values,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
