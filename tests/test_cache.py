"""Tests for src/tonno/cache.py.

The cache stores and retrieves raw JSON-serialisable data.
Encoding/decoding config objects is the caller's responsibility.
"""
import json

import numpy as np
import pytest

from tonno.cache import _make_key, load_best, save_best


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    config_data = {"bm": 32, "bk": 64}
    key_values = {"N": 1024, "D": 256}

    assert load_best("my_fn", "cpu", key_values) is None

    save_best("my_fn", "cpu", key_values, config_data, 1.234)

    loaded = load_best("my_fn", "cpu", key_values)
    assert loaded == config_data

    data = json.loads((tmp_path / "my_fn.json").read_text())
    assert "cpu" in data


def test_load_best_empty_key_values(tmp_path, monkeypatch):
    """load_best with key_values={} and no file returns None without crashing."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert load_best("fn", "cpu", {}) is None


def test_load_best_non_dict_json_returns_none(tmp_path, monkeypatch):
    """Cache file holds valid JSON that is not a dict (list, string) → returns None."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    (tmp_path / "fn.json").write_text(json.dumps([1, 2, 3]))
    assert load_best("fn", "cpu", {"N": 4}) is None

    (tmp_path / "fn.json").write_text(json.dumps("oops"))
    assert load_best("fn", "cpu", {"N": 4}) is None


def test_save_best_numpy_scalar_key_coerced(tmp_path, monkeypatch):
    """numpy scalar key values (e.g. from x.shape[0]) are coerced to Python native."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": np.int64(4)}, {"scale": 1}, 1.0)
    assert load_best("fn", "cpu", {"N": np.int64(4)}) == {"scale": 1}


def test_make_key_nan_inf_pins():
    """float nan/inf produce non-standard JSON literals (not RFC 8259 compliant).
    Pinned: Python's json module is lenient but external parsers will reject this."""
    assert "NaN" in _make_key({"N": float("nan")})
    assert "Infinity" in _make_key({"N": float("inf")})


def test_cache_path_is_directory_raises(tmp_path, monkeypatch):
    """Directory where .json file should be → IsADirectoryError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    (tmp_path / "fn.json").mkdir()
    with pytest.raises((IsADirectoryError, PermissionError, OSError)):
        save_best("fn", "cpu", {"N": 4}, {"scale": 1}, 1.0)


def test_read_only_cache_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    tmp_path.chmod(0o444)
    try:
        with pytest.raises((PermissionError, OSError)):
            save_best("fn", "cpu", {"N": 4}, {"scale": 1}, 1.0)
    finally:
        tmp_path.chmod(0o755)


def test_deep_cache_dir_created(tmp_path, monkeypatch):
    deep = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv("TONNO_CACHE_DIR", str(deep))
    save_best("fn", "cpu", {"N": 4}, {"scale": 1}, 1.0)
    assert (deep / "fn.json").exists()


def test_cache_key_order_independent():
    assert _make_key({"N": 4, "D": 8}) == _make_key({"D": 8, "N": 4})


def test_different_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    key_values = {"N": 512}

    save_best("fn", "gpu_a", key_values, {"bm": 16}, 1.0)
    save_best("fn", "gpu_b", key_values, {"bm": 64}, 2.0)

    assert load_best("fn", "gpu_a", key_values) == {"bm": 16}
    assert load_best("fn", "gpu_b", key_values) == {"bm": 64}
