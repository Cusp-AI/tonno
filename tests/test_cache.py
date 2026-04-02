"""Tests for src/tonno/cache.py.

The cache stores and retrieves raw JSON-serialisable data.
Encoding/decoding config objects is the caller's responsibility.
"""
import json
import platform
import threading
from pathlib import Path

import numpy as np
import pytest

from tonno.cache import _MISSING, _cache_dir, _cache_path, _make_key, load_best, save_best


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    config_data = {"bm": 32, "bk": 64}
    key_values = {"N": 1024, "D": 256}

    assert load_best("my_fn", "cpu", key_values) is _MISSING

    save_best("my_fn", "cpu", key_values, config_data, 1.234)

    loaded = load_best("my_fn", "cpu", key_values)
    assert loaded == config_data

    data = json.loads((tmp_path / "my_fn.json").read_text())
    assert "cpu" in data


def test_load_best_empty_key_values(tmp_path, monkeypatch):
    """load_best with key_values={} and no file returns _MISSING without crashing."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert load_best("fn", "cpu", {}) is _MISSING


def test_load_best_non_dict_json_returns_none(tmp_path, monkeypatch):
    """Cache file holds valid JSON that is not a dict (list, string) → returns None."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    (tmp_path / "fn.json").write_text(json.dumps([1, 2, 3]))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING

    (tmp_path / "fn.json").write_text(json.dumps("oops"))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


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


# ===========================================================================
# Adversarial runtime / environment edge cases
# ===========================================================================

def test_empty_device_name_roundtrip(tmp_path, monkeypatch):
    """Device name '' is a valid JSON key — save and load must round-trip."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 16}
    save_best("fn", "", {"N": 4}, config, 1.0)
    assert load_best("fn", "", {"N": 4}) == config


def test_empty_device_name_isolated_from_named_device(tmp_path, monkeypatch):
    """'' and 'cpu' stored in the same file must not alias each other."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "",    {"N": 4}, {"bm": 1}, 1.0)
    save_best("fn", "cpu", {"N": 4}, {"bm": 2}, 1.0)
    assert load_best("fn", "",    {"N": 4}) == {"bm": 1}
    assert load_best("fn", "cpu", {"N": 4}) == {"bm": 2}


def test_device_name_with_spaces(tmp_path, monkeypatch):
    """'NVIDIA H100 80GB HBM3' is a valid JSON key — must round-trip."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    device = "NVIDIA H100 80GB HBM3"
    config = {"bm": 128}
    save_best("fn", device, {"N": 1024}, config, 0.5)
    assert load_best("fn", device, {"N": 1024}) == config


def test_device_name_with_unicode(tmp_path, monkeypatch):
    """Device names with non-ASCII characters are valid JSON keys."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    device = "加速器\u00a0V100"
    config = {"bm": 64}
    save_best("fn", device, {"N": 8}, config, 2.0)
    assert load_best("fn", device, {"N": 8}) == config


def test_fn_name_path_traversal_stays_in_cache_dir(tmp_path, monkeypatch):
    """fn_name='../../evil' raises ValueError — path traversal is prevented."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="traversal"):
        _cache_path("../../evil")


def test_very_long_fn_name_raises_or_truncates(tmp_path, monkeypatch):
    """fn_name of 300 chars + '.json' suffix = 305 chars — most fs limit is 255."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    long_name = "x" * 300
    try:
        save_best(long_name, "cpu", {"N": 1}, {"bm": 1}, 1.0)
        assert (tmp_path / f"{long_name}.json").exists()
    except OSError:
        pass


def test_load_best_entry_missing_config_key_returns_missing(tmp_path, monkeypatch):
    """entry exists in JSON but has no 'config' field → _MISSING returned gracefully."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    key = _make_key({"N": 4})
    bad = {"cpu": {key: {"time_ms": 0.5, "key_values": {"N": 4}}}}
    (tmp_path / "fn.json").write_text(json.dumps(bad))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


def test_load_best_null_config_returns_none(tmp_path, monkeypatch):
    """JSON null stored as config value → load_best returns None (valid cache hit, not a miss)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    key = _make_key({"N": 4})
    null_entry = {"cpu": {key: {"config": None, "time_ms": 0.1, "key_values": {"N": 4}}}}
    (tmp_path / "fn.json").write_text(json.dumps(null_entry))
    result = load_best("fn", "cpu", {"N": 4})
    assert result is None


def test_concurrent_saves_preserve_both_entries(tmp_path, monkeypatch):
    """Two threads each save a different key — both must survive (no lost-update)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def writer(n: int) -> None:
        try:
            barrier.wait()
            save_best("fn", "cpu", {"N": n}, {"bm": n}, float(n))
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer, args=(4,))
    t2 = threading.Thread(target=writer, args=(8,))
    t1.start(); t2.start()
    t1.join();  t2.join()

    assert errors == [], errors
    r4 = load_best("fn", "cpu", {"N": 4})
    r8 = load_best("fn", "cpu", {"N": 8})
    assert r4 == {"bm": 4}, f"N=4 entry lost — concurrent write race still present: {r4}"
    assert r8 == {"bm": 8}, f"N=8 entry lost — concurrent write race still present: {r8}"


def test_cache_dir_env_var_change_mid_session(tmp_path, monkeypatch):
    """_cache_dir() is lazy — changing TONNO_CACHE_DIR between save and load gives a miss."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir(); dir_b.mkdir()

    monkeypatch.setenv("TONNO_CACHE_DIR", str(dir_a))
    save_best("fn", "cpu", {"N": 4}, {"bm": 16}, 1.0)
    assert load_best("fn", "cpu", {"N": 4}) == {"bm": 16}

    monkeypatch.setenv("TONNO_CACHE_DIR", str(dir_b))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


def test_cache_dir_env_var_unset_falls_back_to_default(monkeypatch):
    """After unsetting TONNO_CACHE_DIR the fallback '.tonno-cache' is used."""
    monkeypatch.delenv("TONNO_CACHE_DIR", raising=False)
    assert _cache_dir() == Path(".tonno-cache")


def test_make_key_list_value_is_deterministic():
    """Lists serialise identically regardless of dict insertion order."""
    k1 = _make_key({"shape": [4, 8], "N": 2})
    k2 = _make_key({"N": 2, "shape": [4, 8]})
    assert k1 == k2


def test_make_key_list_value_order_matters():
    """[4, 8] and [8, 4] must produce different keys (JSON arrays are ordered)."""
    assert _make_key({"shape": [4, 8]}) != _make_key({"shape": [8, 4]})


def test_save_load_roundtrip_list_key_value(tmp_path, monkeypatch):
    """Lists in key_values survive a JSON round-trip and produce matching keys."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    kv = {"shape": [4, 8]}
    config = {"bm": 32}
    save_best("fn", "cpu", kv, config, 1.0)
    assert load_best("fn", "cpu", kv) == config


def test_save_load_roundtrip_dict_key_value(tmp_path, monkeypatch):
    """Nested dict in key_values: sort_keys makes _make_key deterministic."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    kv = {"opts": {"b": 2, "a": 1}}
    config = {"bm": 64}
    save_best("fn", "cpu", kv, config, 1.0)
    assert load_best("fn", "cpu", kv) == config


def test_integer_fn_name_raises_or_coerces(tmp_path, monkeypatch):
    """fn_name=42 — f-string coercion means '42.json' is created without crashing."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    try:
        save_best(42, "cpu", {"N": 1}, {"bm": 1}, 1.0)  # type: ignore[arg-type]
        assert (tmp_path / "42.json").exists()
    except TypeError:
        pass


def test_time_ms_very_large(tmp_path, monkeypatch):
    """round(1e300, 4) does not raise — stored as 1e300."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 1}, {"bm": 1}, 1e300)
    data = json.loads((tmp_path / "fn.json").read_text())
    stored = data["cpu"][_make_key({"N": 1})]["time_ms"]
    assert stored == 1e300


def test_time_ms_very_small(tmp_path, monkeypatch):
    """round(1e-300, 4) rounds to 0.0 — no crash."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 1}, {"bm": 1}, 1e-300)
    data = json.loads((tmp_path / "fn.json").read_text())
    stored = data["cpu"][_make_key({"N": 1})]["time_ms"]
    assert stored == 0.0


def test_time_ms_negative(tmp_path, monkeypatch):
    """Negative time_ms is physically nonsensical but must not crash the cache."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 1}, {"bm": 1}, -5.0)
    data = json.loads((tmp_path / "fn.json").read_text())
    stored = data["cpu"][_make_key({"N": 1})]["time_ms"]
    assert stored == -5.0


def test_time_ms_nan_raises_or_stores(tmp_path, monkeypatch):
    """float('nan') — Python json.dumps writes non-standard 'NaN' literal."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    try:
        save_best("fn", "cpu", {"N": 1}, {"bm": 1}, float("nan"))
        raw = (tmp_path / "fn.json").read_text()
        assert "NaN" in raw
    except (ValueError, OverflowError):
        pass


def test_numpy_int64_save_python_int_load(tmp_path, monkeypatch):
    """Saving with numpy int64 key and loading with Python int hits the same entry."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 32}
    save_best("fn", "cpu", {"N": np.int64(4)}, config, 1.0)
    assert load_best("fn", "cpu", {"N": 4}) == config


def test_python_int_save_numpy_int64_load(tmp_path, monkeypatch):
    """Inverse: save with Python int, load with numpy int64."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 16}
    save_best("fn", "cpu", {"N": 4}, config, 1.0)
    assert load_best("fn", "cpu", {"N": np.int64(4)}) == config


def test_make_key_numpy_int64_equals_python_int():
    """_make_key must produce the same string for np.int64 and int."""
    assert _make_key({"N": np.int64(4)}) == _make_key({"N": 4})


def test_save_overwrites_existing_key(tmp_path, monkeypatch):
    """Saving a better result for the same key replaces the old entry."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    kv = {"N": 4}
    save_best("fn", "cpu", kv, {"bm": 16}, 2.0)
    save_best("fn", "cpu", kv, {"bm": 32}, 1.0)
    assert load_best("fn", "cpu", kv) == {"bm": 32}


def test_different_fn_names_use_separate_files(tmp_path, monkeypatch):
    """fn_name is the cache filename stem — two names must not collide."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    kv = {"N": 4}
    save_best("fn_a", "cpu", kv, {"bm": 1}, 1.0)
    save_best("fn_b", "cpu", kv, {"bm": 2}, 1.0)
    assert load_best("fn_a", "cpu", kv) == {"bm": 1}
    assert load_best("fn_b", "cpu", kv) == {"bm": 2}
    assert (tmp_path / "fn_a.json").exists()
    assert (tmp_path / "fn_b.json").exists()


def test_saved_file_is_valid_json(tmp_path, monkeypatch):
    """After save_best the resulting file must parse as valid JSON."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "gpu", {"N": 128, "D": 64}, {"bm": 32, "bk": 16}, 3.1415)
    raw = (tmp_path / "fn.json").read_text(encoding="utf-8")
    assert isinstance(json.loads(raw), dict)


# ===========================================================================
# Adversarial runtime / environment edge cases — iteration 2
# ===========================================================================

def test_make_key_bool_true_and_false_distinct():
    """True and False produce different JSON keys (true vs false)."""
    assert _make_key({"flag": True}) != _make_key({"flag": False})


def test_make_key_bool_is_not_int_1():
    """JSON bool is distinct from integer 1 — keys must differ."""
    assert _make_key({"v": True}) != _make_key({"v": 1})


def test_make_key_bool_roundtrip(tmp_path, monkeypatch):
    """Boolean key value round-trips through save/load."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 8}
    save_best("fn", "cpu", {"flag": True}, config, 1.0)
    assert load_best("fn", "cpu", {"flag": True}) == config
    assert load_best("fn", "cpu", {"flag": False}) is _MISSING


def test_make_key_numpy_bool_equals_python_bool():
    """numpy bool_ coerces to Python bool via .item(), matching _make_key."""
    assert _make_key({"flag": np.bool_(True)}) == _make_key({"flag": True})


def test_make_key_none_value():
    """None serialises to JSON null — must not raise."""
    assert "null" in _make_key({"N": None})


def test_make_key_none_roundtrip(tmp_path, monkeypatch):
    """None as a key value round-trips through save/load correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 16}
    save_best("fn", "cpu", {"N": None}, config, 1.0)
    assert load_best("fn", "cpu", {"N": None}) == config


def test_make_key_none_distinct_from_zero():
    """None (null) and 0 must produce different cache keys."""
    assert _make_key({"N": None}) != _make_key({"N": 0})


def test_make_key_float_value_deterministic():
    """0.5 serialised to JSON always produces the same key string."""
    assert _make_key({"ratio": 0.5}) == _make_key({"ratio": 0.5})


def test_make_key_float_roundtrip(tmp_path, monkeypatch):
    """Float key value round-trips — same float produces a cache hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 32}
    save_best("fn", "cpu", {"ratio": 0.5}, config, 1.0)
    assert load_best("fn", "cpu", {"ratio": 0.5}) == config


def test_make_key_different_floats_distinct():
    """0.5 and 0.50000000001 must map to different keys."""
    assert _make_key({"ratio": 0.5}) != _make_key({"ratio": 0.50000000001})


def test_make_key_negative_integer():
    """Negative integers are valid JSON and must not raise."""
    assert "-1" in _make_key({"N": -1})


def test_save_load_negative_integer_key_value(tmp_path, monkeypatch):
    """Negative integer key value round-trips correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 4}
    save_best("fn", "cpu", {"N": -1}, config, 1.0)
    assert load_best("fn", "cpu", {"N": -1}) == config


def test_negative_and_positive_int_keys_distinct(tmp_path, monkeypatch):
    """-1 and 1 are different keys and must not collide."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": -1}, {"bm": 1}, 1.0)
    save_best("fn", "cpu", {"N":  1}, {"bm": 2}, 1.0)
    assert load_best("fn", "cpu", {"N": -1}) == {"bm": 1}
    assert load_best("fn", "cpu", {"N":  1}) == {"bm": 2}


def test_save_null_config_file_is_valid_json(tmp_path, monkeypatch):
    """save_best(config_data=None) must write valid JSON without raising."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 4}, None, 1.0)
    assert isinstance(json.loads((tmp_path / "fn.json").read_text()), dict)


def test_save_null_config_stored_in_entry(tmp_path, monkeypatch):
    """Null config is stored; load_best returns None (valid cache hit — callers receive None as the config value)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 4}, None, 1.0)
    assert load_best("fn", "cpu", {"N": 4}) is None


@pytest.mark.skipif(platform.system() == "Windows", reason="symlinks require privileges on Windows")
def test_symlink_cache_file_write_follows_link(tmp_path, monkeypatch):
    """If fn.json is a symlink to a writable target, write_text follows it."""
    real_file = tmp_path / "real_cache.json"
    link_file = tmp_path / "fn.json"
    real_file.write_text("{}\n")
    link_file.symlink_to(real_file)

    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 4}, {"bm": 8}, 1.0)

    assert json.loads(real_file.read_text())["cpu"] is not None
    assert load_best("fn", "cpu", {"N": 4}) == {"bm": 8}


@pytest.mark.skipif(platform.system() == "Windows", reason="symlinks require privileges on Windows")
def test_symlink_pointing_to_dev_null_returns_none_on_load(tmp_path, monkeypatch):
    """/dev/null swallows writes; read returns '' → JSONDecodeError → None."""
    link_file = tmp_path / "fn.json"
    link_file.symlink_to(Path("/dev/null"))

    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 4}, {"bm": 8}, 1.0)
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


def test_make_key_integer_dict_key_coercion():
    """json.dumps({1: 'v'}) yields '{"1":"v"}' — integer dict keys are coerced to strings."""
    assert _make_key({1: "val"}) == _make_key({"1": "val"})  # type: ignore[dict-item]


def test_make_key_tuple_value_serialises_as_array():
    """Tuples serialise as JSON arrays — same as a list with the same elements."""
    assert _make_key({"shape": (4, 8)}) == _make_key({"shape": [4, 8]})


def test_save_load_tuple_key_value_roundtrip(tmp_path, monkeypatch):
    """Tuple key value is stored and retrieved correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 16}
    save_best("fn", "cpu", {"shape": (4, 8)}, config, 1.0)
    assert load_best("fn", "cpu", {"shape": (4, 8)}) == config
    assert load_best("fn", "cpu", {"shape": [4, 8]}) == config


def test_load_best_device_data_is_list_returns_missing(tmp_path, monkeypatch):
    """device_data is a list — not a dict, so treated as a corrupt entry → _MISSING."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    (tmp_path / "fn.json").write_text(json.dumps({"cpu": [1, 2, 3]}))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


def test_load_best_entry_is_integer_returns_missing(tmp_path, monkeypatch):
    """Entry value is an integer — not a dict, treated as corrupt → _MISSING."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    (tmp_path / "fn.json").write_text(json.dumps({"cpu": {_make_key({"N": 4}): 42}}))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


def test_load_best_entry_is_string_returns_missing(tmp_path, monkeypatch):
    """Entry value is a string — not a dict, treated as corrupt → _MISSING."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    (tmp_path / "fn.json").write_text(json.dumps({"cpu": {_make_key({"N": 4}): "oops"}}))
    assert load_best("fn", "cpu", {"N": 4}) is _MISSING


def test_save_load_deeply_nested_config(tmp_path, monkeypatch):
    """Deeply nested config survives JSON serialisation + deserialisation."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    deep = {"a": {"b": {"c": {"d": [1, 2, 3]}}}}
    save_best("fn", "cpu", {"N": 4}, deep, 1.0)
    assert load_best("fn", "cpu", {"N": 4}) == deep


def test_save_load_unicode_in_config_data(tmp_path, monkeypatch):
    """Non-ASCII characters in config_data values are preserved by JSON."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"name": "α-β kernel", "emoji": "\U0001f525"}
    save_best("fn", "cpu", {"N": 4}, config, 1.0)
    assert load_best("fn", "cpu", {"N": 4}) == config


def test_saved_file_unicode_is_valid_utf8(tmp_path, monkeypatch):
    """Cache file containing unicode is readable as valid UTF-8."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    save_best("fn", "cpu", {"N": 1}, {"name": "α"}, 1.0)
    raw = (tmp_path / "fn.json").read_text(encoding="utf-8")
    assert json.loads(raw)["cpu"][_make_key({"N": 1})]["config"] == {"name": "α"}


@pytest.mark.skipif(platform.system() == "Windows", reason="forward slash semantics differ on Windows")
def test_fn_name_with_slash_creates_subdirectory(tmp_path, monkeypatch):
    """fn_name='outer/inner' resolves to cache_dir/outer/inner.json via parents=True mkdir."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    config = {"bm": 64}
    save_best("outer/inner", "cpu", {"N": 8}, config, 1.0)
    assert (tmp_path / "outer" / "inner.json").exists()
    assert load_best("outer/inner", "cpu", {"N": 8}) == config
