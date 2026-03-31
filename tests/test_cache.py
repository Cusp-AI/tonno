import json

from tonno.cache import load_best, save_best
from tonno.config import Config


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    cfg = Config(bm=32, bk=64)
    key_values = {"N": 1024, "D": 256}

    # Miss
    assert load_best("my_fn", "cpu", key_values) is None

    # Save
    save_best("my_fn", "cpu", key_values, cfg, 1.234)

    # Hit
    loaded = load_best("my_fn", "cpu", key_values)
    assert loaded is not None
    assert loaded.to_dict() == cfg.to_dict()

    # Check file is valid JSON
    cache_file = tmp_path / "my_fn.json"
    data = json.loads(cache_file.read_text())
    assert "cpu" in data


def test_different_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    cfg_a = Config(bm=16)
    cfg_b = Config(bm=64)
    key_values = {"N": 512}

    save_best("fn", "gpu_a", key_values, cfg_a, 1.0)
    save_best("fn", "gpu_b", key_values, cfg_b, 2.0)

    assert load_best("fn", "gpu_a", key_values).to_dict() == cfg_a.to_dict()
    assert load_best("fn", "gpu_b", key_values).to_dict() == cfg_b.to_dict()
