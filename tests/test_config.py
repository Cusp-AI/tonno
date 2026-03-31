from tonno import Config


def test_config_roundtrip():
    cfg = Config(bm=16, bk=32, bn=128)
    assert cfg.to_dict() == {"bm": 16, "bk": 32, "bn": 128}
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_config_equality():
    assert Config(a=1, b=2) == Config(a=1, b=2)
    assert Config(a=1, b=2) != Config(a=1, b=3)


def test_config_hash():
    s = {Config(a=1), Config(a=1), Config(a=2)}
    assert len(s) == 2


def test_config_repr():
    cfg = Config(bm=16)
    assert "bm=16" in repr(cfg)
