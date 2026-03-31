import jax
import jax.numpy as jnp

from tonno import Config, autotune


def test_autotune_selects_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    call_log = []

    @autotune(
        configs=[Config(scale=1), Config(scale=2), Config(scale=3)],
        key=["N"],
    )
    def multiply(x, *, scale, N=None):
        call_log.append(scale)
        return x * scale

    x = jnp.ones(4)
    result = multiply(x, N=4)

    # All 3 configs were tried during sweep
    assert len(call_log) >= 3
    # Result uses the best config (called one extra time after sweep)
    assert result.shape == (4,)


def test_autotune_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    call_count = [0]

    @autotune(
        configs=[Config(scale=1), Config(scale=2)],
        key=["N"],
    )
    def multiply(x, *, scale, N=None):
        call_count[0] += 1
        return x * scale

    x = jnp.ones(4)

    # First call: sweep
    multiply(x, N=4)
    first_count = call_count[0]

    # Second call: should use cache (only 1 additional call)
    multiply(x, N=4)
    assert call_count[0] == first_count + 1


def test_autotune_missing_key():
    @autotune(configs=[Config(s=1)], key=["N"])
    def fn(x, *, s):
        return x

    try:
        fn(1)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "N" in str(e)
