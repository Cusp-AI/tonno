"""Tests for src/tonno/tune.py — the autotune decorator (v0.2 API).

Usage pattern::

    @autotune(configs=[{"BN": 32, "BK": 64}, {"BN": 64, "BK": 128}])
    @jax.jit(static_argnames=["BN", "BK"])
    def fn(x: Array, BN: int = 32, BK: int = 64) -> Array:
        return x * BN

    fn(jnp.ones(4))            # BN/BK swept and cached
    fn(jnp.ones(4), BN=16)     # explicit override, autotune bypassed
"""

from __future__ import annotations

import json
import threading
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tonno import autotune

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fn(configs: list[dict[str, Any]] | None = None) -> Any:
    if configs is None:
        configs = [{"BN": 1}, {"BN": 2}]

    @autotune(configs=configs)
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    return fn


# ===========================================================================
# Decoration-time validation
# ===========================================================================


def test_empty_configs_raises() -> None:
    with pytest.raises(ValueError, match="empty"):

        @autotune(configs=[])
        @jax.jit
        def fn(x: jax.Array) -> jax.Array:
            return x


def test_empty_config_dict_raises() -> None:
    with pytest.raises(ValueError, match="empty"):

        @autotune(configs=[{}])
        @jax.jit
        def fn(x: jax.Array) -> jax.Array:
            return x


def test_mismatched_config_keys_raises() -> None:
    with pytest.raises(ValueError, match="different keys"):

        @autotune(configs=[{"BN": 32}, {"BK": 64}])
        @jax.jit(static_argnames=["BN", "BK"])
        def fn(x: jax.Array, BN: int = 32, BK: int = 64) -> jax.Array:
            return x


def test_not_jitted_raises() -> None:
    with pytest.raises(TypeError, match="jax.jit"):

        @autotune(configs=[{"BN": 32}])
        def fn(x: jax.Array, BN: int = 32) -> jax.Array:
            return x


def test_tunable_param_missing_from_signature_raises() -> None:
    with pytest.raises((TypeError, ValueError)):

        @autotune(configs=[{"MISSING": 32}])
        @jax.jit(static_argnames=["MISSING"])
        def fn(x: jax.Array) -> jax.Array:
            return x


def test_tunable_param_without_default_raises() -> None:
    with pytest.raises(TypeError, match="no default"):

        @autotune(configs=[{"BN": 32}])
        @jax.jit(static_argnames=["BN"])
        def fn(x: jax.Array, BN: int) -> jax.Array:
            return x * BN


def test_num_timing_zero_raises() -> None:
    with pytest.raises(ValueError, match="num_timing"):
        autotune(configs=[{"BN": 32}], num_timing=0)


def test_num_timing_negative_raises() -> None:
    with pytest.raises(ValueError, match="num_timing"):
        autotune(configs=[{"BN": 32}], num_timing=-1)


def test_num_warmup_negative_raises() -> None:
    with pytest.raises(ValueError, match="num_warmup"):
        autotune(configs=[{"BN": 32}], num_warmup=-1)


# ===========================================================================
# Core behaviour
# ===========================================================================


def test_autotune_selects_config(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_log: list[int] = []

    @autotune(configs=[{"BN": 1}, {"BN": 2}, {"BN": 3}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        call_log.append(BN)
        return x * BN

    result = fn(jnp.ones(4))
    assert result.shape == (4,)
    assert set(call_log) >= {1, 2, 3}


def test_autotune_uses_cache(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_count = [0]

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        call_count[0] += 1
        return x * BN

    fn(jnp.ones(4))
    after_sweep = call_count[0]
    fn(jnp.ones(4))
    # Cache hit: jax.jit reuses the compiled artifact without re-tracing the
    # Python body, so call_count does not increment.
    assert call_count[0] == after_sweep


def test_explicit_override_bypasses_autotune(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    assert jnp.allclose(fn(jnp.ones(4), BN=7), jnp.full(4, 7.0))


def test_cache_miss_and_hit_same_result(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()
    x = jnp.ones(4)
    assert jnp.allclose(fn(x), fn(x))


# ===========================================================================
# Cache key — shapes, dtypes, non-tunable static args
# ===========================================================================


def test_different_shapes_independent_sweeps(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        sweep_count[0] += 1
        return x * BN

    fn(jnp.ones(4))
    after_first = sweep_count[0]
    fn(jnp.ones(8))
    assert sweep_count[0] > after_first


def test_different_dtypes_independent_sweeps(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        sweep_count[0] += 1
        return x * BN

    fn(jnp.ones(4, dtype=jnp.float32))
    after_first = sweep_count[0]
    fn(jnp.ones(4, dtype=jnp.float16))
    assert sweep_count[0] > after_first


def test_non_tunable_static_kwarg_in_cache_key(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN", "transpose"])
    def fn(x: jax.Array, transpose: bool = False, BN: int = 1) -> jax.Array:
        sweep_count[0] += 1
        return jnp.flip(x) * BN if transpose else x * BN

    fn(jnp.ones(4), transpose=False)
    after_first = sweep_count[0]
    fn(jnp.ones(4), transpose=True)
    assert sweep_count[0] > after_first


def test_same_inputs_reuses_cache(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        sweep_count[0] += 1
        return x * BN

    fn(jnp.ones(4))
    after_sweep = sweep_count[0]
    for _ in range(4):
        fn(jnp.ones(4))
    # Cache hits don't re-trace the Python body.
    assert sweep_count[0] == after_sweep


# ===========================================================================
# Edge cases
# ===========================================================================


def test_single_config(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 5}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    assert jnp.allclose(fn(jnp.ones(4)), jnp.full(4, 5.0))


def test_num_warmup_zero(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}], num_warmup=0)
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    assert fn(jnp.ones(4)).shape == (4,)


def test_num_timing_one(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}], num_timing=1)
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    assert fn(jnp.ones(4)).shape == (4,)


def test_large_num_configs(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": i} for i in range(1, 101)])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    assert fn(jnp.ones(4)).shape == (4,)


def test_multiple_tunable_params(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 32, "BK": 64}, {"BN": 64, "BK": 128}])
    @jax.jit(static_argnames=["BN", "BK"])
    def fn(x: jax.Array, BN: int = 32, BK: int = 64) -> jax.Array:
        return x * (BN + BK)

    result = fn(jnp.ones(4))
    assert result.shape == (4,)
    assert float(result[0]) in {96.0, 192.0}


def test_wrapper_preserves_metadata() -> None:
    @autotune(configs=[{"BN": 1}])
    @jax.jit(static_argnames=["BN"])
    def my_kernel(x: jax.Array, BN: int = 1) -> jax.Array:
        """Docstring."""
        return x * BN

    assert my_kernel.__name__ == "my_kernel"
    assert "my_kernel" in my_kernel.__qualname__
    assert my_kernel.__doc__ == "Docstring."


def test_fn_returning_tuple(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> tuple[jax.Array, jax.Array]:
        return x * BN, x * BN * 2

    r = fn(jnp.ones(4))
    assert isinstance(r, tuple) and len(r) == 2
    assert jnp.allclose(r[1], r[0] * 2)


def test_no_array_args(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(BN: int = 1) -> jax.Array:
        return jnp.array(float(BN))

    result = fn()
    assert result.shape == ()
    assert float(result) in {1.0, 2.0}


def test_numpy_array_arg(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert _make_fn()(np.ones(4, dtype=np.float32)).shape == (4,)  # type: ignore[arg-type]


def test_zero_sized_array(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert _make_fn()(jnp.zeros((0, 4))).shape == (0, 4)


def test_complex_array(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    x = jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64)
    assert _make_fn()(x).dtype == jnp.complex64


# ===========================================================================
# JAX integration
# ===========================================================================


def test_jax_jit_of_wrapper(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()
    fn(jnp.ones(4))
    assert jax.jit(fn)(jnp.ones(4)).shape == (4,)


def test_jax_grad_warm_cache(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return jnp.sum(x * BN)

    fn(jnp.ones(4))
    g = jax.grad(fn)(jnp.ones(4))
    assert g.shape == (4,) and jnp.all(g > 0)


def test_jax_grad_cold_cache(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 2) -> jax.Array:
        return jnp.sum(x * BN)

    g = jax.grad(fn)(jnp.ones(4))
    assert jnp.allclose(g, jnp.full(4, 2.0))


def test_jax_vmap_warm(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()
    fn(jnp.ones(4))
    assert jax.vmap(fn)(jnp.ones((3, 4))).shape == (3, 4)


def test_jax_vmap_cold(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert jax.vmap(_make_fn())(jnp.ones((3, 4))).shape == (3, 4)


def test_lax_scan(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()

    def body(carry: None, x: jax.Array) -> tuple[None, jax.Array]:
        return carry, fn(x)

    _, ys = jax.lax.scan(body, None, jnp.ones((5, 4)))
    assert ys.shape == (5, 4)


def test_jax_remat(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return jnp.sum(x * BN)

    fn(jnp.ones(4))
    g = jax.grad(jax.remat(fn))(jnp.ones(4))
    assert g.shape == (4,) and jnp.all(g > 0)


def test_second_order_grad(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}, {"BN": 2}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return jnp.sum(x**2 * BN)

    fn(jnp.array(1.0))
    g2 = jax.grad(jax.grad(fn))(jnp.array(3.0))
    assert float(g2) in {2.0, 4.0}


# ===========================================================================
# Partial compilation failures
# ===========================================================================


def test_failing_config_skipped_good_config_wins(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A config that raises during compilation is skipped; the good one wins."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 0}, {"BN": 2}])  # BN=0 will raise in the kernel
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        if BN == 0:
            raise ValueError("BN=0 is invalid")
        return x * BN

    result = fn(jnp.ones(4))
    assert jnp.allclose(result, jnp.full(4, 2.0))


def test_all_configs_fail_raises_runtime_error(tmp_path: Any, monkeypatch: Any) -> None:
    """If every config fails to compile, RuntimeError is raised with details."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 0}, {"BN": -1}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        raise ValueError(f"BN={BN} always fails")

    with pytest.raises(RuntimeError, match="configs failed"):
        fn(jnp.ones(4))


def test_failing_config_not_cached(tmp_path: Any, monkeypatch: Any) -> None:
    """A sweep that skips a failing config caches only the passing config's result."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 0}, {"BN": 3}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        if BN == 0:
            raise ValueError("bad config")
        return x * BN

    fn(jnp.ones(4))
    import json as _json

    data = _json.loads((tmp_path / f"{fn.__qualname__}.json").read_text())
    cached = next(iter(next(iter(data.values())).values()))["config"]
    assert cached == {"BN": 3}


# ===========================================================================
# Cache edge cases
# ===========================================================================


def test_corrupt_cache_degrades_gracefully(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()
    (tmp_path / f"{fn.__qualname__}.json").write_text("<<<NOT JSON>>>")
    assert fn(jnp.ones(4)).shape == (4,)


def test_stale_cache_wrong_keys_raises(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[{"BN": 1}])
    @jax.jit(static_argnames=["BN"])
    def fn(x: jax.Array, BN: int = 1) -> jax.Array:
        return x * BN

    # Key must use sort_keys=True to match _make_key's output (dtype before shape).
    bad = {
        "cpu": {
            '{"__arg0":{"dtype":"float32","shape":[4]}}': {
                "config": {"WRONG": 99},
                "time_ms": 0.1,
                "key_values": {},
            }
        }
    }
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(bad))
    with pytest.raises((TypeError, Exception)):
        fn(jnp.ones(4))


def test_concurrent_first_call_no_exception(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()
    errors: list[Exception] = []

    def call() -> None:
        try:
            fn(jnp.ones(4))
        except Exception as e:
            errors.append(e)

    t1, t2 = threading.Thread(target=call), threading.Thread(target=call)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []


def test_cache_file_created_with_qualname(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _make_fn()
    fn(jnp.ones(4))
    assert (tmp_path / f"{fn.__qualname__}.json").exists()
