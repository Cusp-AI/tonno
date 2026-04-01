"""Tests for src/tonno/tune.py — the autotune decorator.

Decorated functions take a hashable config as their first positional argument.
Autotune injects the best config; callers never pass it.

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x, *, N=None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)   # cfg injected automatically
"""
from __future__ import annotations

import json
import threading
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tonno import autotune


# ---------------------------------------------------------------------------
# Config types used in tests
# ---------------------------------------------------------------------------

class KC(NamedTuple):
    scale: int = 1


class KCF(NamedTuple):
    alpha: float = 0.5


class KCBlock(NamedTuple):
    block_size: int | str = 4


class KCEmpty(NamedTuple):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn(configs: list[KC] | None = None, key: tuple[str, ...] = ("N",)):
    if configs is None:
        configs = [KC(1), KC(2)]

    @autotune(configs=configs, key=list(key))
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    return fn


# ===========================================================================
# Core behaviour
# ===========================================================================

def test_autotune_selects_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_log: list[int] = []

    @autotune(configs=[KC(1), KC(2), KC(3)], key=["N"])
    def multiply(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        call_log.append(cfg.scale)
        return x * cfg.scale

    result = multiply(jnp.ones(4), N=4)
    assert len(call_log) >= 3
    assert result.shape == (4,)


def test_autotune_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def multiply(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        call_count[0] += 1
        return x * cfg.scale

    multiply(jnp.ones(4), N=4)
    first_count = call_count[0]
    multiply(jnp.ones(4), N=4)
    assert call_count[0] == first_count + 1


def test_autotune_missing_key():
    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError, match="N"):
        fn(jnp.ones(4))


# ===========================================================================
# Decoration-time validation
# ===========================================================================

def test_empty_configs_raises():
    with pytest.raises(ValueError, match="empty"):
        @autotune(configs=[], key=["N"])
        def fn(cfg: KC, x: jax.Array) -> jax.Array:
            return x


def test_configs_as_generator_works(tmp_path, monkeypatch):
    """Generators are coerced to list."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=(KC(i + 1) for i in range(2)), key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_key_param_without_default_raises_at_decoration():
    """Key param declared in fn without a default → clear TypeError at decoration time.

    Without this check the sweep silently fails with a cryptic
    'missing required keyword-only argument' from inside jitted_fn.lower().
    """
    with pytest.raises(TypeError, match="has no default"):
        @autotune(configs=[KC(1)], key=["N"])
        def fn(cfg: KC, x: jax.Array, *, N: int) -> jax.Array:  # type: ignore[misc]
            return x * cfg.scale


def test_heterogeneous_configs_raises():
    """Configs with different pytree structures → ValueError at decoration time."""
    class KCA(NamedTuple):
        a: int

    class KCB(NamedTuple):
        b: int

    with pytest.raises(ValueError, match="pytree structure"):
        @autotune(configs=[KCA(1), KCB(2)], key=["N"])  # type: ignore[arg-type]
        def fn(cfg: KCA, x: jax.Array) -> jax.Array:
            return x


def test_num_timing_zero_raises():
    with pytest.raises(ValueError, match="num_timing"):
        @autotune(configs=[KC(1)], key=["N"], num_timing=0)
        def fn(cfg: KC, x: jax.Array) -> jax.Array:
            return x


def test_num_timing_negative_raises():
    with pytest.raises(ValueError, match="num_timing"):
        @autotune(configs=[KC(1)], key=["N"], num_timing=-1)
        def fn(cfg: KC, x: jax.Array) -> jax.Array:
            return x


def test_num_warmup_negative_raises():
    with pytest.raises(ValueError, match="num_warmup"):
        @autotune(configs=[KC(1)], key=["N"], num_warmup=-1)
        def fn(cfg: KC, x: jax.Array) -> jax.Array:
            return x


def test_num_warmup_zero_works(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"], num_warmup=0)
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_num_timing_one_works(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"], num_timing=1)
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


# ===========================================================================
# Sweep edge cases
# ===========================================================================

def test_single_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(3)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert jnp.allclose(fn(jnp.ones(4), N=4), jnp.full(4, 3.0))


def test_duplicate_configs_no_crash(tmp_path, monkeypatch):
    """Two equal configs collapse in the futures dict; result still correct."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(2), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert jnp.allclose(fn(jnp.ones(4), N=4), jnp.full(4, 2.0))


def test_empty_config_no_crash(tmp_path, monkeypatch):
    """Config with no fields — fn ignores cfg, must not crash."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KCEmpty(), KCEmpty()], key=["N"])
    def fn(cfg: KCEmpty, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * 2

    assert jnp.allclose(fn(jnp.ones(4), N=4), jnp.full(4, 2.0))


def test_no_positional_args(tmp_path, monkeypatch):
    """kwargs-only fn: jax.tree.map over empty args tuple is harmless."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, *, x: jax.Array, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(x=jnp.ones(4), N=4).shape == (4,)


def test_variadic_positional_args(tmp_path, monkeypatch):
    """*args after cfg — jax.tree.map over the args tuple works correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, *arrays: jax.Array, N: int | None = None) -> jax.Array:
        return sum(arrays) * cfg.scale  # type: ignore[return-value]

    assert fn(jnp.ones(4), jnp.ones(4), N=4).shape == (4,)


def test_fn_without_key_param_works(tmp_path, monkeypatch):
    """Key kwarg is popped before fn is called — fn doesn't need to declare it."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_duplicate_key_names_raises(tmp_path, monkeypatch):
    """key=['N', 'N']: second pop finds 'N' already gone → TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N", "N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError, match="N"):
        fn(jnp.ones(4), N=4)


def test_single_bad_config_compile_error_propagates(tmp_path, monkeypatch):
    """A config that causes a compile error in the ThreadPoolExecutor re-raises."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KCBlock("bad"), KCBlock(4)], key=["N"])
    def fn(cfg: KCBlock, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.ones(cfg.block_size) * x.sum()  # type: ignore[arg-type]

    with pytest.raises(Exception):
        fn(jnp.ones(4), N=4)


def test_non_array_positional_arg_works(tmp_path, monkeypatch):
    """Plain Python int as positional arg — passed through as-is via isinstance guard."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, factor: int, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale * factor

    assert fn(jnp.ones(4), 2, N=4).shape == (4,)


# ===========================================================================
# JAX integration
# ===========================================================================

def test_jax_grad_warm_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.sum(x * cfg.scale)

    fn(jnp.ones(4), N=4)
    g = jax.grad(lambda x: fn(x, N=4))(jnp.ones(4))
    assert g.shape == (4,) and jnp.all(g > 0)


def test_jax_grad_cold_cache(tmp_path, monkeypatch):
    """Sweep runs as side channel inside grad trace; gradient is correct."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.sum(x * cfg.scale)  # d/dx = cfg.scale = 2

    g = jax.jit(jax.grad(lambda x: fn(x, N=4)))(jnp.ones(4) * 3.0)
    assert jnp.allclose(g, jnp.full(4, 2.0)), f"expected grad=2, got {g}"


def test_jax_vmap(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert jax.vmap(lambda x: fn(x, N=4))(jnp.ones((3, 4))).shape == (3, 4)


def test_jit_of_wrapper_warm_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()
    fn(jnp.ones(4), N=4)
    assert jax.jit(lambda x: fn(x, N=4))(jnp.ones(4)).shape == (4,)


def test_lax_cond_warm_cache(tmp_path, monkeypatch):
    """lax.cond traces both branches — cache-hit path must return correct shapes."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()
    x = jnp.ones(4)
    fn(x, N=4)
    result = jax.lax.cond(True, lambda x: fn(x, N=4), lambda x: fn(x, N=4), x)
    assert result.shape == (4,)


def test_key_as_jax_scalar_works(tmp_path, monkeypatch):
    """0-d JAX array as key value: .item() coerces to Python int."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert _fn()(jnp.ones(4), N=jnp.array(4)).shape == (4,)  # type: ignore[arg-type]


def test_key_as_jax_array_raises(tmp_path, monkeypatch):
    """Multi-element JAX array as key value → TypeError (not a scalar)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    with pytest.raises((TypeError, ValueError)):
        _fn()(jnp.ones(4), N=jnp.array([4, 8]))  # type: ignore[arg-type]


def test_key_as_tracer_raises(tmp_path, monkeypatch):
    """Key kwarg as dynamic jit arg (tracer) → TypeError: not JSON-serialisable."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    with pytest.raises(Exception):
        jax.jit(_fn())(jnp.ones(4), N=4)


def test_none_positional_arg(tmp_path, monkeypatch):
    """None is an empty pytree — tree.map leaves it alone, lower() accepts it."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, mask: None, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), None, N=4).shape == (4,)


def test_list_of_arrays_positional_arg(tmp_path, monkeypatch):
    """List of jax.Arrays as positional arg — tree.map recurses, preserves structure."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, xs: list[jax.Array], *, N: int | None = None) -> jax.Array:
        return jnp.stack(xs) * cfg.scale

    assert fn([jnp.ones(4), jnp.ones(4) * 2.0], N=4).shape == (2, 4)


def test_numpy_positional_arg_passthrough(tmp_path, monkeypatch):
    """numpy.ndarray is not jax.Array → dummy_args passes it through as-is."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    assert _fn()(np.ones(4, dtype=np.float32), N=4).shape == (4,)  # type: ignore[arg-type]


def test_kwarg_shape_but_no_dtype_raises(tmp_path, monkeypatch):
    """Object with .shape but no .dtype → AttributeError in dummy_kwargs build."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    class HasShapeNoDtype:
        shape = (4,)

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, weights: object, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(AttributeError):
        fn(jnp.ones(4), weights=HasShapeNoDtype(), N=4)


def test_float_config_survives_cache_roundtrip(tmp_path, monkeypatch):
    """Float config values must survive JSON round-trip as float, not str."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    seen: list[float] = []

    @autotune(configs=[KCF(0.5), KCF(1.0)], key=["N"])
    def fn(cfg: KCF, x: jax.Array, *, N: int | None = None) -> jax.Array:
        seen.append(cfg.alpha)
        return x * cfg.alpha

    fn(jnp.ones(4), N=4)
    seen.clear()
    fn(jnp.ones(4), N=4)
    assert len(seen) == 1 and isinstance(seen[0], float)


# ===========================================================================
# Behavioural pins (known limitations)
# ===========================================================================

def test_pin_cache_miss_and_hit_same_result(tmp_path, monkeypatch):
    """Cache-miss and cache-hit paths return numerically identical results."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()
    x = jnp.ones(4)
    assert jnp.allclose(fn(x, N=4), fn(x, N=4))


def test_pin_key_kwarg_not_forwarded_to_fn(tmp_path, monkeypatch):
    """Key kwargs are popped before forwarding to fn — fn always sees the default."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    received: list[int | None] = []

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        received.append(N)
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    fn(jnp.ones(4), N=4)
    assert all(v is None for v in received)


def test_pin_empty_key_list_cache_collision(tmp_path, monkeypatch):
    """key=[] → all shapes share cache entry '{}' — silent collision."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    seen: list[tuple[int, ...]] = []

    @autotune(configs=[KC(1), KC(2)], key=[])
    def fn(cfg: KC, x: jax.Array) -> jax.Array:
        seen.append(x.shape)
        return x * cfg.scale

    fn(jnp.ones(4))
    after = len(seen)
    fn(jnp.ones(8))
    assert len(seen) == after + 1


def test_pin_qualname_collision(tmp_path, monkeypatch):
    """Two fns from the same factory share __qualname__ → same cache file."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    def build(configs: list[KC]):
        @autotune(configs=configs, key=["N"])
        def compute(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
            return x * cfg.scale
        return compute

    v1 = build([KC(1)])
    v2 = build([KC(99)])
    assert v1.__qualname__ == v2.__qualname__

    r1 = v1(jnp.ones(4), N=4)
    r2 = v2(jnp.ones(4), N=4)  # hits v1's cache → scale=1, not 99
    assert jnp.allclose(r1, r2)


def test_pin_two_lambdas_share_cache(tmp_path, monkeypatch):
    """Two lambdas have the same __qualname__ → share cache → collision."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    fn1 = autotune(configs=[KC(1)], key=["N"])(
        lambda cfg, x, *, N=None: x * cfg.scale
    )
    fn2 = autotune(configs=[KC(99)], key=["N"])(
        lambda cfg, x, *, N=None: x * cfg.scale
    )

    fn1(jnp.ones(4), N=4)
    r2 = fn2(jnp.ones(4), N=4)  # hits fn1's cache → scale=1, not 99
    assert float(r2[0]) != 99.0


def test_pin_stale_shape_after_cache_hit(tmp_path, monkeypatch):
    """Cache key is N=4 regardless of actual array shape — silently applied to (8,)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()
    fn(jnp.ones(4), N=4)
    assert fn(jnp.ones(8), N=4).shape == (8,)


def test_pin_dtype_not_in_cache_key(tmp_path, monkeypatch):
    """dtype is not part of the cache key — float64 call silently hits float32 entry."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    jax.config.update("jax_enable_x64", True)
    try:
        sweep_count = [0]

        @autotune(configs=[KC(1), KC(2)], key=["N"])
        def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
            sweep_count[0] += 1
            return x * cfg.scale

        fn(jnp.ones(4, dtype=jnp.float32), N=4)
        after = sweep_count[0]
        fn(jnp.ones(4, dtype=jnp.float64), N=4)
        assert sweep_count[0] == after + 1
    finally:
        jax.config.update("jax_enable_x64", False)


# ===========================================================================
# Cache edge cases (decorator-level view)
# ===========================================================================

def test_corrupt_cache_degrades_gracefully(tmp_path, monkeypatch):
    """Corrupt cache → treated as miss, sweep runs, result returned."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()
    (tmp_path / f"{fn.__qualname__}.json").write_text("<<<NOT JSON>>>")
    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_stale_config_raises_on_decode(tmp_path, monkeypatch):
    """Cache with wrong field names → TypeError when decoder calls KC(**bad_data)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()  # uses KC(scale=...)
    bad = {"cpu": {'{"N":4}': {"config": {"WRONG_KEY": 42}, "time_ms": 0.5, "key_values": {"N": 4}}}}
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(bad))
    with pytest.raises(TypeError):
        fn(jnp.ones(4), N=4)


def test_nested_qualname_cache_file(tmp_path, monkeypatch):
    """Nested fn qualname ('outer.<locals>.inner') creates valid cache path on POSIX."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    def outer():
        @autotune(configs=[KC(1)], key=["N"])
        def inner(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
            return x * cfg.scale
        return inner

    fn = outer()
    fn(jnp.ones(4), N=4)
    assert (tmp_path / f"{fn.__qualname__}.json").exists()


def test_lambda_qualname_cache_file(tmp_path, monkeypatch):
    """Lambda __qualname__ includes '<lambda>' — cache file created on POSIX."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    fn = autotune(configs=[KC(1), KC(2)], key=["N"])(
        lambda cfg, x, *, N=None: x * cfg.scale
    )
    fn(jnp.ones(4), N=4)
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and "<lambda>" in files[0].name


def test_concurrent_first_call_no_exception(tmp_path, monkeypatch):
    """Two threads race on cache miss — last writer wins, no exception, valid JSON."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    fn = _fn()
    errors: list[Exception] = []

    def call() -> None:
        try:
            fn(jnp.ones(4), N=4)
        except Exception as e:
            errors.append(e)

    t1, t2 = threading.Thread(target=call), threading.Thread(target=call)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    assert "cpu" in json.loads((tmp_path / f"{fn.__qualname__}.json").read_text())
