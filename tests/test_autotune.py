"""Tests for src/tonno/tune.py — the autotune decorator.

Decorated functions take a hashable config as their first positional argument.
Autotune injects the best config; callers never pass it.

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x, *, N=None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)   # cfg injected automatically
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, NamedTuple

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


class KCTwo(NamedTuple):
    bm: int = 32
    bk: int = 64


class _Pair:
    """Minimal custom pytree node for testing."""
    def __init__(self, a: jax.Array, b: jax.Array) -> None:
        self.a = a
        self.b = b


jax.tree_util.register_pytree_node(
    _Pair,
    lambda p: ([p.a, p.b], None),
    lambda _, children: _Pair(*children),
)


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

    with pytest.raises((TypeError, AttributeError)):
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


# ===========================================================================
# Adversarial API contract scenarios
# ===========================================================================

def test_unhashable_list_config_raises():
    """list config → TypeError when used as dict key in futures dict."""
    with pytest.raises(TypeError):
        @autotune(configs=[[1, 2], [3, 4]], key=["N"])  # type: ignore[arg-type]
        def fn(cfg, x: jax.Array, *, N: int | None = None) -> jax.Array:
            return x

        fn(jnp.ones(4), N=4)


def test_unhashable_set_config_raises():
    """set config → TypeError at hashtable insertion."""
    with pytest.raises(TypeError):
        @autotune(configs=[{1, 2}, {3, 4}], key=["N"])  # type: ignore[arg-type]
        def fn(cfg, x: jax.Array, *, N: int | None = None) -> jax.Array:
            return x

        fn(jnp.ones(4), N=4)


def test_unhashable_dict_config_raises():
    """dict config → TypeError at hashtable insertion."""
    with pytest.raises(TypeError):
        @autotune(configs=[{"a": 1}, {"a": 2}], key=["N"])  # type: ignore[arg-type]
        def fn(cfg, x: jax.Array, *, N: int | None = None) -> jax.Array:
            return x

        fn(jnp.ones(4), N=4)


def test_key_as_bare_single_char_string_works_by_accident(tmp_path, monkeypatch):
    """key='N' iterates to ['N'] — works by accident for 1-char strings."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key="N")  # type: ignore[arg-type]
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_key_as_multichar_string_wrong_behaviour(tmp_path, monkeypatch):
    """key='block' iterates to ['b','l','o','c','k'] — each char is a key lookup."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key="block")  # type: ignore[arg-type]
    def fn(cfg: KC, x: jax.Array, *, block: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError, match="'b'"):
        fn(jnp.ones(4), block=4)


def test_encode_not_callable_raises(tmp_path, monkeypatch):
    """encode=42 (not callable) → TypeError when encode(best_config) is called."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"], encode=42)  # type: ignore[arg-type]
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError):
        fn(jnp.ones(4), N=4)


def test_decode_not_callable_raises(tmp_path, monkeypatch):
    """decode='oops' → TypeError when called on cache hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"], decode="oops")  # type: ignore[arg-type]
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    with pytest.raises(TypeError):
        fn(jnp.ones(4), N=4)


def test_encode_returns_numpy_array_raises(tmp_path, monkeypatch):
    """encode returning a numpy array → TypeError in json.dumps inside save_best."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(
        configs=[KC(1), KC(2)],
        key=["N"],
        encode=lambda cfg: np.array([cfg.scale]),
    )
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises((TypeError, ValueError)):
        fn(jnp.ones(4), N=4)


def test_encode_returns_non_serialisable_object_raises(tmp_path, monkeypatch):
    """encode returning an arbitrary object → TypeError in json.dumps."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    class Opaque:
        pass

    @autotune(
        configs=[KC(1)],
        key=["N"],
        encode=lambda _cfg: Opaque(),
    )
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises((TypeError, ValueError)):
        fn(jnp.ones(4), N=4)


def test_encode_raises_propagates(tmp_path, monkeypatch):
    """encode that raises → exception propagates out of wrapper."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    def bad_encode(cfg: KC) -> None:
        raise RuntimeError("encode exploded")

    @autotune(configs=[KC(1)], key=["N"], encode=bad_encode)
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(RuntimeError, match="encode exploded"):
        fn(jnp.ones(4), N=4)


def test_decode_raises_propagates(tmp_path, monkeypatch):
    """decode that raises → exception propagates out of wrapper on cache hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    def bad_decode(raw: Any) -> KC:
        raise ValueError("decode exploded")

    @autotune(configs=[KC(1)], key=["N"], decode=bad_decode)
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    with pytest.raises(ValueError, match="decode exploded"):
        fn(jnp.ones(4), N=4)


def test_cached_null_config_raises_decode_error(tmp_path, monkeypatch):
    """Cache entry config=null for a KC-typed fn → TypeError on decode (null ≠ KC).

    load_best now returns the actual null value (None) as a valid cache hit.
    _decode(None) for a NamedTuple type has no valid construction path → TypeError.
    """
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    payload = {"cpu": {'{"N":4}': {"config": None, "time_ms": 0.1, "key_values": {"N": 4}}}}
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(payload))

    with pytest.raises(TypeError):
        fn(jnp.ones(4), N=4)


def test_cached_int_for_namedtuple_raises(tmp_path, monkeypatch):
    """config=42 in cache → KC(*42) or KC(**42) → TypeError on decode."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    payload = {"cpu": {'{"N":4}': {"config": 42, "time_ms": 0.1, "key_values": {"N": 4}}}}
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(payload))

    with pytest.raises((TypeError, Exception)):
        fn(jnp.ones(4), N=4)


def test_cached_dict_with_extra_fields_raises(tmp_path, monkeypatch):
    """config dict has unexpected key → KC(**{scale:1, extra:9}) → TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    payload = {"cpu": {'{"N":4}': {
        "config": {"scale": 1, "unexpected_field": 99},
        "time_ms": 0.1,
        "key_values": {"N": 4},
    }}}
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(payload))

    with pytest.raises(TypeError):
        fn(jnp.ones(4), N=4)


def test_cached_list_for_namedtuple_decodes_correctly(tmp_path, monkeypatch):
    """config stored as list → KC(*[1]) uses the tuple branch of _decode."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    payload = {"cpu": {'{"N":4}': {"config": [1], "time_ms": 0.1, "key_values": {"N": 4}}}}
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(payload))

    result = fn(jnp.ones(4), N=4)
    assert result.shape == (4,)


def test_plain_int_config_type(tmp_path, monkeypatch):
    """Config is a plain int — _decode fallback calls int(cached_value)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[1, 2, 4], key=["N"])
    def fn(cfg: int, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg

    result = fn(jnp.ones(4), N=4)
    assert result.shape == (4,)


def test_plain_str_config_type(tmp_path, monkeypatch):
    """Config is a plain str — _decode falls to the cfg_type(d) branch: str(d)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=["fast", "slow"], key=["N"])
    def fn(cfg: str, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * (2.0 if cfg == "fast" else 1.0)

    result = fn(jnp.ones(4), N=4)
    assert result.shape == (4,)


def test_namedtuple_with_cached_dict_decodes_correctly(tmp_path, monkeypatch):
    """NamedTuple cached as dict decodes via cfg_type(**d) — dict path checked first.

    KC(*{"bm":32,"bk":64}) would iterate keys; KC(**{"bm":32,"bk":64}) uses values.
    The dict branch now runs before the tuple branch.
    """
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KCTwo(32, 64), KCTwo(64, 128)], key=["N"])
    def fn(cfg: KCTwo, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.bm

    payload = {"cpu": {'{"N":4}': {
        "config": {"bm": 32, "bk": 64},
        "time_ms": 0.1,
        "key_values": {"N": 4},
    }}}
    (tmp_path / f"{fn.__qualname__}.json").write_text(json.dumps(payload))

    result = fn(jnp.ones(4), N=4)
    assert jnp.allclose(result, jnp.full(4, 32.0))


def test_empty_string_key_works(tmp_path, monkeypatch):
    """key=[''] — empty string is a valid kwarg key in Python."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=[""])
    def fn(cfg: KC, x: jax.Array, **kw: Any) -> jax.Array:
        return x * cfg.scale

    result = fn(jnp.ones(4), **{"": 4})
    assert result.shape == (4,)


def test_key_name_shadowing_builtin(tmp_path, monkeypatch):
    """key=['type'] — shadows Python builtin, but is a valid kwarg name."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["type"])
    def fn(cfg: KC, x: jax.Array, *, type: int | None = None) -> jax.Array:  # noqa: A002
        return x * cfg.scale

    assert fn(jnp.ones(4), type=4).shape == (4,)


def test_key_name_with_dots_in_cache_key_not_path(tmp_path, monkeypatch):
    """key=['a.b'] — dots appear in the JSON cache key, not in the file path."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["a.b"])
    def fn(cfg: KC, x: jax.Array, **kw: Any) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), **{"a.b": 4})
    cache_files = list(tmp_path.iterdir())
    assert len(cache_files) == 1
    data = json.loads(cache_files[0].read_text())
    device_data = next(iter(data.values()))
    cache_key = next(iter(device_data.keys()))
    assert "a.b" in cache_key


def test_key_kwarg_passed_positionally_not_found(tmp_path, monkeypatch):
    """N passed as positional arg → not in kw dict → TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError, match="N"):
        fn(jnp.ones(4), 4)


def test_multiple_keys_happy_path(tmp_path, monkeypatch):
    """Two distinct keys: M and N both required, both appear in the cache entry."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["M", "N"])
    def fn(cfg: KC, x: jax.Array, *, M: int | None = None, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), M=8, N=4)
    data = json.loads(list(tmp_path.iterdir())[0].read_text())
    cache_key = next(iter(next(iter(data.values())).keys()))
    parsed = json.loads(cache_key)
    assert "M" in parsed and "N" in parsed


def test_multiple_keys_missing_one_raises(tmp_path, monkeypatch):
    """Two keys declared but only one supplied → TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["M", "N"])
    def fn(cfg: KC, x: jax.Array, *, M: int | None = None, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError, match="M|N"):
        fn(jnp.ones(4), N=4)


def test_decode_none_fallback_for_int_config(tmp_path, monkeypatch):
    """decode=None with plain int config → _decode calls int(cached_value)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    seen: list[int] = []

    @autotune(configs=[1, 2], key=["N"])
    def fn(cfg: int, x: jax.Array, *, N: int | None = None) -> jax.Array:
        seen.append(cfg)
        return x * cfg

    fn(jnp.ones(4), N=4)
    seen.clear()
    fn(jnp.ones(4), N=4)
    assert len(seen) == 1 and isinstance(seen[0], int)


def test_custom_encode_decode_round_trip(tmp_path, monkeypatch):
    """encode serialises to int, decode reconstructs KC — full round-trip."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    seen: list[KC] = []

    @autotune(
        configs=[KC(1), KC(2)],
        key=["N"],
        encode=lambda cfg: cfg.scale,
        decode=lambda raw: KC(scale=int(raw)),
    )
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        seen.append(cfg)
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    seen.clear()
    fn(jnp.ones(4), N=4)
    assert len(seen) == 1 and isinstance(seen[0], KC)


def test_encode_returns_none_survives_roundtrip(tmp_path, monkeypatch):
    """encode returning None → JSON null → decode(None) called on cache hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(
        configs=[KC(1)],
        key=["N"],
        encode=lambda _cfg: None,
        decode=lambda raw: KC(scale=1),
    )
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    result = fn(jnp.ones(4), N=4)
    assert jnp.allclose(result, jnp.ones(4))


def test_large_num_timing_no_crash(tmp_path, monkeypatch):
    """num_timing=10, num_warmup=2 runs more iterations but must not crash."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"], num_timing=10, num_warmup=2)
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_single_int_config_cache_roundtrip(tmp_path, monkeypatch):
    """Single plain-int config: sweep, save, reload via int() decode."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[7], key=["N"])
    def fn(cfg: int, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x * cfg

    fn(jnp.ones(4), N=4)
    after_sweep = sweep_count[0]
    fn(jnp.ones(4), N=4)
    assert sweep_count[0] == after_sweep + 1


# ===========================================================================
# Adversarial JAX execution model scenarios
# ===========================================================================

def test_jit_cold_no_retrace_on_second_call(tmp_path, monkeypatch):
    """First jit call triggers sweep; second call reuses cached jit trace."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        call_count[0] += 1
        return x * cfg.scale

    outer = jax.jit(lambda x: fn(x, N=4))
    r1 = outer(jnp.ones(4))
    after_cold = call_count[0]
    r2 = outer(jnp.ones(4))
    after_warm = call_count[0]

    assert after_warm == after_cold
    assert jnp.allclose(r1, r2)


def test_double_jit_stable(tmp_path, monkeypatch):
    """jax.jit(jax.jit(wrapper)) produces consistent results across two calls."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    double_jitted = jax.jit(jax.jit(lambda x: fn(x, N=4)))
    r1 = double_jitted(jnp.ones(4))
    r2 = double_jitted(jnp.ones(4))

    assert r1.shape == (4,) and jnp.allclose(r1, r2)


def test_vmap_cold_cache_correct(tmp_path, monkeypatch):
    """vmap cold cache: inside vmap trace x.shape strips the batch axis, sweep compiles correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x * cfg.scale

    result = jax.vmap(lambda x: fn(x, N=4))(jnp.ones((3, 4)))
    assert result.shape == (3, 4)
    assert sweep_count[0] > 0


def test_lax_scan_cold_cache(tmp_path, monkeypatch):
    """Sweep fires inside scan body trace; scan output has correct stacked shape."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x * cfg.scale

    def body(carry: None, x: jax.Array) -> tuple[None, jax.Array]:
        return carry, fn(x, N=4)

    _, ys = jax.lax.scan(body, None, jnp.ones((5, 4)))
    assert ys.shape == (5, 4)
    assert sweep_count[0] > 0


def test_lax_scan_warm_cache_traces_body_once(tmp_path, monkeypatch):
    """scan body is traced exactly once; with warm cache exactly one fn call occurs."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    baseline = sweep_count[0]

    def body(carry: None, x: jax.Array) -> tuple[None, jax.Array]:
        return carry, fn(x, N=4)

    _, ys = jax.lax.scan(body, None, jnp.ones((5, 4)))
    assert ys.shape == (5, 4)
    assert sweep_count[0] == baseline + 1


def test_grad_cold_cache_no_jit_wrapper(tmp_path, monkeypatch):
    """Pure eager jax.grad (no jit) with cold cache: sweep runs, gradient is correct."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.sum(x * cfg.scale)

    g = jax.grad(lambda x: fn(x, N=4))(jnp.ones(4))
    assert jnp.allclose(g, jnp.full(4, 2.0))


def test_static_argnums_artifact_reuse_per_config(tmp_path, monkeypatch):
    """3 configs → 3 sweep traces + 1 final call = exactly 4 total traces."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    trace_log: list[int] = []

    @autotune(configs=[KC(1), KC(2), KC(3)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        trace_log.append(cfg.scale)
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    assert sorted(set(trace_log)) == [1, 2, 3]
    assert len(trace_log) == 4

    trace_log.clear()
    fn(jnp.ones(4), N=4)
    assert len(trace_log) == 1


def test_remat_cold_cache_correct_gradient(tmp_path, monkeypatch):
    """jax.remat with cold cache: sweep fires on first grad call, gradient is correct."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.sum(x * cfg.scale)

    remated = jax.remat(lambda x: fn(x, N=4))
    g = jax.grad(remated)(jnp.ones(4))
    assert g.shape == (4,) and jnp.all(g > 0)


def test_remat_warm_cache_gradient_consistent(tmp_path, monkeypatch):
    """jax.remat with warm cache: gradient matches across two calls."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.sum(x * cfg.scale)

    fn(jnp.ones(4), N=4)
    remated = jax.remat(lambda x: fn(x, N=4))
    assert jnp.allclose(jax.grad(remated)(jnp.ones(4)), jax.grad(remated)(jnp.ones(4)))


def test_inconsistent_output_shapes_sweep_does_not_crash(tmp_path, monkeypatch):
    """Configs producing different output shapes: sweep completes, result is valid."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    class KCS(NamedTuple):
        out_size: int = 4

    @autotune(configs=[KCS(4), KCS(8)], key=["N"])
    def fn(cfg: KCS, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.ones(cfg.out_size)

    result = fn(jnp.ones(4), N=4)
    assert result.ndim == 1 and result.shape[0] in {4, 8}


def test_jax_array_in_config_field_raises(tmp_path, monkeypatch):
    """Config containing a jax.Array field is unhashable → TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    class KCArr(NamedTuple):
        weights: jax.Array

    @autotune(
        configs=[KCArr(jnp.ones(4)), KCArr(jnp.zeros(4))],
        key=["N"],
    )
    def fn(cfg: KCArr, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.weights

    with pytest.raises(TypeError, match="unhashable"):
        fn(jnp.ones(4), N=4)


def test_no_array_args_only_key_kwarg(tmp_path, monkeypatch):
    """fn takes only cfg and key kwarg — empty args tuple handled correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, *, N: int | None = None) -> jax.Array:
        return jnp.array(float(cfg.scale))

    result = fn(N=4)
    assert result.shape == () and float(result) in {1.0, 2.0}


def test_per_key_value_sweep_isolation(tmp_path, monkeypatch):
    """Two distinct key values each trigger an independent sweep."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        call_count[0] += 1
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    after_first = call_count[0]
    fn(jnp.ones(8), N=8)
    after_second = call_count[0]
    fn(jnp.ones(4), N=4)
    after_hit1 = call_count[0]
    fn(jnp.ones(8), N=8)
    after_hit2 = call_count[0]

    assert after_second > after_first
    assert after_hit1 == after_second + 1
    assert after_hit2 == after_hit1 + 1


def test_custom_encode_decode_non_json_config(tmp_path, monkeypatch):
    """frozenset config with custom encode/decode survives cache roundtrip."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(
        configs=[frozenset({1}), frozenset({2})],
        key=["N"],
        encode=lambda cfg: sorted(cfg),
        decode=lambda d: frozenset(d),
    )
    def fn(cfg: frozenset, x: jax.Array, *, N: int | None = None) -> jax.Array:
        val = next(iter(cfg))
        return x * val

    fn(jnp.ones(4), N=4)
    r = fn(jnp.ones(4), N=4)
    assert r.shape == (4,) and float(r[0]) in {1.0, 2.0}


def test_pin_jit_commits_to_config_at_trace_time(tmp_path, monkeypatch):
    """jit compiles one config; tampering with the cache file has no effect."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    outer = jax.jit(lambda x: fn(x, N=4))
    r_before = outer(jnp.ones(4))

    for f in tmp_path.iterdir():
        data = json.loads(f.read_text())
        for dev in data:
            for k in data[dev]:
                data[dev][k]["config"] = [99]
        f.write_text(json.dumps(data))

    r_after = outer(jnp.ones(4))
    assert jnp.allclose(r_before, r_after)
    assert float(r_after[0]) != 99.0


def test_second_order_grad_warm_cache(tmp_path, monkeypatch):
    """jax.grad(jax.grad(fn)) with warm cache returns correct Hessian diagonal."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jnp.sum(x ** 2 * cfg.scale)

    fn(jnp.array(1.0), N=1)
    g2 = jax.grad(jax.grad(lambda x: fn(x, N=1)))(jnp.array(3.0))
    assert float(g2) in {2.0, 4.0}


def test_lax_while_loop_cold_cache(tmp_path, monkeypatch):
    """Sweep fires inside lax.while_loop body trace; final output is correct."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        i, x = carry
        return (i + 1, fn(x, N=4))

    def cond(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
        i, _ = carry
        return i < 3

    _, result = jax.lax.while_loop(cond, body, (jnp.array(0), jnp.ones(4)))
    assert result.shape == (4,) and jnp.all(result > 0)


def test_plain_int_config_jax(tmp_path, monkeypatch):
    """Plain int config: hashable, JSON-serializable, reconstructed correctly on cache hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[1, 2, 4], key=["N"])
    def fn(cfg: int, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg

    r = fn(jnp.ones(4), N=4)
    assert r.shape == (4,) and float(r[0]) in {1.0, 2.0, 4.0}
    assert jnp.allclose(r, fn(jnp.ones(4), N=4))


# ===========================================================================
# Adversarial API contract scenarios — iteration 2
# ===========================================================================

def test_none_config_round_trips_correctly(tmp_path, monkeypatch):
    """None config: encodes to null, load_best returns None (valid hit), _decode returns None.

    The _MISSING sentinel fixes the perpetual-miss bug: null is now a valid cache
    entry, not a miss indicator.  The isinstance(d, type(None)) fast-path in _decode
    returns None directly, so fn(None, ...) is called on the second call (cache hit).
    """
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    call_count = [0]

    @autotune(configs=[None], key=["N"])  # type: ignore[arg-type]
    def fn(cfg: None, x: jax.Array, *, N: int | None = None) -> jax.Array:
        call_count[0] += 1
        return x

    fn(jnp.ones(4), N=4)          # cache miss — sweep runs
    after_sweep = call_count[0]
    fn(jnp.ones(4), N=4)          # cache hit — no sweep, one call via decode path
    assert call_count[0] == after_sweep + 1  # exactly one call, not a re-sweep


def test_none_config_with_custom_encode_decode(tmp_path, monkeypatch):
    """None config works when encode/decode avoid the null sentinel."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(
        configs=[None],  # type: ignore[arg-type]
        key=["N"],
        encode=lambda _cfg: "__none__",
        decode=lambda raw: None,
    )
    def fn(cfg: None, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_async_fn_raises_at_decoration_time(tmp_path, monkeypatch):
    """@autotune on async def fn raises TypeError at decoration time."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    with pytest.raises(TypeError, match="async"):
        @autotune(configs=[KC(1)], key=["N"])
        async def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:  # type: ignore[misc]
            return x * cfg.scale


def test_fn_returning_none(tmp_path, monkeypatch):
    """fn returning None: block_until_ready(None) is a no-op; wrapper returns None."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> None:
        _ = x * cfg.scale

    assert fn(jnp.ones(4), N=4) is None


def test_configs_as_range(tmp_path, monkeypatch):
    """range() is a valid Iterable[int]: list(range(1, 4)) == [1, 2, 3]."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=range(1, 4), key=["N"])  # type: ignore[arg-type]
    def fn(cfg: int, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_key_integer_element_raises(tmp_path, monkeypatch):
    """key=[1] — integer is not a valid kwargs key → TypeError at call time."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=[1])  # type: ignore[arg-type]
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises((TypeError, KeyError)):
        fn(jnp.ones(4), N=4)


def test_double_autotune_raises(tmp_path, monkeypatch):
    """Stacking two @autotune decorators raises at call time."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def base_fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    double_wrapped = autotune(configs=[KC(2)], key=["N"])(base_fn)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        double_wrapped(jnp.ones(4), N=4)  # type: ignore[call-arg]


def test_wrapper_has_wrapped_attribute():
    """functools.wraps sets __wrapped__ pointing to the original function."""

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert hasattr(fn, "__wrapped__")
    wrapped = getattr(fn, "__wrapped__")
    assert wrapped is not fn
    assert callable(wrapped)


def test_qualname_with_angle_brackets_creates_cache_file(tmp_path, monkeypatch):
    """__qualname__ containing '<locals>' creates a valid cache file on POSIX."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    def factory() -> Any:
        @autotune(configs=[KC(1)], key=["N"])
        def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
            return x * cfg.scale
        return fn

    fn = factory()
    try:
        assert fn(jnp.ones(4), N=4).shape == (4,)
        assert len(list(tmp_path.iterdir())) >= 1
    except OSError:
        pytest.xfail("Filename with '<'/'>' chars not supported on this OS")


def test_large_number_of_configs(tmp_path, monkeypatch):
    """100 configs compiled in parallel via ThreadPoolExecutor: no deadlock."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(s) for s in range(1, 101)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_empty_tuple_config_round_trips(tmp_path, monkeypatch):
    """() config: hashable, empty pytree, encodes as [] and decodes back to ()."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[()], key=["N"])  # type: ignore[arg-type]
    def fn(cfg: tuple, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x

    fn(jnp.ones(4), N=4)
    after_sweep = sweep_count[0]
    fn(jnp.ones(4), N=4)
    assert sweep_count[0] == after_sweep + 1


def test_identical_timing_returns_first_config(tmp_path, monkeypatch):
    """On timing tie (all 0ms) the first config is chosen (loop uses strict <)."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    chosen: list[int] = []

    @autotune(configs=[KC(1), KC(2), KC(3)], key=["N"], num_timing=1, num_warmup=0)
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        chosen.append(cfg.scale)
        return x * cfg.scale

    import time as _time
    monkeypatch.setattr(_time, "perf_counter", lambda: 0.0)
    fn(jnp.ones(4), N=4)
    assert chosen[-1] == 1


def test_wrapper_preserves_metadata():
    """functools.wraps copies __name__, __qualname__, __doc__ to the wrapper."""

    @autotune(configs=[KC(1)], key=["N"])
    def my_kernel(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        """Important docstring."""
        return x * cfg.scale

    assert my_kernel.__name__ == "my_kernel"
    assert "my_kernel" in my_kernel.__qualname__
    assert my_kernel.__doc__ == "Important docstring."


def test_positional_only_param_as_key_raises(tmp_path, monkeypatch):
    """Key param declared positional-only (before /) never appears in kwargs → TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1)], key=["N"])
    def fn(cfg: KC, x: jax.Array, N: int = 4, /) -> jax.Array:  # type: ignore[misc]
        return x * cfg.scale

    with pytest.raises(TypeError, match="N"):
        fn(jnp.ones(4), 4)


def test_empty_key_list_no_kwargs_call(tmp_path, monkeypatch):
    """key=[] with zero kwargs: extraction loop is a no-op, call proceeds."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=[])
    def fn(cfg: KC, x: jax.Array) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.ones(4)).shape == (4,)


def test_fn_execution_error_during_sweep_propagates(tmp_path, monkeypatch):
    """RuntimeError from encode after sweep propagates out of wrapper."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(
        configs=[KC(1)],
        key=["N"],
        encode=lambda cfg: (_ for _ in ()).throw(RuntimeError("sweep-time error")),
    )
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(RuntimeError, match="sweep-time error"):
        fn(jnp.ones(4), N=4)


# ===========================================================================
# Adversarial JAX execution model scenarios — iteration 2
# ===========================================================================

def test_eval_shape_cold_cache_returns_correct_abstract_shape(tmp_path, monkeypatch):
    """eval_shape on cold-cache fn triggers sweep and returns the right abstract shape."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    result = jax.eval_shape(lambda x: fn(x, N=4), jnp.ones(4))
    assert result.shape == (4,)
    assert result.dtype == jnp.float32


def test_eval_shape_cold_cache_populates_cache(tmp_path, monkeypatch):
    """Sweep runs inside eval_shape → cache written → next real call is a hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x * cfg.scale

    jax.eval_shape(lambda x: fn(x, N=4), jnp.ones(4))
    count_after_eval = sweep_count[0]
    assert count_after_eval >= 2

    fn(jnp.ones(4), N=4)
    assert sweep_count[0] == count_after_eval + 1


def test_eval_shape_warm_cache(tmp_path, monkeypatch):
    """eval_shape on warm-cache fn returns correct abstract shape without a sweep."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    assert jax.eval_shape(lambda x: fn(x, N=4), jnp.ones(4)).shape == (4,)


def test_make_jaxpr_cold_cache_produces_valid_jaxpr(tmp_path, monkeypatch):
    """make_jaxpr on cold-cache fn triggers sweep and returns a valid jaxpr."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    jaxpr = jax.make_jaxpr(lambda x: fn(x, N=4))(jnp.ones(4))
    assert len(jaxpr.jaxpr.eqns) >= 1


def test_make_jaxpr_warm_cache_uses_best_config(tmp_path, monkeypatch):
    """make_jaxpr on warm-cache fn captures the best config's constant."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    jaxpr_str = str(jax.make_jaxpr(lambda x: fn(x, N=4))(jnp.ones(4)))
    assert "1.0" in jaxpr_str or "2.0" in jaxpr_str


def test_make_jaxpr_cold_cache_populates_cache(tmp_path, monkeypatch):
    """Sweep inside make_jaxpr writes the cache; subsequent real call is a hit."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))
    sweep_count = [0]

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        sweep_count[0] += 1
        return x * cfg.scale

    jax.make_jaxpr(lambda x: fn(x, N=4))(jnp.ones(4))
    count_after = sweep_count[0]
    assert count_after >= 2
    fn(jnp.ones(4), N=4)
    assert sweep_count[0] == count_after + 1


def test_jit_of_wrapper_static_argnums_0_raises(tmp_path, monkeypatch):
    """jax.jit(fn, static_argnums=0) tries to hash a jax.Array → ValueError/TypeError."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.ones(4), N=4)
    jit_fn = jax.jit(fn, static_argnums=(0,))
    with pytest.raises((ValueError, TypeError)):
        jit_fn(jnp.ones(4), N=4)


def test_zero_sized_array_sweep_and_result_shape(tmp_path, monkeypatch):
    """Shape (0, 4) array: dummy construction, compile, timing all succeed."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    assert fn(jnp.zeros((0, 4)), N=4).shape == (0, 4)


def test_zero_sized_array_warm_cache(tmp_path, monkeypatch):
    """Zero-sized array on warm cache hit returns correct shape."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    fn(jnp.zeros((0, 4)), N=4)
    assert fn(jnp.zeros((0, 4)), N=4).shape == (0, 4)


def test_bool_dtype_sweep_and_result(tmp_path, monkeypatch):
    """bool arrays: jnp.empty(shape, bool_) is valid; sweep compiles and runs."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x & jnp.ones_like(x, dtype=jnp.bool_)

    x = jnp.array([True, False, True, False])
    r = fn(x, N=4)
    assert r.dtype == jnp.bool_ and r.shape == (4,)


def test_bool_dtype_warm_cache(tmp_path, monkeypatch):
    """bool array warm cache hit returns correct dtype and shape."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x & jnp.ones_like(x, dtype=jnp.bool_)

    x = jnp.array([True, False, True, False])
    fn(x, N=4)
    r = fn(x, N=4)
    assert r.dtype == jnp.bool_ and r.shape == (4,)


def test_complex64_sweep_and_result(tmp_path, monkeypatch):
    """complex64 arrays: dummy construction and timing succeed."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    x = jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64)
    r = fn(x, N=2)
    assert r.dtype == jnp.complex64 and r.shape == (2,)


def test_complex64_warm_cache(tmp_path, monkeypatch):
    """complex64 array warm cache hit returns correct dtype."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    x = jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64)
    fn(x, N=2)
    assert fn(x, N=2).dtype == jnp.complex64


def test_custom_pytree_arg_sweep(tmp_path, monkeypatch):
    """Custom pytree as positional arg: tree.map recurses into it correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, pair: _Pair, *, N: int | None = None) -> jax.Array:
        return (pair.a + pair.b) * cfg.scale

    assert fn(_Pair(jnp.ones(4), jnp.ones(4) * 2.0), N=4).shape == (4,)


def test_custom_pytree_arg_warm_cache(tmp_path, monkeypatch):
    """Custom pytree warm cache hit returns same result as cold call."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, pair: _Pair, *, N: int | None = None) -> jax.Array:
        return (pair.a + pair.b) * cfg.scale

    pair = _Pair(jnp.ones(4), jnp.ones(4) * 2.0)
    assert jnp.allclose(fn(pair, N=4), fn(pair, N=4))


def test_mixed_shape_dtype_args_sweep(tmp_path, monkeypatch):
    """Two positional arrays with different shapes and dtypes get correct dummies."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, y: jax.Array, *, N: int | None = None) -> jax.Array:
        return (x + y.astype(x.dtype)[: x.shape[0]]) * cfg.scale

    r = fn(jnp.ones(4, dtype=jnp.float32), jnp.ones(8, dtype=jnp.int32), N=4)
    assert r.shape == (4,) and r.dtype == jnp.float32


def test_mixed_shape_dtype_args_warm_cache(tmp_path, monkeypatch):
    """Warm cache hit with mixed-type args returns numerically consistent result."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, y: jax.Array, *, N: int | None = None) -> jax.Array:
        return (x + y.astype(x.dtype)[: x.shape[0]]) * cfg.scale

    x, y = jnp.ones(4, dtype=jnp.float32), jnp.ones(8, dtype=jnp.int32)
    assert jnp.allclose(fn(x, y, N=4), fn(x, y, N=4))


def test_inner_jit_in_fn_cold_cache(tmp_path, monkeypatch):
    """fn that calls jax.jit internally: sweep compiles and times correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jax.jit(lambda a: a * cfg.scale)(x)

    assert fn(jnp.ones(4), N=4).shape == (4,)


def test_inner_jit_in_fn_warm_cache(tmp_path, monkeypatch):
    """Warm cache hit with inner-jit fn: result consistent."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jax.jit(lambda a: a * cfg.scale)(x)

    x = jnp.ones(4)
    assert jnp.allclose(fn(x, N=4), fn(x, N=4))


def test_inner_jit_in_fn_inside_outer_jit(tmp_path, monkeypatch):
    """Warm cache hit inside an outer jax.jit with inner-jit fn computes correctly."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return jax.jit(lambda a: a * cfg.scale)(x)

    fn(jnp.ones(4), N=4)
    assert jax.jit(lambda x: fn(x, N=4))(jnp.ones(4)).shape == (4,)


def test_shape_dtype_struct_as_positional_arg_raises(tmp_path, monkeypatch):
    """ShapeDtypeStruct is not computable data — passing it directly raises TypeError.

    The sweep phase succeeds (dummy_args converts it via hasattr), but the final
    fn(best_config, struct) call can't compute x * cfg.scale on an abstract descriptor.
    Use jax.eval_shape if you need abstract output shape information.
    """
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> jax.Array:
        return x * cfg.scale

    with pytest.raises(TypeError):
        fn(jax.ShapeDtypeStruct((4,), jnp.float32), N=4)  # type: ignore[arg-type]


def test_tuple_output_cold_cache(tmp_path, monkeypatch):
    """fn returning a tuple: both elements have correct shapes on cache miss."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> tuple[jax.Array, jax.Array]:
        return (x * cfg.scale, x * cfg.scale * 2)

    r = fn(jnp.ones(4), N=4)
    assert isinstance(r, tuple) and len(r) == 2
    assert r[0].shape == (4,) and r[1].shape == (4,)


def test_tuple_output_warm_cache(tmp_path, monkeypatch):
    """Warm cache hit with tuple output: both elements match cold-call values."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> tuple[jax.Array, jax.Array]:
        return (x * cfg.scale, x * cfg.scale * 2)

    x = jnp.ones(4)
    r_cold = fn(x, N=4)
    r_warm = fn(x, N=4)
    assert jnp.allclose(r_cold[0], r_warm[0]) and jnp.allclose(r_cold[1], r_warm[1])


def test_tuple_output_values_are_consistent(tmp_path, monkeypatch):
    """Tuple output: r[1] is always exactly 2 * r[0]."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> tuple[jax.Array, jax.Array]:
        return (x * cfg.scale, x * cfg.scale * 2)

    r = fn(jnp.ones(4), N=4)
    assert jnp.allclose(r[1], r[0] * 2)


def test_tuple_output_inside_jit(tmp_path, monkeypatch):
    """Tuple output fn inside jax.jit returns correct shapes on warm cache."""
    monkeypatch.setenv("TONNO_CACHE_DIR", str(tmp_path))

    @autotune(configs=[KC(1), KC(2)], key=["N"])
    def fn(cfg: KC, x: jax.Array, *, N: int | None = None) -> tuple[jax.Array, jax.Array]:
        return (x * cfg.scale, x * cfg.scale * 2)

    fn(jnp.ones(4), N=4)
    r = jax.jit(lambda x: fn(x, N=4))(jnp.ones(4))
    assert isinstance(r, tuple) and r[0].shape == (4,) and r[1].shape == (4,)
