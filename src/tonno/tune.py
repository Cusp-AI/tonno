# Copyright 2025 CuspAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from tonno import cache as _cache
from tonno.cache import MISSING
from tonno.hardware import get_device_name

_F = TypeVar("_F", bound=Callable[..., Any])


def _make_cache_key(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    tunable_names: frozenset[str],
) -> dict[str, Any]:
    """Derive a JSON-serialisable cache key from non-tunable arguments.

    Arrays are keyed by (shape, dtype) — the same information jax.jit uses to
    decide whether to recompile.  All other values are included as-is.
    """
    key: dict[str, Any] = {}
    for i, a in enumerate(args):
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            key[f"__arg{i}"] = {"shape": list(a.shape), "dtype": str(a.dtype)}
        else:
            key[f"__arg{i}"] = a
    for k, v in sorted(kwargs.items()):
        if k in tunable_names:
            continue
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            key[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            key[k] = v
    return key


def autotune(
    configs: list[dict[str, Any]],
    *,
    num_warmup: int = 1,
    num_timing: int = 3,
) -> Callable[[_F], _F]:
    """Sweep block-size configs on a ``jax.jit``-wrapped function and cache the best.

    Stack ``@autotune`` on top of ``@jax.jit``.  The tunable parameters (the
    keys of each config dict) must be declared as ``static_argnames`` in the
    ``jax.jit`` call and must have defaults in the function signature — autotune
    injects the best values at call time, callers never pass them.

    The cache key is derived automatically from the non-tunable arguments using
    the same logic as ``jax.jit``: arrays are identified by ``(shape, dtype)``,
    all other static args by their value.  No explicit ``key=`` is needed.

    Example::

        @autotune(configs=[{"BN": 32, "BK": 64}, {"BN": 64, "BK": 128}])
        @jax.jit(static_argnames=["BN", "BK"])
        def matmul(x: Array, w: Array, BN: int = 32, BK: int = 64) -> Array:
            return pallas_matmul(x, w, BN=BN, BK=BK)

        matmul(x, w)                # BN/BK swept once, best config cached
        matmul(x, w, BN=16, BK=32)  # explicit override, autotune bypassed

    Args:
        configs: Non-empty list of dicts mapping tunable kwarg names to candidate
                 values.  All dicts must have identical keys.  Values must be
                 JSON-serialisable (int, float, str, bool).
        num_warmup: Warmup calls per config after compilation.  Default 1.
        num_timing: Timed calls per config; median elapsed time is used.  Default 3.
    """
    if not configs:
        raise ValueError("configs must not be empty")
    if num_timing < 1:
        raise ValueError(f"num_timing must be >= 1, got {num_timing}")
    if num_warmup < 0:
        raise ValueError(f"num_warmup must be >= 0, got {num_warmup}")

    ref_keys = frozenset(configs[0].keys())
    if not ref_keys:
        raise ValueError("config dicts must not be empty")
    for i, cfg in enumerate(configs[1:], 1):
        if frozenset(cfg.keys()) != ref_keys:
            raise ValueError(
                f"configs[{i}] has different keys than configs[0]: "
                f"{set(cfg.keys())} vs {set(configs[0].keys())}"
            )

    tunable_names = ref_keys

    def decorator(fn: _F) -> _F:
        if not hasattr(fn, "lower"):
            raise TypeError(
                f"autotune expects a jax.jit-wrapped function, "
                f"got {type(fn).__name__!r}.\n"
                f"Stack @autotune on top of @jax.jit:\n\n"
                f"    @autotune(configs=[...])\n"
                f"    @jax.jit(static_argnames=[...])\n"
                f"    def fn(...): ..."
            )

        fn_name = fn.__qualname__

        # Validate tunable params against the underlying function's signature.
        underlying = getattr(fn, "__wrapped__", None)
        if underlying is not None:
            sig = inspect.signature(underlying)
            for k in tunable_names:
                param = sig.parameters.get(k)
                if param is None:
                    raise TypeError(
                        f"tunable parameter {k!r} from configs not found "
                        f"in {fn.__qualname__!r}"
                    )
                if param.default is inspect.Parameter.empty:
                    raise TypeError(
                        f"tunable parameter {k!r} in {fn.__qualname__!r} has no "
                        f"default. Autotune injects it at call time — add a "
                        f"default, e.g. {k}: int = 0"
                    )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # User supplied tunable kwargs explicitly — bypass autotune.
            if any(k in kwargs for k in tunable_names):
                return fn(*args, **kwargs)

            device_name = get_device_name()
            key_values = _make_cache_key(args, kwargs, tunable_names)

            raw = _cache.load_best(fn_name, device_name, key_values)
            if raw is not MISSING:
                return fn(*args, **kwargs, **raw)

            # Build concrete dummy inputs so the sweep is safe inside any JAX
            # trace context (jax.grad, jax.vmap, jax.lax.scan, etc.).
            # ensure_compile_time_eval makes jnp.empty concrete even when args
            # are abstract tracers — shape and dtype are always concrete.
            with jax.ensure_compile_time_eval():
                dummy_args = jax.tree.map(
                    lambda a: jnp.empty(a.shape, a.dtype)  # type: ignore[reportUnknownLambdaType]
                    if hasattr(a, "shape") and hasattr(a, "dtype")
                    else a,
                    args,
                )
                dummy_kw = {
                    k: jnp.empty(v.shape, v.dtype)
                    if hasattr(v, "shape") and hasattr(v, "dtype")
                    else v
                    for k, v in kwargs.items()
                }

            # Compile all configs in parallel — XLA is CPU-bound and thread-safe.
            def _compile(cfg: dict[str, Any]) -> None:
                jax.block_until_ready(fn(*dummy_args, **dummy_kw, **cfg))

            with ThreadPoolExecutor() as pool:
                list(pool.map(_compile, configs))

            # Time each config sequentially for accurate device measurements.
            best_config = configs[0]
            best_time = float("inf")

            for cfg in configs:
                for _ in range(num_warmup):
                    jax.block_until_ready(fn(*dummy_args, **dummy_kw, **cfg))
                times = []
                for _ in range(num_timing):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(*dummy_args, **dummy_kw, **cfg))
                    times.append((time.perf_counter() - t0) * 1000)
                median = sorted(times)[len(times) // 2]
                if median < best_time:
                    best_time = median
                    best_config = cfg

            _cache.save_best(fn_name, device_name, key_values, best_config, best_time)
            return fn(*args, **kwargs, **best_config)

        return wrapper  # type: ignore[return-value]

    return decorator
