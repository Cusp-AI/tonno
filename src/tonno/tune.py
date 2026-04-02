from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Concatenate, ParamSpec, TypeVar

import jax
import jax.numpy as jnp

from tonno import cache as _cache
from tonno.cache import _MISSING
from tonno.hardware import get_device_name

_C = TypeVar("_C", bound=Hashable)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def autotune(
    configs: Iterable[_C],
    key: Sequence[str],
    *,
    num_warmup: int = 1,
    num_timing: int = 3,
    encode: Callable[[_C], Any] | None = None,
    decode: Callable[[Any], _C] | None = None,
) -> Callable[[Callable[Concatenate[_C, _P], _R]], Callable[_P, _R]]:
    """Decorator that automates finding the best config for a JAX/Pallas function.

    The decorated function must accept a config as its first positional argument.
    Autotune injects the best config at call time; callers never pass it.

    ``NamedTuple`` is the recommended config type: it is hashable, fully typed,
    and JSON-serialisable out of the box (json encodes tuples as lists; the
    default decoder reconstructs via ``cfg_type(*loaded_list)``).  For types
    that are not JSON-serialisable by default, supply ``encode`` and ``decode``.

    Example::

        class KC(NamedTuple):
            bm: int
            bk: int

        @autotune(configs=[KC(32, 64), KC(64, 32)], key=["N"])
        def kernel(cfg: KC, x, *, N=None):
            return x * cfg.bm  # cfg.bm: int — fully typed

    Args:
        configs: Configs to sweep.  Must be hashable (required by JAX's
                 ``static_argnums``) and share the same pytree structure.
        key: Names of kwargs that identify the problem shape.  Must be
             passed by the caller; used as the cache key.
        num_warmup: Warmup iterations before timing.
        num_timing: Timed iterations per config; median is used.
        encode: Converts a config to a JSON-serialisable value for the cache.
                Defaults to identity — works for NamedTuple/tuple and any type
                that json can serialise directly.
        decode: Reconstructs a config from the json-loaded value.  Defaults to
                ``cfg_type(*data)`` for list data (covers NamedTuple/tuple) or
                ``cfg_type(**data)`` for dict data.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("configs must not be empty")
    if num_timing < 1:
        raise ValueError(f"num_timing must be >= 1, got {num_timing}")
    if num_warmup < 0:
        raise ValueError(f"num_warmup must be >= 0, got {num_warmup}")

    # All configs must share the same pytree structure so they produce comparable
    # compiled artifacts (same input/output shapes) and can be fairly timed.
    ref_structure = jax.tree.structure(configs[0])
    for i, cfg in enumerate(configs[1:], 1):
        if jax.tree.structure(cfg) != ref_structure:
            raise ValueError(
                f"configs[{i}] has a different pytree structure than configs[0]. "
                f"All configs must be the same type with the same fields.\n"
                f"  configs[0]: {configs[0]!r}\n"
                f"  configs[{i}]: {cfg!r}"
            )

    cfg_type = type(configs[0])
    _encode: Callable[[_C], Any] = encode if encode is not None else (lambda x: x)

    def _decode(d: Any) -> _C:  # type: ignore[misc]
        if decode is not None:
            return decode(d)
        # Fast path: if the cached value is already the right type, return it
        # directly.  This handles plain scalars (int, str) and None configs.
        if isinstance(d, cfg_type):
            return d  # type: ignore[return-value]
        # Check dict before tuple: a NamedTuple is-a tuple, so without this guard
        # KC(*{"bm": 32}) would iterate dict keys ("bm") not values (32).
        if isinstance(d, dict):
            return cfg_type(**d)  # type: ignore[call-arg]
        if issubclass(cfg_type, tuple):
            return cfg_type(*d)  # type: ignore[call-arg]
        return cfg_type(d)  # type: ignore[call-arg]

    def decorator(
        fn: Callable[Concatenate[_C, _P], _R],
    ) -> Callable[_P, _R]:
        if inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"autotune does not support async functions ({fn.__qualname__!r}). "
                f"JAX operations are synchronous — remove 'async' from the definition."
            )

        fn_name = fn.__qualname__

        # Validate that any key param declared in fn has a default.
        # Key params are popped before fn is called, so a required (no-default)
        # key param causes a cryptic "missing keyword argument" error in the sweep.
        sig = inspect.signature(fn)
        for k in key:
            param = sig.parameters.get(k)
            if param is not None and param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"key parameter {k!r} in {fn.__qualname__!r} has no default. "
                    f"Autotune pops key params before calling fn, so they are never "
                    f"forwarded — add a default: {k}: ... = None"
                )

        # The config is the first positional arg and is static (static_argnums=0).
        # Because configs are hashable, JAX produces one compiled artifact per unique
        # config — exactly what Pallas kernels need (cfg.bm / cfg.bk determine the grid).
        jitted_fn = jax.jit(fn, static_argnums=(0,))

        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            kw: dict[str, Any] = dict(kwargs)

            # Extract key values — must be concrete Python values (problem shape).
            key_values: dict[str, Any] = {}
            for k in key:
                if k not in kw:
                    raise TypeError(
                        f"autotune key {k!r} not found in kwargs. "
                        f"Pass it as a keyword argument."
                    )
                key_values[k] = kw.pop(k)

            device_name = get_device_name()

            raw = _cache.load_best(fn_name, device_name, key_values)
            if raw is not _MISSING:
                return fn(_decode(raw), *args, **kw)  # type: ignore[call-arg]

            # Build concrete dummy inputs from args' abstract properties.
            # ensure_compile_time_eval makes jnp.empty concrete even inside a jit
            # trace: tracer.shape and tracer.dtype are always concrete.
            with jax.ensure_compile_time_eval():
                dummy_args = jax.tree.map(
                    lambda x: jnp.empty(x.shape, x.dtype) if hasattr(x, "shape") and hasattr(x, "dtype") else x,
                    args,
                )
                dummy_kw = {
                    k: jnp.empty(v.shape, v.dtype) if hasattr(v, "shape") and hasattr(v, "dtype") else v
                    for k, v in kw.items()
                }

            # Compile all configs in parallel — XLA is CPU-bound and thread-safe.
            # The config is static_argnums=0; the compiled artifact has no config input.
            def _compile(cfg: _C) -> Any:
                return jitted_fn.lower(cfg, *dummy_args, **dummy_kw).compile()

            with ThreadPoolExecutor() as pool:
                futures = {cfg: pool.submit(_compile, cfg) for cfg in configs}
            compiled_map = {cfg: f.result() for cfg, f in futures.items()}

            # Time each compiled artifact sequentially for accurate device timing.
            best_config = configs[0]
            best_time = float("inf")

            for cfg, exe in compiled_map.items():
                for _ in range(num_warmup):
                    jax.block_until_ready(exe(*dummy_args, **dummy_kw))
                times = []
                for _ in range(num_timing):
                    t0 = time.perf_counter()
                    jax.block_until_ready(exe(*dummy_args, **dummy_kw))
                    times.append((time.perf_counter() - t0) * 1000)
                median = sorted(times)[len(times) // 2]
                if median < best_time:
                    best_time = median
                    best_config = cfg

            _cache.save_best(fn_name, device_name, key_values, _encode(best_config), best_time)

            # Call fn with the original args + best config as first positional.
            # When inside a jit trace this is the only call that lands in the jaxpr.
            return fn(best_config, *args, **kw)  # type: ignore[call-arg]

        return wrapper  # type: ignore[return-value]

    return decorator
