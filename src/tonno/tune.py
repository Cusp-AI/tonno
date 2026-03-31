from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax

from tonno import cache as _cache
from tonno.config import Config
from tonno.hardware import get_device_name


def autotune(
    configs: Sequence[Config],
    key: Sequence[str],
    *,
    num_warmup: int = 1,
    num_timing: int = 3,
) -> Callable:
    """Decorator that automates finding the best config for a JAX function.

    Args:
        configs: List of Config objects to sweep over.
        key: Names of keyword arguments that identify the problem shape.
              The caller must pass these as kwargs.
        num_warmup: Number of warmup iterations before timing.
        num_timing: Number of timed iterations (median is used).
    """
    if not configs:
        raise ValueError("configs must not be empty")

    def decorator(fn: Callable) -> Callable:
        fn_name = fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract key values from kwargs
            key_values = {}
            for k in key:
                if k not in kwargs:
                    raise TypeError(
                        f"autotune key {k!r} not found in kwargs. "
                        f"Pass it as a keyword argument."
                    )
                key_values[k] = kwargs.pop(k)

            device_name = get_device_name()

            # Check cache
            best = _cache.load_best(fn_name, device_name, key_values)
            if best is not None:
                return fn(*args, **kwargs, **best.to_dict())

            # Sweep
            best_config = configs[0]
            best_time = float("inf")

            for cfg in configs:
                merged_kwargs = {**kwargs, **cfg.to_dict()}
                t = _time_fn(fn, args, merged_kwargs, num_warmup, num_timing)
                if t < best_time:
                    best_time = t
                    best_config = cfg

            _cache.save_best(fn_name, device_name, key_values, best_config, best_time)

            return fn(*args, **kwargs, **best_config.to_dict())

        return wrapper

    return decorator


def _time_fn(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    num_warmup: int,
    num_timing: int,
) -> float:
    """Time a function call, returning median time in milliseconds."""
    # Warmup
    for _ in range(num_warmup):
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)

    # Timing
    times = []
    for _ in range(num_timing):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    return times[len(times) // 2]
