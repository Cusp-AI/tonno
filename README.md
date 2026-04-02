# tonno

Autotuning for JAX/Pallas kernels — a lightweight `@autotune` decorator that sweeps
block-size configs, times them on the current device, and caches the winner so the
sweep only runs once per (hardware, problem shape) pair.

Inspired by Triton's `@triton.autotune`. Fills the same gap for Pallas (see
[jax-ml/jax#24340](https://github.com/jax-ml/jax/issues/24340)).

## Install

```bash
pip install tonno          # or: uv add tonno
```

## Usage

Stack `@autotune` on top of `@jax.jit`. The tunable parameters (keys of each config
dict) must be declared as `static_argnames` in the `jax.jit` call and have defaults
in the function signature — autotune injects the best values at call time.

```python
import jax
from tonno import autotune

@autotune(configs=[
    {"BM": 32, "BN": 64},
    {"BM": 64, "BN": 128},
])
@jax.jit(static_argnames=["BM", "BN"])
def matmul(a: jax.Array, b: jax.Array, BM: int = 32, BN: int = 64) -> jax.Array:
    # BM / BN are concrete ints at compile time (static_argnames)
    return pallas_matmul(a, b, BM=BM, BN=BN)
```

### Calling it

```python
# First call: sweeps all configs, compiles in parallel, times sequentially,
# writes the winner to .tonno-cache/matmul.json
c = matmul(a, b)

# Subsequent calls with the same input shapes on the same device: cache hit,
# no sweep, runs immediately with the best config
c = matmul(a, b)

# Explicit override — bypass autotune entirely
c = matmul(a, b, BM=16, BN=32)
```

The cache key is derived from the input shapes and dtypes — exactly the same
information `jax.jit` uses to decide whether to recompile. No explicit `key=`
parameter needed.

### Non-tunable static args

If your kernel has static args beyond the tunable ones, declare them in `jax.jit`
as usual. They are automatically part of the cache key:

```python
@autotune(configs=[{"BM": 32}, {"BM": 64}])
@jax.jit(static_argnames=["BM", "transpose"])
def matmul(a: jax.Array, b: jax.Array, transpose: bool = False, BM: int = 32) -> jax.Array:
    ...
```

## How it works

1. **On first call** (cache miss): dummy inputs are built from the args' shapes/dtypes
   via `jax.ensure_compile_time_eval`. All configs are compiled in parallel via
   `ThreadPoolExecutor` (XLA compilation is CPU-bound). Each compiled artifact is then
   timed sequentially for accurate device timing. The winner is written to
   `.tonno-cache/<fn>.json`.

2. **On subsequent calls** (cache hit): the best config is loaded from disk and
   injected as static kwargs. JAX's own compilation cache takes over from there.

3. **Inside `jax.jit` / `jax.grad` / `jax.vmap`**: the sweep runs as a side channel
   during the first trace, then the winning config is baked into the jaxpr as a
   compile-time constant.

## API reference

```python
autotune(
    configs: list[dict[str, Any]],  # configs to sweep; all dicts must share keys
    *,
    num_warmup: int = 1,            # warmup calls per config after compilation
    num_timing: int = 3,            # timed calls per config (median used)
)
```

**Contract:**

- `@autotune` must wrap a `@jax.jit`-decorated function.
- The tunable param keys must appear in `static_argnames` of the `jax.jit` call.
- Tunable params must have defaults in the function signature.
- Config values must be JSON-serialisable (`int`, `float`, `str`, `bool`).

## Cache

Results are stored in `.tonno-cache/<qualname>.json` (or `$TONNO_CACHE_DIR`).
The file is human-readable JSON; you can inspect or delete entries manually.

```json
{
  "NVIDIA H100 80GB": {
    "{\"__arg0\":{\"dtype\":\"float32\",\"shape\":[4096,4096]}, ...}": {
      "config": {"BM": 64, "BN": 128},
      "time_ms": 0.312,
      "key_values": {...}
    }
  }
}
```

## Example

See [`examples/matmul.py`](examples/matmul.py) for a complete autotuned tiled GEMM.
