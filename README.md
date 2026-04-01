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

### 1. Define a config type

Use a `NamedTuple` — hashable, fully typed, JSON-serialisable out of the box:

```python
from typing import NamedTuple

class GemmConfig(NamedTuple):
    bm: int  # output tile rows
    bn: int  # output tile cols
```

### 2. Decorate your Pallas kernel

The config is the **first positional argument**. Autotune injects it; callers never
pass it directly. Key kwargs identify the problem shape and are used as the cache key.

```python
from jax.experimental import pallas as pl
from tonno import autotune

@autotune(
    configs=[
        GemmConfig(bm=16, bn=64),
        GemmConfig(bm=32, bn=128),
        GemmConfig(bm=64, bn=128),
    ],
    key=["M", "K", "N"],
)
def matmul(
    cfg: GemmConfig,
    a: jax.Array,
    b: jax.Array,
    *,
    M: int | None = None,   # key param — must have a default
    K: int | None = None,
    N: int | None = None,
) -> jax.Array:
    # cfg.bm / cfg.bn are concrete ints at JIT compile time (static_argnums=0)
    # Derive grid from array shapes, not from key params (those are popped)
    return pl.pallas_call(
        lambda a_ref, b_ref, c_ref: ...,
        out_shape=jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), a.dtype),
        grid=(a.shape[0] // cfg.bm, b.shape[1] // cfg.bn),
        ...
    )(a, b)
```

### 3. Call it

```python
# First call: sweeps all configs, compiles in parallel, times sequentially,
# writes the best GemmConfig to .tonno-cache/matmul.json
c = matmul(a, b, M=4096, K=4096, N=4096)

# Subsequent calls with the same (M, K, N) on the same device: cache hit,
# no sweep, runs immediately with the best config
c = matmul(a, b, M=4096, K=4096, N=4096)
```

The cache is per-device (`H100-80GB`, `TPU-v4`, `cpu`, …) so configs transfer
correctly across runs on the same hardware.

## How it works

1. **On first call** (cache miss): dummy inputs are built from the args' shapes/dtypes.
   All configs are compiled in parallel via `ThreadPoolExecutor` (XLA compilation is
   CPU-bound). Each compiled artifact is then timed sequentially on the dummy inputs
   for accurate device timing. The winner is written to `.tonno-cache/<fn>.json`.

2. **On subsequent calls** (cache hit): the best config is loaded from disk and
   injected as `static_argnums=0`. JAX's own compilation cache takes over from there.

3. **Inside `jax.jit`**: the sweep runs as a side channel during the first trace
   (via `jax.ensure_compile_time_eval`), then the winning config is baked into the
   jaxpr as a compile-time static.

## Config types

Any **hashable** type works. `NamedTuple` is recommended because it is:
- Hashable → required by `static_argnums`
- Typed → `cfg.bm: int`, not `cfg.bm: int | float | str | bool`
- JSON-serialisable natively (tuple → list; default decoder reconstructs via `T(*data)`)

```python
# NamedTuple — recommended
class KC(NamedTuple):
    bm: int
    bk: int

# frozen dataclass — works with explicit encode/decode
from dataclasses import dataclass
import dataclasses

@dataclass(frozen=True)
class KC:
    bm: int
    bk: int

@autotune(
    configs=[KC(32, 64), KC(64, 32)],
    key=["N"],
    encode=dataclasses.asdict,
    decode=lambda d: KC(**d),
)
def kernel(cfg: KC, x, *, N=None): ...
```

## API reference

```python
autotune(
    configs: Iterable[_C],          # configs to sweep, must be hashable
    key: Sequence[str],             # kwargs naming the problem shape
    *,
    num_warmup: int = 1,            # warmup iterations before timing
    num_timing: int = 3,            # timed iterations (median used)
    encode: Callable | None = None, # config → JSON-serialisable (default: identity)
    decode: Callable | None = None, # JSON-loaded → config (default: T(*data))
)
```

**Rules for the decorated function:**

- Config is the **first positional argument**, typed as `_C`.
- Key params must have a **default value** (`N: int | None = None`) — they are
  popped by autotune and never forwarded to the function body.
- Derive Pallas grids from **array shapes** (`a.shape[0] // cfg.bm`), not from
  key params (which are `None` inside the function).
- All configs must have the **same pytree structure** (same type, same fields).

## Cache

Results are stored in `.tonno-cache/<qualname>.json` (or `$TONNO_CACHE_DIR`).
The file is human-readable JSON; you can inspect or delete entries manually.

```json
{
  "NVIDIA H100 80GB": {
    "{\"M\":4096,\"K\":4096,\"N\":4096}": {
      "config": [64, 128],
      "time_ms": 0.312,
      "key_values": {"M": 4096, "K": 4096, "N": 4096}
    }
  }
}
```

## Example

See [`examples/matmul.py`](examples/matmul.py) for a complete autotuned tiled GEMM.
