# Autotune Library — Handover

## Context

The Pallas kernels in this repo (`segmented_matmul_pallas.py`) use fixed block sizes
because Pallas has **no built-in autotuning** (jax-ml/jax#24340, still open as of 2026-03).

Triton's `@triton.autotune` sweeps configs at runtime keyed on problem dimensions.
We need an equivalent for Pallas — a small standalone library, not tied to this repo.

## Requirements

1. **Decorator API** similar to Triton's `@autotune`:
   ```python
   @autotune(
       configs=[
           Config(bm=16, bk=32, bn=128),
           Config(bm=32, bk=32, bn=128),
           Config(bm=64, bk=32, bn=256),
       ],
       key=["N", "D_in", "D_out", "S"],
   )
   def my_pallas_kernel(x, w, seg, *, bm, bk, bn):
       ...
   ```

2. **Cache results** per (hardware, key-values) so the sweep only runs once per
   unique problem shape. Persist to disk (JSON/pickle) so restarts are free.

3. **Parallel compilation** — JAX compiles different configs independently; launch
   all compilations concurrently, then time sequentially.

4. **Hardware detection** — key the cache on device type (e.g. `H100-80GB`, `TPU-v4`)
   so configs port across runs on the same hardware.

5. **Warmup + timing** — configurable number of warmup iterations and timing
   iterations. Use `jax.block_until_ready()` for accurate GPU timing.

## Existing Work

- **tune-jax** (https://github.com/rdyro/tune-jax) — closest prior art. Provides a
  `@tune` decorator with parallel compilation. Worth studying but last updated Oct 2024.
- **Tokamax** (https://github.com/openxla/tokamax) — Google's curated kernel library
  with autotuning infra, but it's a kernel library not a general decorator.
- **ejkernel** (https://github.com/erfanzar/ejkernel) — another library with config
  hierarchy for Pallas/Triton.

## Design Notes

- Keep it zero-dependency beyond JAX itself.
- The cache file should be human-readable (JSON) so users can inspect/override.
- Consider a `fallback` config for when no cache exists yet (first config in list).
- For this repo's segmented matmul, the tuning keys are `(N, D_in, D_out, S)` —
  same as the Triton version.

## Block Size Candidates (from Triton configs)

Forward / grad_x:
| BM  | BK  | BN  |
|-----|-----|-----|
| 16  | 32  | 128 |
| 32  | 32  | 64  |
| 32  | 32  | 128 |
| 32  | 32  | 256 |
| 32  | 64  | 128 |
| 64  | 32  | 128 |
| 64  | 32  | 256 |
| 64  | 64  | 128 |

Grad_w:
| BN  | BK  | BD  |
|-----|-----|-----|
| 16  | 32  | 128 |
| 32  | 32  | 128 |
| 32  | 64  | 128 |
| 64  | 32  | 128 |
| 32  | 32  | 256 |
| 16  | 32  | 256 |
