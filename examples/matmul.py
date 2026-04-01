"""Autotuned tiled matrix multiply — Pallas kernel example.

This is the canonical use case for tonno: finding the best (bm, bk, bn)
block-size triple for a tiled GEMM on the current GPU.

The first call to ``matmul`` sweeps all configs, compiles each in parallel,
times them sequentially, and writes the winner to ``.tonno-cache/``.
Every subsequent call with the same (M, K, N) hits the cache and runs
immediately with the best config.

Requirements: JAX with GPU or TPU backend.

    python examples/matmul.py
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tonno import autotune


class GemmConfig(NamedTuple):
    bm: int  # tile rows (output)
    bk: int  # tile depth (reduction)
    bn: int  # tile cols (output)


def _matmul_kernel(a_ref, b_ref, c_ref):
    """Accumulate one (bm, bk) × (bk, bn) tile into c_ref."""
    c_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(c_ref.dtype)


@autotune(
    configs=[
        GemmConfig(bm=16, bk=32, bn=128),
        GemmConfig(bm=32, bk=32, bn=64),
        GemmConfig(bm=32, bk=32, bn=128),
        GemmConfig(bm=32, bk=32, bn=256),
        GemmConfig(bm=32, bk=64, bn=128),
        GemmConfig(bm=64, bk=32, bn=128),
        GemmConfig(bm=64, bk=32, bn=256),
        GemmConfig(bm=64, bk=64, bn=128),
    ],
    key=["M", "K", "N"],
)
def matmul(
    cfg: GemmConfig,
    a: jax.Array,
    b: jax.Array,
    *,
    M: int,
    K: int,
    N: int,
) -> jax.Array:
    """C = A @ B, tiled with block sizes from cfg.

    Args:
        a: (M, K) matrix.
        b: (K, N) matrix.
        M, K, N: problem dimensions — must match a.shape and b.shape;
                 used as the cache key.
    """
    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // cfg.bm, N // cfg.bn),
        in_specs=[
            pl.BlockSpec((cfg.bm, cfg.bk), lambda i, j: (i, 0)),
            pl.BlockSpec((cfg.bk, cfg.bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((cfg.bm, cfg.bn), lambda i, j: (i, j)),
    )(a, b)


if __name__ == "__main__":
    M, K, N = 4096, 4096, 4096
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.float16)
    b = jax.random.normal(key, (K, N), dtype=jnp.float16)

    print(f"Autotuning matmul ({M}×{K}) @ ({K}×{N}) on {jax.devices()[0]}…")
    c = matmul(a, b, M=M, K=K, N=N)
    jax.block_until_ready(c)
    print(f"Result shape : {c.shape}, dtype: {c.dtype}")

    # Second call — cache hit, no sweep
    c2 = matmul(a, b, M=M, K=K, N=N)
    jax.block_until_ready(c2)
    print("Cache hit on second call ✓")

    # Sanity check against jnp.matmul
    ref = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)).astype(jnp.float16)
    err = float(jnp.max(jnp.abs(c - ref)))
    print(f"Max error vs jnp.matmul: {err:.4f}")
