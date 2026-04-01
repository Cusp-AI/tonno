"""Autotuned tiled matrix multiply — Pallas kernel example.

This is the canonical use case for tonno: finding the best (bm, bn)
output-tile shape for a tiled GEMM on the current device.

Each Pallas program computes one (bm, bn) output tile by loading the
full (bm, K) strip of A and (K, bn) strip of B and calling jnp.dot.
The optimal tile shape varies by GPU architecture and problem size.

The first call to ``matmul`` sweeps all configs, compiles each in
parallel, times them sequentially, and writes the winner to
``.tonno-cache/``.  Every subsequent call with the same (M, K, N) hits
the cache and runs immediately with the best config.

Run on CPU (interpret mode, for correctness verification):
    python examples/matmul.py

Run on GPU (real autotuning):
    XLA_FLAGS=--xla_gpu_enable_custom_fusions=false python examples/matmul.py
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tonno import autotune


class GemmConfig(NamedTuple):
    bm: int  # output tile rows
    bn: int  # output tile cols


def _matmul_kernel(a_ref, b_ref, c_ref):
    """One Pallas program: compute one (bm, bn) output tile."""
    c_ref[...] = jnp.dot(
        a_ref[...], b_ref[...], preferred_element_type=jnp.float32
    ).astype(c_ref.dtype)


def make_matmul(interpret: bool = False):
    """Return an autotuned matmul function.

    Args:
        interpret: run Pallas in interpreted (CPU-compatible) mode.
                   On GPU leave this False for real hardware performance.
    """

    @autotune(
        configs=[
            GemmConfig(bm=16, bn=64),
            GemmConfig(bm=16, bn=128),
            GemmConfig(bm=32, bn=64),
            GemmConfig(bm=32, bn=128),
            GemmConfig(bm=64, bn=64),
            GemmConfig(bm=64, bn=128),
        ],
        key=["M", "K", "N"],
    )
    def matmul(
        cfg: GemmConfig,
        a: jax.Array,
        b: jax.Array,
        *,
        M: int | None = None,
        K: int | None = None,
        N: int | None = None,
    ) -> jax.Array:
        """C = A @ B, tiled with the output tile shape from cfg.

        Args:
            a: (M, K) matrix.
            b: (K, N) matrix.
            M, K, N: problem dimensions — must match a/b shapes; used as
                     the cache key so the tuned config is device-specific.
        """
        # Derive grid from concrete array shapes (always known at trace time).
        # Do NOT use M/K/N here — they are popped by autotune before fn is called.
        rows, k_dim = a.shape
        _, cols = b.shape

        return pl.pallas_call(
            _matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((rows, cols), a.dtype),
            grid=(rows // cfg.bm, cols // cfg.bn),
            in_specs=[
                pl.BlockSpec((cfg.bm, k_dim), lambda i, j: (i, 0)),
                pl.BlockSpec((k_dim, cfg.bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((cfg.bm, cfg.bn), lambda i, j: (i, j)),
            interpret=interpret,
        )(a, b)

    return matmul


if __name__ == "__main__":
    on_cpu = jax.devices()[0].platform == "cpu"
    # Use a small problem on CPU (interpret mode is not optimised).
    M, K, N = (256, 256, 256) if on_cpu else (4096, 4096, 4096)

    matmul = make_matmul(interpret=on_cpu)

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.float32)
    b = jax.random.normal(key, (K, N), dtype=jnp.float32)

    print(f"Autotuning matmul ({M}×{K}) @ ({K}×{N}) on {jax.devices()[0]}…")
    c = matmul(a, b, M=M, K=K, N=N)
    jax.block_until_ready(c)
    print(f"Result shape : {c.shape}, dtype: {c.dtype}")

    # Cache hit — no sweep
    c2 = matmul(a, b, M=M, K=K, N=N)
    jax.block_until_ready(c2)
    print("Cache hit on second call ✓")

    # Correctness check
    ref = a @ b
    err = float(jnp.max(jnp.abs(c - ref)))
    print(f"Max error vs a @ b : {err:.6f}")
    assert err < 1e-3, f"too large: {err}"
    print("Correctness check passed ✓")
