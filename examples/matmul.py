"""Autotuned tiled matrix multiply — Pallas kernel example.

This is the canonical use case for tonno: finding the best (BM, BN)
output-tile shape for a tiled GEMM on the current device.

The first call to ``matmul`` sweeps all configs, compiles each in
parallel, times them sequentially, and writes the winner to
``.tonno-cache/``.  Every subsequent call with the same input shapes
hits the cache and runs immediately with the best config.

Run on CPU (interpret mode, for correctness verification):
    python examples/matmul.py

Run on GPU (real autotuning):
    XLA_FLAGS=--xla_gpu_enable_custom_fusions=false python examples/matmul.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tonno import autotune


def _matmul_kernel(a_ref, b_ref, c_ref):
    """One Pallas program: compute one (BM, BN) output tile."""
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
            {"BM": 16, "BN": 64},
            {"BM": 16, "BN": 128},
            {"BM": 32, "BN": 64},
            {"BM": 32, "BN": 128},
            {"BM": 64, "BN": 64},
            {"BM": 64, "BN": 128},
        ]
    )
    @jax.jit(static_argnames=["BM", "BN"])
    def matmul(
        a: jax.Array,
        b: jax.Array,
        BM: int = 32,
        BN: int = 64,
    ) -> jax.Array:
        """C = A @ B, tiled with the output tile shape (BM, BN).

        Args:
            a: (M, K) matrix.
            b: (K, N) matrix.
            BM, BN: output tile dimensions — swept by autotune, injected
                    automatically; pass explicitly to override.
        """
        rows, k_dim = a.shape
        _, cols = b.shape

        return pl.pallas_call(
            _matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((rows, cols), a.dtype),
            grid=(rows // BM, cols // BN),
            in_specs=[
                pl.BlockSpec((BM, k_dim), lambda i, j: (i, 0)),
                pl.BlockSpec((k_dim, BN), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            interpret=interpret,
        )(a, b)

    return matmul


if __name__ == "__main__":
    on_cpu = jax.devices()[0].platform == "cpu"
    M, K, N = (256, 256, 256) if on_cpu else (4096, 4096, 4096)

    matmul = make_matmul(interpret=on_cpu)

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.float32)
    b = jax.random.normal(key, (K, N), dtype=jnp.float32)

    print(f"Autotuning matmul ({M}×{K}) @ ({K}×{N}) on {jax.devices()[0]}…")
    c = matmul(a, b)
    jax.block_until_ready(c)
    print(f"Result shape : {c.shape}, dtype: {c.dtype}")

    c2 = matmul(a, b)
    jax.block_until_ready(c2)
    print("Cache hit on second call ✓")

    ref = a @ b
    err = float(jnp.max(jnp.abs(c - ref)))
    print(f"Max error vs a @ b : {err:.6f}")
    assert err < 1e-3, f"too large: {err}"
    print("Correctness check passed ✓")
