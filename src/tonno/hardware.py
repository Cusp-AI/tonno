# Copyright 2025 CuspAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax


def get_device_name() -> str:
    """Return a string identifying the current JAX device (e.g. 'NVIDIA H100 80GB', 'TPU v4', 'cpu')."""
    device = jax.devices()[0]
    platform = device.platform  # 'gpu', 'tpu', 'cpu'

    if platform == "cpu":
        return "cpu"

    return device.device_kind
