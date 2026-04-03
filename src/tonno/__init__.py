# Copyright 2025 CuspAI
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version as _version

from tonno.tune import autotune

__version__ = _version("tonno")
__all__ = ["autotune", "__version__"]
