"""Shared types for the vtc module.

Must never import optional dependencies (vtc_inference / segma / torch).
"""

from __future__ import annotations

import typing as t

Device = t.Literal["cpu", "cuda", "gpu", "mps"]
