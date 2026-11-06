"""Utility functions."""

from __future__ import annotations

import shutil

from .errors import MissingBinaryError


def require_binary(name: str) -> None:
    """Ensure an external binary is available on PATH.

    Raises:
        MissingBinaryError: If `name` is not found on PATH.

    """
    if shutil.which(name) is None:
        raise MissingBinaryError(name)
