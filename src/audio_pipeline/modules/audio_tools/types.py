"""Shared types for the audio_tools module.

Must never import optional dependencies.
"""

from __future__ import annotations

import typing as t

AudioFormat = t.Literal["wav", "flac", "ogg", "opus", "mp3"]
