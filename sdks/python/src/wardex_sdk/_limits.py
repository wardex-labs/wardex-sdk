"""Typed mirror of the core's resource limits.

This dataclass intentionally holds no values: every field defaults to None,
meaning "use the core default". The core (crates/wardex-limits) owns both the
schema and the values, so a limit can never disagree between two declaration
sites. test_limits.py asserts both properties.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from . import _wardex_native


@dataclass(frozen=True, slots=True)
class CaptureLimits:
    """Resource limit overrides. None means the core default is used."""

    max_headers: int | None = None
    max_body_bytes: int | None = None
    max_opaque_body_bytes: int | None = None
    max_stream_buffer_bytes: int | None = None
    max_decoded_bytes: int | None = None
    max_streams: int | None = None
    max_ws_frame_bytes: int | None = None
    ws_sample_bytes: int | None = None
    max_connections: int | None = None
    max_sessions: int | None = None
    max_session_entries: int | None = None
    max_units: int | None = None
    max_entries_per_unit: int | None = None
    mcp_sniff_bytes: int | None = None
    max_buffer_spans: int | None = None
    max_buffer_bytes: int | None = None
    replay_buffer_size: int | None = None
    zstd_level: int | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and v < 1:
                raise ValueError(f"{f.name} must be >= 1, got {v}")

    def to_native(self) -> Any:
        """Build the native Limits object, applying only the overrides set here."""
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        return _wardex_native.Limits(**{k: v for k, v in kwargs.items() if v is not None})

    def resolved(self) -> dict[str, int]:
        """Effective values (overrides merged onto core defaults) for host-side use."""
        out = dict(_wardex_native.limits_defaults())
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out
