from __future__ import annotations

from .._types import InternalEnvelope
from ._base import Transport


class NoOpTransport(Transport):
    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> None:
        return None
