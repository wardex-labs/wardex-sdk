from __future__ import annotations

from .._types import Envelope
from ._base import Transport


class NoOpTransport(Transport):
    def export(self, envelope: Envelope, *, timeout: float | None = None) -> None:
        return None
