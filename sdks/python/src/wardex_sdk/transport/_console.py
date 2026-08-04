from __future__ import annotations

import sys
from typing import TextIO

from .._types import InternalEnvelope
from ._base import DEFAULT_TIMEOUT, Transport


class ConsoleTransport(Transport):
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> None:
        # No bounded I/O here: a stream write has nothing to time out against.
        # The parameter exists only to keep signature parity with the base.
        self._stream.write(repr(envelope) + "\n")

    def flush(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._stream.flush()
