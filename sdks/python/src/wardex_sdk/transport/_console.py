from __future__ import annotations

import sys
from typing import TextIO

from .._types import InternalEnvelope
from ._base import Transport


class ConsoleTransport(Transport):
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def export(self, envelope: InternalEnvelope) -> None:
        self._stream.write(repr(envelope) + "\n")

    def flush(self, timeout: float = 5.0) -> None:
        self._stream.flush()
