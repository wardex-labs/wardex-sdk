from __future__ import annotations

import sys
from typing import TextIO

from .._types import Envelope
from ._base import DEFAULT_TIMEOUT, Transport


class ConsoleTransport(Transport):
    """Print each envelope to a stream. A LOCAL DEBUGGING TOOL, nothing more.

    IT PRINTS THE ENVELOPE RAW — PRE-MASKING. `export()` receives data before
    any PII masking runs (masking lives in the encoder, behind
    `Transport.encode()`, and this transport never encodes), so whatever the
    host captured — emails, keys, payloads — reaches the stream verbatim. Use
    it to see what wardex captures on a developer machine; never point it at
    anything that leaves the process.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> None:
        # No bounded I/O here: a stream write has nothing to time out against.
        # The parameter exists only to keep signature parity with the base.
        self._stream.write(repr(envelope) + "\n")

    def flush(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._stream.flush()
