"""Span export transports — the last hop out of the SDK, and the ONE home for
everything a transport implementer needs.

`Transport` (subclass it, implement `export()`, reach bytes through
`encode()`), the three shipped transports, the `Envelope` the hook and
`export()` receive, the `UNDELIVERED` decline sentinel and its `Undelivered`
type for the return annotation, `CallerBudget` for the Python-only budget
diagnostic, and `DEFAULT_TIMEOUT`, the shared default every budget on this
path starts from.

`Transport` and the shipped transports are ALSO importable from the package
root — that is the beginner surface `init(transport=...)` is typed against.
This module used to exclude them here on the grounds that the contract-only
names (`UNDELIVERED`, `CallerBudget`) were the ones with no other home; that
rationale loses to one-home discoverability, so an implementer can now import
the whole SPI from this module and nothing else.
"""

from .._types import Envelope
from ._base import DEFAULT_TIMEOUT, UNDELIVERED, CallerBudget, Transport, Undelivered
from ._console import ConsoleTransport
from ._noop import NoOpTransport
from ._otlp_http import OtlpHttpTransport

__all__ = [
    "Transport",
    "NoOpTransport",
    "ConsoleTransport",
    "OtlpHttpTransport",
    "Envelope",
    "UNDELIVERED",
    "Undelivered",
    "CallerBudget",
    "DEFAULT_TIMEOUT",
]
