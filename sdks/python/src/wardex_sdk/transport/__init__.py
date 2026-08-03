"""Span export transports — the last hop out of the SDK."""

from ._base import UNDELIVERED, CallerBudget
from ._otlp_http import OtlpHttpTransport

# `UNDELIVERED` and `CallerBudget` are here and not in the package root on
# purpose: both are part of the `Transport` contract, so a third-party transport
# that wants to report a decline, or to tell a budget its caller named from one
# wardex derived, must be able to import them by a public name -- but both are
# inert (a sentinel and a float, nothing to call, no reach into the core), so
# neither belongs on the root surface that `wardex.init()`'s degraded-mode
# checklist walks.
__all__ = ["UNDELIVERED", "CallerBudget", "OtlpHttpTransport"]
