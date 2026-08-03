"""Span export transports — the last hop out of the SDK."""

from ._base import UNDELIVERED
from ._otlp_http import OtlpHttpTransport

# `UNDELIVERED` is here and not in the package root on purpose: it is part of
# the `Transport` contract, so a third-party transport that wants to report a
# decline must be able to import it by a public name -- but it is inert (a
# sentinel, nothing to call, no reach into the core), so it does not belong on
# the root surface that `wardex.init()`'s degraded-mode checklist walks.
__all__ = ["UNDELIVERED", "OtlpHttpTransport"]
