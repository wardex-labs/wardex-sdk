from __future__ import annotations

import abc

from .._types import InternalEnvelope


class _Undelivered:
    """The type of `UNDELIVERED`. A singleton, so `is` is the whole test."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNDELIVERED"


UNDELIVERED = _Undelivered()
"""What `Transport.export` returns to say "I did not send this one".

The client cannot see inside a transport, and every attempt to work out from the
outside whether a send would happen has been wrong: the check runs at a moment
that is not the moment of the send, and the work in between (a `before_send`
that outlives the budget, most of all) invalidates it. So the transport says so
itself, at the one moment it knows.

Precise meaning, because the client acts on it: *this envelope was not put on
the wire, nothing about it was consumed, and an identical attempt later with a
fresh budget could succeed.* That is what makes it safe for a non-final
`flush()` to keep the spans and re-send them -- a decline that had already
half-sent something would be a duplicate, and one that can never succeed would
be an infinite retry. A transport that fails mid-POST, or that cannot encode at
all, must NOT return this; it has its own reporting and its own reasons.

Anything else -- above all `None`, which is what every transport written before
this existed returns -- means the transport took the envelope. That default is
the safe direction: delivering unconditionally is a legal transport, and a
client guessing "not delivered" from the outside would announce losses that
never happened and burn the one-line-per-process report budget doing it.
"""


class Transport(abc.ABC):
    # PII policy applied by the native encoders on wire paths (design §4.2).
    # Class-level defaults are secure-by-default: a transport used without
    # init() still masks.
    _pii_mode: str = "mask"
    _pii_disabled: tuple[str, ...] = ()

    def set_pii_policy(self, mode: str, disabled: tuple[str, ...]) -> None:
        """Install the PII policy resolved from WardexConfig (called by init)."""
        self._pii_mode = mode
        self._pii_disabled = disabled

    @abc.abstractmethod
    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> object | None:
        """Ship one envelope. `timeout` is the caller's remaining budget, in
        seconds, or None when the caller imposed no deadline.

        Keyword-only and defaulted on purpose: `Transport` is public, and
        subclasses written before this parameter existed still declare
        `export(self, envelope)`. The client probes the signature and withholds
        the keyword from those, so an old subclass keeps working -- but it also
        keeps stalling, because only the transport can bound its own I/O. A
        transport that performs blocking I/O should honour this by *narrowing*
        its configured timeout, never widening it: a caller asking for 99s must
        not get more than the transport was configured for.

        Return `UNDELIVERED` when the envelope was NOT put on the wire and an
        identical attempt later could still succeed -- a spent `timeout` is the
        case wardex's own transport uses it for. Returning anything else, `None`
        included, means "taken": the client then neither retries nor reports it.
        A transport that declines silently is indistinguishable from one that
        delivered, and is treated as the latter; see `UNDELIVERED` for why that
        is the safe default and not a shrug.
        """

    def flush(self, timeout: float = 5.0) -> None:
        return None

    def close(self, timeout: float = 5.0) -> None:
        return None
