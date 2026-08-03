from __future__ import annotations

import abc

from .._types import InternalEnvelope


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
    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> None:
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
        """

    def flush(self, timeout: float = 5.0) -> None:
        return None

    def close(self, timeout: float = 5.0) -> None:
        return None
