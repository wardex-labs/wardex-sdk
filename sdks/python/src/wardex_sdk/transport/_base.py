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
    def export(self, envelope: InternalEnvelope) -> None: ...

    def flush(self, timeout: float = 5.0) -> None:
        return None

    def close(self, timeout: float = 5.0) -> None:
        return None
