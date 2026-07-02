from __future__ import annotations

import abc

from .._types import InternalEnvelope


class Transport(abc.ABC):
    @abc.abstractmethod
    def export(self, envelope: InternalEnvelope) -> None: ...

    def flush(self, timeout: float = 5.0) -> None:
        return None

    def close(self, timeout: float = 5.0) -> None:
        return None
