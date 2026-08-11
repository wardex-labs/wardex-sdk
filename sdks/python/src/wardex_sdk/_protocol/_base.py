"""Protocol parser interface — spec §5.2."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .._types import ParsedMessage


class ProtocolParserInterface(ABC):
    """Incrementally parses a byte stream into structured protocol messages."""

    @abstractmethod
    def protocol_name(self) -> str: ...

    @abstractmethod
    def feed(self, data: bytes) -> list[ParsedMessage]:
        """Accumulates bytes and returns zero or more completed messages."""
        ...

    @abstractmethod
    def flush(self) -> ParsedMessage | None:
        """On connection close, detaches any incomplete message as truncated
        (None if there is none)."""
        ...
