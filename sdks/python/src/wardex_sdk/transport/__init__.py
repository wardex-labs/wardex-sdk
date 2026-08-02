"""Span export transports — the last hop out of the SDK."""

from ._otlp_http import OtlpHttpTransport

__all__ = ["OtlpHttpTransport"]
