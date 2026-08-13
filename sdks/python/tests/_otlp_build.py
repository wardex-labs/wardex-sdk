"""Hand-rolled OTLP ExportTraceServiceRequest bytes for bridge tests.

A deterministic protobuf writer over the handful of field numbers the
opentelemetry-proto trace schema pins (vendored under
crates/wardex-codec/proto). Deliberately NOT wardex's own encoder — reusing it
would couple these fixtures to wardex semantics and make the round trip
self-certifying — and deliberately not a test dependency on
opentelemetry-proto, which the suite does not otherwise carry.

Only what the bridge consumes is spelled: string/int/double/bool attributes,
ids, times, status, resource attributes and scope name. Everything is built
from the CLI's observed span shapes (claude_code.*).
"""

from __future__ import annotations

import struct


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        bits = n & 0x7F
        n >>= 7
        if n:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _len_delimited(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _string(field: int, value: str) -> bytes:
    return _len_delimited(field, value.encode())


def _bytes_field(field: int, value: bytes) -> bytes:
    return _len_delimited(field, value)


def _varint_field(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _fixed64(field: int, value: int) -> bytes:
    return _tag(field, 1) + struct.pack("<Q", value)


def _any_value(value) -> bytes:
    """opentelemetry.proto.common.v1.AnyValue."""
    if isinstance(value, bool):
        return _varint_field(2, int(value))
    if isinstance(value, int):
        return _varint_field(3, value)
    if isinstance(value, float):
        return _tag(4, 1) + struct.pack("<d", value)
    return _string(1, str(value))


def _key_value(key: str, value) -> bytes:
    return _string(1, key) + _len_delimited(2, _any_value(value))


def _attributes(field: int, attrs: dict) -> bytes:
    return b"".join(_len_delimited(field, _key_value(k, v)) for k, v in attrs.items())


def span(
    *,
    name: str,
    trace_id: str,
    span_id: str,
    parent_span_id: str = "",
    start_ns: int = 0,
    end_ns: int = 0,
    attrs: dict | None = None,
    status_code: int = 0,
    status_message: str = "",
) -> bytes:
    """One opentelemetry.proto.trace.v1.Span. Ids are hex strings."""
    out = bytearray()
    out += _bytes_field(1, bytes.fromhex(trace_id))
    out += _bytes_field(2, bytes.fromhex(span_id))
    if parent_span_id:
        out += _bytes_field(4, bytes.fromhex(parent_span_id))
    out += _string(5, name)
    out += _varint_field(6, 1)  # SPAN_KIND_INTERNAL
    out += _fixed64(7, start_ns)
    out += _fixed64(8, end_ns)
    out += _attributes(9, attrs or {})
    if status_code or status_message:
        status = b""
        if status_message:
            status += _string(2, status_message)
        status += _varint_field(3, status_code)
        out += _len_delimited(15, status)
    return bytes(out)


def request(
    spans: list[bytes],
    *,
    resource_attrs: dict | None = None,
    scope_name: str = "claude_code.tracing",
) -> bytes:
    """ExportTraceServiceRequest holding all `spans` under one resource/scope."""
    scope = _string(1, scope_name)
    scope_spans = _len_delimited(1, scope) + b"".join(_len_delimited(2, s) for s in spans)
    resource = _attributes(1, resource_attrs or {})
    resource_spans = _len_delimited(1, resource) + _len_delimited(2, scope_spans)
    return _len_delimited(1, resource_spans)
