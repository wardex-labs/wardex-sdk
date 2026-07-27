"""Native limits surface: defaults, construction, and parser wiring."""

from __future__ import annotations

from wardex_sdk import _wardex_native


def test_limits_defaults_returns_every_field():
    d = _wardex_native.limits_defaults()
    assert d["max_headers"] == 96
    assert d["max_body_bytes"] == 32 * 1024 * 1024
    assert d["max_opaque_body_bytes"] == 256 * 1024
    assert d["zstd_level"] == 3
    assert len(d) == 16


def test_limits_construction_defaults_unspecified_fields():
    lim = _wardex_native.Limits(max_body_bytes=1024)
    assert lim.max_body_bytes == 1024
    assert lim.max_headers == 96  # untouched field keeps the core default


def test_parser_accepts_limits():
    lim = _wardex_native.Limits(max_headers=1)
    p = _wardex_native.protocol.Http1Parser(False, lim)
    raw = b"HTTP/1.1 200 OK\r\nA: 1\r\nB: 2\r\nContent-Length: 0\r\n\r\n"
    assert p.feed(raw) == []


def test_parser_without_limits_uses_defaults():
    p = _wardex_native.protocol.Http1Parser(False)
    raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"
    assert len(p.feed(raw)) == 1
