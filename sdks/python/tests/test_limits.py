"""Native limits surface: defaults, construction, and parser wiring."""

from __future__ import annotations

from wardex_sdk import CaptureLimits, WardexConfig, _wardex_native


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


def test_mirror_field_set_matches_core():
    core = _wardex_native.limits_defaults()
    assert set(CaptureLimits.__dataclass_fields__) == set(core)


def test_mirror_holds_no_values():
    # None means "use the core default". A mirror that carries its own values
    # is exactly the drift this design exists to prevent.
    lim = CaptureLimits()
    assert all(getattr(lim, f) is None for f in CaptureLimits.__dataclass_fields__)


def test_to_native_applies_overrides_only():
    native = CaptureLimits(max_body_bytes=1024).to_native()
    assert native.max_body_bytes == 1024
    assert native.max_headers == 96


def test_config_rejects_non_positive_limit():
    import pytest

    with pytest.raises(ValueError, match="max_body_bytes"):
        WardexConfig(limits=CaptureLimits(max_body_bytes=0))


def test_moved_fields_raise_helpful_error():
    import pytest

    with pytest.raises(TypeError, match="limits=CaptureLimits"):
        WardexConfig(max_buffer_spans=100)


def test_body_cap_reaches_the_parser_end_to_end():
    """A limit set on the config must reach the native parser via _Http1Tracker,
    not just sit in config.

    Uses max_headers rather than max_body_bytes/max_opaque_body_bytes: those two
    fields are declared on Limits but are not yet enforced anywhere on the HTTP/1
    path (crates/wardex-protocol/src/http1.rs only wires max_headers today — see
    Task 2's brief; max_body_bytes truncation exists only in http2.rs, and its own
    comment documents the opaque-body-aware cap as a "later change", not yet
    landed). A truncation-based assertion on HTTP/1 would pass or fail identically
    whether or not the config value actually reached the tracker, which is exactly
    the kind of non-discriminating test the task's self-review explicitly warns
    against. max_headers, by contrast, is genuinely enforced by the native
    parser (crates/wardex-protocol/src/http1.rs::try_parse_one), so this test
    still proves the value travels from CaptureLimits through _Http1Tracker into
    the native Http1Parser.
    """
    import wardex_sdk
    from wardex_sdk import _hub

    wardex_sdk.init(limits=CaptureLimits(max_headers=1))
    try:
        from wardex_sdk.interceptors._trackers import _Http1Tracker

        client = _hub.get_client()
        assert client is not None
        t = _Http1Tracker(client.config.limits.to_native())
        t.on_request_bytes(b"GET / HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
        # Two headers exceeds the configured cap of one — with the default
        # max_headers=96 this response would parse into one transaction, so
        # getting zero here proves the cap reached the native parser.
        raw = b"HTTP/1.1 200 OK\r\nA: 1\r\nB: 2\r\nContent-Length: 0\r\n\r\n"
        txns = t.on_response_bytes(raw)
        assert txns == [], "max_headers=1 must block parsing of a 2-header response"
    finally:
        wardex_sdk.close()
