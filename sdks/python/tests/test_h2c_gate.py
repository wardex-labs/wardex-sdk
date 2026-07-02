"""h2c gate unit tests — preface detection → _Http2Tracker SWAP + protocol classification."""

from __future__ import annotations

from wardex_sdk.interceptors._seam import _ConnectionState
from wardex_sdk.interceptors._socket import _H2_PREFACE, RawSocketInterceptor
from wardex_sdk.interceptors._trackers import _Http1Tracker, _Http2Tracker


def _st() -> _ConnectionState:
    return _ConnectionState(_Http1Tracker(), "127.0.0.1", 8080)


def test_preface_latches_h2c_and_swaps_tracker():
    itc = RawSocketInterceptor()
    st = _st()
    # An h2 client sends the preface + SETTINGS frame together in the first transmission.
    data = _H2_PREFACE + b"\x00\x00\x00\x04\x00\x00\x00\x00\x00"
    assert itc._gate(st, data, "request") is True
    assert st.gate == "h2c"
    assert isinstance(st.tracker, _Http2Tracker)


def test_http_method_latches_http_no_swap():
    itc = RawSocketInterceptor()
    st = _st()
    assert itc._gate(st, b"POST /v1/chat HTTP/1.1\r\n", "request") is True
    assert st.gate == "http"
    assert isinstance(st.tracker, _Http1Tracker)


def test_tls_record_ignored():
    itc = RawSocketInterceptor()
    st = _st()
    # TLS ClientHello: 0x16(handshake) 0x03 0x01 ...
    assert itc._gate(st, b"\x16\x03\x01\x00\xff", "request") is False
    assert st.gate == "ignore"


def test_redis_bytes_ignored():
    itc = RawSocketInterceptor()
    st = _st()
    assert itc._gate(st, b"*1\r\n$4\r\nPING\r\n", "request") is False
    assert st.gate == "ignore"


def test_partial_preface_ignored_documented():
    # Documented false-negative: a partial preface under 24 bytes is not detected.
    # This doesn't happen in practice because real h2 libraries send the preface atomically.
    itc = RawSocketInterceptor()
    st = _st()
    assert itc._gate(st, b"PRI * HTTP/2.0", "request") is False
    assert st.gate == "ignore"


def test_latch_is_sticky_after_h2c():
    # No re-evaluation after the first decision: once latched to h2c, arbitrary
    # frame bytes still pass through.
    itc = RawSocketInterceptor()
    st = _st()
    itc._gate(st, _H2_PREFACE, "request")
    assert st.gate == "h2c"
    assert itc._gate(st, b"\x00\x00\x10\x01\x04", "request") is True
    assert st.gate == "h2c"


def test_response_first_ignored():
    # Response before request (server-side socket nature) → cannot decide → ignore.
    itc = RawSocketInterceptor()
    st = _st()
    assert itc._gate(st, _H2_PREFACE, "response") is False
    assert st.gate == "ignore"
