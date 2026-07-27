from wardex_sdk._enums import Protocol
from wardex_sdk.protocol._http1 import Http1RequestParser, Http1ResponseParser


def test_response_parser_maps_to_parsed_message():
    p = Http1ResponseParser()
    msgs = p.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi")
    assert len(msgs) == 1
    m = msgs[0]
    assert m.protocol == Protocol.HTTP
    assert m.status_code == 200
    assert m.body == b"hi"
    assert ("Content-Length", "2") in m.headers


def test_request_parser_maps_method_and_path():
    p = Http1RequestParser()
    msgs = p.feed(b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    assert len(msgs) == 1
    assert msgs[0].method == "POST"
    assert msgs[0].url == "/v1/messages"


def test_flush_returns_truncated_response():
    p = Http1ResponseParser()
    assert p.feed(b"HTTP/1.1 200 OK\r\nServer: x\r\n\r\npart") == []
    m = p.flush()
    assert m is not None
    assert m.body == b"part"


def test_response_parser_propagates_header_len():
    p = Http1ResponseParser()
    msgs = p.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi")
    assert len(msgs) == 1
    assert msgs[0].header_len == len(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n")


def test_wrapper_delegates_disabled_reason_to_the_native_parser():
    # The thin Python wrapper (_Http1Parser) must forward disabled_reason()
    # to the native parser rather than silently dropping it — no span exists
    # to carry it, so this is the only place a caller can observe the latch.
    p = Http1ResponseParser()
    assert p.disabled_reason() is None
    assert p.feed(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n") == []
    assert p.disabled_reason() == "not_http"
