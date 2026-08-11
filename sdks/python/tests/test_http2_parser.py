from wardex_sdk._protocol._http2 import Http2Parser


def test_http2_parser_feed_returns_tuple():
    p = Http2Parser()
    opened, txns = p.feed(True, b"")
    assert opened == []
    assert txns == []
