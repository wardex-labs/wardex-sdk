from wardex_sdk._protocol._http2 import Http2Parser


def test_http2_parser_feed_returns_tuple():
    p = Http2Parser()
    opened, txns = p.feed(True, b"")
    assert opened == []
    assert txns == []


def test_truncated_hpack_size_update_latches_off_instead_of_raising():
    """The ten bytes a server can send that used to raise inside the host's
    `recv()`: a HEADERS frame whose HPACK block is one dynamic-table size
    update (`0x3f`) with its integer cut off. `fluke-hpack` unwraps on it, and
    before the parser caught that, the panic crossed PyO3 as a BaseException
    no guard in the SDK catches. Now it is the same latch-off as any other
    bad block: nothing raised, nothing captured, and the connection stays off.
    """
    poison = bytes.fromhex("0000010104000000013f")
    # A well-formed request on stream 3 (`:method GET`, `:path /` from the
    # static table), END_HEADERS | END_STREAM. On a fresh parser it opens.
    request = bytes.fromhex("0000020105000000038284")
    assert Http2Parser().feed(True, request)[0] == [3]

    p = Http2Parser()
    assert p.feed(False, poison) == ([], [])
    assert p.feed(True, request) == ([], []), "the parser must stay latched off"
