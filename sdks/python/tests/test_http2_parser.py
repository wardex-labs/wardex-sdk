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


# A HEADERS frame (END_HEADERS, stream 1) whose one-octet HPACK block is an
# indexed field with a saturated 7-bit prefix and no continuation octets: the
# block cannot be decoded. This is what the first header block on a pooled
# keep-alive connection looks like to a parser that attached after the
# connection's HPACK dynamic table was built.
_HPACK_POISON = bytes.fromhex("000001010400000001ff")


def test_the_native_parser_reports_why_it_latched_off():
    """The h2 latch used to be the one parser latch with no reason anywhere:
    no accessor in Rust, none across the FFI, so a connection went silent and
    `init(debug=True)` had nothing to print."""
    from wardex_sdk import _wardex_native

    native = _wardex_native.protocol.Http2Parser(None)
    assert native.disabled_reason() is None
    native.feed(False, _HPACK_POISON)
    assert native.disabled_reason() == "hpack_decode_failed"


def test_the_wrapper_and_the_tracker_delegate_the_reason():
    from wardex_sdk._interceptors._trackers import _Http2Tracker

    p = Http2Parser()
    p.feed(False, _HPACK_POISON)
    assert p.disabled_reason() == "hpack_decode_failed"

    tracker = _Http2Tracker()
    assert tracker.disabled_reason() is None
    assert tracker.on_response_bytes(_HPACK_POISON) == []
    assert tracker.disabled_reason() == "hpack_decode_failed"
