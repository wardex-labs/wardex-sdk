from wardex_sdk._assembly import Limitation
from wardex_sdk._interceptors._trackers import _WebSocketTracker


def _frame(fin: bool, opcode: int, payload: bytes) -> bytes:
    out = bytearray()
    out.append((0x80 if fin else 0) | opcode)
    out.append(len(payload))  # assumes < 126
    out += payload
    return bytes(out)


def test_close_emits_span_with_counts_and_sample():
    t = _WebSocketTracker(path="/realtime", deflate=False, parent=None, start_ns=1)
    # client sends 2 text messages
    assert t.on_request_bytes(_frame(True, 0x1, b'{"a":1}')) == []
    assert t.on_request_bytes(_frame(True, 0x1, b"hello")) == []
    # server sends 1 text message
    assert t.on_response_bytes(_frame(True, 0x1, b"world")) == []
    # server close(1000)
    close_payload = (1000).to_bytes(2, "big")
    out = t.on_response_bytes(_frame(True, 0x8, close_payload))
    assert len(out) == 1
    txn = out[0]
    assert txn.version == "websocket"
    assert txn.path == "/realtime"
    assert txn.ws_close_code == 1000
    assert txn.ws_messages_sent == 2
    assert txn.ws_messages_received == 1
    assert b"hello" in txn.request_body
    assert txn.response_body == b"world"
    assert Limitation.WS_NO_CLOSE not in txn.ws_markers


def test_flush_emits_with_no_close_marker():
    t = _WebSocketTracker(path="/x", deflate=True, parent=None, start_ns=1)
    t.on_request_bytes(_frame(True, 0x1, b"hi"))
    # `ws_markers` carries Limitation MEMBERS, not free strings:
    # the tracker is where `ws_compressed` and `ws_parse_failed` were produced,
    # and both are pre-rename spellings that SpanDraft.finish() would reject.
    out = t.flush(Limitation.WS_NO_CLOSE)
    assert len(out) == 1
    assert Limitation.WS_NO_CLOSE in out[0].ws_markers
    assert Limitation.PAYLOAD_COMPRESSED in out[0].ws_markers  # deflate=True
    # the second flush returns an empty list (no duplicate emission)
    assert t.flush(Limitation.WS_NO_CLOSE) == []
