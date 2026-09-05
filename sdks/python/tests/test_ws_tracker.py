import json

import pytest

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


# --- the WebSocket LLM-transport question ---------------------------------
#
# The tracker only DECIDES; it reports the decision as `_Txn.ws_llm_call` /
# `_Txn.ws_llm_unconfirmed` at close, and the seam counts from those
# (test_ws_interceptor.py). No counter moves in here.

_CLOSE_1001 = _frame(True, 0x8, (1001).to_bytes(2, "big"))


def _tracker(llm_upgrade: str | None, *, deflate: bool) -> _WebSocketTracker:
    return _WebSocketTracker(
        path="/v1/responses", deflate=deflate, parent=None, start_ns=1, llm_upgrade=llm_upgrade
    )


def test_known_provider_confirms_on_first_client_message():
    t = _tracker("known_provider", deflate=True)
    assert t._llm_call is False
    # any bytes: the provider host is the corroboration, so deflate is moot
    assert t.on_request_bytes(_frame(True, 0x1, b"\x8b\x00\x01")) == []
    assert t._llm_call is True
    # decided once: a second message cannot flip or re-make the decision
    assert t.on_request_bytes(_frame(True, 0x1, b'{"op":"ping"}')) == []
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is True
    assert txn.ws_llm_unconfirmed is False
    assert Limitation.WS_LLM_SEMANTICS_UNREAD in txn.ws_markers


def test_unknown_host_confirms_from_the_responses_envelope():
    t = _tracker("unknown_host", deflate=False)
    t.on_request_bytes(_frame(True, 0x1, b'{"type": "response.create", "model": "gpt-4o-mini"}'))
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is True
    assert txn.ws_llm_unconfirmed is False
    assert Limitation.WS_LLM_SEMANTICS_UNREAD in txn.ws_markers


@pytest.mark.parametrize(
    "first",
    [
        # `sort_keys=True` puts `model` before `type`
        json.dumps({"type": "response.create", "model": "gpt-4o-mini"}, sort_keys=True).encode(),
        # a UTF-8 BOM ahead of the envelope
        b"\xef\xbb\xbf" + b'{"type":"response.create"}',
        # the type key deep in a large envelope
        b'{"input":"' + b"x" * 4000 + b'","type" : "response.create"}',
    ],
    ids=["sort_keys", "bom", "deep"],
)
def test_unknown_host_envelope_is_found_anywhere_in_the_first_message(first: bytes):
    """The corroboration is a bounded search for the `response.create` type
    over the whole first client message — the message is already capped by
    the frame parser — not a prefix match that any other key order, a BOM
    or leading noise would defeat for the life of the connection."""
    t = _tracker("unknown_host", deflate=False)
    header = bytes([0x81, 126]) + len(first).to_bytes(2, "big")
    t.on_request_bytes(header + first)
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is True
    assert txn.ws_llm_unconfirmed is False


def test_unknown_host_with_deflate_stays_unconfirmed():
    t = _tracker("unknown_host", deflate=True)
    t.on_request_bytes(_frame(True, 0x1, b'{"type": "response.create"}'))
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is False
    assert txn.ws_llm_unconfirmed is True
    assert Limitation.WS_LLM_SEMANTICS_UNREAD not in txn.ws_markers


def test_unknown_host_non_responses_message_stays_unconfirmed():
    t = _tracker("unknown_host", deflate=False)
    t.on_request_bytes(_frame(True, 0x1, b'{"op":"ping"}'))
    # decided once: a later `response.create` does not reopen the question
    t.on_request_bytes(_frame(True, 0x1, b'{"type":"response.create"}'))
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is False
    assert txn.ws_llm_unconfirmed is True
    assert Limitation.WS_LLM_SEMANTICS_UNREAD not in txn.ws_markers


#: A Text frame header claiming a 2 MiB payload — above the default
#: `max_ws_frame_bytes` — so the client-direction parser disables on its
#: first frame and never yields a message.
_OVERSIZE_FIRST_FRAME = bytes([0x81, 127]) + (2 * 1024 * 1024).to_bytes(8, "big")


def test_known_provider_confirms_when_the_first_frame_kills_the_parser():
    """The decision must not be starved by a parse failure: bytes crossed
    the provider's connection, so the call happened whether or not the
    parser could frame it."""
    t = _tracker("known_provider", deflate=False)
    assert t.on_request_bytes(_OVERSIZE_FIRST_FRAME) == []
    assert t._llm_call is True
    # single-shot: the disabled parser keeps reporting disabled, and the
    # decision is not made again
    assert t.on_request_bytes(b"\x81\x02hi") == []
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is True
    assert txn.ws_llm_unconfirmed is False
    assert Limitation.WS_LLM_SEMANTICS_UNREAD in txn.ws_markers
    assert Limitation.FRAME_PARSE_FAILED in txn.ws_markers


def test_unknown_host_is_unconfirmed_when_the_first_frame_kills_the_parser():
    """Nothing readable ever crossed, so the envelope cannot corroborate —
    but the connection is counted rather than vanishing."""
    t = _tracker("unknown_host", deflate=False)
    assert t.on_request_bytes(_OVERSIZE_FIRST_FRAME) == []
    assert t.on_request_bytes(b"\x81\x02hi") == []
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is False
    assert txn.ws_llm_unconfirmed is True
    assert Limitation.WS_LLM_SEMANTICS_UNREAD not in txn.ws_markers
    assert Limitation.FRAME_PARSE_FAILED in txn.ws_markers


def test_no_client_message_confirms_nothing():
    """A connection that never sent a client message carried no call."""
    t = _tracker("known_provider", deflate=False)
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is False
    assert txn.ws_llm_unconfirmed is False
    assert Limitation.WS_LLM_SEMANTICS_UNREAD not in txn.ws_markers


def test_unrecognised_upgrade_claims_nothing():
    t = _tracker(None, deflate=False)
    t.on_request_bytes(_frame(True, 0x1, b'{"type": "response.create"}'))
    (txn,) = t.on_response_bytes(_CLOSE_1001)
    assert txn.ws_llm_call is False
    assert txn.ws_llm_unconfirmed is False
    assert Limitation.WS_LLM_SEMANTICS_UNREAD not in txn.ws_markers
