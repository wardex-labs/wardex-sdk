"""An HTTP/2 model call whose header block the stream bound evicted before its request ended.

Two tasks writing to one connection interleave their frames: HEADERS 1, HEADERS 3, DATA 1
(END_STREAM), DATA 3 (END_STREAM). With `max_streams=1`, stream 3's header block evicts stream 1's
before stream 1's request ends, so when its END_STREAM arrives the parser no longer holds the block
that named its path. It used to announce the request with no path: the model call's answer shipped
as `HTTP ? /` with no usage, the seal witnessed nothing, and the bridge chat unit above it kept the
framework's usage demoted with no wire span carrying the call.

The parser now keeps the path of a header block it evicts before the request ends, bounded by the
same `max_streams`, and reports it when the request does end. Driven over a real h2c loopback with
a raw `h2` client, so the four frames really arrive in that order in one write.
"""

from __future__ import annotations

import json
import socket
import threading

import h2.config
import h2.connection
import h2.events
import pytest

import wardex_sdk as wardex
from test_h2c_evicted_in_one_feed import (
    _PREFIX,
    _USAGE,
    _describe_chat,
    _h2c_server,
    _limitations,
)
from wardex_sdk import _hub
from wardex_sdk._adapters._context import Placement
from wardex_sdk._adapters._registry import context_for
from wardex_sdk._assembly import Limitation, SpanIntent, UnitKind, counters
from wardex_sdk._limits import LimitsConfig
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")


def _two_requests_interleaved(host: str, port: int) -> None:
    """Stream 1 is the model call, stream 3 a plain POST; their header blocks go first and their
    bodies after, all in ONE `sendall`, so the bound of one stream evicts stream 1's header block
    before its body ends."""
    s = socket.create_connection((host, port))
    s.settimeout(2.0)
    try:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        conn.initiate_connection()
        s.sendall(conn.data_to_send())  # the preface on its own: the gate detects h2 from it
        authority = f"{host}:{port}"
        streams = ((1, "/v1/chat/completions"), (3, "/health"))
        for sid, path in streams:
            headers = [(":method", "POST"), (":authority", authority), (":scheme", "http")]
            conn.send_headers(sid, [*headers, (":path", path)])
        body = json.dumps({"model": "m", "messages": []}).encode()
        for sid, _path in streams:
            conn.send_data(sid, body, end_stream=True)
        s.sendall(conn.data_to_send())
        ended: set[int] = set()
        while ended != {1, 3}:
            data = s.recv(65535)
            if not data:
                break
            for ev in conn.receive_data(data):
                if isinstance(ev, h2.events.StreamEnded):
                    ended.add(ev.stream_id)
            if out := conn.data_to_send():
                s.sendall(out)
        assert ended == {1, 3}, "precondition: both streams were answered"
    finally:
        s.close()


def test_a_model_call_whose_header_block_was_evicted_before_its_body_ended_keeps_its_path():
    transport = RecordingTransport()
    wardex.init(
        transport=transport,
        intercept=True,
        intercept_hosts=["127.0.0.1"],
        limits=LimitsConfig(max_streams=1),
    )
    try:
        ctx = context_for("h2probe", _hub.get_client())
        chat = ctx.open_run(
            UnitKind.CALL,
            intent=SpanIntent.CHAT,
            placement=Placement.ROOT,
            subject="m",
            describe=_describe_chat,
        )
        chat.draft.set_extra(_PREFIX + "gen_ai.usage.input_tokens", _USAGE[0])
        chat.draft.set_extra(_PREFIX + "gen_ai.usage.output_tokens", _USAGE[1])
        assert ctx.hold_framework_usage(chat, prefix=_PREFIX) is True
        assert chat.pin(driver=threading.current_thread()) is True
        with _h2c_server() as (host, port):
            _two_requests_interleaved(host, port)
        chat.unpin()
        chat.close()
        _hub.get_client()._settle()
        wardex.flush()
        spans = [span for envelope in transport.envelopes for span in envelope.spans]
    finally:
        wardex.close()

    (unit,) = [s for s in spans if s.name == "chat m"]
    (wire,) = [s for s in spans if s.name.endswith("/v1/chat/completions")]
    assert wire.name == "HTTP ? /v1/chat/completions"
    assert Limitation.H2_REQUEST_EVICTED in _limitations(wire)
    carriers = [s for s in spans if s.gen_ai is not None and s.gen_ai.input_tokens]
    assert [(s.gen_ai.input_tokens, s.gen_ai.output_tokens) for s in carriers] == [_USAGE]
    assert carriers[0] is wire, "the wire span carries the call; the unit keeps its number demoted"
    assert Limitation.NO_WIRE_EVIDENCE not in _limitations(unit)
    assert counters.get("protocol.http2.evicted_witnessed") == 1
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    assert counters.get("assembly._usage_hold.usage_promoted") == 0
