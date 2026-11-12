"""An HTTP/2 model call whose request the stream bound evicted in the same write that sent it.

With `max_streams=1`, a client that writes two whole requests in one `sendall` hands the parser
both streams in one `feed`. The stream table evicts stream 1 the moment stream 3 opens, inside that
call, so a question about stream 1's path asked after the call returns could no longer be answered:
the model call's answer shipped as `HTTP ? /` with no model-call path, the seal witnessed nothing,
and the bridge chat unit above it promoted the framework's usage as if no wire had seen the call.

The parser now reports each opened stream WITH the path it read as the request ended, so the
tracker keeps the path even when the stream is gone before the call returns. Driven over a real h2c
loopback with a raw `h2` client, so the two requests really arrive in one write.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from collections.abc import Iterator
from typing import Any

import h2.config
import h2.connection
import h2.events
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._adapters._context import Placement
from wardex_sdk._adapters._registry import context_for
from wardex_sdk._assembly import GenAIAttributes, Limitation, SpanIntent, UnitKind, counters
from wardex_sdk._enums import OperationName
from wardex_sdk._limits import LimitsConfig
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_PREFIX = "wardex.h2probe.otel."
_USAGE = (11, 4)
_CHAT_BODY = json.dumps(
    {
        "id": "c1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": _USAGE[0], "completion_tokens": _USAGE[1], "total_tokens": 15},
    }
).encode()
_HEALTH_BODY = b'{"ok":true}'


@contextlib.contextmanager
def _h2c_server() -> Iterator[tuple[str, int]]:
    """A loopback h2c server answering every request whole: a chat completion with usage on a
    model-call path, a small JSON body anywhere else."""

    def handle(sock: socket.socket) -> None:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        conn.initiate_connection()
        sock.sendall(conn.data_to_send())
        paths: dict[int, str] = {}
        while True:
            try:
                data = sock.recv(65535)
            except OSError:
                return
            if not data:
                return
            for ev in conn.receive_data(data):
                if isinstance(ev, h2.events.RequestReceived):
                    paths[ev.stream_id] = dict(ev.headers)[b":path"].decode()
                elif isinstance(ev, h2.events.StreamEnded):
                    chat = paths.get(ev.stream_id) == "/v1/chat/completions"
                    conn.send_headers(
                        ev.stream_id, [(":status", "200"), ("content-type", "application/json")]
                    )
                    conn.send_data(
                        ev.stream_id, _CHAT_BODY if chat else _HEALTH_BODY, end_stream=True
                    )
            with contextlib.suppress(OSError):
                sock.sendall(conn.data_to_send())

    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                raw, _ = listener.accept()
            except (TimeoutError, OSError):
                continue
            threading.Thread(target=handle, args=(raw,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    try:
        yield listener.getsockname()[:2]
    finally:
        stop.set()
        listener.close()


def _two_requests_in_one_write(host: str, port: int) -> None:
    """Stream 1 is the model call, stream 3 a plain GET; both whole, both in ONE `sendall`, so the
    parser sees them in one `feed` and the bound of one stream evicts stream 1 inside it."""
    s = socket.create_connection((host, port))
    s.settimeout(2.0)
    try:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        conn.initiate_connection()
        s.sendall(conn.data_to_send())  # the preface on its own: the gate detects h2 from it
        authority = f"{host}:{port}"
        for sid, method, path in ((1, "POST", "/v1/chat/completions"), (3, "GET", "/health")):
            headers = [(":method", method), (":authority", authority), (":scheme", "http")]
            conn.send_headers(sid, [*headers, (":path", path)], end_stream=True)
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


def _limitations(span: Any) -> tuple[Limitation, ...]:
    return tuple(span.capture_integrity.limitations) if span.capture_integrity else ()


def _describe_chat(handle: Any) -> None:
    handle.draft.set_gen_ai(GenAIAttributes(operation=OperationName.CHAT, request_model="m"))


def test_a_model_call_evicted_in_the_write_that_sent_it_keeps_its_path_and_is_counted_once():
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
            _two_requests_in_one_write(host, port)
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
