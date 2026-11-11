"""A streamed model call over HTTP/2 whose END_STREAM the client never read.

The OpenAI SDK stops reading a stream at its `[DONE]` event and, on HTTP/2, closes the stream
without draining it: the connection stays usable, so draining would only add latency. A server or
proxy that sends END_STREAM in a separate, later, empty DATA frame therefore leaves the stream
open on the client side until the connection closes. The response's status and every event up to
`[DONE]` were read, usage included, but the parser only completed a transaction on END_STREAM,
so the call produced no wire span at all and the usage it carried was lost from the wire record.

At connection close such a stream now ships as what arrived, marked as a body whose end was not
seen (`FRAME_PARSE_FAILED`), the same shape HTTP/1 gives a response cut by the close. The server
here holds END_STREAM back until the client has stopped reading: an h2c loopback on the `h2`
package, the OpenAI SDK on `httpx.AsyncClient(http1=False, http2=True)`.

Under a bridge chat unit the call is still counted once; the last test says how.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import Iterator
from typing import Any

import h2.config
import h2.connection
import h2.events
import httpx
import pytest

import wardex_sdk as wardex
from _pydantic_ai_support import (
    MODES,
    STREAM_USAGE,
    extras,
    instrumentation_mode,
    one,
    spans_by_source,
)
from _pydantic_ai_support import _sse as stream_body
from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, OperationName, StatusCode
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_DEMOTED = "wardex.pydantic_ai.otel.gen_ai.usage."
_LATE = "wardex.usage_late."


@contextlib.contextmanager
def _h2c_stream_never_ended() -> Iterator[str]:
    """A loopback Chat Completions API speaking only h2c. Every request is answered with a
    streamed body ending at `[DONE]` and NO END_STREAM; the empty DATA frame that would end the
    stream is sent only once the client has gone quiet, so the client never reads it."""

    def handle(sock: socket.socket) -> None:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        conn.initiate_connection()
        sock.sendall(conn.data_to_send())
        sock.settimeout(0.3)
        held: list[int] = []
        while True:
            try:
                data = sock.recv(65535)
            except TimeoutError:
                # The client stopped reading after `[DONE]`: send the end it will never read.
                for sid in held:
                    conn.end_stream(sid)
                held.clear()
                with contextlib.suppress(OSError):
                    sock.sendall(conn.data_to_send())
                continue
            except OSError:
                return
            if not data:
                return
            for ev in conn.receive_data(data):
                if isinstance(ev, h2.events.DataReceived):
                    conn.acknowledge_received_data(ev.flow_controlled_length, ev.stream_id)
                elif isinstance(ev, h2.events.StreamEnded):
                    conn.send_headers(
                        ev.stream_id, [(":status", "200"), ("content-type", "text/event-stream")]
                    )
                    conn.send_data(ev.stream_id, stream_body(), end_stream=False)
                    held.append(ev.stream_id)
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
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
    finally:
        stop.set()
        listener.close()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(http1=False, http2=True)


def _shipped(transport: RecordingTransport) -> list[Any]:
    _hub.get_client()._settle()
    wardex.flush()
    return [span for envelope in transport.envelopes for span in envelope.spans]


def _wire_chats(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.gen_ai is not None and s.gen_ai.operation is OperationName.CHAT]


def _assert_cut_wire_chat(wire: Any, *, usage: tuple[int, int] | None = STREAM_USAGE) -> None:
    got = (wire.gen_ai.input_tokens, wire.gen_ai.output_tokens)
    assert got == (usage or (None, None))
    assert Limitation.FRAME_PARSE_FAILED in wire.capture_integrity.limitations
    assert ("network.protocol.version", "2") in wire.extra
    assert wire.status is StatusCode.UNSET and wire.error_type is None  # how it ended: not seen


def test_a_stream_read_to_done_ships_its_usage_when_the_connection_closes():
    """The OpenAI SDK alone, under a host span: one wire CHAT with the stream's usage, marked as a
    body whose end was not seen, under the span the request was issued in."""
    from openai import AsyncOpenAI

    transport = RecordingTransport()
    wardex.init(transport=transport)
    try:
        with _h2c_stream_never_ended() as base_url:

            async def main() -> str:
                http_client = _client()
                api = AsyncOpenAI(base_url=base_url, api_key="t", http_client=http_client)
                text = ""
                with wardex.span("caller"):
                    stream = await api.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "hi"}],
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            text += chunk.choices[0].delta.content
                await http_client.aclose()
                return text

            assert asyncio.run(main()) == "hello there"
        spans = _shipped(transport)
    finally:
        wardex.close()
    (wire,) = _wire_chats(spans)
    (caller,) = [s for s in spans if s.name == "caller"]
    _assert_cut_wire_chat(wire)
    assert wire.parent_span_id == caller.context.span_id


@pytest.mark.parametrize("mode", MODES)
def test_under_the_bridge_a_stream_cut_at_close_is_counted_once(mode):
    """The same through pydantic-ai's own OpenAI provider on an h2c client, with the bridge on.

    The chat unit settles first: pydantic-ai's `chat` span ends when the stream reaches `[DONE]`,
    and no model call was sealed under it by then, so it promotes the framework's usage and says
    `no_wire_evidence`. The connection closes later, with the client, and only then is the wire
    CHAT sealed under the already-shipped unit. That seal is told the unit promoted, so the wire
    span still ships, cut and under the unit, but with its usage moved out of the typed block into
    `wardex.usage_late.*`: every sum over the trace counts the call once."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    transport = RecordingTransport()
    with instrumentation_mode(mode):
        wardex.init(
            transport=transport, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,))
        )
        try:
            with _h2c_stream_never_ended() as base_url:

                async def main() -> str:
                    http_client = _client()
                    provider = OpenAIProvider(
                        base_url=base_url, api_key="t", http_client=http_client
                    )
                    model = OpenAIChatModel("gpt-4o-mini", provider=provider)
                    async with Agent(model, name="probe").run_stream("hi") as stream:
                        text = "".join([t async for t in stream.stream_text(delta=True)])
                    await http_client.aclose()
                    return text

                assert asyncio.run(main()) == "hello there"
            spans = _shipped(transport)
        finally:
            wardex.close()
    bridge, other = spans_by_source(spans)
    chat = one(bridge, "chat gpt-4o-mini")
    (wire,) = _wire_chats(other)
    _assert_cut_wire_chat(wire, usage=None)
    assert wire.parent_span_id == chat.context.span_id
    late = extras(wire)
    assert (late[_LATE + "input_tokens"], late[_LATE + "output_tokens"]) == STREAM_USAGE
    assert late[_LATE + "demoted"] is True
    assert (chat.gen_ai.input_tokens, chat.gen_ai.output_tokens) == STREAM_USAGE
    assert _DEMOTED + "input_tokens" not in extras(chat)
    assert Limitation.NO_WIRE_EVIDENCE in chat.capture_integrity.limitations
    totals = [
        (s.gen_ai.input_tokens, s.gen_ai.output_tokens)
        for s in spans
        if s.gen_ai is not None and s.gen_ai.input_tokens
    ]
    assert totals == [STREAM_USAGE], "one call, one count: on the chat unit"
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 0
    assert counters.get("assembly._usage_hold.usage_promoted") == 1
    assert counters.get("assembly._usage_hold.late_wire_usage_demoted") == 1
