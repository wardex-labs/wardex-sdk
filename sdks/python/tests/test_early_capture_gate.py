"""No capture, no cost — and the two conditions that are NOT allowed to gate early.

The byte seam used to pay for every socket in the process before it knew
whether it wanted any of it: `ssl.SSLSocket.send` materialized the whole send
buffer and only then asked, a connection the sniff-latch had already
classified as "not HTTP" kept paying that on every call for its whole life,
and a seam with no client to emit into still accumulated both bodies in a
tracker it holds until the connection dies.

The fix is a split, not a switch, and this file is the half of the split that a
performance change is most likely to get wrong. Two conditions are INVARIANT
for a connection and may be answered before a byte is copied. Two are about ONE
transaction and must not be latched onto the connection at all — an agent unit
can activate after a pooled socket was opened, and a per-connection early-out
on "no agent unit is ambient right now" would silently drop everything issued
on that socket afterwards. That is data loss dressed as an optimization, so it
gets a test of its own here rather than a sentence in a docstring.
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import _FakeSSLSocket
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureMode
from wardex_sdk._suppress import suppress_capture
from wardex_sdk._types import SpanContext, SpanId, TraceId
from wardex_sdk.context._contextvar import fork_active_span
from wardex_sdk.interceptors import _seam
from wardex_sdk.interceptors._ssl import SSLInterceptor
from wardex_sdk.transport._noop import NoOpTransport


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()


# --- the traffic ---------------------------------------------------------


def _http(head: bytes, body: bytes) -> bytes:
    """`head` + a Content-Length that matches `body`, so the parser sees a whole message."""
    return head + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


_LLM_REQUEST = _http(
    b"POST /v1/chat/completions HTTP/1.1\r\nHost: api.openai.com\r\n"
    b"Content-Type: application/json\r\n",
    b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}',
)
_LLM_RESPONSE = _http(
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n",
    b'{"id":"chatcmpl-1","model":"gpt-4o","usage":{"prompt_tokens":1,"completion_tokens":2},'
    b'"choices":[{"message":{"role":"assistant","content":"hey"}}]}',
)
_PLAIN_REQUEST = _http(
    b"POST /health HTTP/1.1\r\nHost: api.example.com\r\nContent-Type: application/json\r\n",
    b'{"foo":"bar"}',
)
_PLAIN_RESPONSE = _http(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n", b'{"ok":true}')
# A real Redis client's first write: enough for the sniff-latch to classify the
# connection as not-HTTP and never revisit it.
_REDIS_WRITE = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n"


# --- the seam under test -------------------------------------------------


class _RecordingClient:
    def __init__(self, mode: CaptureMode = CaptureMode.AGENT) -> None:
        self.config = WardexConfig(api_key="k", capture_mode=mode)
        self.spans: list[Any] = []

    def capture_span(self, span: Any) -> None:
        self.spans.append(span)


def _seam_for(client: Any) -> SSLInterceptor:
    """A TLS seam wired to `client` but never `.install()`-ed — no monkeypatch.

    `_load_limits` is called the way `install()` calls it, so the semantic
    parser runs against a resolved limits record rather than the core default.
    """
    itc = SSLInterceptor()
    itc._client = client
    itc._load_limits(client)
    return itc


def _socket(host: str = "api.openai.com") -> _FakeSSLSocket:
    sock = _FakeSSLSocket(None)
    sock.server_hostname = host
    return sock


class _Payload:
    """A send buffer that counts every full materialization taken of it.

    `bytes(x)` dispatches to `__bytes__`, which is exactly the copy the seam
    used to take unconditionally in `_mk_send` (`bytes(data)[:ret]`). Counting
    it is the only way to assert on work that is otherwise invisible — the
    bytes are identical either way, and only the allocation differs.
    """

    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.copies = 0

    def __bytes__(self) -> bytes:
        self.copies += 1
        return self.raw

    def __len__(self) -> int:
        return len(self.raw)


def _send_wrapper(itc: SSLInterceptor):  # noqa: ANN202
    """`ssl.SSLSocket.send`'s wrapper, over a stub that reports a full write."""

    def real(this: Any, data: Any, *args: Any, **kwargs: Any) -> int:
        return len(data)

    return itc._mk_send("send", real)


def _recv_wrapper(itc: SSLInterceptor, payload: bytes):  # noqa: ANN202
    def real(this: Any, *args: Any, **kwargs: Any) -> bytes:
        return payload

    return itc._mk_recv(real)


@pytest.fixture
def parse_spy(monkeypatch):
    """Counts `parse_llm_semantics` calls; answers None, as a parser that found nothing does."""
    calls: list[tuple[Any, ...]] = []

    def spy(*args: Any) -> None:
        calls.append(args)
        return None

    monkeypatch.setattr(_seam, "parse_llm_semantics", spy)
    return calls


# --- (a) nothing to capture into -----------------------------------------


def test_no_client_means_no_parse_and_no_body_accumulation(parse_spy):
    """A seam with no client cannot emit, so it must not pay to find out.

    The accumulation is the half that was actually being paid, and the
    connection table is how it is asserted: `_ConnectionState` owns the
    tracker, and the tracker is what buffers request and response bodies for
    the life of the connection. No entry means no buffer.

    The parse was already skipped here before the gate existed — `_emit_span`
    returns on a `None` client above `_build_span` — so that assertion is a
    ratchet rather than a repair: it fails if a later refactor lifts the parse
    above the emit boundary, which is the shape this whole file is about.
    """
    itc = _seam_for(None)
    sock = _socket()

    itc._on_request_bytes(sock, _LLM_REQUEST)
    itc._on_response_bytes(sock, _LLM_RESPONSE)

    assert parse_spy == [], "the semantic parser ran for a seam that cannot emit a span"
    assert itc._conns == {}, "bodies were accumulated for a seam that cannot emit a span"


def test_a_latched_off_connection_never_reaches_the_parser(parse_spy):
    """Redis over TLS: classified once, and then it must cost nothing forever.

    The seam's own entry points have short-circuited on the latch since it
    shipped; the wrapper ABOVE them had not, which is the copy test below.
    This states the entry-point half so the two read as one rule rather than
    as an assertion with an unexplained companion.
    """
    client = _RecordingClient(CaptureMode.ALL)
    itc = _seam_for(client)
    sock = _socket()

    itc._on_request_bytes(sock, _REDIS_WRITE)
    assert itc._conns[id(sock)].gate == "ignore"

    itc._on_response_bytes(sock, b"$4096\r\n" + b"v" * 4096 + b"\r\n")

    assert parse_spy == []
    assert client.spans == []


# --- (b) the send-buffer copy --------------------------------------------


def test_a_suppressed_send_does_not_copy_the_buffer():
    """The exporter's own POST pays nothing, not even the copy.

    Suppression already stopped the PARSE; the copy happened above it, in the
    patch wrapper, so every outbound OTLP batch was materialized twice on its
    way to being ignored. This is the assertion that keeps the gate above the
    copy rather than merely above the tracker.
    """
    itc = _seam_for(_RecordingClient(CaptureMode.ALL))
    send = _send_wrapper(itc)
    sock = _socket()

    suppressed = _Payload(_LLM_REQUEST)
    with suppress_capture():
        send(sock, suppressed)
    assert suppressed.copies == 0
    assert itc._conns == {}

    # Control: the same wrapper, the same buffer, outside the guard. Without
    # this the assertion above would also pass on a wrapper that never copies
    # anything, i.e. on a seam that captures nothing at all.
    observed = _Payload(_LLM_REQUEST)
    send(sock, observed)
    assert observed.copies == 1


def test_a_latched_off_send_does_not_copy_the_buffer():
    """The same, for the case that lasts the life of the process.

    A TLS-backed Redis or Postgres client is not suppressed and never will be;
    it is simply not HTTP. Before the gate moved, every one of its writes was
    copied out in full, forever, so that the seam could re-derive a verdict it
    had already latched on the first one.
    """
    itc = _seam_for(_RecordingClient(CaptureMode.ALL))
    send = _send_wrapper(itc)
    sock = _socket()

    send(sock, _Payload(_REDIS_WRITE))
    assert itc._conns[id(sock)].gate == "ignore"

    later = _Payload(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
    send(sock, later)
    assert later.copies == 0


# --- (c) what must NOT be latched onto the connection --------------------


def test_an_agent_unit_activating_after_the_connection_opened_is_still_captured():
    """The early-out that would have been a silent data loss.

    Under the AGENT default a request with no LLM semantics is captured only
    when a local wardex span was ambient at the moment it was ISSUED. That is
    an answer about one transaction and about one instant — a pooled keep-alive
    connection outlives any single agent run, and a connection opened before
    the run started carries the run's requests just the same.

    So: two byte-identical exchanges on ONE socket, the first outside any span
    and the second inside one. If the early gate ever learns to ask about the
    ambient scope, the second exchange disappears and this is the test that
    says so.
    """
    client = _RecordingClient(CaptureMode.AGENT)
    itc = _seam_for(client)
    sock = _socket("api.example.com")

    itc._on_request_bytes(sock, _PLAIN_REQUEST)
    itc._on_response_bytes(sock, _PLAIN_RESPONSE)
    assert client.spans == [], "generic traffic outside any span is not AGENT-mode traffic"

    local = SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), is_remote=False)
    with fork_active_span(local):
        itc._on_request_bytes(sock, _PLAIN_REQUEST)
        itc._on_response_bytes(sock, _PLAIN_RESPONSE)

    assert len(client.spans) == 1, (
        "a unit that activated AFTER this connection was opened was not captured — "
        "a context-dependent condition has been latched onto the connection"
    )
    assert client.spans[0].parent_span_id == local.span_id


def test_a_noop_transport_is_not_treated_as_capture_being_off(parse_spy):
    """Recorded deliberately, because it is the obvious next early-out and it is wrong.

    A `NoOpTransport` does discard everything handed to it — but `init()`
    resolves a missing `transport=` argument to exactly one, so it is the
    out-of-the-box configuration rather than a statement that capture is off.
    Gating on it would turn `wardex.init(intercept=True)` into a silent no-op
    for every user who has not wired an exporter yet, and would take most of
    this suite with it.

    If a "the sink is a black hole" early-out is ever wanted, it needs to be
    something the host ASKED for, not something it got by omission.
    """
    client = Client(WardexConfig(api_key="k", capture_mode=CaptureMode.ALL), NoOpTransport())
    _hub.set_client(client)  # conftest's autouse fixture joins its worker thread
    itc = _seam_for(client)
    sock = _socket()

    itc._on_request_bytes(sock, _LLM_REQUEST)
    itc._on_response_bytes(sock, _LLM_RESPONSE)

    assert parse_spy, "a NoOpTransport silently disabled capture"


# --- (d) when capture is on, nothing changed -----------------------------


def _shape(span: Any) -> dict[str, Any]:
    """Everything about a span that two runs of the same exchange must agree on.

    Ids, wall clocks and `connection_id` are excluded because they are supposed
    to differ between two runs; every field that describes what was OBSERVED is
    in.
    """
    http = span.transport.http
    return {
        "name": span.name,
        "kind": span.kind,
        "status": span.status,
        "error_type": span.error_type,
        "server": (span.server_address, span.server_port),
        "input_data": span.input_data,
        "output_data": span.output_data,
        "http": (http.method, http.url, http.status_code),
        "sizes": (span.transport.request_size, span.transport.response_size),
        "reused": span.transport.connection_reused,
        "gen_ai": span.gen_ai,
        "extra": span.extra,
        "capture_sources": span.capture_sources,
        "integrity": span.capture_integrity,
    }


def test_capture_active_output_is_unchanged_through_the_patch_wrappers():
    """The gate is a skip, never a filter: what survives it is byte-for-byte the same.

    One exchange driven two ways — straight into the seam's entry points, and
    through the real `send`/`recv` wrappers that now ask the gate first. A gate
    that dropped a byte, truncated a body or lost a marker on the way would
    show up as a difference here rather than as a quieter span nobody compares.
    """
    direct_client = _RecordingClient(CaptureMode.AGENT)
    direct = _seam_for(direct_client)
    direct_sock = _socket()
    direct._on_request_bytes(direct_sock, _LLM_REQUEST)
    direct._on_response_bytes(direct_sock, _LLM_RESPONSE)

    wrapped_client = _RecordingClient(CaptureMode.AGENT)
    wrapped = _seam_for(wrapped_client)
    wrapped_sock = _socket()
    _send_wrapper(wrapped)(wrapped_sock, _Payload(_LLM_REQUEST))
    _recv_wrapper(wrapped, _LLM_RESPONSE)(wrapped_sock)

    assert len(direct_client.spans) == 1
    assert len(wrapped_client.spans) == 1
    assert _shape(wrapped_client.spans[0]) == _shape(direct_client.spans[0])
    assert direct_client.spans[0].gen_ai is not None, (
        "precondition: this exchange must be identified as an LLM call, or the "
        "comparison above is between two spans that both lost their semantics"
    )
