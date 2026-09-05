import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import CaptureMode, SpanKind
from wardex_sdk._protocol import classify_ws_upgrade
from wardex_sdk._types import StatusCode


@pytest.fixture(autouse=True)
def _fresh_counters():
    counters.reset()
    yield
    counters.reset()


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


class _FakeSSLObj:
    """Minimal fake object that only provides server_hostname (for span host)."""

    def __init__(self, host: str) -> None:
        self.server_hostname = host

    def selected_alpn_protocol(self):
        return None


def _frame(fin: bool, opcode: int, payload: bytes) -> bytes:
    return bytes([(0x80 if fin else 0) | opcode, len(payload)]) + payload


def _ws_spans():
    return [
        s
        for s in _hub.get_client()._spans
        if s.kind == SpanKind.CLIENT and (s.name or "").startswith("WS ")
    ]


def test_ws_upgrade_to_close_emits_one_span():
    # capture_mode=ALL: WS spans carry no LLM semantics (sem is always None in
    # _emit_ws), so under the AGENT-mode default this generic-traffic test
    # needs an active local span to latch onto, which it deliberately has
    # none of — it targets WS frame/tracker capture, not the policy gate.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    from wardex_sdk._interceptors._registry import get_registry

    interceptor = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("api.example")

    # Upgrade request
    interceptor._on_request_bytes(
        obj,
        b"GET /realtime HTTP/1.1\r\nHost: api.example\r\n"
        b"Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    # 101 + leftover WS text
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
        b"Connection: Upgrade\r\n\r\n" + _frame(True, 0x1, b"hello"),
    )
    # Client sends text
    interceptor._on_request_bytes(obj, _frame(True, 0x1, b'{"x":1}'))
    # Server close(1000)
    interceptor._on_response_bytes(obj, _frame(True, 0x8, (1000).to_bytes(2, "big")))

    spans = _ws_spans()
    assert len(spans) == 1
    sp = spans[0]
    assert sp.name == "WS /realtime"
    assert ("ws.messages.received", 1) in sp.extra  # "hello"
    assert ("ws.messages.sent", 1) in sp.extra  # {"x":1}
    assert b"hello" in sp.output_data
    assert ("ws.close_code", 1000) in sp.extra


def test_no_close_flushes_on_uninstall_with_marker():
    # capture_mode=ALL: see rationale in test_ws_upgrade_to_close_emits_one_span.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    from wardex_sdk._interceptors._registry import get_registry

    registry = get_registry()
    interceptor = registry._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("api.example")
    interceptor._on_request_bytes(
        obj,
        b"GET /rt HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_request_bytes(obj, _frame(True, 0x1, b"hi"))
    # uninstall without close → flush
    registry.uninstall_all()

    spans = _ws_spans()
    assert len(spans) == 1
    assert Limitation.WS_NO_CLOSE in spans[0].capture_integrity.limitations


def test_deflate_negotiation_marks_compressed():
    # capture_mode=ALL: see rationale in test_ws_upgrade_to_close_emits_one_span.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    from wardex_sdk._interceptors._registry import get_registry

    registry = get_registry()
    interceptor = registry._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("api.example")
    interceptor._on_request_bytes(
        obj,
        b"GET /rt HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Extensions: permessage-deflate\r\n\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Extensions: permessage-deflate\r\n\r\n",
    )
    interceptor._on_response_bytes(obj, _frame(True, 0x8, (1000).to_bytes(2, "big")))
    spans = _ws_spans()
    assert len(spans) == 1
    # Census merge (design §6.5.1): `ws_compressed` and `grpc_compressed` folded
    # into PAYLOAD_COMPRESSED; TransportAttributes.protocol already carries
    # which protocol it was.
    assert Limitation.PAYLOAD_COMPRESSED in spans[0].capture_integrity.limitations


def test_client_close_error_code_maps_error_status():
    # capture_mode=ALL: see rationale in test_ws_upgrade_to_close_emits_one_span.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    from wardex_sdk._enums import StatusCode
    from wardex_sdk._interceptors._registry import get_registry

    interceptor = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("api.example")
    interceptor._on_request_bytes(
        obj,
        b"GET /rt HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    # Client sends close(1008 policy_violation) → _on_request_bytes' WS branch + ERROR mapping
    interceptor._on_request_bytes(obj, _frame(True, 0x8, (1008).to_bytes(2, "big")))

    spans = _ws_spans()
    assert len(spans) == 1
    assert spans[0].status == StatusCode.ERROR
    assert spans[0].error_type == "policy_violation"


def test_close_flushes_ws_span_to_transport():
    class _RecordingTransport:
        def __init__(self):
            self.envelopes = []

        def export(self, envelope):
            self.envelopes.append(envelope)

        def flush(self, timeout: float = 5.0):
            pass

        def close(self, timeout: float = 5.0):
            pass

    # capture_mode=ALL: see rationale in test_ws_upgrade_to_close_emits_one_span.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    rec = _RecordingTransport()
    _hub.get_client()._transport = rec
    from wardex_sdk._interceptors._registry import get_registry

    interceptor = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("api.example")
    interceptor._on_request_bytes(
        obj,
        b"GET /rt HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_request_bytes(obj, _frame(True, 0x1, b"hi"))  # one message, no close
    # Public close path: the WS-close flush span must reach the transport
    wardex.close()

    ws_spans = [
        s
        for env in rec.envelopes
        for s in env.spans
        if s.kind == SpanKind.CLIENT and (s.name or "").startswith("WS ")
    ]
    assert len(ws_spans) == 1
    assert Limitation.WS_NO_CLOSE in ws_spans[0].capture_integrity.limitations


# --- the WebSocket LLM-transport question, through the seam ---------------

_UNREAD = "interceptors.seam.ws_llm_semantics_unread"
_UNCONFIRMED = "interceptors.seam.ws_llm_endpoint_unconfirmed"


def _session(host: str, path: str, *, deflate: bool, first: bytes) -> None:
    """One WebSocket session under the DEFAULT capture mode (AGENT, no local
    span): upgrade, one client Text, two server Texts, server close 1001."""
    wardex.init(intercept=True)
    from wardex_sdk._interceptors._registry import get_registry

    interceptor = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj(host)
    ext = b"Sec-WebSocket-Extensions: permessage-deflate\r\n" if deflate else b""
    interceptor._on_request_bytes(
        obj,
        b"GET " + path.encode() + b" HTTP/1.1\r\nHost: " + host.encode() + b"\r\n"
        b"Upgrade: websocket\r\nConnection: Upgrade\r\n" + ext + b"\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
        b"Connection: Upgrade\r\n" + ext + b"\r\n",
    )
    interceptor._on_request_bytes(obj, _frame(True, 0x1, first))
    interceptor._on_response_bytes(obj, _frame(True, 0x1, b'{"type":"response.created"}'))
    interceptor._on_response_bytes(obj, _frame(True, 0x1, b'{"type":"response.completed"}'))
    interceptor._on_response_bytes(obj, _frame(True, 0x8, (1001).to_bytes(2, "big")))


def test_responses_websocket_ships_marked_under_default_agent_mode():
    """The openai-agents opt-in transport: every call crosses one
    `wss://api.openai.com/v1/responses` connection. It used to produce no
    span and no counter under the default mode."""
    _session("api.openai.com", "/v1/responses", deflate=True, first=b"\x8b\x00\x01")
    spans = _ws_spans()
    assert len(spans) == 1
    sp = spans[0]
    assert sp.name == "WS /v1/responses"
    assert sp.status == StatusCode.OK
    assert sp.gen_ai is None
    markers = sp.capture_integrity.limitations
    assert Limitation.WS_LLM_SEMANTICS_UNREAD in markers
    assert Limitation.PAYLOAD_COMPRESSED in markers
    assert ("ws.messages.sent", 1) in sp.extra
    assert ("ws.messages.received", 2) in sp.extra
    assert counters.get(_UNREAD) == 1
    assert counters.get(_UNCONFIRMED) == 0


def test_loopback_responses_websocket_confirms_from_the_envelope():
    _session(
        "127.0.0.1",
        "/v1/responses",
        deflate=False,
        first=b'{"type": "response.create", "model": "gpt-4o-mini"}',
    )
    spans = _ws_spans()
    assert len(spans) == 1
    assert Limitation.WS_LLM_SEMANTICS_UNREAD in spans[0].capture_integrity.limitations
    assert counters.get(_UNREAD) == 1


def test_loopback_responses_websocket_with_deflate_is_unconfirmed():
    """The gate refuses the span under the default mode; the count happens
    before the gate, so the refused connection is still counted."""
    _session("127.0.0.1", "/v1/responses", deflate=True, first=b'{"type": "response.create"}')
    assert _ws_spans() == []
    assert counters.get(_UNCONFIRMED) == 1
    assert counters.get(_UNREAD) == 0


def test_ws_llm_counters_move_at_connection_close():
    """Counted from the seam when the connection's one span is built, not
    from the tracker on the first message: after the first client message
    nothing has moved, after the close exactly one count has."""
    wardex.init(intercept=True)
    from wardex_sdk._interceptors._registry import get_registry

    interceptor = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("api.openai.com")
    interceptor._on_request_bytes(
        obj,
        b"GET /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\n"
        b"Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_request_bytes(obj, _frame(True, 0x1, b'{"type":"response.create"}'))
    assert counters.get(_UNREAD) == 0
    interceptor._on_response_bytes(obj, _frame(True, 0x8, (1000).to_bytes(2, "big")))
    assert counters.get(_UNREAD) == 1
    assert len(_ws_spans()) == 1


def test_ordinary_messages_websocket_is_not_an_llm_call():
    """`/messages` is the commonest WebSocket path name; the row is not
    WebSocket-capable, so no claim is made and nothing is counted."""
    _session("chat.example.com", "/messages", deflate=False, first=b'{"type": "response.create"}')
    assert _ws_spans() == []
    assert counters.get(_UNREAD) == 0
    assert counters.get(_UNCONFIRMED) == 0


def test_substring_provider_host_is_not_the_provider():
    """`openai-mock.corp` contains the provider's name and is not the
    provider: a custom protocol on its `/v1/responses` must not ship under
    the default mode with payload samples on the strength of its hostname.
    Same refusal the HTTP path makes — a hostname is not semantics."""
    _session("openai-mock.corp", "/v1/responses", deflate=False, first=b'{"op":"custom"}')
    assert _ws_spans() == []
    assert counters.get(_UNCONFIRMED) == 1
    assert counters.get(_UNREAD) == 0


@pytest.mark.parametrize(
    ("host", "want"),
    [
        ("api.openai.com", "known_provider"),
        ("eu.api.openai.com", "known_provider"),
        ("API.OPENAI.COM", "known_provider"),
        ("openai-mock.corp", "unknown_host"),
        ("openai-shim.corp", "unknown_host"),
        ("api.openai.com.evil.example", "unknown_host"),
        ("127.0.0.1", "unknown_host"),
    ],
)
def test_classify_ws_upgrade_host_rule(host: str, want: str):
    """Mirrors the Rust table test in endpoint.rs: the provider's own
    hosts are `api.<domain>` and subdomains of `<domain>`, nothing else."""
    assert classify_ws_upgrade(host, "/v1/responses") == want
    assert classify_ws_upgrade(host, "/v1/chat/completions") is None


def test_realtime_websocket_is_still_dropped_under_agent_mode():
    """The residual silence, documented: OpenAI Realtime is not a
    WebSocket-capable row in the endpoint table, so outside a local span it
    is still dropped without a marker or a counter."""
    _session("api.openai.com", "/v1/realtime", deflate=False, first=b'{"type":"session.update"}')
    assert _ws_spans() == []
    assert counters.get(_UNREAD) == 0
    assert counters.get(_UNCONFIRMED) == 0
