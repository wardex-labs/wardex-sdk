import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import CaptureMode, SpanKind


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
