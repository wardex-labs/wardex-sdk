"""Anthropic Agent SDK OTel bridge — receiver, merge, injection and drain.

Everything here runs WITHOUT a CLI: the receiver is a real loopback HTTP
server POSTed to with http.client, the OTLP bodies are hand-built protobuf
(`_otlp_build`, deliberately not wardex's own encoder), and the assembler is
driven directly the way `test_agent_sdk_assembler.py` drives it. That is the
same CLI-less rule the adapter's own suite adopted, and the first of the
design's test gates requires it by name.
"""

from __future__ import annotations

import gzip
import http.client

import pytest

import _otlp_build
from wardex_sdk._adapters._otel_receiver import _OtelBridgeReceiver
from wardex_sdk._assembly import counters

TRACE = "aa" * 16


@pytest.fixture(autouse=True)
def _fresh_counters():
    counters.reset()
    yield
    counters.reset()


@pytest.fixture
def receiver():
    r = _OtelBridgeReceiver(max_body_bytes=64 * 1024, max_spans_per_session=64, max_sessions=8)
    yield r
    r.close()


def _post(
    receiver,
    body: bytes,
    *,
    token: str | None = None,
    path: str = "/v1/traces",
    method: str = "POST",
    headers: dict | None = None,
) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", receiver.port, timeout=5)
    try:
        sent = {"x-wardex-bridge": receiver.token if token is None else token}
        sent.update(headers or {})
        conn.request(method, path, body=body, headers=sent)
        return conn.getresponse().status
    finally:
        conn.close()


def _one_span_body(trace_id: str = TRACE, **kw) -> bytes:
    kw.setdefault("name", "claude_code.hook")
    kw.setdefault("span_id", "0d" * 8)
    kw.setdefault("start_ns", 1)
    kw.setdefault("end_ns", 2)
    return _otlp_build.request([_otlp_build.span(trace_id=trace_id, **kw)])


# --------------------------------------------------------------------------
# receiver — token, paths, methods, caps (no CLI anywhere)
# --------------------------------------------------------------------------


def test_a_post_without_the_token_is_rejected_and_creates_no_span(receiver):
    """Design gate 2. Loopback is not authorization: any process on the
    machine can reach this port, and only the CLI wardex spawned holds the
    token. A refused POST must leave NO state — a slot created for an
    unauthenticated sender would let it grow memory without the token too."""
    receiver.reserve(TRACE)
    body = _one_span_body()

    assert _post(receiver, body, token="") == 403
    assert _post(receiver, body, token="wrong-" + receiver.token[6:]) == 403

    slot = receiver.take(TRACE, None)
    assert slot is not None and slot.spans == []  # the reservation, untouched
    assert receiver.take(TRACE, "s-1") is None
    assert counters.get("adapters.anthropic.otel_bridge.token_rejected") == 2


def test_the_receiver_rejects_wrong_paths_and_methods(receiver):
    body = _one_span_body()
    assert _post(receiver, body, path="/v1/metrics") == 404
    assert _post(receiver, body, path="/") == 404
    assert _post(receiver, body, method="GET") == 405
    receiver.reserve(TRACE)
    slot = receiver.take(TRACE, None)
    assert slot is not None and slot.spans == []


def test_an_oversized_or_bomb_body_is_rejected_and_counted(receiver):
    tight = _OtelBridgeReceiver(max_body_bytes=64, max_spans_per_session=64, max_sessions=8)
    try:
        tight.reserve(TRACE)
        big = _one_span_body(attrs={"pad": "x" * 128})
        assert len(big) > 64
        assert _post(tight, big) == 413
        assert counters.get("adapters.anthropic.otel_bridge.body_rejected") == 1

        # A gzip body whose WIRE size fits the cap but which inflates past it:
        # the decompression-bomb half of the bound.
        bomb = gzip.compress(b"\x00" * 4096)
        assert len(bomb) <= 64
        assert _post(tight, bomb, headers={"Content-Encoding": "gzip"}) == 413
        assert counters.get("adapters.anthropic.otel_bridge.body_rejected") == 2
        assert tight.take(TRACE, None).spans == []
    finally:
        tight.close()

    # A valid gzip body under the cap decodes and routes normally — the
    # Content-Encoding the injected exporter may legitimately use.
    receiver.reserve(TRACE)
    assert (
        _post(receiver, gzip.compress(_one_span_body()), headers={"Content-Encoding": "gzip"})
        == 200
    )
    slot = receiver.take(TRACE, None)
    assert slot is not None and len(slot.spans) == 1


def test_spans_beyond_the_session_cap_are_dropped_and_counted():
    receiver = _OtelBridgeReceiver(
        max_body_bytes=64 * 1024, max_spans_per_session=3, max_sessions=8
    )
    try:
        receiver.reserve(TRACE)
        body = _otlp_build.request(
            [
                _otlp_build.span(
                    name="claude_code.hook",
                    trace_id=TRACE,
                    span_id=f"{i:016x}",
                    start_ns=1,
                    end_ns=2,
                )
                for i in range(5)
            ]
        )
        assert _post(receiver, body) == 200
        slot = receiver.take(TRACE, None)
        assert slot is not None
        assert len(slot.spans) == 3
        assert slot.dropped == 2
        assert counters.get("adapters.anthropic.otel_bridge.span_dropped") == 2
    finally:
        receiver.close()


def test_identity_pii_is_scrubbed_at_the_receiver_boundary(receiver):
    """R12's denylist half, applied BEFORE anything is stored: the CLI stamps
    identity PII on every span, so it must never sit in wardex memory."""
    receiver.reserve(TRACE)
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.hook",
                trace_id=TRACE,
                span_id="0d" * 8,
                start_ns=1,
                end_ns=2,
                attrs={
                    "user.email": "a@b.c",
                    "user.id": "u-1",
                    "user.account_uuid": "uuid-1",
                    "organization.id": "o-1",
                    "user_prompt": "raw text",
                    "tool_name": "Bash",
                },
            )
        ],
        resource_attrs={"user.email": "a@b.c", "service.version": "2.1.226"},
    )
    assert _post(receiver, body) == 200
    slot = receiver.take(TRACE, None)
    (span,) = slot.spans
    assert span["attributes"] == {"tool_name": "Bash"}
    assert slot.resource == {"service.version": "2.1.226"}


def test_a_span_routes_by_session_id_when_the_trace_is_unknown(receiver):
    """The fallback route (spike discovery 1: session.id is unconditionally on
    trace spans). A session whose injection read-back failed still converges
    on a slot, which is what keeps its spans mergeable."""
    body = _one_span_body(trace_id="bb" * 16, attrs={"session.id": "s-9"})
    assert _post(receiver, body) == 200
    assert receiver.take("cc" * 16, "s-9") is not None
    assert receiver.take("cc" * 16, "s-9") is None  # popped from both indexes


def test_an_undecodable_post_answers_200_and_flags_the_sole_live_slot(receiver):
    receiver.reserve(TRACE)
    assert _post(receiver, b"\xff\xfenot otlp at all") == 200
    assert counters.get("adapters.anthropic.otel_bridge.undecodable") == 1
    slot = receiver.take(TRACE, None)
    assert slot is not None and slot.schema_failed is True


def test_an_undecodable_post_with_several_live_slots_is_only_counted(receiver):
    receiver.reserve(TRACE)
    receiver.reserve("bb" * 16)
    assert _post(receiver, b"\xff\xfenot otlp at all") == 200
    assert counters.get("adapters.anthropic.otel_bridge.undecodable") == 1
    assert receiver.take(TRACE, None).schema_failed is False
    assert receiver.take("bb" * 16, None).schema_failed is False
