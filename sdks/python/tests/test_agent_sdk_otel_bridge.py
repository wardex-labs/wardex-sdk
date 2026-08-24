"""Anthropic Agent SDK OTel bridge — receiver, merge, injection and drain.

Everything here runs WITHOUT a CLI: the receiver is a real loopback HTTP
server POSTed to with http.client, the OTLP bodies are hand-built protobuf
(`_otlp_build`, deliberately not wardex's own encoder), and the assembler is
driven directly the way `test_agent_sdk_assembler.py` drives it. That is the
same CLI-less rule the adapter's own suite adopted, and the first of the
design's test gates requires it by name.
"""

from __future__ import annotations

import asyncio
import gzip
import http.client
import json
import time

import pytest

import _otlp_build
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._adapters._otel_receiver import _OtelBridgeReceiver
from wardex_sdk._adapters._session_state import _BridgeBinding
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._enums import CaptureSource, StatusCode

TRACE = "aa" * 16

_TIMING = Limitation.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS

INIT = {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-sonnet-5"}
ASSISTANT = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m1",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 25},
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}}
        ],
    },
}
ASSISTANT_2 = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m2",
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 30},
        "content": [{"type": "text", "text": "done"}],
    },
}
TOOL_RESULT = {
    "type": "user",
    "session_id": "s-1",
    "parent_tool_use_id": "toolu_01",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}],
    },
}
RESULT = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "total_cost_usd": 0.01,
    "duration_ms": 100,
    "duration_api_ms": 80,
}


class FakeClient:
    def __init__(self):
        self.spans = []

    def capture_span(self, span):
        self.spans.append(span)


def _binding(confirmed: bool = True, trace: str | None = TRACE) -> _BridgeBinding:
    return _BridgeBinding(trace_id_hex=trace, confirmed=confirmed)


def _outbound(asm, key=1, bridge=None, text="go"):
    asm.on_outbound(
        key,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": text}}
        ),
        bridge=bridge,
    )


def _limitations(span) -> tuple:
    return span.capture_integrity.limitations if span.capture_integrity else ()


def _named(spans, name):
    return next(s for s in spans if s.name == name)


@pytest.fixture(autouse=True)
def _fresh_counters():
    counters.reset()
    yield
    counters.reset()


@pytest.fixture(autouse=True)
def _fresh_reports():
    reset_reports_for_test()
    yield
    reset_reports_for_test()


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


# --------------------------------------------------------------------------
# the merge (design gate 1) — assembler driven directly, POSTs over real HTTP
# --------------------------------------------------------------------------


def test_a_hand_built_otlp_post_merges_into_the_session_tree(receiver):
    """Design gate 1, end to end without a CLI: a scripted session plus one
    hand-built OTLP POST, asserted on edges, sources, times and markers."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    t0 = time.time_ns()
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        "toolu_01",
    )
    asm.on_hook(
        "PostToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"}, "toolu_01"
    )
    asm.on_inbound(1, TOOL_RESULT)
    asm.on_inbound(1, RESULT)
    t1 = time.time_ns()

    llm_start, llm_end = t0 + 1_000, t1 + 2_000_000
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.interaction",
                trace_id=TRACE,
                span_id="01" * 8,
                start_ns=t0,
                end_ns=t1,
                attrs={"session.id": "s-1"},
            ),
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id="02" * 8,
                parent_span_id="01" * 8,
                start_ns=llm_start,
                end_ns=llm_end,
                attrs={"gen_ai.response.id": "req_011", "ttft_ms": 250},
            ),
            _otlp_build.span(
                name="claude_code.tool",
                trace_id=TRACE,
                span_id="03" * 8,
                parent_span_id="02" * 8,
                start_ns=t0,
                end_ns=t1,
                attrs={"tool_use_id": "toolu_01", "tool_name": "Bash"},
            ),
            _otlp_build.span(
                name="claude_code.tool.execution",
                trace_id=TRACE,
                span_id="04" * 8,
                parent_span_id="03" * 8,
                start_ns=t0 + 5_000_000,
                end_ns=t0 + 10_000_000,
                attrs={"tool_use_id": "toolu_01"},
            ),
            _otlp_build.span(
                name="claude_code.hook",
                trace_id=TRACE,
                span_id="05" * 8,
                parent_span_id="01" * 8,
                start_ns=t0,
                end_ns=t0 + 1_000_000,
            ),
        ],
        resource_attrs={"service.version": "2.1.226"},
    )
    assert _post(receiver, body) == 200
    asm.on_close(1, None)

    root = _named(client.spans, "invoke_agent")
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    tool = _named(client.spans, "execute_tool Bash")
    step = _named(client.spans, "execute_step hook")

    # Merged chat: the CLI's interval and ttft, BOTH sources, and the two
    # timing markers gone — the headline deliverable.
    assert chat.start_time_ns == llm_start
    assert chat.end_time_ns == llm_end
    assert chat.gen_ai.time_to_first_chunk_s == pytest.approx(0.25)
    assert chat.capture_sources == (CaptureSource.ADAPTER, CaptureSource.OTEL_BRIDGE)
    assert _TIMING not in _limitations(chat)
    assert Limitation.TTFT_IPC_APPROXIMATION not in _limitations(chat)
    assert ("wardex.anthropic_agent_sdk.otel.request_id", "req_011") in chat.extra
    assert b"go" in chat.input_data  # content authority: still the stream's

    # Merged tool: source + additive CLI duration, but IPC times and the
    # timing marker KEPT — marker removal is the merged-LLM deliverable only.
    assert CaptureSource.OTEL_BRIDGE in tool.capture_sources
    assert _TIMING in _limitations(tool)
    assert dict(tool.extra)["wardex.anthropic_agent_sdk.otel.tool_duration_ms"] == pytest.approx(
        5.0
    )
    assert t0 <= tool.start_time_ns <= t1  # hook-observed, not rewritten

    # Pure increment: CLI-measured times, bridge-only source, no timing marker.
    assert step.parent_span_id == root.context.span_id
    assert step.capture_sources == (CaptureSource.OTEL_BRIDGE,)
    assert ("wardex.step.name", "hook") in step.extra
    assert ("wardex.anthropic_agent_sdk.otel.span", "claude_code.hook") in step.extra
    assert step.start_time_ns == t0
    assert step.end_time_ns == t0 + 1_000_000
    assert _TIMING not in _limitations(step)

    # Root cross-check: source + CLI version, no fail-open marker.
    assert CaptureSource.OTEL_BRIDGE in root.capture_sources
    assert ("wardex.anthropic_agent_sdk.otel.cli_version", "2.1.226") in root.extra
    assert Limitation.OTEL_BRIDGE_NO_DATA not in _limitations(root)
    assert Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN not in _limitations(root)

    # Pre-existing edges unchanged: the bridge adds, never re-parents.
    assert chat.parent_span_id == root.context.span_id
    assert tool.parent_span_id == root.context.span_id


def _evicted_then_completed(asm, receiver, t0, t1, *, cap_fill=5):
    """Open `cap_fill` tools over a 4-entry table so `t1` is evicted, then close
    it — two pending records under one `tool_use_id` — and hand the CLI a tool
    span for it."""
    receiver.reserve(TRACE)
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    for n in range(1, cap_fill + 1):
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
            f"t{n}",
        )
    asm.on_hook(
        "PostToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_response": "done"}, "t1"
    )
    return _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.interaction",
                trace_id=TRACE,
                span_id="01" * 8,
                start_ns=t0,
                end_ns=t1,
                attrs={"session.id": "s-1"},
            ),
            _otlp_build.span(
                name="claude_code.tool",
                trace_id=TRACE,
                span_id="03" * 8,
                parent_span_id="01" * 8,
                start_ns=t0,
                end_ns=t0 + 7_000_000,
                attrs={"tool_use_id": "t1", "tool_name": "Bash"},
            ),
            # A CLI child of the tool span, so "which half became the anchor"
            # is observable rather than inferred.
            _otlp_build.span(
                name="claude_code.hook",
                trace_id=TRACE,
                span_id="04" * 8,
                parent_span_id="03" * 8,
                start_ns=t0 + 1_000_000,
                end_ns=t0 + 6_000_000,
            ),
        ]
    )


def test_the_bridge_merges_the_completion_half_not_the_stub(receiver):
    """The join pops by `tool_use_id`, and an eviction puts two records under one.

    The stub is pended FIRST, so without a qualification field it wins the pop:
    it would take the CLI's duration and the bridge source and become the anchor
    for the CLI's children, while the half that holds the output and the real
    interval fell through unmerged. The CLI measured the WHOLE call, so its
    number belongs on the half that represents the whole call.
    """
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver, max_session_entries=4)
    t0 = time.time_ns()
    body = _evicted_then_completed(asm, receiver, t0, t0 + 20_000_000)
    assert _post(receiver, body) == 200
    asm.on_close(1, None)

    halves = [s for s in client.spans if s.tool is not None and s.tool.call_id == "t1"]
    assert len(halves) == 2
    stub = next(s for s in halves if s.status is StatusCode.UNSET)
    completion = next(s for s in halves if s is not stub)
    for half in halves:
        assert Limitation.SESSION_ENTRY_TABLE_FULL in _limitations(half)

    key = "wardex.anthropic_agent_sdk.otel.tool_duration_ms"
    assert dict(completion.extra)[key] == pytest.approx(7.0)
    assert CaptureSource.OTEL_BRIDGE in completion.capture_sources
    assert key not in dict(stub.extra)
    assert CaptureSource.OTEL_BRIDGE not in stub.capture_sources

    # ...and the CLI's own child hangs off the half that was merged.
    child = _named(client.spans, "execute_step hook")
    assert child.parent_span_id == completion.context.span_id
    assert child.parent_span_id != stub.context.span_id


def test_at_most_one_mergeable_tool_record_per_call_id(receiver):
    """INV: within `sess.pending`, `tool_use_id` identifies at most one join
    target. Asserted on runtime STATE rather than by reading the source, so a
    duplicate arriving later for some other reason dies here first — the pop-key
    join is structurally fragile to duplicates, and this is where that shows."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver, max_session_entries=4)
    t0 = time.time_ns()
    _evicted_then_completed(asm, receiver, t0, t0 + 20_000_000)

    pending = asm._by_key[1].pending
    tool_ids = [r.tool_use_id for r in pending if r.kind == "tool" and r.tool_use_id]
    assert tool_ids.count("t1") == 2, "the two halves must both be pending"
    mergeable = [r.tool_use_id for r in pending if r.kind == "tool" and r.mergeable]
    assert len(mergeable) == len(set(mergeable))


def test_a_rejected_post_leaves_a_confirmed_session_with_the_no_data_marker(receiver):
    """Gate 2's second half: the 403'd sender created no state, so a session
    whose injection was CONFIRMED closes with zero bridge sources and exactly
    one `otel_bridge_no_data` on the root."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    _outbound(asm, 1, bridge=_binding())
    for msg in (INIT, ASSISTANT, RESULT):
        asm.on_inbound(1, msg)
    assert _post(receiver, _one_span_body(), token="") == 403
    asm.on_close(1, None)

    assert client.spans, "the session must still emit"
    for span in client.spans:
        assert CaptureSource.OTEL_BRIDGE not in span.capture_sources
    root = _named(client.spans, "invoke_agent")
    assert list(_limitations(root)).count(Limitation.OTEL_BRIDGE_NO_DATA) == 1


# --------------------------------------------------------------------------
# fail-open equivalence (design gate 3) and schema drift (marker 42)
# --------------------------------------------------------------------------


def _scripted_session(asm, key=1, bridge=None):
    _outbound(asm, key, bridge=bridge)
    for msg in (INIT, ASSISTANT):
        asm.on_inbound(key, msg)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        "toolu_01",
    )
    asm.on_hook(
        "PostToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"}, "toolu_01"
    )
    asm.on_inbound(key, TOOL_RESULT)
    asm.on_inbound(key, RESULT)
    asm.on_close(key, None)


def _normalized(spans, drop_root_marker=None):
    """Span-id- and wall-clock-independent view of a captured tree."""
    by_id = {s.context.span_id: s for s in spans}

    def path(s):
        names = []
        cur = s
        while cur is not None:
            names.append(cur.name)
            cur = by_id.get(cur.parent_span_id) if cur.parent_span_id else None
        return tuple(names)

    out = []
    for s in spans:
        markers = [m.value for m in _limitations(s)]
        if drop_root_marker is not None and s.name == "invoke_agent":
            markers = [m for m in markers if m != drop_root_marker.value]
        out.append(
            (
                s.name,
                path(s),
                s.status,
                s.error_type,
                s.input_data,
                s.output_data,
                tuple(s.capture_sources),
                tuple(sorted(markers)),
                tuple(sorted(s.extra)),
            )
        )
    return sorted(out)


def test_a_bridge_session_with_no_data_matches_the_bridge_off_tree_plus_one_marker(receiver):
    """Design gate 3, doubling as the off-path equivalence proof: one scripted
    session through two assemblers — bridge off, and bridge on with zero POSTs
    — differs by exactly one root marker and NOTHING else. (The whole existing
    agent-sdk suite, which runs bridge-off, is the standing byte-level
    regression net for the off path itself.)"""
    off_client, on_client = FakeClient(), FakeClient()
    _scripted_session(SessionAssembler(off_client))
    receiver.reserve(TRACE)
    on_asm = SessionAssembler(on_client, bridge=receiver)
    _scripted_session(on_asm, bridge=_binding())

    on_root = _named(on_client.spans, "invoke_agent")
    assert list(_limitations(on_root)).count(Limitation.OTEL_BRIDGE_NO_DATA) == 1
    assert _normalized(on_client.spans, drop_root_marker=Limitation.OTEL_BRIDGE_NO_DATA) == (
        _normalized(off_client.spans)
    )


def test_schema_drift_marks_the_root_and_changes_nothing_else(receiver, capsys):
    """R11 / marker 42, both routes: decodable spans that classify as nothing,
    and an undecodable POST attributed to the sole live bridge session."""
    # Route 1 — decodable, unrecognized names.
    off_client, on_client = FakeClient(), FakeClient()
    _scripted_session(SessionAssembler(off_client))
    receiver.reserve(TRACE)
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.unrecognized_thing",
                trace_id=TRACE,
                span_id=f"{i:016x}",
                start_ns=1,
                end_ns=2,
            )
            for i in range(3)
        ]
    )
    assert _post(receiver, body) == 200
    on_asm = SessionAssembler(on_client, bridge=receiver)
    _scripted_session(on_asm, bridge=_binding())

    on_root = _named(on_client.spans, "invoke_agent")
    assert list(_limitations(on_root)).count(Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN) == 1
    assert Limitation.OTEL_BRIDGE_NO_DATA not in _limitations(on_root)
    assert _normalized(on_client.spans, drop_root_marker=Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN) == (
        _normalized(off_client.spans)
    )

    # Route 2 — an undecodable POST, sole live session, same marker + the
    # counter + one report_once line.
    client2 = FakeClient()
    trace2 = "cc" * 16
    receiver.reserve(trace2)
    capsys.readouterr()
    assert _post(receiver, b"\x00garbage, not otlp") == 200
    assert counters.get("adapters.anthropic.otel_bridge.undecodable") == 1
    assert capsys.readouterr().err.count("did not decode") == 1
    asm2 = SessionAssembler(client2, bridge=receiver)
    _scripted_session(asm2, bridge=_binding(trace=trace2))
    root2 = _named(client2.spans, "invoke_agent")
    assert list(_limitations(root2)).count(Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN) == 1


def test_user_prompt_from_the_cli_is_dropped_at_merge(receiver):
    """R12: the interaction span merges (root gains the source) but its
    user_prompt value appears NOWHERE; the chat's input stays the stream's."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    _outbound(asm, 1, bridge=_binding())
    for msg in (INIT, ASSISTANT, RESULT):
        asm.on_inbound(1, msg)
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.interaction",
                trace_id=TRACE,
                span_id="01" * 8,
                start_ns=1,
                end_ns=2,
                attrs={"user_prompt": "raw prompt text the CLI logged"},
            )
        ]
    )
    assert _post(receiver, body) == 200
    asm.on_close(1, None)

    root = _named(client.spans, "invoke_agent")
    assert CaptureSource.OTEL_BRIDGE in root.capture_sources
    for span in client.spans:
        assert b"raw prompt text" not in span.input_data
        assert b"raw prompt text" not in span.output_data
        assert "raw prompt text" not in repr(span.extra)
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert b"go" in chat.input_data


def test_identity_pii_never_reaches_a_wardex_span(receiver):
    """R12 end to end: identity PII on EVERY posted span, scanned for on every
    EMITTED span — while the merge itself demonstrably worked (the allowlisted
    request_id extra is present), so the scrub is a policy, not a join failure."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    t0 = time.time_ns()
    _outbound(asm, 1, bridge=_binding())
    for msg in (INIT, ASSISTANT, RESULT):
        asm.on_inbound(1, msg)
    t1 = time.time_ns()
    pii = {
        "user.email": "victim@example.com",
        "user.id": "user-123456",
        "user.account_uuid": "uuid-abcdef",
        "organization.id": "org-654321",
        "user_prompt": "secret prompt body",
    }
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.interaction",
                trace_id=TRACE,
                span_id="01" * 8,
                start_ns=t0,
                end_ns=t1,
                attrs=dict(pii),
            ),
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id="02" * 8,
                parent_span_id="01" * 8,
                start_ns=t0 + 1_000,
                end_ns=t1,
                attrs={"gen_ai.response.id": "req_9", "ttft_ms": 100, **pii},
            ),
            _otlp_build.span(
                name="claude_code.hook",
                trace_id=TRACE,
                span_id="03" * 8,
                parent_span_id="01" * 8,
                start_ns=t0,
                end_ns=t0 + 500,
                attrs=dict(pii),
            ),
        ]
    )
    assert _post(receiver, body) == 200
    asm.on_close(1, None)

    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert ("wardex.anthropic_agent_sdk.otel.request_id", "req_9") in chat.extra
    for span in client.spans:
        blob = repr(span)
        for value in pii.values():
            assert value not in blob, f"{value!r} leaked into {span.name}"


# --------------------------------------------------------------------------
# ambiguity (design gate 5) — never guess a parent
# --------------------------------------------------------------------------


def test_an_ambiguous_join_never_guesses_a_parent(receiver):
    """Two chats over one llm_request AND two llm_requests over one chat: no
    party merges, the drafts keep their markers, and every unplaced
    llm_request ships as a SIBLING step span saying correlation_conflict."""
    # Case A: two chat windows, one overlapping llm_request.
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    t0 = time.time_ns()
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)
    _outbound(asm, 1, text="and then?")
    asm.on_inbound(1, ASSISTANT_2)
    asm.on_inbound(1, RESULT)
    t1 = time.time_ns()
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id="02" * 8,
                start_ns=t0,
                end_ns=t1,
                attrs={"gen_ai.response.id": "req_amb"},
            )
        ]
    )
    assert _post(receiver, body) == 200
    asm.on_close(1, None)

    root = _named(client.spans, "invoke_agent")
    chats = [s for s in client.spans if s.name.startswith("chat")]
    assert len(chats) == 2
    for chat in chats:
        assert chat.capture_sources == (CaptureSource.ADAPTER,)
        assert _TIMING in _limitations(chat)
        assert chat.parent_span_id == root.context.span_id  # unchanged edge
    sibling = _named(client.spans, "execute_step llm_request")
    assert sibling.parent_span_id == root.context.span_id
    assert Limitation.CORRELATION_CONFLICT in _limitations(sibling)
    assert sibling.capture_sources == (CaptureSource.OTEL_BRIDGE,)

    # Case B: one chat window, two overlapping llm_requests — same refusal.
    client_b = FakeClient()
    asm_b = SessionAssembler(client_b, bridge=receiver)
    trace_b = "bb" * 16
    receiver.reserve(trace_b)
    t0 = time.time_ns()
    _outbound(asm_b, 2, bridge=_binding(trace=trace_b))
    asm_b.on_inbound(2, INIT)
    asm_b.on_inbound(2, ASSISTANT_2)
    asm_b.on_inbound(2, RESULT)
    t1 = time.time_ns()
    body_b = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=trace_b,
                span_id=f"{i:016x}",
                start_ns=t0,
                end_ns=t1,
                attrs={"gen_ai.response.id": f"req_{i}"},
            )
            for i in range(2)
        ]
    )
    assert _post(receiver, body_b) == 200
    asm_b.on_close(2, None)
    chat_b = next(s for s in client_b.spans if s.name.startswith("chat"))
    assert chat_b.capture_sources == (CaptureSource.ADAPTER,)
    assert _TIMING in _limitations(chat_b)
    siblings = [s for s in client_b.spans if s.name == "execute_step llm_request"]
    assert len(siblings) == 2


# --------------------------------------------------------------------------
# injection — never-hijack (design gate 4), slice 1, copy-on-write + routing
# --------------------------------------------------------------------------


def _adapter_with(receiver, propagation=False):
    from wardex_sdk._adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter

    adapter = AnthropicAgentSdkAdapter()
    adapter._bridge = receiver
    adapter._propagation_enabled = propagation
    return adapter


_WARDEX_ENV_KEYS = ("TRACEPARENT", "TRACESTATE")


def _wardex_written_keys(env: dict) -> set:
    return {
        k
        for k in env
        if k.startswith("OTEL_") or k.startswith("CLAUDE_CODE") or k in _WARDEX_ENV_KEYS
    }


def test_user_otel_env_disables_injection_with_one_warning(receiver, monkeypatch, capsys):
    """Design gate 4. Any user telemetry key — in os.environ or options.env —
    means ZERO injected keys, the user's options untouched, and exactly one
    warning; with the bridge off the same environment produces silence."""
    import claude_agent_sdk

    from wardex_sdk._adapters._anthropic_agent_sdk import (
        AnthropicAgentSdkAdapter,
        _prepare_options,
    )

    adapter = _adapter_with(receiver)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318")
    opts = claude_agent_sdk.ClaudeAgentOptions()
    capsys.readouterr()

    out = _prepare_options(opts, adapter)
    assert _wardex_written_keys(out.env or {}) == set()
    assert opts.env in (None, {})  # the user's object, untouched
    assert capsys.readouterr().err.count("never-hijack") == 1

    _prepare_options(claude_agent_sdk.ClaudeAgentOptions(), adapter)
    assert capsys.readouterr().err == ""  # once per process, not per session

    # The same verdict when the key lives only in options.env.
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    reset_reports_for_test()
    opts2 = claude_agent_sdk.ClaudeAgentOptions(env={"OTEL_TRACES_EXPORTER": "otlp"})
    out2 = _prepare_options(opts2, adapter)
    assert (out2.env or {}) == {"OTEL_TRACES_EXPORTER": "otlp"}
    assert capsys.readouterr().err.count("never-hijack") == 1

    # Bridge OFF: a user with their own OTEL_* hears nothing at all.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318")
    reset_reports_for_test()
    _prepare_options(claude_agent_sdk.ClaudeAgentOptions(), AnthropicAgentSdkAdapter())
    assert capsys.readouterr().err == ""


def test_traceparent_is_injected_for_slice_one_only_under_its_full_gate(receiver, monkeypatch):
    """Slice 1's exact gate, as a matrix: injection happens ONLY at
    (propagation on, user CLI telemetry present, ambient context live, no user
    TRACEPARENT) — and what it injects is TRACEPARENT/TRACESTATE alone, never
    an endpoint or a toggle (never-hijack by construction)."""
    import claude_agent_sdk

    from wardex_sdk import _hub
    from wardex_sdk._adapters._anthropic_agent_sdk import (
        AnthropicAgentSdkAdapter,
        _prepare_options,
    )
    from wardex_sdk._types import SpanContext, SpanId, TraceId

    ambient_ctx = SpanContext(
        trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8), trace_flags=1
    )
    expected = "00-" + "01" * 16 + "-" + "02" * 8 + "-01"

    def run(propagation, telemetry, ambient, preset):
        adapter = AnthropicAgentSdkAdapter()  # slice 1 needs no receiver
        adapter._propagation_enabled = propagation
        if telemetry:
            monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
        else:
            monkeypatch.delenv("CLAUDE_CODE_ENABLE_TELEMETRY", raising=False)
        env = {"TRACEPARENT": "00-" + "ff" * 16 + "-" + "ee" * 8 + "-01"} if preset else {}
        opts = claude_agent_sdk.ClaudeAgentOptions(env=env)
        with _hub.isolation_scope():
            if ambient:
                _hub.get_current_scope().active_span_context = ambient_ctx
            out = _prepare_options(opts, adapter)
        return out.env or {}

    for propagation in (False, True):
        for telemetry in (False, True):
            for ambient in (False, True):
                for preset in (False, True):
                    got = run(propagation, telemetry, ambient, preset)
                    should_inject = propagation and telemetry and ambient and not preset
                    if should_inject:
                        assert got["TRACEPARENT"] == expected
                        # STRICTLY less than slice 2: context only, no
                        # endpoint, no exporter, no telemetry toggle.
                        assert _wardex_written_keys(got) <= {"TRACEPARENT", "TRACESTATE"}
                    elif preset:
                        assert got["TRACEPARENT"].startswith("00-" + "ff" * 16)
                    else:
                        assert "TRACEPARENT" not in got

    # TRACESTATE rides along exactly when the ambient context carries one.
    adapter = AnthropicAgentSdkAdapter()
    adapter._propagation_enabled = True
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    with _hub.isolation_scope():
        scope = _hub.get_current_scope()
        scope.active_span_context = ambient_ctx
        scope.tracestate = "vendor=1"
        out = _prepare_options(claude_agent_sdk.ClaudeAgentOptions(), adapter)
    assert out.env["TRACESTATE"] == "vendor=1"


def test_bridge_injection_is_copy_on_write_and_reservation_routes(receiver):
    """Slice 2's injection contract end to end: a fresh options object with all
    eight keys and the user's env preserved; the minted TRACEPARENT is
    reserved in the receiver; the env read-back CONFIRMS the binding and the
    trace id is the primary route; a broken read-back degrades to session.id
    routing and never claims NO_DATA."""
    import claude_agent_sdk

    from wardex_sdk._adapters._anthropic_agent_sdk import _prepare_options

    adapter = _adapter_with(receiver)
    opts = claude_agent_sdk.ClaudeAgentOptions(env={"MY_VAR": "1"})
    out = _prepare_options(opts, adapter)

    assert out is not opts
    assert opts.env == {"MY_VAR": "1"}  # copy-on-write: the user's dict intact
    env = out.env
    assert env["MY_VAR"] == "1"
    for key in (
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA",
        "OTEL_TRACES_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_TRACES_EXPORT_INTERVAL",
        "TRACEPARENT",
    ):
        assert key in env, key
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == receiver.endpoint

    # The minted trace id is reserved: the receiver already holds its slot.
    trace_hex = env["TRACEPARENT"].split("-")[1]

    class _FakeTransport:
        def __init__(self, options):
            self._options = options

    binding = adapter._bridge_binding_for(_FakeTransport(out))
    assert binding is not None and binding.confirmed and binding.trace_id_hex == trace_hex

    # Primary route: a POST under the reserved trace id, NO session.id
    # attribute anywhere, reaches the session.
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    _outbound(asm, 1, bridge=binding)
    for msg in (INIT, ASSISTANT, RESULT):
        asm.on_inbound(1, msg)
    assert _post(receiver, _one_span_body(trace_id=trace_hex)) == 200
    asm.on_close(1, None)
    assert any(s.name == "execute_step hook" for s in client.spans)

    # Read-back broken: an unconfirmed binding, session.id fallback routing,
    # and NO NO_DATA claim on an empty slot.
    broken = adapter._bridge_binding_for(object())
    assert broken is not None and not broken.confirmed and broken.trace_id_hex is None
    assert counters.get("adapters.anthropic.otel_bridge.readback_failed") == 1

    client2 = FakeClient()
    asm2 = SessionAssembler(client2, bridge=receiver)
    _outbound(asm2, 2, bridge=broken)
    for msg in (INIT, ASSISTANT, RESULT):
        asm2.on_inbound(2, msg)
    body = _one_span_body(trace_id="dd" * 16, attrs={"session.id": "s-1"})
    assert _post(receiver, body) == 200
    asm2.on_close(2, None)
    assert any(s.name == "execute_step hook" for s in client2.spans)

    client3 = FakeClient()
    asm3 = SessionAssembler(client3, bridge=receiver)
    _outbound(asm3, 3, bridge=_binding(confirmed=False, trace=None))
    for msg in (INIT, RESULT):
        asm3.on_inbound(3, msg)
    asm3.on_close(3, None)
    root3 = _named(client3.spans, "invoke_agent")
    assert Limitation.OTEL_BRIDGE_NO_DATA not in _limitations(root3)
    assert counters.get("adapters.assembler.otel_bridge_unconfirmed_no_data") == 1


def test_a_user_transport_env_that_is_not_ours_means_a_bridge_off_session(receiver):
    """The read-back's third verdict: a READABLE env without our endpoint is a
    spawn that got no injection — no binding, today's behavior exactly."""
    adapter = _adapter_with(receiver)

    class _FakeTransport:
        class _options:
            env = {"PATH": "/usr/bin"}

    assert adapter._bridge_binding_for(_FakeTransport()) is None


# --------------------------------------------------------------------------
# the drain (design gate 6) and the paths that must never wait
# --------------------------------------------------------------------------


def test_a_session_with_no_bridge_data_closes_without_the_drain(receiver, monkeypatch):
    """Design gate 6: with a 5s drain configured, a bridge session whose slot
    never received a span pays NOTHING at close — the plan is None before any
    poll, and `last_arrival` is never even consulted."""
    adapter = _adapter_with(receiver)
    adapter._drain_seconds = 5.0
    client = FakeClient()
    adapter._assembler = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    _outbound(adapter._assembler, 1, bridge=_binding())
    adapter._assembler.on_inbound(1, INIT)

    polled = []
    monkeypatch.setattr(receiver, "last_arrival", lambda *a: polled.append(a))
    start = time.monotonic()
    asyncio.run(adapter._drain_bridge(1))
    assert time.monotonic() - start < 0.1
    assert polled == []

    # Companion: a bridge-off session's close path never touches any bridge
    # attribute — the merge hook is structurally unreachable without a binding.
    touched = []
    monkeypatch.setattr(
        SessionAssembler, "_merge_bridge", lambda self, sess, now: touched.append(sess)
    )
    off_client = FakeClient()
    off = SessionAssembler(off_client)
    _outbound(off, 9)
    off.on_inbound(9, INIT)
    off.on_close(9, None)
    assert touched == []


def test_a_session_with_data_drains_until_quiescent(receiver):
    """The drain's positive half: data arrived, so the close waits — and exits
    on QUIESCENCE (0.1s of silence), far before a generous deadline."""
    adapter = _adapter_with(receiver)
    adapter._drain_seconds = 5.0
    client = FakeClient()
    adapter._assembler = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    _outbound(adapter._assembler, 1, bridge=_binding())
    adapter._assembler.on_inbound(1, INIT)
    assert _post(receiver, _one_span_body()) == 200

    start = time.monotonic()
    asyncio.run(adapter._drain_bridge(1))
    elapsed = time.monotonic() - start
    assert 0.1 <= elapsed < 1.0  # quiescence exit, nowhere near the 5s deadline


def test_close_all_sessions_never_drains(receiver):
    """The structural skip the flush budgets rely on: `close_all_sessions`
    (atexit/signal/uninstall) returns immediately even with a 5s drain
    configured and data pending — while still merging what already arrived."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    t0 = time.time_ns()
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)
    t1 = time.time_ns()
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id="02" * 8,
                start_ns=t0 + 1_000,
                end_ns=t1,
                attrs={"gen_ai.response.id": "req_x", "ttft_ms": 50},
            )
        ]
    )
    assert _post(receiver, body) == 200

    start = time.monotonic()
    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)
    assert time.monotonic() - start < 0.1  # test_flush_budget's premise, kept

    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert CaptureSource.OTEL_BRIDGE in chat.capture_sources  # merged anyway
    assert _TIMING not in _limitations(chat)


# --------------------------------------------------------------------------
# pending-buffer conservation: overflow and the eviction retirement path
# --------------------------------------------------------------------------


def test_pending_buffer_overflow_emits_oldest_unmerged(receiver):
    """The pending buffer is bounded by max_session_entries: the third pended
    chat pushes the first out — emitted IMMEDIATELY, unmerged, deferred
    markers applied, sources (adapter,) — and nothing is ever dropped."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver, max_session_entries=2)
    receiver.reserve(TRACE)
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    for i in range(3):
        asm.on_inbound(1, ASSISTANT_2)
        _outbound(asm, 1, text=f"turn {i}")

    overflowed = [s for s in client.spans if s.name.startswith("chat")]
    assert len(overflowed) == 1  # the oldest, emitted at overflow time
    assert overflowed[0].capture_sources == (CaptureSource.ADAPTER,)
    assert _TIMING in _limitations(overflowed[0])
    assert counters.get("adapters.assembler.otel_bridge_pending_overflow") == 1

    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)
    chats = [s for s in client.spans if s.name.startswith("chat")]
    assert len(chats) == 3  # conservation: every turn shipped exactly once


def test_an_evicted_bridge_sessions_pended_spans_still_ship(receiver):
    """The retirement half of span conservation (the review's first fix): the
    registry evicts a bridge session's root mid-pend — `max_units=1` and a
    second transport take the only slot — and every pended span still ships,
    unmerged, deferred markers applied."""
    client = FakeClient()
    from wardex_sdk._adapters._sink import _ClientSink
    from wardex_sdk._assembly import UnitRegistry

    asm = SessionAssembler(
        client, bridge=receiver, units=UnitRegistry(sink=_ClientSink(client), max_units=1)
    )
    receiver.reserve(TRACE)
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)  # one chat now PENDS
    assert not any(s.name.startswith("chat") for s in client.spans)

    # A second transport takes the registry's only root slot...
    _outbound(asm, 2)
    asm.on_inbound(2, {"type": "system", "subtype": "init", "session_id": "s-2", "model": "m"})
    retired_dead = not asm._by_key[1].unit.is_live
    assert retired_dead, "the registry did not evict — the bound moved"

    # ...and the next event on transport 1 retires the session. The pended
    # chat MUST ship here — the root already shipped as an eviction stub, so
    # finalize will never run for it.
    asm.on_inbound(1, RESULT)
    chats = [s for s in client.spans if s.name.startswith("chat")]
    assert len(chats) == 1
    assert chats[0].capture_sources == (CaptureSource.ADAPTER,)
    assert _TIMING in _limitations(chats[0])
    assert counters.get("adapters.assembler.otel_bridge_retired") == 1

    # The receiver slot went with it: nothing left filed under the trace.
    assert receiver.take(TRACE, None) is None
    asm.on_close(1, None)
    asm.on_close(2, None)


def test_a_multi_turn_agentic_loop_joins_each_llm_request_uniquely(receiver):
    """The live-CLI regression, pinned: chats of one agentic loop share ONE
    host write, so their raw turn starts collide — sequencing each window
    from the previous same-scope chat's arrival is what lets two real
    llm_requests join two of three chats uniquely, with the duplicate-view
    chat unmerged and NO conflict siblings."""
    client = FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    _outbound(asm, 1, bridge=_binding())
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)  # chat 1 (text/tool_use view of response 1)
    asm.on_inbound(1, ASSISTANT_2)  # chat 2 — SAME turn, no host write between
    asm.on_inbound(1, TOOL_RESULT)
    asm.on_inbound(1, ASSISTANT_2)  # chat 3 — the post-tool response
    asm.on_inbound(1, RESULT)

    # The REAL recorded windows (their raw starts all collide on the one host
    # write — the exact shape the live CLI produced). Each llm_request starts
    # just before its own chat's arrival, as a real one does.
    raw_windows = [rec.window for rec in asm._by_key[1].pending if rec.kind == "chat"]
    assert len(raw_windows) == 3
    assert raw_windows[0][0] == raw_windows[1][0] == raw_windows[2][0]  # collision
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id="0a" * 8,
                start_ns=raw_windows[0][1] - 1_000,  # strictly inside chat 1's window
                end_ns=raw_windows[0][1],
                attrs={"gen_ai.response.id": "req_1"},
            ),
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id="0b" * 8,
                start_ns=raw_windows[2][1] - 1_000,  # strictly inside SEQUENCED chat 3
                end_ns=raw_windows[2][1],
                attrs={"gen_ai.response.id": "req_2"},
            ),
        ]
    )
    assert _post(receiver, body) == 200
    asm.on_close(1, None)

    chats = [s for s in client.spans if s.name.startswith("chat")]
    assert len(chats) == 3
    merged = [s for s in chats if "otel_bridge" in {m.value for m in s.capture_sources}]
    request_ids = sorted(
        dict(s.extra).get("wardex.anthropic_agent_sdk.otel.request_id") for s in merged
    )
    assert request_ids == ["req_1", "req_2"]
    for s in merged:
        assert _TIMING not in _limitations(s)
    unmerged = [s for s in chats if s not in merged]
    assert len(unmerged) == 1 and _TIMING in _limitations(unmerged[0])
    assert not any(s.name == "execute_step llm_request" for s in client.spans)
