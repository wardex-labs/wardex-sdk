"""Vocabulary gaps the SDK deliberately does NOT close, pinned as live assertions.

A deferral that lives only in a comment is a deferral the next reader has no way
to learn about, and a deferral nothing asserts is one a later change can close —
or widen — without anybody noticing. Each test below states a known-wrong
behaviour, names what has to land before it can be fixed, and fails the moment
that behaviour changes. That is the point: the fix is *forced* to come here and
rewrite the assertion, so closing the gap is a visible edit rather than a silent
one.

None of these is a bug report. Each is a decision, recorded where a decision can
be checked.
"""

from __future__ import annotations

import json

import pytest

from wardex_sdk import _hub
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureMode, SpanKind, StatusCode
from wardex_sdk._tracing import span as manual_span
from wardex_sdk._types import (
    EnvelopeHeader,
    InternalEnvelope,
    InternalSpan,
    InternalSpanLink,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.adapters._assembler import SessionAssembler
from wardex_sdk.transport import _codec


class _FakeClient:
    def __init__(self) -> None:
        self.config = WardexConfig(api_key="k", capture_mode=CaptureMode.ALL, debug=True)
        self.spans: list = []
        self.snapshots: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def capture_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self, timeout: float = 5.0) -> None:
        """conftest's autouse fixture closes whatever the hub holds."""


@pytest.fixture
def client() -> _FakeClient:
    _hub.reset_for_test()
    c = _FakeClient()
    _hub.set_client(c)
    return c


# --------------------------------------------------------------------------
# 1. `gen_ai.operation.name` is NOT closed on the manual path — retired when the
#    published `SpanBuilder.operation` setter can stop accepting a bare `str`.
# --------------------------------------------------------------------------


def test_a_manual_span_may_still_name_an_operation_outside_the_vocabulary(client):
    """§6.2 declares the operation axis CLOSED. On the manual path it is not.

    `SpanBuilder.operation` is published API typed `OperationName | str | None`,
    and `SpanDraft.set_operation_label` takes whatever it is handed — so a host
    can put an arbitrary string on `gen_ai.operation.name` and give a dashboard
    grouping by that key unbounded cardinality on the one axis §6.2 says is
    bounded.

    Not closed, and the reason is not oversight: closing it changes what a
    published setter accepts, and §6.5 tier 1's declaration mechanism
    (`FRAMEWORK_EXTRAS`, where the unmapped original would go) does not exist
    yet. Rejecting the string before there is anywhere to put it would delete
    the user's span to enforce a namespace wardex has not yet given them a way
    to declare — the same trade `_check_extra` already refuses to make for
    `set_attribute`.

    Whatever lands `FRAMEWORK_EXTRAS` rewrites this test.
    """
    with manual_span("custom") as s:
        s.operation = "my_custom_operation"

    (sp,) = client.spans
    assert ("gen_ai.operation.name", "my_custom_operation") in sp.extra


# --------------------------------------------------------------------------
# 2. The adapter root ships the MODEL ID as the agent name — retired by the
#    Anthropic adapter rewrite.
# --------------------------------------------------------------------------


def test_the_adapter_root_still_reports_the_model_as_the_agent_name(client):
    """§6.3: `gen_ai.agent.name` is not the model id — the model belongs in
    `gen_ai.request.model`.

    `_build_root` ships `AgentAttributes(name=sess.model)`, so a dashboard
    grouping `invoke_agent` spans by agent name renders `claude-sonnet-5` as an
    agent, and a session whose model changes mid-run renders as two agents.

    Left as-is because the extraction that moved these sites onto `assembly/`
    deliberately kept every span FIELD identical, and this is a field a
    dashboard groups by: changing it silently re-partitions existing charts.
    It rides the adapter rewrite with the rest of the Anthropic
    semantics, where the root also gains the `gen_ai` block the model belongs in
    — moving the value with nowhere to move it TO would just lose it.
    """
    asm = SessionAssembler(client)
    asm.on_outbound(
        1, json.dumps({"type": "user", "session_id": "s-1", "message": {"content": "hi"}})
    )
    asm.on_inbound(
        1,
        {
            "type": "system",
            "subtype": "init",
            "session_id": "s-1",
            "model": "claude-sonnet-5",
        },
    )
    asm.on_close(1, None)

    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.agent is not None
    assert root.agent.name == "claude-sonnet-5"  # the known §6.3 violation
    assert root.gen_ai is None  # ...and nowhere to move it to yet


# --------------------------------------------------------------------------
# 3. The ENCODE direction of `LinkReason` still flattens an unmapped string —
#    retired by giving `LinkReason` the unmapped value `Limitation` already has
#    (design R11).
# --------------------------------------------------------------------------


def _envelope(span: InternalSpan) -> InternalEnvelope:
    return InternalEnvelope(
        header=EnvelopeHeader(
            event_id="evt-1",
            api_key="k",
            sdk=SdkInfo(
                name="wardex.python",
                version="0.1.0",
                python_version="3.12",
                os="mac",
                arch="arm64",
            ),
            sent_at_ns=42,
        ),
        spans=(span,),
    )


def test_an_unmapped_link_reason_is_still_dropped_on_encode():
    """The decode direction distinguishes "unset" from "unknown"; encode does not.

    `map_link_reason` returns UNSPECIFIED for a string it does not know, with no
    breadcrumb — so a reason that came from outside the SDK arrives on the wire
    indistinguishable from no reason at all. Design R11 names the fix, and
    `Limitation` already has it: an unrecognized marker becomes
    `LIMITATION_VOCABULARY_UNMAPPED` with the original preserved in
    `extra["wardex.limitation.unmapped"]`. `LinkReason` has no such value in
    `common.proto`, so adding one is a further schema change.

    The hole is bounded meanwhile, and this test says how: only a caller reaching
    past `SpanDraft.add_link` (which takes the enum) can produce one, i.e. a
    third party constructing `InternalSpanLink` by hand — which is what this test
    has to do to reach it at all.
    """
    span = InternalSpan(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="execute_step planner",
        kind=SpanKind.INTERNAL,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
        links=(
            InternalSpanLink(
                trace_id=TraceId(b"\xaa" * 16),
                span_id=SpanId(b"\xbb" * 8),
                reason="my_custom_reason",
            ),
        ),
    )

    decoded = _codec.decode(_codec.encode(_envelope(span)))["items"][0]["span"]

    (link,) = decoded["links"]
    assert link["reason"] == ""  # the string is gone, and says nothing about itself
