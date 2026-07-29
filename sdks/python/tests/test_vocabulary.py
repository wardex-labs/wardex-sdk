"""The closed span vocabulary — design §6.1-§6.4, §6.6, and `SpanDraft.finish()`.

Two things are asserted here and they belong together:

  * the RULES `finish()` enforces. Each one exists because the shipped tree
    violated it somewhere, so each test names the span it would have caught.
  * the AGREEMENT between the Python enums and the proto enums. §6.6's whole
    argument is that a multi-language SDK whose vocabulary lives only in Python
    will have Node and Java re-derive it from prose and drift; `common.proto`
    is the source of truth, and `_wardex_native.codec.vocabulary_tables()`
    reads the generated proto enums back so "they agree" is a build failure
    rather than a claim.
"""

from __future__ import annotations

import pytest

from wardex_sdk import _wardex_native
from wardex_sdk._enums import (
    CaptureSource,
    OperationName,
    SnapshotType,
    SpanKind,
    StatusCode,
    ToolExecutionType,
)
from wardex_sdk._types import (
    AgentAttributes,
    ConversationContext,
    GenAIAttributes,
    SpanContext,
    SpanId,
    ToolAttributes,
    TraceId,
)
from wardex_sdk.assembly import (
    AMBIENT,
    Ambient,
    LinkReason,
    SpanDraft,
    SpanIntent,
    TransportLabel,
    VocabularyError,
    resolve_parentage,
    vocabulary_name,
)


def _parentage(parent: SpanContext | None = None):
    return resolve_parentage(Ambient(parent, None, None), AMBIENT)


def _draft(intent=SpanIntent.CHAT, subject="gpt-4o", **kw):
    return SpanDraft(
        _parentage(),
        intent=intent,
        subject=subject,
        source=CaptureSource.ADAPTER,
        start_ns=1,
        **kw,
    )


def _gen_ai():
    return GenAIAttributes(operation=OperationName.CHAT, request_model="gpt-4o")


# --------------------------------------------------------------------------
# §6.1 — the name grammar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intent", "subject", "expected"),
    [
        (SpanIntent.CHAT, "gpt-4o", "chat gpt-4o"),
        (SpanIntent.INVOKE_AGENT, None, "invoke_agent"),
        (SpanIntent.EXECUTE_TOOL, "Bash", "execute_tool Bash"),
        (SpanIntent.EXECUTE_STEP, "  planner  ", "execute_step planner"),
    ],
)
def test_the_name_is_operation_plus_subject(intent, subject, expected):
    assert vocabulary_name(intent, subject) == expected


@pytest.mark.parametrize("subject", [None, "", "   ", "None"])
def test_a_missing_subject_yields_the_bare_operation(subject):
    """`"chat None"` was a real span name in a shipped adapter.

    It came from `f"chat {ev.model}"` on a turn whose stream never reported a
    model. The literal string "None" is in this list because that is exactly
    what interpolation produces, and a grammar that only handled `None` the
    object would still let the bug through from any site that pre-formatted.
    """
    assert vocabulary_name(SpanIntent.CHAT, subject) == "chat"

    draft = _draft(subject=subject)
    draft.set_gen_ai(_gen_ai())
    assert draft.finish(2).name == "chat"


def test_a_transport_observation_keeps_a_closed_label():
    draft = SpanDraft.transport(
        _parentage(),
        label=TransportLabel.HTTP,
        subject="POST /v1/messages",
        source=CaptureSource.SSL,
        start_ns=1,
    )
    assert draft.finish(2).name == "HTTP POST /v1/messages"


def test_relabel_is_transport_only():
    """A VOCABULARY draft whose intent could change later would make the
    required-block check meaningless."""
    draft = _draft()
    with pytest.raises(VocabularyError):
        draft.relabel(TransportLabel.GRPC, "/pkg.Svc/Do")


# --------------------------------------------------------------------------
# §6.2 / §6.3 — the structural rules `finish()` enforces
# --------------------------------------------------------------------------


def test_an_llm_intent_without_a_gen_ai_block_is_rejected():
    with pytest.raises(VocabularyError, match="gen_ai"):
        _draft().finish(2)


def test_execute_tool_without_a_tool_block_is_rejected():
    with pytest.raises(VocabularyError, match="tool"):
        _draft(intent=SpanIntent.EXECUTE_TOOL, subject="Bash").finish(2)


def test_invoke_agent_without_an_agent_block_is_rejected():
    with pytest.raises(VocabularyError, match="agent"):
        _draft(intent=SpanIntent.INVOKE_AGENT, subject=None).finish(2)


def test_execute_step_requires_its_namespaced_key():
    draft = _draft(intent=SpanIntent.EXECUTE_STEP, subject="planner")
    with pytest.raises(VocabularyError, match="wardex.step.name"):
        draft.finish(2)
    draft.set_extra("wardex.step.name", "planner")
    assert draft.finish(2).name == "execute_step planner"


def test_status_error_requires_an_error_type():
    """The untyped-failure defect, as a mechanism rather than a review note.

    The adapter shipped `is_error=true` spans with no `error.type` while the
    hook payload that carries the reason sat unread.
    """
    draft = _draft(intent=SpanIntent.EXECUTE_TOOL, subject="Bash")
    draft.set_tool(ToolAttributes(name="Bash"))
    draft.set_status(StatusCode.ERROR)
    with pytest.raises(VocabularyError, match="error_type"):
        draft.finish(2)
    draft.set_error("tool_error")
    assert draft.finish(2).error_type == "tool_error"


def test_an_empty_conversation_id_is_rejected():
    """§6.3: `""` collides across every session in any store that keys on it."""
    draft = _draft()
    draft.set_gen_ai(_gen_ai())
    draft.set_conversation(ConversationContext(conversation_id=""))
    with pytest.raises(VocabularyError, match="conversation_id"):
        draft.finish(2)


def test_an_undeclared_extra_key_is_rejected():
    draft = _draft()
    draft.set_gen_ai(_gen_ai())
    draft.set_extra("random.vendor.thing", 1)
    with pytest.raises(VocabularyError, match="undeclared"):
        draft.finish(2)


def test_a_manual_span_may_carry_any_extra_key():
    """`SpanBuilder.set_attribute` is published and has always taken any key.

    Rejecting one would delete the user's span to enforce a namespace wardex
    has not yet given them a way to declare (`FRAMEWORK_EXTRAS`, step 8).
    """
    draft = SpanDraft.manual(
        _parentage(), name="anything at all", kind=SpanKind.INTERNAL, start_ns=1
    )
    draft.set_extra("random.vendor.thing", 1)
    assert draft.finish(2).extra == (("random.vendor.thing", 1),)


def test_a_draft_cannot_leave_its_parentages_trace():
    draft = _draft()
    draft.set_gen_ai(_gen_ai())
    draft.context = SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8))
    with pytest.raises(VocabularyError, match="trace"):
        draft.finish(2)


def test_a_string_limitation_is_rejected():
    """Typed setters make this normally unrepresentable; `finish()` re-checks
    because the builder is reachable from duck-typed code in adapters/."""
    draft = _draft()
    draft.set_gen_ai(_gen_ai())
    draft.integrity.limitation("payload_compressed")  # type: ignore[arg-type]
    with pytest.raises(VocabularyError, match="Limitation enum"):
        draft.finish(2)


# --------------------------------------------------------------------------
# the one decision `finish()` folds together
# --------------------------------------------------------------------------


def test_the_operation_is_mirrored_into_gen_ai_and_not_double_carried():
    draft = _draft(intent=SpanIntent.CHAT, subject="gpt-4o")
    draft.set_gen_ai(GenAIAttributes(operation=OperationName.EMBEDDINGS, request_model="gpt-4o"))

    span = draft.finish(2)

    assert span.gen_ai.operation is OperationName.CHAT
    assert [k for k, _ in span.extra if k == "gen_ai.operation.name"] == []


def test_the_operation_is_stamped_into_extra_when_there_is_no_gen_ai_block():
    draft = _draft(intent=SpanIntent.INVOKE_AGENT, subject=None)
    draft.set_agent(AgentAttributes(name="researcher"))

    span = draft.finish(2)

    assert ("gen_ai.operation.name", "invoke_agent") in span.extra


def test_capture_sources_is_always_set():
    """The `_tracing.span()` path shipped spans with an empty tuple here — one
    of the three fields that made two structurally different execute_tool
    shapes (§6.4)."""
    span = SpanDraft.manual(_parentage(), name="x", kind=SpanKind.INTERNAL, start_ns=1).finish(2)
    assert span.capture_sources == (CaptureSource.MANUAL,)
    assert span.correlation is not None


def test_integrity_captured_means_attempted_not_non_empty():
    """`request_body_captured=bool(tool.input_data)` reported a tool called with
    `{}` as a capture FAILURE, on the field a dashboard reads to decide whether
    a replay is trustworthy."""
    draft = _draft(intent=SpanIntent.EXECUTE_TOOL, subject="Bash")
    draft.set_tool(ToolAttributes(name="Bash"))
    draft.set_io(input_data=b"", output_data=b"")

    integrity = draft.finish(2).capture_integrity

    assert integrity.request_body_captured is True
    assert integrity.response_body_captured is True


def test_integrity_is_absent_when_nothing_was_recorded():
    span = SpanDraft.manual(_parentage(), name="x", kind=SpanKind.INTERNAL, start_ns=1).finish(2)
    assert span.capture_integrity is None


# --------------------------------------------------------------------------
# §6.3 — links are causality
# --------------------------------------------------------------------------


def test_a_link_carries_its_reason():
    other = SpanContext(trace_id=TraceId(b"\x03" * 16), span_id=SpanId(b"\x04" * 8))
    draft = _draft()
    draft.set_gen_ai(_gen_ai())
    draft.add_link(other, LinkReason.TRIGGERED_BY)

    (link,) = draft.finish(2).links

    assert link.trace_id == other.trace_id
    assert link.span_id == other.span_id
    # The MEMBER. `== "triggered_by"` passed while the builder was storing
    # `reason.value`, which meant `link.reason is LinkReason.HANDOFF_FROM` — the
    # check a renderer needs in order to draw a sibling instead of nesting — was
    # always False.
    assert link.reason is LinkReason.TRIGGERED_BY


def test_an_event_carries_its_timestamp_and_attributes():
    draft = _draft()
    draft.set_gen_ai(_gen_ai())
    draft.add_event("cache_hit", 42, key="k")

    (event,) = draft.finish(2).events

    assert event.name == "cache_hit"
    assert event.timestamp_ns == 42
    assert event.attributes == (("key", "k"),)


# --------------------------------------------------------------------------
# §6.6 — proto is the single source of truth
# --------------------------------------------------------------------------


def _tables():
    return _wardex_native.codec.vocabulary_tables()


def test_span_intent_is_one_to_one_with_operation_name():
    assert {i.value for i in SpanIntent} == {o.value for o in OperationName}
    for intent in SpanIntent:
        assert intent.operation.value == intent.value


@pytest.mark.parametrize(
    ("table", "enum"),
    [
        ("OperationName", OperationName),
        ("ToolExecutionType", ToolExecutionType),
        ("LinkReason", LinkReason),
        ("SnapshotType", SnapshotType),
    ],
)
def test_every_python_enum_value_is_declared_in_proto(table, enum):
    """The §6.6 check: a value the Python SDK can emit that proto does not
    declare is a value Node and Java cannot represent."""
    declared = _tables()[table]

    assert {m.value for m in enum} == set(declared), (
        f"{table} disagrees between wardex_sdk._enums/assembly and "
        f"proto/wardex/v1/common.proto:\n"
        f"  python only: {sorted({m.value for m in enum} - set(declared))}\n"
        f"  proto only:  {sorted(set(declared) - {m.value for m in enum})}"
    )


@pytest.mark.parametrize(
    "table", ["OperationName", "ToolExecutionType", "LinkReason", "SnapshotType"]
)
def test_no_declared_value_maps_to_unspecified(table):
    """A value that maps to 0 is a value the codec silently loses.

    `map_*` returns UNSPECIFIED for anything it does not know, so a member the
    Rust table forgot would look identical to an absent field on the wire.
    """
    declared = _tables()[table]

    zeros = sorted(name for name, number in declared.items() if number == 0)

    assert zeros == [], f"{table} values map to UNSPECIFIED: {zeros}"


@pytest.mark.parametrize(
    "table", ["OperationName", "ToolExecutionType", "LinkReason", "SnapshotType"]
)
def test_declared_numbers_are_unique(table):
    numbers = list(_tables()[table].values())
    assert len(set(numbers)) == len(numbers)


def test_the_three_new_operations_and_two_new_execution_types_are_declared():
    """The additions §6.2 budgets, named so a later edit cannot quietly drop one."""
    ops = _tables()["OperationName"]
    assert {"execute_step", "handoff", "evaluate"} <= set(ops)
    tools = _tables()["ToolExecutionType"]
    assert {"ipc", "unknown"} <= set(tools)
