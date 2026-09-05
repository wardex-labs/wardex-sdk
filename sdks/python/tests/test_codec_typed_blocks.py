"""Round-trip tests for the typed blocks' flattening (agent, tool,
conversation, evaluation)."""

import pytest

from test_codec import _env, _span  # reuse the existing envelope/span helpers
from wardex_sdk._enums import AgentType, ToolExecutionType, ToolType
from wardex_sdk._types import (
    AgentAttributes,
    ConversationContext,
    EvaluationAttributes,
    ToolAttributes,
)
from wardex_sdk.transport import _codec


def _extra_dict(span: dict) -> dict:
    return {kv["key"]: kv["value"] for kv in span.get("extra", [])}


def test_agent_attributes_flatten_to_extra():
    agent = AgentAttributes(
        name="researcher",
        id="agent-1",
        description="finds things",
        agent_type=AgentType.SUB_AGENT,
        parent_agent="root",
    )
    out = _codec.decode(_codec.encode(_env(_span(agent=agent))))
    extra = _extra_dict(out["items"][0]["span"])
    assert extra["gen_ai.agent.name"] == "researcher"
    assert extra["gen_ai.agent.id"] == "agent-1"
    assert extra["wardex.agent.type"] == "sub_agent"
    assert extra["wardex.agent.parent"] == "root"


def test_tool_attributes_flatten_to_extra():
    tool = ToolAttributes(
        name="Bash",
        call_id="toolu_01",
        type=ToolType.FUNCTION,
        execution_type=ToolExecutionType.IN_PROCESS,
    )
    out = _codec.decode(_codec.encode(_env(_span(tool=tool))))
    extra = _extra_dict(out["items"][0]["span"])
    assert extra["gen_ai.tool.name"] == "Bash"
    assert extra["gen_ai.tool.call.id"] == "toolu_01"
    assert extra["wardex.tool.execution_type"] == "in_process"


def test_conversation_context_flattens_to_extra():
    """The id every span of a conversation is grouped by, and the two
    wardex-side fields only when they carry something: a `turn_index` of 0
    is the dataclass default, not a fact."""
    conv = ConversationContext(conversation_id="conv-123")
    out = _codec.decode(_codec.encode(_env(_span(conversation=conv))))
    extra = _extra_dict(out["items"][0]["span"])
    assert extra["gen_ai.conversation.id"] == "conv-123"
    assert not any(k.startswith("wardex.conversation.") for k in extra)

    full = ConversationContext(conversation_id="conv-123", session_id="sess-1", turn_index=3)
    out = _codec.decode(_codec.encode(_env(_span(conversation=full))))
    extra = _extra_dict(out["items"][0]["span"])
    assert extra["wardex.conversation.session_id"] == "sess-1"
    assert extra["wardex.conversation.turn_index"] == 3


def test_evaluation_attributes_flatten_to_extra():
    ev = EvaluationAttributes(
        name="block_input", explanation="why", score_value=0.25, score_label="tripwire"
    )
    out = _codec.decode(_codec.encode(_env(_span(evaluation=ev))))
    extra = _extra_dict(out["items"][0]["span"])
    assert extra["gen_ai.evaluation.name"] == "block_input"
    assert extra["gen_ai.evaluation.explanation"] == "why"
    assert extra["gen_ai.evaluation.score.value"] == 0.25
    assert extra["gen_ai.evaluation.score.label"] == "tripwire"

    partial = EvaluationAttributes(name="pass_only", score_label="pass")
    out = _codec.decode(_codec.encode(_env(_span(evaluation=partial))))
    extra = _extra_dict(out["items"][0]["span"])
    assert set(k for k in extra if k.startswith("gen_ai.evaluation.")) == {
        "gen_ai.evaluation.name",
        "gen_ai.evaluation.score.label",
    }


def test_absent_blocks_add_no_keys():
    out = _codec.decode(_codec.encode(_env(_span())))
    extra = _extra_dict(out["items"][0]["span"])
    assert not any(
        k.startswith(
            (
                "gen_ai.agent.",
                "gen_ai.tool.",
                "gen_ai.conversation.",
                "gen_ai.evaluation.",
                "wardex.agent.",
                "wardex.tool.",
                "wardex.conversation.",
            )
        )
        for k in extra
    )


# --------------------------------------------------------------------------
# a wrong-typed value in a typed block costs one attribute, never the batch
# --------------------------------------------------------------------------


def _unchecked(cls, **fields):  # noqa: ANN001, ANN202
    """The dataclass with its constructor coercion BYPASSED — the shape a
    caller reaches by any route that skips `__init__`, and the one the
    marshaller has to survive on its own."""
    obj = object.__new__(cls)
    for key, value in fields.items():
        object.__setattr__(obj, key, value)
    return obj


def test_a_uuid_conversation_id_and_a_none_turn_index_export_cleanly():
    """The constructor: a UUID id is its text, `turn_index=None` is the
    default. The marshaller, with the constructor bypassed: the same two
    values still leave as one attribute and no error."""
    import uuid

    ident = uuid.uuid4()
    conv = ConversationContext(conversation_id=ident, turn_index=None)
    assert conv.conversation_id == str(ident) and conv.turn_index == 0
    out = _codec.decode(_codec.encode(_env(_span(conversation=conv))))
    assert _extra_dict(out["items"][0]["span"])["gen_ai.conversation.id"] == str(ident)

    raw = _unchecked(ConversationContext, conversation_id=ident, session_id=None, turn_index=None)
    out = _codec.decode(_codec.encode(_env(_span(conversation=raw))))
    extra = _extra_dict(out["items"][0]["span"])
    assert extra["gen_ai.conversation.id"] == str(ident)
    assert "wardex.conversation.turn_index" not in extra
    assert "wardex.codec.unmarshalled" not in extra

    with pytest.raises(TypeError):
        ConversationContext(conversation_id="c", turn_index="three")


def test_a_score_value_the_marshaller_cannot_read_is_named_not_fatal():
    """`"0.9"` is a score, by `float()`'s rule at the constructor and by the
    marshaller's when the constructor was bypassed; `"high"` is a
    `ValueError` at the host's line, and — bypassed — an omitted attribute
    NAMED under `wardex.codec.unmarshalled`, with the rest of the block and
    the span intact."""
    ev = EvaluationAttributes(name="judge", score_value="0.9")
    assert ev.score_value == 0.9
    with pytest.raises(ValueError):
        EvaluationAttributes(name="judge", score_value="high")

    raw = _unchecked(
        EvaluationAttributes, name="judge", explanation=None, score_value="0.9", score_label=None
    )
    out = _codec.decode(_codec.encode(_env(_span(evaluation=raw))))
    assert _extra_dict(out["items"][0]["span"])["gen_ai.evaluation.score.value"] == 0.9

    bad = _unchecked(
        EvaluationAttributes, name="judge", explanation=None, score_value="high", score_label="x"
    )
    out = _codec.decode(_codec.encode(_env(_span(evaluation=bad))))
    extra = _extra_dict(out["items"][0]["span"])
    assert "gen_ai.evaluation.score.value" not in extra
    assert extra["gen_ai.evaluation.name"] == "judge"
    assert extra["gen_ai.evaluation.score.label"] == "x"
    assert extra["wardex.codec.unmarshalled"] == "gen_ai.evaluation.score.value"


@pytest.mark.usefixtures("fresh_counters")
def test_a_batch_with_one_unmarshallable_span_ships_the_other_spans():
    """The EXPORT path. One span whose tool block holds an int for a name
    cannot be marshalled at all; before, that raise reached the client's
    drain and the whole batch was dropped in silence. Now the two good spans
    leave, the bad one is counted, and one line names it."""
    from wardex_sdk import _wardex_native
    from wardex_sdk._assembly import counters
    from wardex_sdk._types import Envelope
    from wardex_sdk.testing import RecordingTransport

    good_a = _span(name="good-a")
    bad = _span(name="bad", tool=ToolAttributes(name=123))  # type: ignore[arg-type]
    good_b = _span(name="good-b")
    env = Envelope(header=_env(good_a).header, spans=(good_a, bad, good_b))
    bodies = RecordingTransport().encode(env, compress=False)
    names = [
        sp["name"]
        for body in bodies
        for rs in _wardex_native.codec.decode_otlp_traces(body)["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
    ]
    assert names == ["good-a", "good-b"]
    assert counters.get("transport.otlp.span_unmarshalled") == 1
