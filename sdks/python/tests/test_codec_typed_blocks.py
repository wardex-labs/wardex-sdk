"""Round-trip tests for the typed blocks' flattening (agent, tool,
conversation, evaluation)."""

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
