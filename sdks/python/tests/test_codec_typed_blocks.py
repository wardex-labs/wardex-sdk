"""Round-trip tests for AgentAttributes/ToolAttributes flattening (typed blocks)."""

from test_codec import _env, _span  # reuse the existing envelope/span helpers
from wardex_sdk._enums import AgentType, ToolExecutionType, ToolType
from wardex_sdk._types import AgentAttributes, ToolAttributes
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


def test_absent_blocks_add_no_keys():
    out = _codec.decode(_codec.encode(_env(_span())))
    extra = _extra_dict(out["items"][0]["span"])
    assert not any(k.startswith(("gen_ai.agent.", "gen_ai.tool.", "wardex.agent.", "wardex.tool.")) for k in extra)
