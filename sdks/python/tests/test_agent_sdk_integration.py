"""End-to-end integration through claude_agent_sdk.query with a FakeTransport."""

import anyio
import claude_agent_sdk

from test_agent_sdk_adapter_install import (
    ASSISTANT_LINE,
    INIT_LINE,
    RESULT_LINE,
    FakeTransport,
)
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureSource, StatusCode
from wardex_sdk.adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter
from wardex_sdk.assembly import Limitation
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes = []

    def export(self, envelope):
        self.envelopes.append(envelope)


class FakeClient:
    def __init__(self):
        self.spans = []

    def capture_span(self, span):
        self.spans.append(span)


def _run_query(adapter_client, script):
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(adapter_client)
    received = []
    try:

        async def main():
            fake = FakeTransport(script)
            async for msg in claude_agent_sdk.query(prompt="2+2?", transport=fake):
                received.append(msg)

        anyio.run(main)
    finally:
        adapter.uninstall()
    return received


def test_query_produces_span_tree():
    client = FakeClient()
    received = _run_query(client, [INIT_LINE, ASSISTANT_LINE, RESULT_LINE])
    assert len(received) == 3  # pass-through intact
    names = sorted(s.name for s in client.spans)
    assert "chat claude-sonnet-5" in names
    assert "invoke_agent" in names
    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.status is StatusCode.OK
    assert CaptureSource.ADAPTER in root.capture_sources


def test_transport_death_marks_session_aborted():
    class DyingTransport(FakeTransport):
        def read_messages(self):
            async def gen():
                yield INIT_LINE
                raise RuntimeError("process died")

            return gen()

    client = FakeClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client)
    try:

        async def main():
            async for _ in claude_agent_sdk.query(prompt="x", transport=DyingTransport([])):
                pass

        try:
            anyio.run(main)
        except Exception:
            pass  # the user-facing error still propagates — we only observe
    finally:
        adapter.uninstall()
    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.status is StatusCode.ERROR
    assert Limitation.SESSION_ABORTED in root.capture_integrity.limitations


def test_parallel_sessions_do_not_cross():
    client = FakeClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client)
    try:
        init2 = dict(INIT_LINE, session_id="s-2")
        asst2 = {**ASSISTANT_LINE, "session_id": "s-2"}
        result2 = dict(RESULT_LINE, session_id="s-2")

        async def main():
            async def one(script):
                async for _ in claude_agent_sdk.query(prompt="x", transport=FakeTransport(script)):
                    pass

            async with anyio.create_task_group() as tg:
                tg.start_soon(one, [INIT_LINE, ASSISTANT_LINE, RESULT_LINE])
                tg.start_soon(one, [init2, asst2, result2])

        anyio.run(main)
    finally:
        adapter.uninstall()
    roots = [s for s in client.spans if s.name == "invoke_agent"]
    assert len(roots) == 2
    assert {r.conversation.session_id for r in roots} == {"s-1", "s-2"}
    chats = [s for s in client.spans if s.name.startswith("chat")]
    by_trace = {r.context.trace_id: r.conversation.session_id for r in roots}
    for c in chats:
        assert c.context.trace_id in by_trace  # each chat belongs to exactly one root's trace


def test_repeated_sdk_tool_registration_does_not_double_wrap():
    """The regression: a host that builds fresh options per query while reusing
    its `@tool` definitions. Calling the patched `create_sdk_mcp_server` twice
    with the same `SdkMcpTool` object must not nest the wrapper, which would emit
    duplicate `execute_tool` spans for a single handler invocation.

    `install(client)` and not `install(None)`, and that is a real change to what
    this test sets up. The tool wrapper used to reach the hub directly through
    `_tracing.span()` — the one span this adapter emitted that did NOT go through
    the client it was installed with — so the test could install with None and
    still see the span. Now every span the adapter produces, this one included,
    is handed to the adapter's own client, which is what `install_configured_
    adapters` always passes in production. The regression the test defends is
    untouched: one handler invocation, exactly one span.
    """
    _hub.reset_for_test()
    t = _Recording()
    client = Client(WardexConfig(api_key="k"), t)
    _hub.set_client(client)

    async def handler(args):
        return {"content": [{"type": "text", "text": "ok"}]}

    tool_def = claude_agent_sdk.SdkMcpTool(
        name="greet", description="d", input_schema={}, handler=handler
    )

    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client)
    try:
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool_def])
        first_handler = tool_def.handler
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool_def])
        second_handler = tool_def.handler
        # No re-wrap on the second registration: same handler object.
        assert first_handler is second_handler

        anyio.run(first_handler, {"x": 1})
    finally:
        adapter.uninstall()

    _hub.get_client().flush()
    spans = [s for s in t.envelopes[0].spans if s.name == "execute_tool greet"]
    assert len(spans) == 1
