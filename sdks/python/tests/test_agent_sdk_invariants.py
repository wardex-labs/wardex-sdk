"""First-invariant regression: the adapter must never break the host app."""

from unittest import mock

import anyio
import claude_agent_sdk

from test_agent_sdk_adapter_install import (
    ASSISTANT_LINE,
    INIT_LINE,
    RESULT_LINE,
    FakeTransport,
)
from wardex_sdk.adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter


class FakeClient:
    def __init__(self):
        self.spans = []

    def capture_span(self, span):
        self.spans.append(span)


def _collect(script):
    received = []

    async def main():
        async for msg in claude_agent_sdk.query(prompt="x", transport=FakeTransport(script)):
            received.append(msg)

    anyio.run(main)
    return received


def test_parser_exception_does_not_break_user_stream():
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(FakeClient())
    try:
        with mock.patch(
            "wardex_sdk.adapters._assembler.parse_line",
            side_effect=RuntimeError("boom"),
        ):
            received = _collect([INIT_LINE, ASSISTANT_LINE, RESULT_LINE])
        assert len(received) == 3  # user saw everything despite total parser failure
    finally:
        adapter.uninstall()


def test_hook_callback_exception_returns_empty_output():
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(FakeClient())
    try:
        from wardex_sdk.adapters._anthropic_agent_sdk import _make_hook

        hook = _make_hook(adapter, "PreToolUse")
        with mock.patch.object(adapter, "_on_hook", side_effect=RuntimeError("boom")):
            result = anyio.run(hook, {"session_id": "s-1"}, "toolu_1", {})
        assert result == {}  # never blocks, never raises
    finally:
        adapter.uninstall()


def test_no_session_leak_after_many_runs():
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(FakeClient())
    try:
        for _ in range(20):
            _collect([INIT_LINE, ASSISTANT_LINE, RESULT_LINE])
        assert adapter._assembler.open_session_count() == 0
    finally:
        adapter.uninstall()


def test_uninstalled_adapter_leaves_sdk_untouched():
    before_query = claude_agent_sdk.query
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(FakeClient())
    adapter.uninstall()
    assert claude_agent_sdk.query is before_query
    received = _collect([INIT_LINE, RESULT_LINE])
    assert len(received) == 2
