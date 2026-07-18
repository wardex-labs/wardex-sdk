"""Install/uninstall + hook-merge + transport-tee pass-through tests.

Uses the real claude-agent-sdk package (dev dependency) with a FakeTransport;
no CLI subprocess is ever spawned.
"""

import json

import anyio
import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from claude_agent_sdk._internal.transport import Transport

from wardex_sdk.adapters._anthropic_agent_sdk import (
    AnthropicAgentSdkAdapter,
    _prepare_options,
)

RESULT_LINE = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "duration_ms": 10,
    "duration_api_ms": 5,
}
INIT_LINE = {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-sonnet-5"}
ASSISTANT_LINE = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m1",
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "content": [{"type": "text", "text": "4"}],
    },
}


class FakeTransport(Transport):
    """Replays a scripted message list; records writes.

    The real Query control protocol (claude_agent_sdk._internal.query.Query)
    performs a control_request/control_response handshake for initialize()
    before any script messages should be delivered: it writes a
    ``{"type": "control_request", "request_id": ..., "request": {"subtype":
    "initialize", ...}}`` frame and blocks on a matching control_response.
    A purely static script (as used by earlier SDK protocol versions) never
    satisfies that wait and the handshake times out after 60s. So write()
    detects the initialize control_request and read_messages() answers it
    with a synthetic control_response before replaying the script.
    """

    def __init__(self, script):
        self.script = script
        self.written = []
        self.closed = False
        self._init_seen = anyio.Event()
        self._init_request_id: str | None = None

    async def connect(self):
        pass

    async def write(self, data):
        self.written.append(data)
        try:
            msg = json.loads(data)
        except ValueError:
            return
        if (
            msg.get("type") == "control_request"
            and msg.get("request", {}).get("subtype") == "initialize"
        ):
            self._init_request_id = msg["request_id"]
            self._init_seen.set()

    def read_messages(self):
        async def gen():
            await self._init_seen.wait()
            yield {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": self._init_request_id},
            }
            for msg in self.script:
                yield msg

        return gen()

    async def close(self):
        self.closed = True

    def is_ready(self):
        return True

    async def end_input(self):
        pass


def test_install_uninstall_restores_surface():
    adapter = AnthropicAgentSdkAdapter()
    orig_query = claude_agent_sdk.query
    orig_create_sdk_mcp_server = claude_agent_sdk.create_sdk_mcp_server
    assert adapter._assembler is None
    adapter.install(None)
    assert claude_agent_sdk.query is not orig_query
    assert claude_agent_sdk.create_sdk_mcp_server is not orig_create_sdk_mcp_server
    assert adapter._assembler is not None
    adapter.uninstall()
    assert claude_agent_sdk.query is orig_query
    assert claude_agent_sdk.create_sdk_mcp_server is orig_create_sdk_mcp_server
    assert adapter._assembler is None


def test_hook_merge_preserves_user_hooks():
    async def user_hook(input_data, tool_use_id, context):
        return {}

    opts = ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[user_hook])]})
    adapter = AnthropicAgentSdkAdapter()
    merged = _prepare_options(opts, adapter)
    # user matcher survives, ours is appended
    assert merged.hooks["PreToolUse"][0].hooks[0] is user_hook
    assert len(merged.hooks["PreToolUse"]) == 2
    # other wardex events exist
    assert "SubagentStart" in merged.hooks
    # the original options object is untouched
    assert len(opts.hooks["PreToolUse"]) == 1


def test_query_passthrough_with_fake_transport():
    class RecordingClient:
        def __init__(self):
            self.spans = []

        def capture_span(self, span):
            self.spans.append(span)

    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client)
    try:
        received = []

        async def main():
            fake = FakeTransport([INIT_LINE, ASSISTANT_LINE, RESULT_LINE])
            async for msg in claude_agent_sdk.query(prompt="2+2?", transport=fake):
                received.append(msg)

        anyio.run(main)
        # user sees every message despite the tee
        assert len(received) == 3
        # the tee fed the assembler enough to close the session cleanly
        assert adapter._assembler.open_session_count() == 0
        # and to emit at least the root span plus one chat turn
        names = [s.name for s in client.spans]
        assert "invoke_agent" in names
        assert any(n.startswith("chat") for n in names)
    finally:
        adapter.uninstall()
