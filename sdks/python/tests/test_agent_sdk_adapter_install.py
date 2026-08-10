"""Install/uninstall + hook-merge + transport-tee pass-through tests.

Uses the real claude-agent-sdk package (dev dependency) with a FakeTransport;
no CLI subprocess is ever spawned.
"""

import json

import anyio
import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from claude_agent_sdk._internal.transport import Transport

from wardex_sdk._adapters._anthropic_agent_sdk import (
    AnthropicAgentSdkAdapter,
    _prepare_options,
)
from wardex_sdk._adapters._registry import context_for
from wardex_sdk._assembly import Limitation

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


def test_installing_with_no_client_at_all_still_patches_and_unpatches():
    """`install(None)` — the shape the registry uses for a client-less install.

    Seam identity in both directions is a shared invariant and lives in
    `test_agent_sdk_conformance.py`; what is asserted here is the ASSEMBLER's
    own lifecycle, which is this adapter's alone. It is built by `install()`
    and dropped by `uninstall()`, and every callback into the adapter gates on
    it — so an assembler that outlived a teardown would let a read still in
    flight open a fresh root in a table nothing will ever close again.
    """
    adapter = AnthropicAgentSdkAdapter()
    assert adapter._assembler is None
    adapter.install(None)
    assert adapter._assembler is not None
    adapter.uninstall()
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
    adapter.install(client, context_for(adapter.name(), client))
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


def test_uninstall_emits_the_span_of_a_run_that_never_finished():
    """The real Ctrl-C path, end to end through the adapter.

    An interpreter exit reaches `atexit` -> `_teardown` -> this `uninstall()`,
    and until it closed the units a run interrupted mid-flight left NOTHING —
    not a truncated span, not a marked one. The `_assembler is None` assertion
    below is the interesting half: the drain has to survive the latch that the
    same method sets, because the latch has to come first.
    """

    class RecordingClient:
        def __init__(self):
            self.spans = []

        def capture_span(self, span):
            self.spans.append(span)

    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        adapter._assembler.on_outbound(
            1,
            json.dumps(
                {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
            ),
        )
        adapter._assembler.on_inbound(1, INIT_LINE)  # ...and no RESULT_LINE ever arrives
        assert adapter._assembler.open_session_count() == 1

        adapter.uninstall()

        assert adapter._assembler is None
        roots = [s for s in client.spans if s.name == "invoke_agent"]
        assert len(roots) == 1
        assert roots[0].conversation.session_id == "s-1"
        assert Limitation.ADAPTER_UNINSTALLED in roots[0].capture_integrity.limitations
    finally:
        adapter.uninstall()  # idempotent; must not emit a second root


def test_a_read_still_in_flight_cannot_reopen_a_session_during_uninstall():
    """Why the latch is set BEFORE the drain rather than after.

    The reader task can be mid-`read_messages` when teardown starts, so the
    window that matters is the one DURING the walk, not after it — after it,
    both orderings look identical and a test written that way proves nothing.
    Here the straggler is fired from inside the drain, which is where a real one
    lands.

    Every callback gates on `self._assembler is not None`, so latching first
    leaves the straggler no table to open a fresh root in. Drain first and it
    opens one behind the walk: live, in a registry nothing will close again —
    the original bug, reintroduced by the code that fixes it.
    """

    class RecordingClient:
        def __init__(self):
            self.spans = []

        def capture_span(self, span):
            self.spans.append(span)

    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    assembler = adapter._assembler
    try:
        adapter._assembler.on_outbound(
            1,
            json.dumps(
                {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
            ),
        )
        adapter._assembler.on_inbound(1, INIT_LINE)

        original = assembler._stamp_root

        def stamp_then_straggle(sess, error):
            adapter._on_outbound(
                2,
                json.dumps(
                    {
                        "type": "user",
                        "session_id": "s-2",
                        "message": {"role": "user", "content": "x"},
                    }
                ),
            )
            return original(sess, error)

        assembler._stamp_root = stamp_then_straggle
        adapter.uninstall()

        assert assembler.open_session_count() == 0, "a straggler opened a root behind the drain"
        assert [s.name for s in client.spans].count("invoke_agent") == 1
    finally:
        adapter.uninstall()
