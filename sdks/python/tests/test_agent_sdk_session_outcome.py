"""How an Agent SDK session, and the calls in it, END — through the real SDK.

Every test here drives the real `claude_agent_sdk` (its `Query` control loop,
its `ClaudeSDKClient`, its hook dispatch) against `LiveCli`, a scripted stand-in
for the `claude` subprocess. What makes it a stand-in for a LIVE one is the part
the older `FakeTransport` lacks: after its script it stays open, the way the CLI
does between turns, so `ClaudeSDKClient`'s `disconnect()` has to cancel the
reader task to end the session. That cancellation is the whole first defect.

The rules under test, in the words a dashboard user would use:

  * a session the CLI reported as finished reads OK, however the host closed it;
  * a session closed before the CLI answered the turn in flight never reads OK;
  * a tool call whose closing hook never came says what the CLI's own result
    said, or UNSET when nothing reported it — never an invented failure;
  * a stream line nothing in the assembler has a rule for is counted.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import anyio
import claude_agent_sdk
import pytest
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from claude_agent_sdk._internal.transport import Transport

from wardex_sdk._adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter
from wardex_sdk._adapters._registry import context_for
from wardex_sdk._adapters._session_outcome import (
    IGNORED_STREAM_TYPES,
    PARSED_STREAM_TYPES,
    reader_stopped,
)
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import CaptureSource, StatusCode
from wardex_sdk._protocol._claude_stream import parse_line

pytestmark = pytest.mark.usefixtures("fresh_counters")

SESSION = "s-live"
INIT = {"type": "system", "subtype": "init", "session_id": SESSION, "model": "claude-sonnet-5"}


def assistant(content, msg_id="m1", stop_reason="end_turn"):
    return {
        "type": "assistant",
        "session_id": SESSION,
        "parent_tool_use_id": None,
        "message": {
            "id": msg_id,
            "model": "claude-sonnet-5",
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "content": content,
        },
    }


def tool_use(call_id, name="mcp__srv__lookup"):
    return assistant(
        [{"type": "tool_use", "id": call_id, "name": name, "input": {"key": "k"}}],
        msg_id=f"m-{call_id}",
        stop_reason="tool_use",
    )


def tool_result(call_id, text, *, is_error=False):
    block = {"type": "tool_result", "tool_use_id": call_id, "content": text}
    if is_error:
        block["is_error"] = True
    return {
        "type": "user",
        "session_id": SESSION,
        "parent_tool_use_id": None,
        "message": {"role": "user", "content": [block]},
    }


def result(*, is_error=False):
    return {
        "type": "result",
        "subtype": "error_during_execution" if is_error else "success",
        "session_id": SESSION,
        "is_error": is_error,
        "num_turns": 1,
        "duration_ms": 10,
        "duration_api_ms": 5,
    }


def pre_tool_use(call_id, name="mcp__srv__lookup"):
    """A `PreToolUse` hook the CLI sends, as the SDK will hand it to the hook."""
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": SESSION,
        "tool_name": name,
        "tool_input": {"key": "k"},
        "tool_use_id": call_id,
    }
    return ("hook", "PreToolUse", call_id, payload)


class LiveCli(Transport):
    """A scripted `claude` subprocess that stays open between turns.

    Answers the SDK's `initialize`, plays one scripted turn per user message
    the host writes, and asks hooks the way the CLI does: a `hook_callback`
    control request, then nothing further until the SDK has answered it — so a
    hook's observation lands before the stream line that follows it, exactly
    the order the real CLI enforces by blocking on the answer. After the last
    turn it waits for input that never comes, which is what makes the SDK end
    the session by cancelling its reader.
    """

    def __init__(self, *turns, fail_after=None):
        self.turns = list(turns)
        self.fail_after = fail_after
        self._send, self._recv = anyio.create_memory_object_stream(100)
        self._hook_ids: dict[str, str] = {}
        self._answered: dict[str, anyio.Event] = {}
        self._seq = 0

    async def connect(self):
        pass

    def is_ready(self):
        return True

    async def end_input(self):
        # `query()` closes stdin once the run is over, and the CLI exits on EOF.
        self._send.close()

    async def close(self):
        self._send.close()

    async def write(self, data):
        msg = json.loads(data)
        kind = msg.get("type")
        if kind == "control_request" and msg["request"].get("subtype") == "initialize":
            # wardex's own matcher is appended LAST, after any of the host's.
            for event, matchers in (msg["request"].get("hooks") or {}).items():
                self._hook_ids[event] = matchers[-1]["hookCallbackIds"][-1]
            reply = {"subtype": "success", "request_id": msg["request_id"], "response": {}}
            await self._send.send([{"type": "control_response", "response": reply}])
        elif kind == "control_response":
            event = self._answered.get(msg["response"].get("request_id"))
            if event is not None:
                event.set()
        elif kind == "user" and self.turns:
            await self._send.send(self.turns.pop(0))

    def read_messages(self):
        async def gen():
            async for batch in self._recv:
                for step in batch:
                    if isinstance(step, tuple):
                        _, event, call_id, payload = step
                        self._seq += 1
                        request_id = f"cli-{self._seq}"
                        answered = self._answered[request_id] = anyio.Event()
                        yield {
                            "type": "control_request",
                            "request_id": request_id,
                            "request": {
                                "subtype": "hook_callback",
                                "callback_id": self._hook_ids[event],
                                "input": payload,
                                "tool_use_id": call_id,
                            },
                        }
                        await answered.wait()
                    else:
                        yield step
                        if self.fail_after is not None and step is self.fail_after:
                            raise RuntimeError("the CLI process died")

        return gen()


class _Spans:
    config = None

    def __init__(self):
        self.spans = []

    def capture_span(self, span, *, scope=None):
        self.spans.append(span)

    def root(self):
        (root,) = [s for s in self.spans if s.name == "invoke_agent"]
        return root

    def tools(self):
        return [s for s in self.spans if s.name.startswith("execute_tool")]


def _installed():
    client = _Spans()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    return client, adapter


def _client_session(cli, prompts, *, leave_turn=None):
    """Run `prompts` through `async with ClaudeSDKClient`, the documented way.

    `leave_turn` is the index of a turn the host abandons at its first
    assistant message, before the result — the shape of a user pressing stop.
    """
    client, adapter = _installed()

    async def main():
        async with ClaudeSDKClient(options=ClaudeAgentOptions(), transport=cli) as sdk:
            for turn, prompt in enumerate(prompts):
                await sdk.query(prompt)
                async for message in sdk.receive_response():
                    if turn == leave_turn and isinstance(message, AssistantMessage):
                        break
                    if isinstance(message, ResultMessage):
                        break

    try:
        anyio.run(main)
    finally:
        adapter.uninstall()
    return client


def _query_session(cli):
    client, adapter = _installed()

    async def main():
        async for _ in claude_agent_sdk.query(prompt="go", transport=cli):
            pass

    try:
        anyio.run(main)
    finally:
        adapter.uninstall()
    return client


def _limits(span):
    return set(span.capture_integrity.limitations)


# --------------------------------------------------------------------------
# 1. The root: OK only when the CLI answered the turn in flight
# --------------------------------------------------------------------------


def test_a_client_session_closed_after_its_result_is_ok():
    """`async with ClaudeSDKClient`, one turn, the CLI says success, the host
    leaves the block. That close cancels the SDK's reader task, and wardex used
    to record the cancellation as a transport error: ERROR, `session_error`,
    `session_aborted` on a run whose answer was right."""
    cli = LiveCli([INIT, assistant([{"type": "text", "text": "4"}]), result()])
    root = _client_session(cli, ["2+2?"]).root()

    assert root.status is StatusCode.OK
    assert root.error_type is None
    assert Limitation.SESSION_ABORTED not in _limits(root)


def test_the_clis_own_error_result_still_reads_error_after_a_clean_close():
    """Cancelling the reader removes the invented error, not the CLI's own."""
    cli = LiveCli([INIT, assistant([{"type": "text", "text": "no"}]), result(is_error=True)])
    root = _client_session(cli, ["2+2?"]).root()

    assert root.status is StatusCode.ERROR
    assert root.error_type == "agent_error"
    assert Limitation.SESSION_ABORTED not in _limits(root)


def test_a_client_session_left_before_its_result_is_unset_and_aborted():
    """The counterexample the fix must not create: the host leaves mid-turn,
    the same cancellation ends the reader, and no result ever arrived. Nobody
    observed how the run ended, so it is neither OK nor an invented ERROR."""
    cli = LiveCli([INIT, assistant([{"type": "text", "text": "thinking"}])])
    root = _client_session(cli, ["2+2?"], leave_turn=0).root()

    assert root.status is StatusCode.UNSET
    assert root.error_type is None
    assert Limitation.SESSION_ABORTED in _limits(root)


def test_a_second_turn_abandoned_after_the_first_finished_is_not_ok():
    """Turn one's result is in hand when turn two is cut short. Reading the
    root off "a result arrived" would call this session a success; the result
    must belong to the turn that was in flight when the session ended."""
    cli = LiveCli(
        [INIT, assistant([{"type": "text", "text": "4"}]), result()],
        [assistant([{"type": "text", "text": "still working"}], msg_id="m2")],
    )
    root = _client_session(cli, ["2+2?", "and 3+3?"], leave_turn=1).root()

    assert root.status is StatusCode.UNSET
    assert Limitation.SESSION_ABORTED in _limits(root)


def test_a_turn_the_cli_started_by_itself_and_left_unfinished_is_not_ok():
    """The CLI can open a turn with no write from the host: a background
    agent that finished after the result wakes the session. The host leaving
    during that turn cuts it short exactly as leaving a turn it asked for
    would, and the earlier result must not vouch for it."""
    cli = LiveCli(
        [
            INIT,
            assistant([{"type": "text", "text": "4"}]),
            result(),
            assistant([{"type": "text", "text": "a background agent finished"}], msg_id="m2"),
        ]
    )
    client, adapter = _installed()

    async def main():
        async with ClaudeSDKClient(options=ClaudeAgentOptions(), transport=cli) as sdk:
            await sdk.query("2+2?")
            answered = False
            async for message in sdk.receive_messages():
                if isinstance(message, ResultMessage):
                    answered = True
                elif answered and isinstance(message, AssistantMessage):
                    break  # the host leaves inside the turn the CLI started

    try:
        anyio.run(main)
    finally:
        adapter.uninstall()
    root = client.root()

    assert root.status is StatusCode.UNSET
    assert Limitation.SESSION_ABORTED in _limits(root)


def test_a_transport_that_really_fails_after_the_result_stays_error():
    """A dead CLI is still an error — the cancel rule must not absorb it."""
    last = result()
    cli = LiveCli([INIT, assistant([{"type": "text", "text": "4"}]), last], fail_after=last)
    client, adapter = _installed()

    async def main():
        async with ClaudeSDKClient(options=ClaudeAgentOptions(), transport=cli) as sdk:
            await sdk.query("2+2?")
            async for _ in sdk.receive_messages():
                pass

    try:
        with pytest.raises(BaseException):  # noqa: B017 — the SDK's own error type
            anyio.run(main)
    finally:
        adapter.uninstall()
    root = client.root()

    assert root.status is StatusCode.ERROR
    assert root.error_type == "session_error"
    assert Limitation.SESSION_ABORTED in _limits(root)


def test_reader_stopped_tells_a_cancel_from_a_failure(monkeypatch):
    """Cancellation in either async library the SDK runs on, and the consumer
    closing the iterator, are stops; anything else is the transport failing."""

    class FakeTrioCancelled(BaseException):
        pass

    assert reader_stopped(asyncio.CancelledError())
    assert reader_stopped(GeneratorExit())
    assert not reader_stopped(RuntimeError("pipe closed"))
    assert not reader_stopped(FakeTrioCancelled())  # trio not loaded: not its cancel
    monkeypatch.setitem(sys.modules, "trio", types.SimpleNamespace(Cancelled=FakeTrioCancelled))
    assert reader_stopped(FakeTrioCancelled())


# --------------------------------------------------------------------------
# 2. A tool call whose closing hook never came
# --------------------------------------------------------------------------


def test_a_tool_whose_closing_hook_never_came_closes_from_the_streams_result():
    """`PreToolUse` fired, `PostToolUse` never did, and the stream carried a
    successful result. It used to ship as ERROR `tool_unclosed` with no
    result — a success reported as a failure."""
    cli = LiveCli(
        [
            INIT,
            tool_use("call_1"),
            pre_tool_use("call_1"),
            tool_result("call_1", "value-1"),
            assistant([{"type": "text", "text": "done"}], msg_id="m2"),
            result(),
        ]
    )
    (tool,) = _query_session(cli).tools()

    assert tool.status is StatusCode.OK
    assert tool.error_type is None
    assert tool.tool.call_id == "call_1"
    assert b"value-1" in bytes(tool.output_data)
    assert Limitation.CHILD_SPAN_UNCLOSED not in _limits(tool)
    # The hook announced the call and stdout reported its end: both observed it.
    assert CaptureSource.STDIO in tool.capture_sources


def test_a_tool_the_cli_refused_reports_the_clis_own_failure():
    """What a tool left out of `allowed_tools` actually does with the real CLI:
    the permission check blocks it, no `PostToolUse` fires, and the stream's
    result says `is_error` with the CLI's reason. The span now says the same
    thing, instead of claiming the call was left unclosed."""
    cli = LiveCli(
        [
            INIT,
            tool_use("call_1"),
            pre_tool_use("call_1"),
            tool_result("call_1", "blocked for safety", is_error=True),
            assistant([{"type": "text", "text": "sorry"}], msg_id="m2"),
            result(),
        ]
    )
    (tool,) = _query_session(cli).tools()

    assert tool.status is StatusCode.ERROR
    assert tool.error_type == "tool_error"
    assert b"blocked for safety" in bytes(tool.output_data)
    assert Limitation.CHILD_SPAN_UNCLOSED not in _limits(tool)


def test_a_tool_nothing_reported_on_is_unset_not_an_invented_failure():
    """No closing hook and no stream result: the outcome was never observed,
    so UNSET, and the marker says the session ended with the call open."""
    cli = LiveCli([INIT, tool_use("call_1"), pre_tool_use("call_1"), result()])
    (tool,) = _query_session(cli).tools()

    assert tool.status is StatusCode.UNSET
    assert tool.error_type is None
    assert Limitation.CHILD_SPAN_UNCLOSED in _limits(tool)


# --------------------------------------------------------------------------
# 3. Stream lines nothing has a rule for
# --------------------------------------------------------------------------


def test_a_stream_line_of_an_unknown_type_is_counted():
    """The silent-loss case: a line type the assembler has no rule for, e.g.
    the day the CLI renames `assistant`. Lines it knowingly ignores are not
    counted, so the counter stays a signal rather than noise."""
    known_but_ignored = [
        {"type": "tool_progress", "tool_use_id": "x", "elapsed_time_seconds": 30},
        {"type": "keep_alive"},
        {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "allowed"},
            "uuid": "u",
            "session_id": SESSION,
        },
        {"type": "system", "subtype": "informational", "content": "notice"},
    ]
    renamed = {"type": "assistant_v2", "session_id": SESSION, "message": {"content": []}}
    cli = LiveCli([INIT, *known_but_ignored, renamed, result()])
    _query_session(cli)

    assert counters.get("adapters.assembler.stream_line_unrecognized") == 1


def test_the_known_type_lists_agree_with_the_native_parser():
    """`PARSED_STREAM_TYPES` must be exactly what the parser builds events
    from, or a type it starts dropping is silently filed as "known"; and an
    ignored type the parser does consume is an entry that hides nothing."""
    minimal = {
        "system": INIT,
        "assistant": assistant([]),
        "user": tool_result("call_1", "x"),
        "stream_event": {"type": "stream_event", "session_id": SESSION, "event": {}},
        "result": result(),
    }
    assert set(minimal) == set(PARSED_STREAM_TYPES)
    for kind, line in minimal.items():
        assert parse_line(json.dumps(line).encode(), outbound=False) is not None, kind
    for kind in IGNORED_STREAM_TYPES:
        line = json.dumps({"type": kind, "session_id": SESSION}).encode()
        assert parse_line(line, outbound=False) is None, kind
