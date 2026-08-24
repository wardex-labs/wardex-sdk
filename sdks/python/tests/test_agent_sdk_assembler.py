"""SessionAssembler unit tests — synthetic events, no SDK involved."""

import json
import time

import pytest

from wardex_sdk._adapters._anthropic_names import McpToolCatalog
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._assembly import Limitation, UnitKey, UnitKind, counters
from wardex_sdk._enums import CaptureSource, StatusCode


@pytest.fixture(autouse=True)
def _fresh_counters():
    """`counters` is a process-global dict, so without this a bump from one test
    is readable by the next — which is how an assertion passes on evidence its
    own test never produced."""
    counters.reset()
    yield
    counters.reset()


INIT = {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-sonnet-5"}
ASSISTANT = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m1",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 25},
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}}
        ],
    },
}
TOOL_RESULT = {
    "type": "user",
    "session_id": "s-1",
    "parent_tool_use_id": "toolu_01",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}],
    },
}
RESULT = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "total_cost_usd": 0.01,
    "duration_ms": 100,
    "duration_api_ms": 80,
}
DIVERGENT_ASSISTANT = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m2",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 25},
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_03",
                "name": "Bash",
                "input": {"command": "ls -la /stream"},
            }
        ],
    },
}


ASSISTANT_2 = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m2",
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 30},
        "content": [{"type": "text", "text": "done"}],
    },
}


INIT_B = {"type": "system", "subtype": "init", "session_id": "s-2", "model": "claude-sonnet-5"}


class FakeClient:
    def __init__(self):
        self.spans = []

    def capture_span(self, span):
        self.spans.append(span)


def _one_root_registry(client):
    """A registry with room for exactly one root, handed in as a registry.

    `SessionAssembler` has no `max_units` parameter: one bound, one owner, and
    the owner of a registry's bounds is the registry. A test that wants a
    narrow one builds a narrow one — which is also what production does, since
    the adapter hands over the context's registry rather than describing it.
    """
    from wardex_sdk._adapters._sink import _ClientSink
    from wardex_sdk._assembly import UnitRegistry

    return UnitRegistry(sink=_ClientSink(client), max_units=1)


@pytest.fixture
def tallies():
    """Counter DELTAS for one test, without clearing the process-wide table.

    `counters` is a module global shared by every test in the session, so a
    `reset()` here would silently move another file's baseline. Deltas need no
    such cooperation.
    """
    before = counters.snapshot()

    def delta(where):
        return counters.get(where) - before.get(where, 0)

    return delta


def _outbound(asm, key, session="s-1", text="go"):
    asm.on_outbound(
        key,
        json.dumps(
            {"type": "user", "session_id": session, "message": {"role": "user", "content": text}}
        ),
    )


def test_happy_path_emits_root_and_chat_spans():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    for msg in (INIT, ASSISTANT, RESULT):
        asm.on_inbound(1, msg)
    asm.on_close(1, None)

    names = [s.name for s in client.spans]
    assert "chat claude-sonnet-5" in names
    root = client.spans[-1]
    assert root.name == "invoke_agent"
    assert root.status is StatusCode.OK
    assert root.conversation.session_id == "s-1"
    assert CaptureSource.ADAPTER in root.capture_sources
    assert Limitation.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS in root.capture_integrity.limitations
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert chat.gen_ai.input_tokens == 10
    assert chat.gen_ai.output_tokens == 25
    assert chat.parent_span_id == root.context.span_id
    assert asm.open_session_count() == 0


def test_tool_span_from_hooks_joins_stream_content():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        "toolu_01",
    )
    asm.on_hook(
        "PostToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"}, "toolu_01"
    )
    asm.on_inbound(1, TOOL_RESULT)
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.tool.call_id == "toolu_01"
    # No correlation at all, which is what this span has earned: its anchor may
    # have come from a silent fallback, so there is no edge here to price. The
    # framework's id is not lost -- it travels as the tool's own `call_id`, in
    # the field that means "which call", rather than as a hint inside a
    # submessage about parentage.
    assert tool.correlation is None
    assert CaptureSource.STDIO not in tool.capture_sources
    assert b"ls" in tool.input_data


def test_close_tool_stream_input_wins_over_hook_reserialization():
    """Design rule (spec §6.2): stream is content authority. A hook's
    re-serialized tool_input must not shadow the byte-exact stream input_json
    when both are present and diverge."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, DIVERGENT_ASSISTANT)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "hook-version"}},
        "toolu_03",
    )
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"},
        "toolu_03",
    )
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert b"/stream" in tool.input_data
    assert b"hook-version" not in tool.input_data


def test_a_stream_only_tool_span_says_which_channel_saw_it():
    """The hook/stream difference is real and belongs in `capture_sources`.

    It used to travel as `confidence` 1.0 vs 0.7 — the observation channel
    wearing a certainty's clothes, in the field that prices a PARENT EDGE. A
    consumer filtering on low confidence was selecting spans wardex had watched
    from a different vantage point, not spans whose place in the tree was a
    guess.
    """
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    for msg in (INIT, ASSISTANT, TOOL_RESULT, RESULT):
        asm.on_inbound(1, msg)
    asm.on_close(1, None)

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert CaptureSource.STDIO in tool.capture_sources
    assert tool.correlation is None


def test_a_tool_that_failed_does_not_ship_as_a_success():
    """`is_error` sits in the result block the CLI already sends. Nothing read
    it, so the ARRIVAL of a result was taken for the SUCCESS of the call."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    failed = {
        "type": "user",
        "session_id": "s-1",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": "command not found",
                    "is_error": True,
                }
            ],
        },
    }
    for msg in (INIT, ASSISTANT, failed, RESULT):
        asm.on_inbound(1, msg)
    asm.on_close(1, None)

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.status is StatusCode.ERROR
    assert tool.error_type == "tool_error"


def test_subagent_span_attribution():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "SubagentStart", {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"}, None
    )
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Read", "tool_input": {}, "agent_id": "a-1"},
        "toolu_02",
    )
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Read", "tool_response": "x", "agent_id": "a-1"},
        "toolu_02",
    )
    asm.on_hook(
        "SubagentStop", {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"}, None
    )
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    sub = next(s for s in client.spans if s.name == "invoke_agent researcher")
    tool = next(s for s in client.spans if s.name == "execute_tool Read")
    assert sub.agent.id == "a-1"
    assert tool.parent_span_id == sub.context.span_id


def test_abort_closes_open_spans_with_markers():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_09"
    )
    asm.on_close(1, "ProcessError")

    root = next(s for s in client.spans if s.name == "invoke_agent")
    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert root.status is StatusCode.ERROR
    assert Limitation.SESSION_ABORTED in root.capture_integrity.limitations
    # Census rename (design §6.5.1): the free string `tool_span_unclosed` folded
    # into the already-declared member CHILD_SPAN_UNCLOSED, which is what the
    # assembler emits now.
    assert Limitation.CHILD_SPAN_UNCLOSED in tool.capture_integrity.limitations
    assert asm.open_session_count() == 0


def test_a_claimed_tool_gets_no_hook_driven_span():
    """Successor to the `skip_tool_names` regression, and a stronger statement.

    Same rule — a tool whose execution the adapter wrapped must not ALSO get a
    hook-driven span — but the mechanism it tests is the one that works. The
    skip list held BARE names (`greet`, which is all the handler wrapper knows)
    and was compared against the CLI's NAMESPACED `tool_name`
    (`mcp__srv__greet`), so it never matched and every in-process tool shipped
    twice; the old test passed only because it hand-fed the set the namespaced
    spelling nothing produced. Here both observers normalize into one key space
    and the handler's rank 10 beats the hook's 0 whenever it arrives — which is
    always AFTER, since `PreToolUse` fires before the tool body runs.
    """
    client = FakeClient()
    names = McpToolCatalog()
    handle = names.handle_for("srv")
    handle.tools.add("greet")
    handle.instance = object()
    # The token is the mcp_servers KEY, and it is only knowable here.
    names.resolve_tokens({"srv": {"type": "sdk", "instance": handle.instance}})
    asm = SessionAssembler(client, names=names)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    assistant = {
        "type": "assistant",
        "session_id": "s-1",
        "message": {
            "id": "m1",
            "model": "claude-sonnet-5",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 25},
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_greet",
                    "name": "mcp__srv__greet",
                    "input": {"name": "world"},
                }
            ],
        },
    }
    asm.on_inbound(1, assistant)
    # What the handler wrapper does when the tool body starts: it claims the
    # call at its own rank, on the SESSION unit both observers share.
    asm._by_key[1].unit.claim(handle.key_for("greet"), rank=10)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "mcp__srv__greet", "tool_input": {"name": "world"}},
        "toolu_greet",
    )
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "mcp__srv__greet", "tool_response": "hi world"},
        "toolu_greet",
    )
    tool_result = {
        "type": "user",
        "session_id": "s-1",
        "parent_tool_use_id": "toolu_greet",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_greet", "content": "hi world"}
            ],
        },
    }
    asm.on_inbound(1, tool_result)
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    names = [s.name for s in client.spans]
    assert "execute_tool mcp__srv__greet" not in names
    # root/chat spans still emitted normally
    assert "invoke_agent" in names
    assert "chat claude-sonnet-5" in names
    assert asm.open_session_count() == 0


def test_open_entry_cap(tallies):
    """300 opens over a 256-entry table: 44 evicted + 256 drained = 300.

    The arithmetic is the point of the docstring — a reviewer reading the diff
    sees one `== 300` become two numbers and must not read that as spans lost.
    Every open tool still leaves exactly one span; what changed is that the 44
    the BOUND closed and the 256 the TEARDOWN closed now say different things,
    because the reader's next action differs: raise `max_session_entries`, or
    find out why the session ended with tools open.
    """
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for n in range(300):
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-1", "tool_name": f"T{n}", "tool_input": {}},
            f"toolu_{n}",
        )
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    def marked(marker):
        return [
            s
            for s in client.spans
            if s.capture_integrity and marker in s.capture_integrity.limitations
        ]

    evicted = marked(Limitation.SESSION_ENTRY_TABLE_FULL)
    unclosed = marked(Limitation.CHILD_SPAN_UNCLOSED)
    assert len(evicted) == 44
    assert len(unclosed) == 256
    assert len(evicted) + len(unclosed) == 300  # all eventually closed, none leaked
    assert tallies("adapters.assembler.open_tool_table_full") == 44
    # WHICH 44: the FIFO's witness. Ordered by start instant, the flag is a
    # prefix — the oldest opens are the ones the bound closed.
    tools = sorted(
        (s for s in client.spans if s.name.startswith("execute_tool")),
        key=lambda s: s.start_time_ns,
    )
    flags = [
        s.capture_integrity is not None
        and Limitation.SESSION_ENTRY_TABLE_FULL in s.capture_integrity.limitations
        for s in tools
    ]
    assert flags == [True] * 44 + [False] * 256
    assert asm.open_session_count() == 0


def test_open_tool_eviction_names_the_session_entry_knob(tallies):
    """The headline: a full `open_tools` table says which knob closed the span.

    `child_span_unclosed` names no knob at all — it says a parent's teardown
    closed the span, and no teardown happened here — so a reader chasing it
    goes looking for a close that does not exist and concludes the agent
    abandoned the tool. That is wardex blaming its own bound on the agent.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=1)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for n in (1, 2):
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
            f"t{n}",
        )

    evicted = next(s for s in client.spans if s.name == "execute_tool Bash")
    limits = evicted.capture_integrity.limitations
    assert Limitation.SESSION_ENTRY_TABLE_FULL in limits
    assert Limitation.CHILD_SPAN_UNCLOSED not in limits
    assert evicted.status is StatusCode.UNSET
    assert evicted.tool.call_id == "t1"
    assert tallies("adapters.assembler.open_tool_table_full") == 1


def test_the_evicted_tool_is_not_blamed_on_the_agent():
    """UNSET and not ERROR, which is the other way to get this wrong.

    ERROR would report wardex's own full table as a tool failure — status is
    the first field anyone filters an agent run by — and `finish()` refuses an
    ERROR with no type, so "fixing" the OK would arrive with a fabricated
    `error.type` too. The bound stopped watching; it did not observe an outcome.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=1)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for n in (1, 2):
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}},
            f"t{n}",
        )

    evicted = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert evicted.status is StatusCode.UNSET
    assert evicted.status is not StatusCode.ERROR
    assert evicted.error_type is None


def _tool_result(tool_use_id, content="ok", parent=None):
    return {
        "type": "user",
        "session_id": "s-1",
        "parent_tool_use_id": parent or tool_use_id,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
        },
    }


def _evict_one_tool(client=None, cap=1, agent_id=None):
    """Open `cap + 1` tools so the FIRST is evicted, and hand back the assembler.

    The shared arrangement of every completion-after-eviction test: what they
    differ on is which channel delivers the completion afterwards.
    """
    client = client or FakeClient()
    asm = SessionAssembler(client, max_session_entries=cap)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for n in range(cap + 1):
        payload = {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        asm.on_hook("PreToolUse", payload, f"t{n}")
    return client, asm


def _tools(client, call_id):
    return [s for s in client.spans if s.tool is not None and s.tool.call_id == call_id]


def _has(span, marker):
    return span.capture_integrity is not None and marker in span.capture_integrity.limitations


def test_a_late_close_after_an_eviction_is_the_same_bound_twice(tallies):
    """One call, two observations — not one call twice.

    Before the breadcrumb, a `PostToolUse` whose open record had been evicted
    built a brand-new record starting `now`: a second `execute_tool` of ZERO
    duration, carrying no marker, hanging off whatever parent the payload named,
    and tagged `stdio` as though it had been reconstructed from the CLI's
    stdout. Four wrong facts about one call, and the p50 it dragged down was
    the visible one.
    """
    client, asm = _evict_one_tool()
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_response": "late"},
        "t0",
    )

    spans = _tools(client, "t0")
    assert len(spans) == 2
    evicted, completion = spans
    assert evicted.status is StatusCode.UNSET
    assert _has(evicted, Limitation.SESSION_ENTRY_TABLE_FULL)
    assert _has(completion, Limitation.SESSION_ENTRY_TABLE_FULL)
    assert b"late" in completion.output_data
    # The real duration, and the same parent: both come off the breadcrumb.
    assert completion.start_time_ns == evicted.start_time_ns
    assert completion.end_time_ns > completion.start_time_ns
    assert completion.parent_span_id == evicted.parent_span_id
    # A `PostToolUse` IS a hook. Missing the OPEN hook does not make the close
    # a reconstruction from stdout.
    assert CaptureSource.STDIO not in completion.capture_sources
    assert tallies("adapters.assembler.tool_completion_after_evict") == 1


def test_a_stream_result_after_an_eviction_gets_the_same_treatment(tallies):
    """The other completion channel. `stdio` is TRUE here — this one really is
    rebuilt from the CLI's stdout — which is what makes its absence on the hook
    path a statement rather than an accident."""
    client, asm = _evict_one_tool()
    asm.on_inbound(1, _tool_result("t0", "late"))

    spans = _tools(client, "t0")
    assert len(spans) == 2
    evicted, completion = spans
    assert _has(completion, Limitation.SESSION_ENTRY_TABLE_FULL)
    assert b"late" in completion.output_data
    assert completion.start_time_ns == evicted.start_time_ns
    assert completion.parent_span_id == evicted.parent_span_id
    assert CaptureSource.STDIO in completion.capture_sources
    assert tallies("adapters.assembler.tool_completion_after_evict") == 1


def test_both_halves_of_a_subagents_tool_keep_the_same_parent():
    """The stream path hardcoded `agent_id=None`, so the completion half of a
    sub-agent's tool call landed on the session root while the evicted half hung
    under the sub-agent. One call, two parents, and a subtree that reports a
    shape it never had."""
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=2)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": "a1"}, None)
    for n in range(3):
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}, "agent_id": "a1"},
            f"t{n}",
        )
    asm.on_inbound(1, _tool_result("t0", "late"))
    asm.on_hook("SubagentStop", {"session_id": "s-1", "agent_id": "a1"}, None)

    spans = _tools(client, "t0")
    assert len(spans) == 2
    evicted, completion = spans
    assert completion.parent_span_id == evicted.parent_span_id
    sub = next(s for s in client.spans if s.agent is not None and s.agent.id == "a1")
    assert completion.parent_span_id == sub.context.span_id


def test_a_completion_survives_the_stream_metadata_being_gone_too(tallies):
    """The lookup order, asserted where it bites.

    A full session table is a state in which BOTH bounded tables are full, so
    `stream_tool_meta` missing and `open_tools` evicted is the common case
    rather than the corner. Reading the breadcrumb after the `meta is None`
    early return would drop this completion entirely — no span, no marker, no
    counter — precisely when the bound is doing the most work.
    """
    client, asm = _evict_one_tool()
    # Nothing ever put `t0` into `stream_tool_meta`: no assistant turn announced
    # it, which is exactly what a refused metadata entry looks like downstream.
    asm.on_inbound(1, _tool_result("t0", "late"))

    spans = _tools(client, "t0")
    assert len(spans) == 2
    completion = spans[1]
    assert _has(completion, Limitation.SESSION_ENTRY_TABLE_FULL)
    assert completion.name == "execute_tool Bash"  # the name came off the breadcrumb
    assert tallies("adapters.assembler.tool_completion_after_evict") == 1


def test_a_tool_completes_at_most_twice_after_an_eviction(tallies):
    """Hook AND stream both close one call: the second completion is suppressed.

    Two overlapping spans is a documented reading rule; a third would pollute
    the aggregates that rule already asks readers to correct for. What is lost
    is one duplicate copy of a response the surviving half already carries.
    """
    client, asm = _evict_one_tool()
    asm.on_inbound(1, _tool_result("t0", "late"))
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_response": "late"},
        "t0",
    )

    assert len(_tools(client, "t0")) == 2
    assert tallies("adapters.assembler.tool_completion_after_evict") == 1
    assert tallies("adapters.assembler.tool_completion_after_evict_duplicate") == 1


def test_a_close_with_no_open_and_no_breadcrumb_is_counted(tallies):
    """The residue: mid-session install, or a breadcrumb table that itself
    overflowed. The span still ships — with a zero duration, which is the honest
    consequence of not knowing when the call started — and the counter is the
    only place that fact is recorded."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"},
        "orphan",
    )

    span = _tools(client, "orphan")[0]
    assert not _has(span, Limitation.SESSION_ENTRY_TABLE_FULL)
    assert CaptureSource.STDIO not in span.capture_sources
    assert tallies("adapters.assembler.tool_close_without_open") == 1


def _subagents(client):
    return [s for s in client.spans if s.agent is not None and s.name.startswith("invoke_agent ")]


def test_the_subagent_table_evicts_and_emits_instead_of_dropping(tallies):
    """The worse half of the same bound: this one said nothing at all.

    A full `subagents` table simply refused to open the new entry — no span, no
    counter, and the sub-agent's whole subtree quietly re-parented onto the
    session root. One site of a bound reporting the wrong thing while its
    sibling reports nothing is the inconsistency the vocabulary census exists to
    catch.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=1)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for agent_id in ("a1", "a2"):
        asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": agent_id}, None)

    evicted = _subagents(client)
    assert len(evicted) == 1
    assert evicted[0].agent.id == "a1"
    assert _has(evicted[0], Limitation.SESSION_ENTRY_TABLE_FULL)
    assert evicted[0].status is StatusCode.UNSET
    assert tallies("adapters.assembler.subagent_table_full") == 1


def test_an_evicted_subagents_children_keep_their_parent():
    """The eviction must not change the SHAPE of the tree.

    FIFO evicts the oldest, which is the longest-lived, which is the outermost
    sub-agent — the one with the most still-open work beneath it. Its anchors
    are resolved at EMIT time and all three lookups fall silently to the session
    root, so without the breadcrumb this trades one silent drop for a whole
    silently flattened subtree, and nothing in the data says so.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=2)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": "a1"}, None)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}, "agent_id": "a1"},
        "t1",
    )
    for agent_id in ("a2", "a3"):  # a3 evicts a1, whose tool is still open
        asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": agent_id}, None)
    asm.on_hook(
        "PostToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"}, "t1"
    )
    asm.on_close(1, None)  # ships the session root, so "not the root" is checkable

    a1 = next(s for s in _subagents(client) if s.agent.id == "a1")
    tool = _tools(client, "t1")[0]
    assert tool.parent_span_id == a1.context.span_id
    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert tool.parent_span_id != root.context.span_id


def test_a_subagent_stop_after_an_eviction_is_counted_not_dropped(tallies):
    """No completion half, and no silent return either.

    A tool's completion half exists because the close carries OUTPUT bytes. A
    `SubagentStop` carries nothing this SDK reads — the assembler takes the
    `agent_id` off it and no more — so a second span would hold nothing. What is
    lost is the true end instant, and the counter is where that is recorded.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=1)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for agent_id in ("a1", "a2"):
        asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": agent_id}, None)
    before = len(_subagents(client))
    asm.on_hook("SubagentStop", {"session_id": "s-1", "agent_id": "a1"}, None)

    assert len(_subagents(client)) == before
    assert tallies("adapters.assembler.subagent_stop_after_evict") == 1


def _assistant_with(*tool_uses, msg_id="m1"):
    return {
        "type": "assistant",
        "session_id": "s-1",
        "message": {
            "id": msg_id,
            "model": "claude-sonnet-5",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 25},
            "content": [
                {"type": "tool_use", "id": tu_id, "name": "Bash", "input": {"command": cmd}}
                for tu_id, cmd in tool_uses
            ],
        },
    }


def test_the_stream_meta_bound_keeps_what_is_consumed_next_and_counts_the_drop(tallies):
    """The third table under the same bound, and the only one with no span.

    Nothing to mark, so the counter IS the record — a bound that turns
    something away in silence is what this whole change is about. The DIRECTION
    stays refuse-the-newest, against the symmetry argument: both consumers pop
    by the id whose result arrived, and in a turn the results come back broadly
    in announcement order, so the oldest entry is the one most likely to be read
    next. Correctness beats policy symmetry, and there was no measurement
    supporting the swap.

    What the refusal costs is asserted here rather than described, because it
    lands in two different places: a hook-closed call keeps its span and silently
    carries the hook's re-serialized input instead of the byte-exact stream one,
    while a stream-only call has no span at all.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_session_entries=1)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, _assistant_with(("keep", "ls -1"), ("dropped", "ls -2")))

    assert tallies("adapters.assembler.stream_tool_meta_table_full") == 1
    sess = asm._by_key[1]
    assert list(sess.stream_tool_meta) == ["keep"]

    # (i) the entry that survived still supplies byte-exact input
    asm.on_inbound(1, _tool_result("keep", "ok"))
    kept = _tools(client, "keep")
    assert len(kept) == 1
    assert b"ls -1" in kept[0].input_data

    # (ii) the refused one, on the stream-only path, has no span to mark
    asm.on_inbound(1, _tool_result("dropped", "ok"))
    assert _tools(client, "dropped") == []


def test_the_session_id_becomes_a_lookup_alias_for_the_units_own_context():
    """The CLI's `session_id` is registered as a lookup ALIAS (design §5.3-iii).

    What this pins is the DIRECTION of the arrow, which is I2: the id selects a
    unit whose own span context came from a real scope read, and `find()` hands
    back that unit or None. There is no call that turns the string into a span
    context, so the id cannot become a parent.

    It exercises no failover, and there is none to exercise: nothing in the SDK
    calls `find()`/`resolve()`, so the alias is a record in the registry rather
    than a route the adapter falls back on. When the pin does not answer, what
    answers is a lower tier that marks itself — `_session_for_hook` uses the
    adapter's own `_by_session_id` table, and `_open_tool_call` drops to
    `sole_live(SESSION)` at 0.5 with `UNIT_INFERRED_SOLE`.
    """
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)

    unit = asm.units.find(UnitKey("claude.session_id", "s-1"))

    assert unit is asm.unit_for(1)
    assert unit.kind is UnitKind.SESSION


def test_an_evicted_session_emits_its_root_and_says_which_bound_evicted_it():
    """A ceiling that drops state silently is a worse failure than an unenforced
    one (I10). The code this replaces popped the session out of its dict and
    dropped the root span with it — no marker, no counter, no test — so a
    workload that crossed `max_sessions` simply stopped producing traces.
    """
    client = FakeClient()
    asm = SessionAssembler(client, max_sessions=1)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)

    _outbound(asm, key=2)  # evicts the first

    assert asm.open_session_count() == 1
    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert Limitation.UNIT_EVICTED in root.capture_integrity.limitations
    assert root.conversation.session_id == "s-1"


def test_a_hook_is_attributed_to_the_session_in_scope_not_the_one_its_id_names(tallies):
    """I2 on the hook path: a framework identifier is a LOOKUP KEY, not a parent.

    Two sessions are live in one process. A hook fires on session A's reader
    task — the task A's unit is ambient on, which is where `claude_agent_sdk`
    dispatches every hook callback from — and its payload carries session B's
    id. Exactly one of those two facts is evidence about the tree: a callback
    cannot run on a task descended from a session it does not belong to, while
    an id is whatever the CLI wrote in the payload.

    The code this replaces read the id and nothing else, so the tool span landed
    under B's root, in B's TRACE, at confidence 1.0 with no marker — a wrong
    tree with nothing on the wire to reveal it. Shape alone cannot catch that
    (both roots produce a well-formed subtree), so this asserts IDENTITY: whose
    span id the tool actually points at.
    """
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    _outbound(asm, key=2, session="s-2")
    asm.on_inbound(2, INIT_B)
    a, b = asm._by_key[1], asm._by_key[2]

    with a.unit.activate():  # the hook runs on a task descended from A's reader
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-2", "tool_name": "Bash", "tool_input": {"command": "ls"}},
            "toolu_x",
        )
        asm.on_hook(
            "PostToolUse",
            {"session_id": "s-2", "tool_name": "Bash", "tool_response": "ok"},
            "toolu_x",
        )

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.parent_span_id == a.unit.context.span_id
    assert tool.parent_span_id != b.unit.context.span_id
    assert tool.context.trace_id == a.unit.context.trace_id
    # And the disagreement is recorded rather than resolved in silence — twice,
    # once per observation, because each one was independently wrong.
    assert tallies("adapters.assembler.hook_session_conflict") == 2


def test_an_unattributable_hook_is_counted_rather_than_silently_discarded(tallies):
    """The other half of the same defect: data loss with no trace of itself.

    Two sessions live, an id that names neither, and no session in scope. There
    is no defensible parent here — `sole_live`'s rule needs exactly one live
    unit — so the observation really is dropped. What must not happen is
    dropping it INVISIBLY: the shipped code returned None from
    `_session_for_hook` and `on_hook` returned, leaving no span, no limitation
    and no counter, which is indistinguishable from a hook that never fired.
    """
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    _outbound(asm, key=2, session="s-2")
    asm.on_inbound(2, INIT_B)

    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-nobody", "tool_name": "Bash", "tool_input": {}},
        "toolu_y",
    )
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-nobody", "tool_name": "Bash", "tool_response": "ok"},
        "toolu_y",
    )

    assert [s for s in client.spans if s.name == "execute_tool Bash"] == []
    assert tallies("adapters.assembler.hook_session_unresolved") == 2


def test_a_registry_evicted_session_is_retired_and_the_run_resumes_on_a_fresh_root(tallies):
    """The OTHER bound, and the one that had no reconciliation path at all.

    `max_sessions` bounds this assembler's `_by_key`; `max_units` bounds
    `UnitRegistry._roots`. They are independent knobs from different fields of
    the same core struct, so the REGISTRY can evict a session root while this
    side still believes the session is running — and nothing tells it. The test
    above covers the assembler evicting itself, which was always well-behaved
    because `_finalize` runs BEFORE the unit dies. This is the identical
    scenario with the two steps in the opposite order.

    What used to happen: every subsequent event was driven against a dead unit,
    and the semantically complete root — model, conversation id, num_turns,
    cost — was stamped onto its draft and then dropped on the floor, because
    `UnitRegistry.close()` emits nothing for a unit that is already closed. No
    span, no counter, no marker. The only root on the wire was the stub the
    eviction shipped, with `agent.name="agent"` and no conversation at all,
    sitting in a different conversation bucket from its own chat children.
    """
    client = FakeClient()
    asm = SessionAssembler(client, units=_one_root_registry(client))
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    retired = asm._by_key[1]
    issued = retired.issued_conversation_id

    # A second transport takes the registry's only root slot.
    _outbound(asm, key=2, session="s-2")
    asm.on_inbound(2, INIT_B)
    assert not retired.unit.is_live, "the registry did not evict — the bound moved"

    # ...and the first transport keeps talking, as a live CLI subprocess does.
    asm.on_inbound(1, ASSISTANT)
    resumed = asm._by_key[1]
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    assert tallies("adapters.assembler.session_unit_evicted") == 1
    assert resumed is not retired

    # The replacement root ships, and it ships COMPLETE: everything
    # `_stamp_root` writes is on the wire rather than on a garbage-collected
    # draft.
    fresh = client.spans[-1]
    assert fresh.name == "invoke_agent"
    assert fresh.status is StatusCode.OK
    assert fresh.agent.name == "claude-sonnet-5"
    assert ("wardex.agent.num_turns", 1) in fresh.extra
    assert ("wardex.agent.cost_usd", 0.01) in fresh.extra

    # One run, one conversation. The stub and the replacement have to be
    # joinable or the eviction reads downstream as two unrelated agents.
    assert fresh.conversation.session_id == "s-1"
    assert fresh.conversation.conversation_id == "s-1"
    assert resumed.issued_conversation_id == issued
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert chat.conversation.conversation_id == fresh.conversation.conversation_id
    assert chat.parent_span_id == fresh.context.span_id

    # And it says why it exists.
    assert Limitation.UNIT_EVICTED in fresh.capture_integrity.limitations


def test_a_retired_sessions_open_tool_is_still_emitted(tallies):
    """Retiring a session must not become a second, quieter way to lose a span.

    The retired session is the only holder of its open-tool records — they are
    the assembler's state, not the unit's, so the registry's eviction does not
    force-close them — and after this change `on_close` finalizes the
    REPLACEMENT session, which has never heard of them. Draining at retirement
    is what keeps the fix from trading one silent drop for another.
    """
    client = FakeClient()
    asm = SessionAssembler(client, units=_one_root_registry(client))
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_evict"
    )

    _outbound(asm, key=2, session="s-2")  # the registry evicts session one's root
    asm.on_inbound(1, ASSISTANT)  # ...and the assembler notices on the next event

    assert tallies("adapters.assembler.session_unit_evicted") == 1
    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.tool.call_id == "toolu_evict"
    assert Limitation.CHILD_SPAN_UNCLOSED in tool.capture_integrity.limitations


def test_closing_a_retired_transport_does_not_restamp_the_span_it_already_shipped(tallies):
    """The teardown route no other event guards.

    Every other entry point re-enters the liveness check because it goes through
    `_ensure_session` or `_session_for_hook`. `on_close` does not: a transport
    can close after the registry evicted its root with nothing arriving in
    between, and finalizing then reaches `_stamp_root` — which writes the model,
    the conversation id and the result extras onto a draft whose span was
    emitted seconds earlier, because `SpanDraft.finish()` neither freezes the
    draft nor refuses a second call. The write lands on an object nobody will
    read again, and then `UnitRegistry.close()` emits nothing at all, so the
    only visible trace of the whole sequence is a registry-internal counter.

    The stray write itself has no observer — the span object already handed to
    the client is a separate, finished value, so editing the draft behind it
    changes nothing anyone can read. That is the whole hazard, and it is why the
    guard is on the two counters that DO distinguish the paths: retiring names
    the eviction on the assembler's own tally, while finalizing a corpse names
    itself only in the registry's `close_after_close`.
    """
    client = FakeClient()
    asm = SessionAssembler(client, units=_one_root_registry(client))
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    retired = asm._by_key[1]

    _outbound(asm, key=2, session="s-2")  # the registry evicts session one's root
    asm.on_inbound(2, INIT_B)
    assert not retired.unit.is_live, "the registry did not evict — the bound moved"
    shipped = next(s for s in client.spans if s.name == "invoke_agent")

    asm.on_close(1, None)  # ...and nothing arrived for transport 1 in between

    assert tallies("adapters.assembler.session_unit_evicted") == 1
    assert tallies("assembly._units.close_after_close") == 0
    # The stub is what shipped, and nothing ships after it: a second root would
    # mean the retired unit was closed twice.
    assert shipped.conversation is None
    assert [s for s in client.spans if s.name == "invoke_agent"] == [shipped]
    assert asm.open_session_count() == 1  # only the second session is left


# ==========================================================================
# Shutdown — the session that is still live when the process stops
# ==========================================================================
#
# The whole class of bug here is a span that simply does not exist. There is no
# marker to look for, no counter that moves, and no failing assertion anywhere
# else in this file: a run that was interrupted looks exactly like a run that
# was never started. So these tests assert on the presence and the CONTENTS of
# a span that the code they defend is the only reason to expect at all.


def _live_session(client=None):
    """A session mid-run: init seen, one tool open, one subagent unstopped."""
    asm = SessionAssembler(client if client is not None else FakeClient())
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "SubagentStart", {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"}, None
    )
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        "toolu_01",
    )
    return asm


def test_a_live_session_emits_its_root_when_teardown_arrives():
    """The bug in one assertion: before this, the span did not exist.

    A run still in flight at shutdown produced nothing — not a truncated span,
    not a marked one, nothing — because the only thing that closes a session is
    the transport closing, and an interrupted process never gets there. The
    marker is what tells a reader the run did not simply stop being interesting.
    """
    client = FakeClient()
    asm = _live_session(client)

    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    roots = [s for s in client.spans if s.name == "invoke_agent"]
    assert len(roots) == 1
    assert Limitation.ADAPTER_UNINSTALLED in roots[0].capture_integrity.limitations


def test_the_torn_down_root_carries_the_runs_identity_not_a_stub():
    """Which is why teardown runs through the assembler and not the registry.

    `UnitRegistry.close_all` can end a unit, but everything that says WHICH run
    it was — the model, the conversation id, the turn totals — lives in the
    assembler's `_Session`, not in the unit. Closing from below emits an
    `invoke_agent` still carrying the open-time placeholder name and no
    conversation at all: a span that exists but cannot be attributed to
    anything, which is barely better than the one that was missing.
    """
    client = FakeClient()
    asm = _live_session(client)

    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.conversation is not None
    assert root.conversation.session_id == "s-1"
    assert root.agent.name == "claude-sonnet-5"


def test_teardown_drains_open_tools_and_unstopped_subagents_before_the_root():
    """A session's children do not live in the registry, so a registry-only
    teardown drops them silently — the tool call that was running when the
    process died is exactly the one an operator goes looking for.

    Order is asserted too, and it is not cosmetic: a consumer that streams sees
    the subtree before its root, so a child arriving after its parent has
    already been reported closed is a child with nowhere to attach.
    """
    client = FakeClient()
    asm = _live_session(client)

    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    names = [s.name for s in client.spans]
    assert any(n.startswith("execute_tool") for n in names), names
    assert any(n.startswith("invoke_agent researcher") for n in names), names
    assert names.index("invoke_agent") == len(names) - 1, names


def test_teardown_empties_the_assemblers_tables():
    """Left populated, they poison a bound's audit rather than merely leaking.

    A straggler for a session this loop already finalized routes into
    `_live_session`, which reads a retired entry as a registry eviction and
    bumps the counter that exists to tell a user their `max_units` is too low.
    """
    client = FakeClient()
    asm = _live_session(client)

    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    assert asm._by_key == {}
    assert asm._by_session_id == {}
    assert asm.open_session_count() == 0


def test_teardown_declines_when_this_thread_already_holds_the_lock(tallies):
    """The signal-handler case, and the reason the guard is not a try-lock.

    A handler runs on the main thread at an arbitrary bytecode boundary, so it
    can land INSIDE a half-finished mutation. The lock is an `RLock`, so
    re-entering does not deadlock — it walks the tables mid-update and emits
    from state no reader was meant to see. Declining loses a teardown that was
    already racing a dying process; proceeding loses correctness.
    """
    client = FakeClient()
    asm = _live_session(client)

    with asm._lock:
        asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)
        assert client.spans == []

    assert tallies("adapters.assembler.close_all_sessions_reentrant") == 1

    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)
    assert [s.name for s in client.spans if s.name == "invoke_agent"] == ["invoke_agent"]


def test_a_second_teardown_does_not_emit_the_root_twice():
    """Two shutdown paths can both fire — the signal handler closes units and
    then the interpreter runs atexit, which uninstalls. The second must be a
    no-op rather than a duplicate run in the user's trace.
    """
    client = FakeClient()
    asm = _live_session(client)

    asm.close_all_sessions(marker=Limitation.UNIT_INTERRUPTED)
    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    assert [s.name for s in client.spans].count("invoke_agent") == 1


def test_teardown_also_closes_a_call_unit_no_session_owns():
    """Why the registry sweep runs as well, and not as belt-and-braces.

    An in-process tool handler that runs with no session to hang off — the pin
    did not reach it and nothing else is live — opens its CALL unit with no
    parent, which makes it a ROOT of the registry that appears in no `_Session`
    at all. Walking the assembler's own tables cannot find it by construction.
    It is also the span most worth having: a tool call wardex could not attach
    to a run is already the anomalous one.
    """
    from wardex_sdk._assembly import EMPTY_AMBIENT, SpanIntent
    from wardex_sdk._types import ToolAttributes

    client = FakeClient()
    asm = _live_session(client)
    orphan = asm._units.open(
        UnitKind.CALL,
        UnitKey("mcp.tool.call", "srv/greet#1"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.EXECUTE_TOOL,
        subject="greet",
    )
    orphan.draft.set_tool(ToolAttributes(name="greet"))
    assert orphan in asm._units._roots, "the no-holder branch stopped producing a root"

    asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    assert not orphan.is_live
    assert any(s.name == "execute_tool greet" for s in client.spans), [s.name for s in client.spans]


def test_a_second_run_on_a_recycled_transport_key_does_not_pass_for_the_first():
    """One CLI subprocess emits `system/init` once.

    A second one naming a different run, on a key this table still holds live,
    means the earlier subprocess went away without its close reaching us and
    CPython handed its identity to the next object. Everything after it is filed
    under the earlier run's root, so two agent runs share one trace — and
    without this the tree says nothing at all about that.
    """
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, {**INIT, "session_id": "s-OTHER"})
    asm.on_close(1, None)

    root = next(s for s in client.spans if s.name.startswith("invoke_agent"))
    assert Limitation.CORRELATION_CONFLICT in root.capture_integrity.limitations
    assert counters.get("adapters.assembler.session_key_recycled") == 1


def test_one_run_reporting_its_own_id_twice_is_not_a_conflict():
    """The guard is about a DIFFERENT id, not about a repeated line. Keying it
    on "init arrived again" would stamp a conflict on every ordinary re-read."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, INIT)
    asm.on_close(1, None)

    root = next(s for s in client.spans if s.name.startswith("invoke_agent"))
    markers = root.capture_integrity.limitations if root.capture_integrity else ()
    assert Limitation.CORRELATION_CONFLICT not in markers
    assert counters.get("adapters.assembler.session_key_recycled") == 0


# ==========================================================================
# Per-turn prompt capture — the stream is the content authority, the hook
# (UserPromptSubmit) is the turn boundary and the degraded-content fallback
# ==========================================================================


def _chats(client):
    return [s for s in client.spans if s.name.startswith("chat")]


def _submit(asm, prompt, session="s-1"):
    asm.on_hook("UserPromptSubmit", {"session_id": session, "prompt": prompt}, None)


def test_each_turns_prompt_rides_its_own_chat_span(tallies):
    """The defect this replaces: prompts for turns 2+ were parsed off the wire
    and DISCARDED — only the first turn ever carried input_data. Now every
    main-thread chat span ships its own user turn's prompt, byte-exact from
    the stream, with the hook corroborating each boundary."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="first question")
    asm.on_inbound(1, INIT)
    _submit(asm, "first question")
    asm.on_inbound(1, ASSISTANT)
    _outbound(asm, key=1, text="second question")
    _submit(asm, "second question")
    asm.on_inbound(1, ASSISTANT_2)
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    chat1, chat2 = _chats(client)
    assert b"first question" in chat1.input_data
    assert b"second question" not in chat1.input_data
    assert b"second question" in chat2.input_data
    assert b"first question" not in chat2.input_data
    assert chat1.capture_integrity.request_body_captured is True
    assert chat2.capture_integrity.request_body_captured is True
    assert ("wardex.agent.prompt_source", "stream") in chat1.extra
    assert ("wardex.agent.prompt_source", "stream") in chat2.extra
    # Turn numbering stays the per-assistant-message ordinal: uniqueness of
    # (conversation_id, turn_index) is load-bearing for stores that key on it.
    assert chat1.conversation.turn_index == 0
    assert chat2.conversation.turn_index == 1
    assert tallies("adapters.assembler.prompt_overwritten") == 0


def test_an_agentic_loop_between_two_prompts_claims_no_prompt():
    """An intermediate assistant turn of one agentic loop HAS no user prompt.
    It ships `input_attempted=False` — an honest "there was nothing to
    capture", not an empty capture reported as a successful one."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="run the loop")
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)  # stop_reason tool_use — the loop continues
    asm.on_inbound(1, TOOL_RESULT)
    asm.on_inbound(1, ASSISTANT_2)  # end_turn, still the same user turn
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    chat1, chat2 = _chats(client)
    assert chat1.capture_integrity.request_body_captured is True
    assert b"run the loop" in chat1.input_data
    assert chat2.input_data == b""
    assert chat2.capture_integrity.request_body_captured is False
    assert "wardex.agent.prompt_source" not in {k for k, _ in chat2.extra}


def test_the_hook_prompt_stands_down_when_the_stream_saw_the_turn(tallies):
    """The authority rule: the stream's byte-exact message-object JSON is the
    content, and the hook that follows the write it caused only corroborates —
    it must not replace those bytes with the CLI's re-decoded text."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="go")
    asm.on_inbound(1, INIT)
    _submit(asm, "go")
    asm.on_inbound(1, ASSISTANT)

    (chat,) = _chats(client)
    # The stream's shape (a message object), not the hook's bare text.
    assert b'"role"' in chat.input_data
    assert b"go" in chat.input_data
    assert ("wardex.agent.prompt_source", "stream") in chat.extra
    # Corroboration is not an overwrite.
    assert tallies("adapters.assembler.prompt_overwritten") == 0


def test_a_prompt_the_stream_missed_is_captured_from_the_hook_and_says_so():
    """The degraded-content fallback: when no outbound write recorded the turn,
    the hook's re-decoded text is captured — published as exactly the text
    wardex saw (prompt_source "hook"), never dressed up as a message object
    wardex never saw — and the hook arrival sets the turn boundary."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="first")
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)  # consumes turn 1's prompt
    t_mark = time.time_ns()
    _submit(asm, "second")  # turn 2's write never reached the tee
    asm.on_inbound(1, ASSISTANT_2)

    chat1, chat2 = _chats(client)
    assert chat2.input_data == b"second"
    assert chat2.capture_integrity.request_body_captured is True
    assert ("wardex.agent.prompt_source", "hook") in chat2.extra
    # The hook set the turn boundary — without the reset this span would
    # inherit turn 1's start instant.
    assert chat2.start_time_ns >= t_mark
    # replace_correlation(None) discipline holds on the new path too: sub-root
    # spans publish no confidence until the ingestion move earns the field.
    assert chat2.correlation is None
    assert chat1.correlation is None


def test_a_second_stream_prompt_with_no_assistant_between_keeps_the_last_and_counts_the_loss(
    tallies,
):
    """At most one prompt pends per session: a replacement is the bounded,
    honest behavior — and the replaced prompt is a real loss, so it moves a
    counter rather than vanishing."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="first")
    asm.on_inbound(1, INIT)
    _outbound(asm, key=1, text="second")
    asm.on_inbound(1, ASSISTANT)

    (chat,) = _chats(client)
    assert b"second" in chat.input_data
    assert b"first" not in chat.input_data
    assert chat.capture_integrity.request_body_captured is True
    assert tallies("adapters.assembler.prompt_overwritten") == 1


def test_a_second_hook_prompt_while_one_pends_reads_as_a_new_turn_not_a_duplicate(tallies):
    """What the corroboration flag buys. The first submit after a stream write
    IS that write's turn; a second submit while the same prompt still pends can
    only be a NEW user turn whose write the stream missed — treating it as a
    duplicate would silently drop a whole turn's prompt."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="go")
    asm.on_inbound(1, INIT)
    _submit(asm, "go")  # corroborates the pending stream prompt
    # ...turn 1 dies without an assistant message, and the user re-prompts
    # through a write the stream did not record.
    _submit(asm, "retry")
    asm.on_inbound(1, ASSISTANT)

    (chat,) = _chats(client)
    assert chat.input_data == b"retry"
    assert ("wardex.agent.prompt_source", "hook") in chat.extra
    # Turn 1's prompt was seen and lost, and the loss is counted.
    assert tallies("adapters.assembler.prompt_overwritten") == 1


def test_a_subagent_chat_does_not_consume_the_users_pending_prompt():
    """Consumption is gated on `parent_tool_use_id is None`: a
    subagent-attributed chat's input is the subagent's task, not the user's
    session prompt — it must leave the pending prompt for the next main-thread
    turn."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="go")
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "SubagentStart", {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"}, None
    )
    subagent_chat = {
        "type": "assistant",
        "session_id": "s-1",
        "parent_tool_use_id": "a-1",
        "message": {
            "id": "m-sub",
            "model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
            "content": [{"type": "text", "text": "sub result"}],
        },
    }
    asm.on_inbound(1, subagent_chat)  # BEFORE any main-thread assistant turn
    asm.on_inbound(1, ASSISTANT_2)

    sub_chat, main_chat = _chats(client)
    assert sub_chat.input_data == b""
    assert sub_chat.capture_integrity.request_body_captured is False
    assert "wardex.agent.prompt_source" not in {k for k, _ in sub_chat.extra}
    assert b"go" in main_chat.input_data
    assert ("wardex.agent.prompt_source", "stream") in main_chat.extra


def test_an_outbound_line_into_a_subagents_thread_does_not_overwrite_the_pending_prompt(tallies):
    """The install gate is SYMMETRIC with consumption. `_build_chat` consumes
    only when the chat's own `parent_tool_use_id` is None, so a host-written
    user line carrying one — the streaming-input/fork shapes — must not
    install either: it would overwrite the main-thread pending prompt, ride
    the next main chat span, and count an overwrite for a turn that never
    died."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="main question")
    asm.on_inbound(1, INIT)
    asm.on_outbound(
        1,
        json.dumps(
            {
                "type": "user",
                "session_id": "s-1",
                "parent_tool_use_id": "toolu_T",
                "message": {"role": "user", "content": "into the subagent"},
            }
        ),
    )
    asm.on_inbound(1, ASSISTANT_2)

    (chat,) = _chats(client)
    assert b"main question" in chat.input_data
    assert b"into the subagent" not in chat.input_data
    assert ("wardex.agent.prompt_source", "stream") in chat.extra
    assert tallies("adapters.assembler.prompt_overwritten") == 0


def test_an_outbound_line_into_a_subagents_thread_does_not_install_a_pending_prompt(tallies):
    """The other half of the symmetry: with nothing pending, the
    subagent-addressed line still installs nothing — the next main-thread chat
    honestly claims no prompt rather than riding a sub-agent's bytes."""
    client = FakeClient()
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {
                "type": "user",
                "session_id": "s-1",
                "parent_tool_use_id": "toolu_T",
                "message": {"role": "user", "content": "into the subagent"},
            }
        ),
    )
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT_2)

    (chat,) = _chats(client)
    assert chat.input_data == b""
    assert chat.capture_integrity.request_body_captured is False
    assert "wardex.agent.prompt_source" not in {k for k, _ in chat.extra}
    assert tallies("adapters.assembler.prompt_overwritten") == 0


def test_an_interrupted_tool_ships_tool_interrupted():
    """`error.type` refinement from the verified PostToolUseFailure payload:
    `is_interrupt: NotRequired[bool]` -> "tool_interrupted"; an absent key
    keeps the "tool_error" fallback (wire-value change, pre-1.0)."""
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_09"
    )
    asm.on_hook(
        "PostToolUseFailure",
        {
            "session_id": "s-1",
            "tool_name": "Bash",
            "error": "interrupted by user",
            "is_interrupt": True,
        },
        "toolu_09",
    )
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_10"
    )
    asm.on_hook(
        "PostToolUseFailure",
        {"session_id": "s-1", "tool_name": "Bash", "error": "boom"},
        "toolu_10",
    )

    interrupted, plain = (s for s in client.spans if s.name == "execute_tool Bash")
    assert interrupted.status is StatusCode.ERROR
    assert interrupted.error_type == "tool_interrupted"
    assert plain.status is StatusCode.ERROR
    assert plain.error_type == "tool_error"


# ==========================================================================
# the bounds a hand-installed adapter still owes its host
# ==========================================================================


class _ClientWithLimits(FakeClient):
    """`FakeClient` plus the one thing an install reads off a client: a config.

    The doubles above carry none, which is the right shape for a test about
    assembly and the wrong one for a test about bounds — a config-less client
    can only ever exercise the core defaults.
    """

    class _Config:
        debug = False

        def __init__(self, limits) -> None:
            self.limits = limits

    def __init__(self, limits) -> None:
        super().__init__()
        self.config = self._Config(limits)


def test_a_hand_installed_adapter_still_honours_the_configured_bounds():
    """`install(client)` with no context is a SUPPORTED path, and it is bounded.

    `ctx` defaults to None and the adapter says so out loud when it is missing
    (a `report_once` about in-process tool spans), so an adapter installed by
    hand is documented and warned, not unsupported — which means the registry
    it builds for itself is a production table and owes the host the bounds the
    host configured. Two of the four arrived; `max_link_targets` was never
    passed at all and `max_body_bytes` had no parameter to arrive through.
    """
    from wardex_sdk._adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter
    from wardex_sdk._limits import LimitsConfig

    client = _ClientWithLimits(
        LimitsConfig(
            max_units=7,
            max_entries_per_unit=3,
            max_link_targets=5,
            max_body_bytes=4096,
        )
    )
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client)
    try:
        units = adapter._assembler.units
        assert units._max_units == 7
        assert units._max_entries_per_unit == 3
        assert units._max_link_targets == 5
        assert units._max_record_bytes == 4096
    finally:
        adapter.uninstall()
