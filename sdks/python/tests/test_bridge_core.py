"""The two core pieces a bridge adapter stands on: the usage hold and the request latch.

A bridge turns a framework's own `chat` span into a CHAT unit, and the wire CHAT the byte seams
record from the same call nests under it. Both carry usage; shipping both in the typed block counts
one call twice. So the bridge writes the framework's numbers as DEMOTED extras and declares the unit
a holder (`_assembly/_usage_hold.py`); the seam's seal witnesses every transaction to a model-call
endpoint completed under a holder, and the unit's close promotes the demoted numbers only when
nothing was witnessed.

The latch also takes a resolver, because the framework can re-attach its span on a task wardex's
carrier never reached (measured on pydantic-ai's streaming run: the request leaves from the consumer
task, whose wardex ambient is the enclosing run). What is pinned here is that the resolver can move
a request DOWN, to a live descendant of the ambient unit, and nowhere else.

A second resolver answers a tool CALL: a byte seam may see a framework's tool call on a task the
tool never ran on (pydantic-ai's MCP client writes the server's stdin from a task it spawned when
the toolset connected), and `latch_tool_call` parents the wire call to the ONE live tool unit of
that name whose recorded arguments equal the wire's. What is pinned here is that it does so only
for a live `execute_tool` unit, never to a unit whose arguments differ, by name alone only below
the writer's ambient unit, never for an ambiguous call, and never at the request's cost.

No framework and no OpenTelemetry here: a slot token stands in for the framework's current span.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from wardex_sdk import _hub, _runtime
from wardex_sdk._adapters._context import AdapterContext, Placement, RunHandle
from wardex_sdk._assembly import (
    EMPTY_AMBIENT,
    NULL_DRAFT,
    Ambient,
    GenAIAttributes,
    Limitation,
    SpanIntent,
    UnitKind,
    UnitRegistry,
    _bridge_latch,
    _usage_hold,
    canonical_arguments,
    counters,
    hold_framework_usage,
    latch_parent,
    latch_tool_call,
)
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._assembly._units import _ambient_unit
from wardex_sdk._enums import AgentType, OperationName
from wardex_sdk._hub import reset_for_test
from wardex_sdk._interceptors._h2_issuer import StreamIssuers
from wardex_sdk._interceptors._http1_requests import _issued_here
from wardex_sdk._interceptors._mcp_stdio import _ProcState
from wardex_sdk._interceptors._seam import ByteSeamInterceptor, _ConnectionState
from wardex_sdk._interceptors._txn import _Txn
from wardex_sdk._types import AgentAttributes, ConversationContext, ToolAttributes

PREFIX = "wardex.test.otel."


class RecordingSink:
    def __init__(self) -> None:
        self.drafts: list = []

    def emit(self, draft, *, agent_semantic: bool) -> bool:
        self.drafts.append(draft)
        return True

    def span(self, name: str):
        return next(d.finish() for d in self.drafts if d.name == name)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_for_test()
    token = _ambient_unit.set(None)
    counters.reset()
    reset_reports_for_test()
    monkeypatch.setattr(_bridge_latch, "_RESOLVERS", ())
    monkeypatch.setattr(_bridge_latch, "_TOOL_RESOLVERS", ())
    _usage_hold._HOLDERS.clear()
    yield
    _usage_hold._HOLDERS.clear()
    _ambient_unit.reset(token)
    reset_for_test()
    counters.reset()
    reset_reports_for_test()


def context(name: str = "test") -> tuple[AdapterContext, RecordingSink]:
    sink = RecordingSink()
    return AdapterContext(name, units=UnitRegistry(sink=sink), limits={}), sink


def _here() -> object:
    return threading.current_thread()


def _describe_chat(handle) -> None:
    handle.draft.set_gen_ai(GenAIAttributes(operation=OperationName.CHAT, request_model="m"))


def _describe_run(handle) -> None:
    handle.draft.set_agent(AgentAttributes(name="a", agent_type=AgentType.PRIMARY))


def _chat(ctx: AdapterContext, **usage: object) -> RunHandle:
    """A bridge-shaped CHAT unit: ROOT placement, its framework usage written demoted."""
    h = ctx.open_run(
        UnitKind.CALL,
        intent=SpanIntent.CHAT,
        placement=Placement.ROOT,
        subject="m",
        describe=_describe_chat,
    )
    for key, value in usage.items():
        h.draft.set_extra(PREFIX + key, value)
    return h


def _run(ctx: AdapterContext) -> RunHandle:
    h = ctx.open_run(
        UnitKind.SESSION,
        intent=SpanIntent.INVOKE_AGENT,
        placement=Placement.ROOT,
        describe=_describe_run,
    )
    assert h.pin(driver=_here()) is True
    return h


def _latch():
    scope = _hub.get_current_scope()
    return latch_parent(scope.active_span_context, scope.conversation)


def _limitations(span) -> set[Limitation]:
    return set(span.capture_integrity.limitations) if span.capture_integrity else set()


def _extras(span) -> dict[str, object]:
    return dict(span.extra)


# ==========================================================================
# Core change A — the usage hold
# ==========================================================================


class _Seam(ByteSeamInterceptor):
    """The base seam with no transport opinion: `_emit_span` seals on this thread and hands the job
    to the client, which records it and never parses it."""

    def _select_tracker(self, obj):  # pragma: no cover - not reached here
        raise NotImplementedError

    def _resolve_timing(self, obj, st):
        return 0.0, 0.0, False, ()

    def name(self):
        return "test-seam"

    def install(self, client, ctx=None):
        self._client = client

    def uninstall(self):
        pass


class _QueueOnly:
    """A client that keeps the sealed jobs and runs none: the witness may not wait for a parse."""

    config = None

    def __init__(self) -> None:
        self.jobs: list = []

    def capture_deferred(self, job) -> None:
        self.jobs.append(job)

    def capture_span(self, span, *, scope=None) -> None:  # pragma: no cover - gRPC only
        raise AssertionError("nothing here assembles inline")


def _seal(parent, path: str, status: int = 200) -> _QueueOnly:
    """Complete one HTTP/1 transaction latched to `parent` through the seam's real seal."""
    seam, client = _Seam(), _QueueOnly()
    seam._client = client
    txn = _Txn(
        method="POST",
        path=path,
        status=status,
        request_body=b"{}",
        response_body=b"{}",
        parent=parent,
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
    )
    seam._emit_span(SimpleNamespace(), _ConnectionState(None, "api.example.com", 443), txn)
    return client


@pytest.mark.parametrize("status", [200, 500])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_a_sealed_model_call_under_a_held_chat_unit_keeps_its_usage_demoted(path, status):
    """The wire span under the unit is the authority: nothing is promoted, nothing is marked. The
    witness is written at the seal, before the job is queued, so it never waits for the parse; a
    5xx is witnessed too, because its wire span exists and carries the failure."""
    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 3})
    assert ctx.hold_framework_usage(chat, prefix=PREFIX) is True

    client = _seal(chat.draft.context, path, status)
    assert len(client.jobs) == 1 and client.jobs[0].treatment == "llm_call", "queued, not parsed"
    chat.close()

    span = sink.span("chat m")
    assert span.gen_ai.input_tokens is None and span.gen_ai.output_tokens is None
    assert _extras(span)[PREFIX + "gen_ai.usage.input_tokens"] == 10
    assert _extras(span)[PREFIX + "gen_ai.usage.output_tokens"] == 3
    assert Limitation.NO_WIRE_EVIDENCE not in _limitations(span)
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    assert counters.get("assembly._usage_hold.usage_promoted") == 0
    assert not _usage_hold._HOLDERS


@pytest.mark.parametrize("path", ["/v1/anything/else", "/v1/traces/ingest"])
def test_a_sealed_request_wardex_does_not_read_as_a_model_call_promotes_the_usage(path):
    """Bedrock InvokeModel, a custom model's own API, a telemetry upload: the request went out under
    the unit, but its wire span (if any) carries no usage. The framework's numbers are the only
    ones, so they are promoted and the unit says no interceptor read the call."""
    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 3})
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    chat.pin(driver=_here())
    parent = _issued_here().parent
    assert parent == chat.draft.context, "precondition: the request is latched under the unit"
    _seal(parent, path)
    chat.unpin()
    chat.close()

    span = sink.span("chat m")
    assert (span.gen_ai.input_tokens, span.gen_ai.output_tokens) == (10, 3)
    assert PREFIX + "gen_ai.usage.input_tokens" not in _extras(span)
    assert Limitation.NO_WIRE_EVIDENCE in _limitations(span)
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 0
    assert counters.get("assembly._usage_hold.usage_promoted") == 1


class _Recording:
    """A client that records what the seam hands it, inline or deferred, and parses nothing."""

    config = None

    def __init__(self) -> None:
        self.spans: list = []
        self.jobs: list = []

    def capture_deferred(self, job) -> None:
        self.jobs.append(job)

    def capture_span(self, span, *, scope=None) -> None:
        self.spans.append(span)


def test_a_model_call_over_a_websocket_promotes_the_usage():
    """A WebSocket session is not read for usage, so it witnesses nothing: the framework's
    numbers are promoted, marked NO_WIRE_EVIDENCE, and the session span under the unit carries
    no usage of its own, so the call is counted once."""
    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 3})
    assert ctx.hold_framework_usage(chat, prefix=PREFIX) is True
    seam, client = _Seam(), _Recording()
    seam._client = client
    txn = _Txn(
        method="GET",
        path="/v1/responses",
        status=101,
        request_body=b"",
        response_body=b"",
        parent=chat.draft.context,
        start_ns=1,
        end_ns=2_000_000,
        ttfb_ms=None,
        version="websocket",
        ws_close_code=1000,
        ws_messages_sent=2,
        ws_messages_received=3,
        ws_bytes_sent=10,
        ws_bytes_received=20,
        ws_llm_call=True,
    )
    seam._emit_ws(_ConnectionState(None, "api.openai.com", 443), txn)
    chat.close()

    span = sink.span("chat m")
    assert (span.gen_ai.input_tokens, span.gen_ai.output_tokens) == (10, 3)
    assert not [k for k in _extras(span) if k.startswith(PREFIX)]
    assert Limitation.NO_WIRE_EVIDENCE in _limitations(span)
    [ws] = client.spans
    assert ws.gen_ai is None, "the session span carries no usage"
    assert ws.parent_span_id == chat.draft.context.span_id
    assert client.jobs == []
    assert counters.get("assembly._usage_hold.usage_promoted") == 1
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 0


def _h2_latched():
    issuers = StreamIssuers(8)
    assert issuers.record(1) is True
    return issuers.take(1)[0]


def _h2_headers(stream_id: int) -> bytes:
    """One HEADERS frame (END_HEADERS | END_STREAM) that opens `stream_id`."""
    from hpack import Encoder

    block = Encoder().encode([(b":method", b"POST"), (b":path", b"/v1/chat/completions")])
    head = len(block).to_bytes(3, "big") + bytes([0x1, 0x4 | 0x1]) + stream_id.to_bytes(4, "big")
    return head + block


def _h2_writer_latched():
    from wardex_sdk._interceptors._trackers import _Http2Tracker

    tracker = _Http2Tracker()
    tracker.on_request_bytes(_h2_headers(1))
    assert tracker._latch[1][4] is False, "precondition: no issuer was recorded for stream 1"
    return tracker._latch[1][0]


def test_a_request_issued_but_never_sealed_promotes_the_usage():
    """Issuing is not evidence: a model call that timed out or was cancelled before its response
    completed leaves no wire span with usage, so every latch site (HTTP/1, the h2 issuer, the h2
    writer fallback) latches the parent and witnesses nothing."""
    ctx, sink = context()
    for name, issue in (
        ("http1", lambda: _issued_here().parent),
        ("h2 issuer", _h2_latched),
        ("h2 writer", _h2_writer_latched),
    ):
        chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10})
        ctx.hold_framework_usage(chat, prefix=PREFIX)
        chat.pin(driver=_here())
        assert issue() == chat.draft.context, name
        chat.unpin()
        chat.close()
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 0
    assert counters.get("assembly._usage_hold.usage_promoted") == 3
    assert all(Limitation.NO_WIRE_EVIDENCE in _limitations(d.finish()) for d in sink.drafts)


def test_a_held_chat_unit_with_no_request_promotes_every_spelling_and_says_so():
    """No wire saw the call, so the framework's numbers are the only ones: promoted into the typed
    block, the promoted pairs removed (one number, one home), `details.*` kept, and the span marked
    `NO_WIRE_EVIDENCE` — its definition, literally."""
    ctx, sink = context()
    dotted = _chat(
        ctx,
        **{
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.usage.cache_read.input_tokens": 30,
            "gen_ai.usage.cache_creation.input_tokens": 20,
            "gen_ai.usage.reasoning.output_tokens": 10,
            "gen_ai.usage.details.thoughts": 7,
        },
    )
    assert ctx.hold_framework_usage(dotted, prefix=PREFIX) is True
    dotted.close()
    span = sink.span("chat m")
    g = span.gen_ai
    assert (g.input_tokens, g.output_tokens) == (100, 50)
    assert (g.cache_read_input_tokens, g.cache_creation_input_tokens) == (30, 20)
    assert g.reasoning_output_tokens == 10
    assert _extras(span).get(PREFIX + "gen_ai.usage.details.thoughts") == 7
    assert not [
        k for k in _extras(span) if k.startswith(PREFIX + "gen_ai.usage.") and "details" not in k
    ]
    assert Limitation.NO_WIRE_EVIDENCE in _limitations(span)

    sink.drafts.clear()
    underscored = _chat(
        ctx,
        **{
            "gen_ai.usage.input_tokens": 9,
            "gen_ai.usage.cache_read_input_tokens": 4,
            "gen_ai.usage.cache_creation_input_tokens": 2,
        },
    )
    ctx.hold_framework_usage(underscored, prefix=PREFIX)
    underscored.close()
    g = sink.span("chat m").gen_ai
    assert (g.input_tokens, g.cache_read_input_tokens, g.cache_creation_input_tokens) == (9, 4, 2)

    assert counters.get("assembly._usage_hold.usage_promoted") == 2
    assert counters.get("assembly._usage_hold.promote_skipped") == 0
    assert counters.get("assembly.builder.gen_ai_usage_not_inclusive") == 0


def test_promotion_never_writes_a_top_level_usage_extra():
    """The typed block is what the encoder exports as `gen_ai.usage.*`; a same-named extra beside
    it would be the second copy this whole change exists to prevent."""
    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 5, "gen_ai.usage.output_tokens": 1})
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    chat.close()
    span = sink.span("chat m")
    assert span.gen_ai.input_tokens == 5
    assert not [k for k, _ in span.extra if k.startswith("gen_ai.usage")]


def test_promotion_never_overwrites_a_typed_field_and_skips_what_is_not_a_count():
    """A field already filled came from something that read it first; a bool is an int to Python
    and never a token count. Both stay as demoted extras, counted."""
    ctx, sink = context()
    chat = _chat(
        ctx,
        **{
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 3,
            "gen_ai.usage.reasoning.output_tokens": True,
            "gen_ai.usage.cache_read.input_tokens": "4",
        },
    )
    chat.draft.set_gen_ai(
        GenAIAttributes(operation=OperationName.CHAT, request_model="m", input_tokens=99)
    )
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    chat.close()

    span = sink.span("chat m")
    assert span.gen_ai.input_tokens == 99, "never overwritten"
    assert span.gen_ai.output_tokens == 3
    assert span.gen_ai.reasoning_output_tokens is None
    assert span.gen_ai.cache_read_input_tokens is None
    extras = _extras(span)
    assert extras[PREFIX + "gen_ai.usage.input_tokens"] == 10
    assert extras[PREFIX + "gen_ai.usage.reasoning.output_tokens"] is True
    assert extras[PREFIX + "gen_ai.usage.cache_read.input_tokens"] == "4"
    assert PREFIX + "gen_ai.usage.output_tokens" not in extras
    assert counters.get("assembly._usage_hold.promote_skipped") == 3


def test_a_unit_nobody_held_is_untouched():
    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10})
    chat.close()
    span = sink.span("chat m")
    assert span.gen_ai.input_tokens is None
    assert _extras(span)[PREFIX + "gen_ai.usage.input_tokens"] == 10
    assert Limitation.NO_WIRE_EVIDENCE not in _limitations(span)
    for key in ("usage_promoted", "usage_kept_demoted", "promote_skipped"):
        assert counters.get(f"assembly._usage_hold.{key}") == 0


def test_a_held_unit_settles_on_every_close_path():
    """A force-closed child (its parent closed first) and a teardown's `close_all` both pass the
    unit close path, so neither leaves an entry behind."""
    ctx, sink = context()
    run = _run(ctx)
    child = _chat(ctx, **{"gen_ai.usage.input_tokens": 1})
    ctx.hold_framework_usage(child, prefix=PREFIX)
    run.unpin()
    run.close()
    span = sink.span("chat m")
    assert span.gen_ai.input_tokens == 1
    assert Limitation.NO_WIRE_EVIDENCE in _limitations(span)
    assert not _usage_hold._HOLDERS

    sink.drafts.clear()
    orphan = _chat(ctx, **{"gen_ai.usage.input_tokens": 2})
    ctx.hold_framework_usage(orphan, prefix=PREFIX)
    ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)
    assert sink.span("chat m").gen_ai.input_tokens == 2
    assert not _usage_hold._HOLDERS
    assert counters.get("assembly._usage_hold.usage_promoted") == 2


def test_a_hold_is_refused_where_it_could_not_be_settled_or_would_leave_its_namespace():
    ctx, _ = context()
    tool = ctx.open_run(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.ROOT,
        describe=lambda h: h.draft.set_tool(ToolAttributes(name="t")),
    )
    assert ctx.hold_framework_usage(tool, prefix=PREFIX) is False, "not a CHAT unit"
    assert hold_framework_usage(NULL_DRAFT, prefix=PREFIX) is False
    chat = _chat(ctx)
    assert hold_framework_usage(chat.draft, prefix="gen_ai.") is False
    assert hold_framework_usage(chat.draft, prefix="wardex.") is False
    assert counters.get("assembly._usage_hold.hold_refused") == 4

    assert ctx.hold_framework_usage(RunHandle(None, ctx), prefix=PREFIX) is False, "degraded"
    assert ctx.hold_framework_usage(chat, prefix="wardex.other.otel.") is False, "foreign"
    assert counters.get("adapters.test.usage_hold_refused") == 2
    assert not _usage_hold._HOLDERS
    tool.close()
    chat.close()


# ==========================================================================
# Core change B — the resolver in the latch
# ==========================================================================


def _follow(ctx: AdapterContext, handle: RunHandle | None) -> object:
    """Register a resolver whose 'current framework span' is a token mapped to `handle`."""
    token = type("FrameworkSpan", (), {})()
    if handle is not None:
        ctx.slot(token)["handle"] = handle
    ctx.follow_current_span(lambda: token)
    return token


def test_the_resolver_parents_a_request_to_a_live_descendant_of_the_ambient_unit():
    """The streaming shape: the run is ambient on the issuing task, the framework's chat span is
    current there, and the chat unit is below the run. The request goes to the chat unit, and the
    seal witnesses it there."""
    ctx, sink = context()
    run = _run(ctx)
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10})
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    token = _follow(ctx, chat)
    conv = ConversationContext(conversation_id="c")

    parent, conversation = latch_parent(run.draft.context, conv)
    assert parent == chat.draft.context
    assert conversation is (chat._unit.conversation or conv)
    assert counters.get("assembly._bridge_latch.parent_resolved") == 1
    assert _issued_here().parent == chat.draft.context, "the HTTP/1 seam takes the same answer"
    _seal(parent, "/v1/chat/completions")

    chat.close()
    run.unpin()
    run.close()
    span = sink.span("chat m")
    assert span.gen_ai.input_tokens is None, "sealed under the resolved parent, so kept demoted"
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    del token


def test_the_resolver_never_moves_a_request_anywhere_else():
    ctx, _ = context()
    other_ctx, _ = context("other")
    foreign = _chat(other_ctx)  # opened before any pin: a root of another registry
    sibling = _chat(ctx)  # a root of this registry, not under the run
    run = _run(ctx)
    child = _chat(ctx)
    dead = _chat(ctx)
    dead.close()
    conv = ConversationContext(conversation_id="c")
    here = run.draft.context

    for label, target in (
        ("not a descendant", sibling),
        ("a closed unit", dead),
        ("another registry's unit", foreign),
        ("the ambient unit itself", run),
        ("no unit for the span", None),
    ):
        _follow(ctx, target)
        assert latch_parent(here, conv) == (here, conv), label

    _follow(ctx, child)
    elsewhere = sibling.draft.context
    assert latch_parent(elsewhere, conv) == (elsewhere, conv), "the scope's parent is not ambient"
    assert latch_parent(None, conv) == (None, conv)

    run.unpin()
    assert latch_parent(here, conv) == (here, conv), "no ambient unit: nothing is invented"
    assert counters.get("assembly._bridge_latch.parent_resolved") == 0
    for h in (child, run, sibling, foreign):
        h.close()


def test_a_resolver_that_raises_costs_the_deepening_never_the_request():
    ctx, _ = context()
    run = _run(ctx)

    def broken() -> object:
        raise RuntimeError("the framework's context read blew up")

    ctx.follow_current_span(broken)
    here = run.draft.context
    assert latch_parent(here, None) == (here, None)
    assert counters.get("assembly._bridge_latch.resolve") == 1
    run.unpin()
    run.close()


def test_unfollow_removes_the_resolver_and_is_idempotent():
    """What an adapter's uninstall calls: afterwards the latch is the identity again, and calling
    it twice, or for a context that never followed, is harmless."""
    ctx, _ = context()
    other, _ = context("other")
    run = _run(ctx)
    child = _chat(ctx)
    _follow(ctx, child)
    _follow(other, None)
    assert len(_bridge_latch._RESOLVERS) == 2
    _follow(ctx, child)
    assert len(_bridge_latch._RESOLVERS) == 2, "one resolver per owner"
    here = run.draft.context
    assert latch_parent(here, None)[0] == child.draft.context

    ctx.unfollow_current_span()
    ctx.unfollow_current_span()
    context("never")[0].unfollow_current_span()
    assert [owner for owner, _ in _bridge_latch._RESOLVERS] == [other]
    assert latch_parent(here, None) == (here, None)
    other.unfollow_current_span()
    assert _bridge_latch._RESOLVERS == ()
    child.close()
    run.unpin()
    run.close()


# ==========================================================================
# Core change C — the tool-call resolver for a seam that writes from another task
# ==========================================================================


def _tool_call(slot: dict) -> tuple[str, str | None] | None:
    name = slot.get("tool_name")
    return (name, slot.get("tool_arguments")) if name is not None else None


def _tool(
    ctx: AdapterContext,
    name: str,
    *,
    intent: SpanIntent = SpanIntent.EXECUTE_TOOL,
    arguments: str | None = None,
):
    """A bridge-shaped tool unit NESTED under the ambient unit, its slot naming the tool and, when
    given, the JSON arguments the framework recorded at START."""
    h = ctx.open_run(UnitKind.CALL, intent=intent, placement=Placement.NESTED, subject=name)
    token = type("FrameworkSpan", (), {})()
    slot = ctx.slot(token)
    slot.update(handle=h, tool_name=name)
    if arguments is not None:
        slot["tool_arguments"] = canonical_arguments(arguments)
    return h, token


def _ambient_of(handle: RunHandle) -> Ambient:
    return Ambient(span_context=handle.draft.context, conversation=None, tracestate=None)


def test_a_tool_call_is_latched_to_the_one_live_tool_unit_below_the_ambient_unit():
    ctx, _ = context()
    run = _run(ctx)
    tool, token = _tool(ctx, "add")
    ctx.follow_tool_calls(_tool_call)

    ambient, nested = latch_tool_call(_ambient_of(run), "add", {"a": 1})
    assert nested is True, "no recorded arguments: the name decides, below the ambient unit"
    assert ambient.span_context == tool.draft.context
    assert counters.get("assembly._bridge_latch.tool_call_resolved") == 1
    assert latch_tool_call(_ambient_of(run), "mul") == (_ambient_of(run), False), "another name"
    tool.close()
    run.unpin()
    run.close()
    del token


def test_canonical_arguments_spells_the_framework_text_and_the_wire_object_alike():
    assert canonical_arguments('{"b": 3, "a": 2}') == canonical_arguments({"a": 2, "b": 3})
    assert canonical_arguments(b'{"a":[1,"x"]}') == '{"a":[1,"x"]}'
    assert canonical_arguments("{}") == canonical_arguments({}) == "{}"
    assert canonical_arguments({"a": 2}) != canonical_arguments({"a": 3})
    assert canonical_arguments("not json") is None
    assert canonical_arguments({"a": object()}) is None
    assert counters.get("assembly._bridge_latch.canonical_arguments") == 2, "counted, not raised"


def test_a_tool_call_is_never_latched_to_a_unit_whose_arguments_differ():
    """A tool of the same name, called with other arguments (an unrelated MCP client in the same
    process, say), is not the framework's call: it keeps the seam's own placement."""
    ctx, _ = context()
    run = _run(ctx)
    tool, token = _tool(ctx, "add", arguments='{"a": 2, "b": 3}')
    ctx.follow_tool_calls(_tool_call)
    here = _ambient_of(run)
    assert latch_tool_call(here, "add", {"a": 100, "b": 200}) == (here, False)
    assert latch_tool_call(EMPTY_AMBIENT, "add", {"a": 100, "b": 200}) == (EMPTY_AMBIENT, False)
    ambient, nested = latch_tool_call(here, "add", {"b": 3, "a": 2})
    assert nested is True and ambient.span_context == tool.draft.context
    tool.close()
    run.unpin()
    run.close()
    del token


def test_concurrent_calls_of_one_tool_are_told_apart_by_their_arguments():
    ctx, _ = context()
    run = _run(ctx)
    first, t1 = _tool(ctx, "add", arguments='{"a": 2, "b": 3}')
    second, t2 = _tool(ctx, "add", arguments='{"a": 4, "b": 5}')
    ctx.follow_tool_calls(_tool_call)
    here = _ambient_of(run)
    assert latch_tool_call(here, "add", {"a": 2, "b": 3})[0].span_context == first.draft.context
    assert latch_tool_call(here, "add", {"a": 4, "b": 5})[0].span_context == second.draft.context
    twin, t3 = _tool(ctx, "add", arguments='{"a": 2, "b": 3}')
    assert latch_tool_call(here, "add", {"a": 2, "b": 3}) == (here, False), "same arguments"
    assert counters.get("adapters.test.tool_call_ambiguous") == 1
    for h in (twin, second, first):
        h.close()
    run.unpin()
    run.close()
    del t1, t2, t3


def test_a_tool_call_is_never_latched_outside_the_bounds():
    ctx, _ = context()
    run = _run(ctx)
    tool, t1 = _tool(ctx, "add")
    chat, t2 = _tool(ctx, "chatty", intent=SpanIntent.CHAT)
    dead, t3 = _tool(ctx, "gone")
    dead.close()
    ctx.follow_tool_calls(_tool_call)
    here = _ambient_of(run)
    assert latch_tool_call(here, "chatty") == (here, False), "not an execute_tool unit"
    assert latch_tool_call(here, "gone") == (here, False), "a closed unit"

    run.unpin()
    other = _run(ctx)  # a live root of its own: the tool is not below it
    there = _ambient_of(other)
    assert latch_tool_call(there, "add") == (there, False), "by name, not below the ambient unit"
    assert counters.get("assembly._bridge_latch.tool_call_unproved") == 1
    other.unpin()
    other.close()

    run.pin(driver=_here())
    twin, t4 = _tool(ctx, "add")
    assert latch_tool_call(here, "add") == (here, False), "two live calls of one tool"
    assert counters.get("adapters.test.tool_call_ambiguous") == 1
    assert counters.get("assembly._bridge_latch.tool_call_resolved") == 0
    for h in (twin, chat, tool):
        h.close()
    run.unpin()
    run.close()
    del t1, t2, t3, t4


def test_a_tool_call_whose_arguments_match_may_leave_the_writers_run():
    """One MCP toolset shared by two agents: its writer task was spawned by run A, so a call from
    run B is written under A's ambient. The arguments prove B's unit; the name alone would not."""
    ctx, _ = context()
    run_b = _run(ctx)
    tool, token = _tool(ctx, "add", arguments='{"a": 4, "b": 5}')
    ctx.follow_tool_calls(_tool_call)
    run_b.unpin()
    run_a = _run(ctx)
    under_a = _ambient_of(run_a)
    ambient, nested = latch_tool_call(under_a, "add", {"a": 4, "b": 5})
    assert nested is True and ambient.span_context == tool.draft.context
    assert latch_tool_call(under_a, "add", {"a": 6}) == (under_a, False)
    run_a.unpin()
    run_a.close()
    tool.close()
    run_b.close()
    del token


def test_a_tool_call_from_a_writer_whose_ambient_names_no_unit_needs_its_arguments_to_match():
    """The MCP client's writer task predates the run when the agent was entered first, so its
    ambient says nothing. The arguments the framework recorded are the evidence; a tool name
    alone, from a framework told not to record content, is too weak to move the call."""
    ctx, _ = context()
    run = _run(ctx)
    tool, token = _tool(ctx, "add", arguments='{"a": 2}')
    nameless, t2 = _tool(ctx, "mul")
    ctx.follow_tool_calls(_tool_call)
    run.unpin()

    ambient, nested = latch_tool_call(EMPTY_AMBIENT, "add", {"a": 2})
    assert nested is True and ambient.span_context == tool.draft.context
    foreign = Ambient(
        span_context=_chat(context("other")[0]).draft.context, conversation=None, tracestate=None
    )
    ambient, nested = latch_tool_call(foreign, "add", {"a": 2})
    assert nested is True and ambient.span_context == tool.draft.context, "a span, not a unit"
    assert latch_tool_call(EMPTY_AMBIENT, "mul", {"a": 2}) == (EMPTY_AMBIENT, False)
    assert counters.get("assembly._bridge_latch.tool_call_unproved") == 1
    nameless.close()
    tool.close()
    run.close()
    del token, t2


def test_a_tool_call_resolver_that_raises_costs_the_deepening_never_the_request():
    ctx, _ = context()
    run = _run(ctx)
    tool, token = _tool(ctx, "add")

    def broken(slot: dict) -> tuple[str, str | None] | None:
        raise RuntimeError("the adapter's name reader blew up")

    ctx.follow_tool_calls(broken)
    here = _ambient_of(run)
    assert latch_tool_call(here, "add") == (here, False)
    assert counters.get("assembly._bridge_latch.resolve_tool_call") == 1
    tool.close()
    run.unpin()
    run.close()
    del token


def test_unfollow_tool_calls_removes_the_resolver_and_is_idempotent():
    ctx, _ = context()
    other, _ = context("other")
    run = _run(ctx)
    tool, token = _tool(ctx, "add")
    ctx.follow_tool_calls(_tool_call)
    ctx.follow_tool_calls(_tool_call)
    other.follow_tool_calls(_tool_call)
    assert [owner for owner, _ in _bridge_latch._TOOL_RESOLVERS] == [ctx, other]
    ctx.unfollow_tool_calls()
    ctx.unfollow_tool_calls()
    context("never")[0].unfollow_tool_calls()
    assert [owner for owner, _ in _bridge_latch._TOOL_RESOLVERS] == [other]
    here = _ambient_of(run)
    assert latch_tool_call(here, "add") == (here, False)
    other.unfollow_tool_calls()
    assert _bridge_latch._TOOL_RESOLVERS == ()
    tool.close()
    run.unpin()
    run.close()
    del token


_CALL = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{}}}\n'
_RESULT = b'{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"5"}]}}\n'


def test_the_mcp_seam_nests_its_tools_call_under_the_tool_unit_and_does_not_promote_it():
    """Written from a task whose ambient names no unit, as pydantic-ai's MCP client does when the
    agent was entered first: the seam's span is the transport observation under the tool unit."""
    ctx, _ = context()
    run = _run(ctx)
    tool, token = _tool(ctx, "add", arguments="{}")
    ctx.follow_tool_calls(_tool_call)
    run.unpin()

    state = _ProcState()
    state.feed_request(_CALL)
    [(span, _)] = state.feed_response(_RESULT)
    assert span.name == "MCP tools/call"
    assert span.parent_span_id == tool.draft.context.span_id
    assert span.context.trace_id == tool.draft.context.trace_id
    assert span.tool is not None and span.tool.name == "add"
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 1

    ctx.unfollow_tool_calls()
    state.feed_request(_CALL)
    [(span, _)] = state.feed_response(_RESULT)
    assert span.name == "execute_tool add", "unresolved: promoted exactly as before"
    assert span.parent_span_id is None
    tool.close()
    run.close()
    del token


# ==========================================================================
# Cost and process lifecycle
# ==========================================================================


def test_with_no_bridge_the_latch_is_the_identity_and_reaches_nothing(monkeypatch):
    """One falsy module-global read: the resolver path is never entered."""

    def entered(*_a: object) -> object:
        raise AssertionError("the resolver path ran with no resolver registered")

    monkeypatch.setattr(_bridge_latch, "_resolved", entered)
    parent, conv = object(), object()
    out = latch_parent(parent, conv)  # type: ignore[arg-type]
    assert out[0] is parent and out[1] is conv


def test_the_latch_imports_no_opentelemetry():
    """Measured in a fresh interpreter: importing the SDK and latching a request pulls in no
    OpenTelemetry module — the core never names it; only a bridge's own resolver does."""
    script = textwrap.dedent(
        """
        import sys
        import wardex_sdk
        from wardex_sdk._assembly import latch_parent
        p, c = object(), object()
        out = latch_parent(p, c)
        assert out[0] is p and out[1] is c
        print(sorted(m for m in sys.modules if m.split(".")[0] == "opentelemetry"))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def test_a_fork_child_starts_with_no_usage_holds():
    """The child's registries drop their inherited units without closing them, so nothing would
    ever settle the parent's holds there; the runtime's fork path clears the table."""
    ctx, _ = context()
    chat = _chat(ctx)
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    assert _usage_hold._HOLDERS
    _runtime._fork_reinit_module("_assembly._usage_hold")
    assert not _usage_hold._HOLDERS
    source = open(_runtime.__file__, encoding="utf-8").read()
    assert '_fork_reinit_module("_assembly._usage_hold")' in source
    chat.close()
