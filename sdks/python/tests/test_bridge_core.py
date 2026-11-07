"""The two core pieces a bridge adapter stands on: the usage hold and the request latch.

A bridge turns a framework's own `chat` span into a CHAT unit, and the wire CHAT the byte seams
record from the same call nests under it. Both carry usage; shipping both in the typed block counts
one call twice. So the bridge writes the framework's numbers as DEMOTED extras and declares the unit
a holder (`_assembly/_usage_hold.py`); the seams' latch (`_assembly/_bridge_latch.py`) witnesses
every request issued under a holder, and the unit's close promotes the demoted numbers only when
nothing was witnessed.

The latch also takes a resolver, because the framework can re-attach its span on a task wardex's
carrier never reached (measured on pydantic-ai's streaming run: the request leaves from the consumer
task, whose wardex ambient is the enclosing run). What is pinned here is that the resolver can move
a request DOWN, to a live descendant of the ambient unit, and nowhere else.

No framework and no OpenTelemetry here: a slot token stands in for the framework's current span.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading

import pytest

from wardex_sdk import _hub, _runtime
from wardex_sdk._adapters._context import AdapterContext, Placement, RunHandle
from wardex_sdk._assembly import (
    NULL_DRAFT,
    GenAIAttributes,
    Limitation,
    SpanIntent,
    UnitKind,
    UnitRegistry,
    _bridge_latch,
    _usage_hold,
    counters,
    hold_framework_usage,
    latch_parent,
)
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._assembly._units import _ambient_unit
from wardex_sdk._enums import AgentType, OperationName
from wardex_sdk._hub import reset_for_test
from wardex_sdk._interceptors._h2_issuer import StreamIssuers
from wardex_sdk._interceptors._http1_requests import _issued_here
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


def test_a_held_chat_unit_with_a_witnessed_request_keeps_its_usage_demoted():
    """The wire span under the unit is the authority: nothing is promoted, nothing is marked."""
    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 3})
    assert ctx.hold_framework_usage(chat, prefix=PREFIX) is True
    assert chat.pin(driver=_here()) is True

    parent, _ = _latch()
    assert parent == chat.draft.context, "with no resolver the latch is the scope's own parent"
    chat.unpin()
    chat.close()

    span = sink.span("chat m")
    assert span.gen_ai.input_tokens is None and span.gen_ai.output_tokens is None
    assert _extras(span)[PREFIX + "gen_ai.usage.input_tokens"] == 10
    assert _extras(span)[PREFIX + "gen_ai.usage.output_tokens"] == 3
    assert Limitation.NO_WIRE_EVIDENCE not in _limitations(span)
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    assert counters.get("assembly._usage_hold.usage_promoted") == 0
    assert not _usage_hold._HOLDERS


def test_the_http1_and_h2_seams_witness_the_unit_they_latch():
    """The witness is written by the seams themselves, at the scope read that latches the parent."""
    ctx, sink = context()
    for name, issue in (
        ("http1", lambda: _issued_here().parent),
        ("h2", lambda: _h2_latched()),
    ):
        chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10})
        ctx.hold_framework_usage(chat, prefix=PREFIX)
        chat.pin(driver=_here())
        assert issue() == chat.draft.context, name
        chat.unpin()
        chat.close()
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 2
    assert counters.get("assembly._usage_hold.usage_promoted") == 0


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


def test_the_h2_writer_witnesses_a_stream_whose_issuer_was_never_proven():
    """The writer-scope fallback: with no proven issuer the writer IS the stream's parent, so the
    held chat unit that wrote the HEADERS frame keeps its usage demoted."""
    from wardex_sdk._interceptors._trackers import _Http2Tracker

    ctx, _ = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10})
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    chat.pin(driver=_here())
    tracker = _Http2Tracker()
    tracker.on_request_bytes(_h2_headers(1))
    assert tracker._latch[1][4] is False, "precondition: no issuer was recorded for stream 1"
    chat.unpin()
    chat.close()
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    assert counters.get("assembly._usage_hold.usage_promoted") == 0


def test_the_h2_writer_does_not_witness_a_stream_another_task_issued():
    """A shared h2 connection: this task merely flushes a HEADERS frame another task issued. The
    stream's parent is its proven issuer, so the writer's held chat unit saw no request of its
    own and promotes its usage."""
    from wardex_sdk._interceptors._h2_issuer import UNKNOWN_ISSUER
    from wardex_sdk._interceptors._trackers import _Http2Tracker

    ctx, sink = context()
    chat = _chat(ctx, **{"gen_ai.usage.input_tokens": 10})
    ctx.hold_framework_usage(chat, prefix=PREFIX)
    chat.pin(driver=_here())
    tracker = _Http2Tracker()

    class _ProvenElsewhere:
        def see(self, data: bytes) -> None:
            return None

        def latch(self, stream_id, parent, parent_closed):
            return (None, False, None, UNKNOWN_ISSUER, True)

    tracker._issuers = _ProvenElsewhere()
    tracker.on_request_bytes(_h2_headers(1))
    assert tracker._latch[1][4] is True
    chat.unpin()
    chat.close()
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 0
    assert counters.get("assembly._usage_hold.usage_promoted") == 1
    assert sink.span("chat m").gen_ai.input_tokens == 10


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
    witness follows it there."""
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

    chat.close()
    run.unpin()
    run.close()
    span = sink.span("chat m")
    assert span.gen_ai.input_tokens is None, "witnessed through the resolver, so kept demoted"
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
# Cost and process lifecycle
# ==========================================================================


def test_with_no_bridge_the_latch_is_the_identity_and_reaches_nothing(monkeypatch):
    """Two falsy module-global reads: the resolver path is never entered."""

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
