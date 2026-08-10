"""ContextVar fork parenting — repro tests for the gather mis-parenting bug."""

import asyncio
import threading

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._tracing import conversation, span
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


def _setup() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    return t


def _all_spans(t: _Recording):
    _hub.get_client().flush()
    return {sp.name: sp for sp in t.envelopes[0].spans}


def test_gather_children_parent_to_enclosing_span():
    """Three spans started concurrently under one parent must all parent to it,
    not to whichever sibling happened to write the shared scope last."""
    t = _setup()

    async def child(n: int):
        with span(f"child-{n}"):
            await asyncio.sleep(0.01)

    async def main():
        with conversation("root") as root:
            await asyncio.gather(child(1), child(2), child(3))
            return root.context.span_id

    root_sid = asyncio.run(main())
    spans = _all_spans(t)
    for n in (1, 2, 3):
        assert spans[f"child-{n}"].parent_span_id == root_sid, (
            f"child-{n} parented to {spans[f'child-{n}'].parent_span_id}, want root"
        )


def test_gather_interceptor_read_sees_own_span():
    """Trackers latch `get_current_scope().active_span_context` (_trackers.py:106).
    Inside each gather task that read must resolve to the task's own span."""
    _setup()
    seen: dict[int, bool] = {}

    async def child(n: int):
        with span(f"child-{n}") as s:
            await asyncio.sleep(0.01)
            active = _hub.get_current_scope().active_span_context
            seen[n] = active is not None and active.span_id == s.context.span_id

    async def main():
        with conversation("root"):
            await asyncio.gather(child(1), child(2))

    asyncio.run(main())
    assert seen == {1: True, 2: True}


def test_sequential_nesting_unchanged():
    """Regression: sequential nesting must keep the exact parent chain."""
    t = _setup()
    with conversation("root") as root:
        with span("mid") as mid:
            with span("leaf") as leaf:
                pass
    spans = _all_spans(t)
    assert spans["mid"].parent_span_id == root.context.span_id
    assert spans["leaf"].parent_span_id == mid.context.span_id
    assert spans["leaf"].context.trace_id == root.context.trace_id
    assert leaf.context.trace_id == root.context.trace_id


def test_tags_set_before_span_survive_after():
    """Scope mutations made outside spans stay on the shared scope (fork is span-local)."""
    _setup()
    _hub.get_current_scope().set_tag("k", "v")
    with conversation("root"):
        pass
    assert _hub.get_current_scope().tags["k"] == "v"


def test_bind_context_carries_active_span_to_thread():
    _setup()
    results: dict[str, object] = {}

    import wardex_sdk

    with conversation("root") as root:

        def work():
            active = _hub.get_current_scope().active_span_context
            results["sid"] = active.span_id if active else None

        th = threading.Thread(target=wardex_sdk.bind_context(work))
        th.start()
        th.join()
    assert results["sid"] == root.context.span_id


def test_bare_thread_does_not_inherit_context():
    """Documents the Python behavior the helper exists for."""
    _setup()
    results: dict[str, object] = {}
    with conversation("root"):

        def work():
            active = _hub.get_current_scope().active_span_context
            results["sid"] = active.span_id if active else None

        th = threading.Thread(target=work)
        th.start()
        th.join()
    assert results["sid"] is None


# --- activate_span: the general carrier fork_active_span is now an alias of ---
#
# The generalization exists because a logical unit carries a conversation
# identity and a tracestate alongside its span context, and a carrier that
# installs only the context leaves the other two behind on every task the unit
# spans — a sub-agent's spans would silently lose the conversation id its
# session issued. Both extra arguments are OVERRIDES, not assignments: `None`
# means "keep whatever the cloned scope had", which is what keeps this a true
# generalization rather than a replacement that clears a field merely because
# the caller did not restate it.


def _ctx():
    from wardex_sdk._assembly import EMPTY_AMBIENT, resolve_parentage

    return resolve_parentage(EMPTY_AMBIENT).child_context()


def test_fork_active_span_is_activate_span():
    """A delegating wrapper would be a second entry point to grow a second
    opinion in, which is the drift the _assembly/ extraction exists to end."""
    from wardex_sdk.context._contextvar import activate_span, fork_active_span

    assert fork_active_span is activate_span


def test_activate_span_installs_and_restores_the_span_context():
    _setup()
    from wardex_sdk.context._contextvar import activate_span

    ctx = _ctx()
    with activate_span(ctx):
        assert _hub.get_current_scope().active_span_context == ctx
    assert _hub.get_current_scope().active_span_context is None


def test_activate_span_installs_conversation_and_tracestate():
    _setup()
    from wardex_sdk._types import ConversationContext
    from wardex_sdk.context._contextvar import activate_span

    conv = ConversationContext(conversation_id="c-1")
    with activate_span(_ctx(), conversation=conv, tracestate="a=1"):
        scope = _hub.get_current_scope()
        assert scope.conversation == conv
        assert scope.tracestate == "a=1"
    assert _hub.get_current_scope().conversation is None


def test_activate_span_omitting_a_field_keeps_the_inherited_one():
    """The override rule. Writing `None` through would clear a conversation id
    the enclosing scope issued, on every nested activation that did not restate
    it — which is exactly the loss the generalization was introduced to stop."""
    _setup()
    from wardex_sdk._types import ConversationContext
    from wardex_sdk.context._contextvar import activate_span

    conv = ConversationContext(conversation_id="c-1")
    with activate_span(_ctx(), conversation=conv, tracestate="a=1"):
        inner = _ctx()
        with activate_span(inner):
            scope = _hub.get_current_scope()
            assert scope.active_span_context == inner
            assert scope.conversation == conv
            assert scope.tracestate == "a=1"
