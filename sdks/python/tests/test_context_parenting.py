"""Phase 4a — ContextVar fork parenting. Repro tests for the gather mis-parenting bug."""

import asyncio

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._tracing import span, trace
from wardex_sdk._types import InternalEnvelope
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(api_key="k"), t))
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
        with trace("root") as root:
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
        with trace("root"):
            await asyncio.gather(child(1), child(2))

    asyncio.run(main())
    assert seen == {1: True, 2: True}


def test_sequential_nesting_unchanged():
    """Regression: sequential nesting must keep the exact parent chain."""
    t = _setup()
    with trace("root") as root:
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
    with trace("root"):
        pass
    assert _hub.get_current_scope().tags["k"] == "v"
