from wardex_sdk._client import Client, build_sdk_info
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import CaptureLimits
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span(output_data: bytes = b""):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
        output_data=output_data,
    )


def test_build_sdk_info_has_runtime_meta():
    info = build_sdk_info()
    assert info.name == "wardex.python"
    assert info.python_version and info.os and info.arch


def test_flush_emits_buffered_spans_in_one_envelope():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span())
    c.capture_span(_span())
    assert t.envelopes == []
    c.flush()
    assert len(t.envelopes) == 1
    assert len(t.envelopes[0].spans) == 2
    c.close()


def test_close_flushes_and_is_idempotent():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span())
    c.close()
    c.close()
    assert len(t.envelopes) == 1


def test_flush_stamps_sent_at_ns():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span())
    c.flush()
    assert t.envelopes[0].header.sent_at_ns > 0
    c.close()


def test_span_buffer_respects_the_byte_budget():
    """Count-based capping alone cannot bound memory: 2048 large spans is gigabytes."""
    t = _Recording()
    cfg = WardexConfig(
        api_key="k",
        limits=CaptureLimits(max_buffer_bytes=64 * 1024, max_buffer_spans=1000),
    )
    c = Client(cfg, t)
    try:
        for _ in range(50):
            c.capture_span(_span(output_data=b"x" * 8192))
        assert c._buffered_bytes <= 64 * 1024
        assert c._dropped > 0
        assert len(c._spans) < 50
    finally:
        c.close()


def test_byte_budget_leaves_small_spans_alone():
    t = _Recording()
    cfg = WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_bytes=64 * 1024))
    c = Client(cfg, t)
    try:
        for _ in range(10):
            c.capture_span(_span(output_data=b"x" * 100))
        assert c._dropped == 0
        assert len(c._spans) == 10
    finally:
        c.close()


def test_byte_counter_survives_a_reentrant_drain_mid_eviction(monkeypatch):
    """`_buffer_lock` is an RLock specifically so a same-thread signal handler
    can call flush() (and so _drain()) mid-capture. This simulates that: the
    drain fires while capture_span's eviction loop is between popleft() and
    the byte-counter subtraction that accounts for it, swapping self._spans
    for a fresh deque and resetting self._buffered_bytes to 0 out from under
    the in-flight loop. If capture_span still applies the pre-computed delta
    to the (now reset) counter, it goes negative/overstated; if it still
    calls popleft() on the swapped-in empty deque on the next iteration, it
    raises IndexError. Neither must happen.
    """
    import wardex_sdk._client as client_module

    t = _Recording()
    cfg = WardexConfig(
        api_key="k", limits=CaptureLimits(max_buffer_bytes=2000, max_buffer_spans=1000)
    )
    c = Client(cfg, t)
    try:
        for _ in range(3):
            c.capture_span(_span(output_data=b"x" * 100))  # 3 * 612 = 1836 bytes resident

        incoming = _span(output_data=b"y" * 300)  # 812 bytes; forces eviction to fit
        real_span_size = client_module._span_size
        state = {"fired": False}

        def draining_span_size(span):
            # Only the *evicted* spans' size lookups happen inside the loop
            # (the incoming span's size is computed once, before the lock is
            # even acquired) -- fire on the first such call, exactly between
            # popleft() and the counter update in the caller.
            if span is not incoming and not state["fired"]:
                state["fired"] = True
                c._drain(5.0)  # reentrant: same thread, same RLock as capture_span holds
            return real_span_size(span)

        monkeypatch.setattr(client_module, "_span_size", draining_span_size)

        c.capture_span(incoming)  # must not raise, and must not corrupt the counter

        assert c._buffered_bytes >= 0
        actual = sum(real_span_size(s) for s in c._spans)
        assert c._buffered_bytes == actual
        # The mid-flight drain exported whatever was still resident at swap time.
        assert len(t.envelopes) == 1
    finally:
        c.close()


def test_byte_counter_survives_a_reentrant_drain_mid_append():
    """Same reentrancy hazard as above, but for the window between the final
    append() and the increment that accounts for it: a drain firing there
    exports the just-appended span before the increment runs, so the
    increment must not add its size to the (now reset, now-empty) counter.
    """
    from collections import deque as deque_type

    t = _Recording()
    cfg = WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_bytes=5000))
    c = Client(cfg, t)
    try:
        c.capture_span(_span(output_data=b"x" * 100))  # baseline resident span

        class _TripwireDeque(deque_type):
            fired = False

            def append(self, item):
                super().append(item)
                if not _TripwireDeque.fired:
                    _TripwireDeque.fired = True
                    c._drain(5.0)  # reentrant: same thread, same RLock as capture_span holds

        c._spans = _TripwireDeque(c._spans)
        c.capture_span(_span(output_data=b"y" * 50))  # must not raise, must not overstate

        assert c._buffered_bytes >= 0
        actual = sum(len(s.output_data) + 512 for s in c._spans)
        assert c._buffered_bytes == actual
        # The mid-append drain exported both the baseline and the new span.
        assert len(t.envelopes) == 1
        assert len(t.envelopes[0].spans) == 2
    finally:
        c.close()


def test_trailing_append_is_never_lost_to_a_reentrant_drain():
    """The narrowest and most severe reentrancy window: a same-thread signal
    handler's flush() firing at the *bare statement boundary* between the
    eviction loop's last (failing) condition check and `self._spans.append`
    -- no function call happens between those two statements, so no
    monkeypatched call boundary (the technique the other two reentrancy
    tests use) can inject a drain there. This uses sys.settrace's line-level
    hook instead, which fires before a given source line executes regardless
    of whether that line makes a call.

    If the append targeted a loop-cached local instead of self._spans fresh,
    a drain firing in this exact window would swap self._spans out from
    under that stale reference, and the span would land in an orphaned
    deque -- appended, but never resident, never exported, never counted.
    Silent, total data loss, and nothing about the counter's own consistency
    would reveal it. Appending to self._spans fresh instead means the span
    always lands in whatever is live *at the moment of the append call* --
    if a drain preempted it, that's the fresh deque, and the span survives.
    The byte counter may then undercount by one span's size until the next
    drain resets it (bounded, self-healing) -- but the span itself is never
    lost.
    """
    import inspect
    import sys

    import wardex_sdk._client as client_module

    src_lines, start_line = inspect.getsourcelines(client_module.Client.capture_span)
    target_line = next(
        start_line + i
        for i, line in enumerate(src_lines)
        if line.strip() == "self._spans.append(span)"
    )

    t = _Recording()
    cfg = WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_bytes=5000))
    c = Client(cfg, t)
    fired = {"done": False}

    def line_tracer(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is client_module.Client.capture_span.__code__
            and frame.f_lineno == target_line
            and not fired["done"]
        ):
            fired["done"] = True
            sys.settrace(None)  # disable before the reentrant call, avoid tracing _drain too
            c._drain(5.0)  # reentrant: same thread, same RLock capture_span already holds
            return None
        return line_tracer

    def call_tracer(frame, event, arg):
        if event == "call" and frame.f_code is client_module.Client.capture_span.__code__:
            return line_tracer
        return call_tracer

    try:
        c.capture_span(_span(output_data=b"x" * 100))  # baseline resident span, 612 bytes

        sys.settrace(call_tracer)
        try:
            new_span = _span(output_data=b"y" * 50)  # 562 bytes; well under the budget
            c.capture_span(new_span)  # the injected drain fires right before this appends
        finally:
            sys.settrace(None)

        assert fired["done"], "trace hook never reached the target line -- test is stale"
        # The core assertion: the span must be present, never silently dropped.
        assert any(s is new_span for s in c._spans)
        # The injected drain fired *before* the append (self._spans held only
        # the baseline at that point), so it exported just the baseline.
        assert len(t.envelopes) == 1
        assert len(t.envelopes[0].spans) == 1
        # The counter may understate against the freshly-swapped-in deque
        # (bounded, self-healing) but must never go negative or overstate.
        actual = sum(len(s.output_data) + 512 for s in c._spans)
        assert 0 <= c._buffered_bytes <= actual
    finally:
        sys.settrace(None)
        c.close()
