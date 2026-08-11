import inspect

from wardex_sdk._client import Client, build_sdk_info
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import LimitsConfig
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


def _find_line(func, needle: str) -> int:
    """Locate the absolute source line number of `needle` (matched against a
    stripped source line) inside `func`'s body, for sys.settrace-based tests
    that inject a reentrant drain at an exact statement boundary.

    Fails with a clear, diagnostic AssertionError rather than a bare
    StopIteration if the text has drifted -- e.g. after a refactor of the
    line these tests target -- so a future maintainer gets a pointer to what
    to fix instead of an opaque error.
    """
    src_lines, start_line = inspect.getsourcelines(func)
    for i, line in enumerate(src_lines):
        if line.strip() == needle:
            return start_line + i
    raise AssertionError(
        f"could not find {needle!r} in {func.__qualname__}'s source -- "
        "this test's line-search target is stale after a refactor; update the needle"
    )


def test_build_sdk_info_has_runtime_meta():
    info = build_sdk_info()
    assert info.name == "wardex.python"
    assert info.python_version and info.os and info.arch


def test_flush_emits_buffered_spans_in_one_envelope():
    t = _Recording()
    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), t)
    c.capture_span(_span())
    c.capture_span(_span())
    assert t.envelopes == []
    c.flush()
    assert len(t.envelopes) == 1
    assert len(t.envelopes[0].spans) == 2
    c.close()


def test_close_flushes_and_is_idempotent():
    t = _Recording()
    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), t)
    c.capture_span(_span())
    c.close()
    c.close()
    assert len(t.envelopes) == 1


def test_flush_stamps_sent_at_ns():
    t = _Recording()
    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), t)
    c.capture_span(_span())
    c.flush()
    assert t.envelopes[0].header.sent_at_ns > 0
    c.close()


def test_span_buffer_respects_the_byte_budget():
    """Count-based capping alone cannot bound memory: 2048 large spans is gigabytes."""
    t = _Recording()
    cfg = WardexConfig(
        limits=LimitsConfig(max_buffer_bytes=64 * 1024, max_buffer_spans=1000),
        backend=BackendConfig(api_key="k"),
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
    cfg = WardexConfig(
        limits=LimitsConfig(max_buffer_bytes=64 * 1024), backend=BackendConfig(api_key="k")
    )
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
        limits=LimitsConfig(max_buffer_bytes=2000, max_buffer_spans=1000),
        backend=BackendConfig(api_key="k"),
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
    cfg = WardexConfig(
        limits=LimitsConfig(max_buffer_bytes=5000), backend=BackendConfig(api_key="k")
    )
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

        # Reach through _buffer directly: Client._spans is a read-only view,
        # deliberately, so that nothing can replace the deque without its
        # byte total. Swapping in a same-contents subclass here keeps the
        # pair consistent (identical sizes), and being explicit about
        # touching the internal object is honest about what this test does.
        c._buffer.spans = _TripwireDeque(c._buffer.spans)
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
    handler's flush() firing at the *bare statement boundary* right before
    `self._buffer.append(span, size)` -- no function call happens between
    the eviction loop's last (failing) condition check and that statement,
    so no monkeypatched call boundary (the technique the other reentrancy
    tests use) can inject a drain there. This uses sys.settrace's line-level
    hook instead, which fires before a given source line executes regardless
    of whether that line makes a call.

    self._buffer.append(...) resolves self._buffer fresh, right there in the
    call -- unlike an earlier version of this method, which appended to a
    loop-cached local and could orphan the span if a drain swapped
    self._buffer out from under that stale reference in this exact window.
    With the fresh resolve, a drain landing here still exports without the
    new span, but the append that follows then targets whatever buffer is
    live *at that point* -- the fresh, post-drain one -- so the span lands
    there and is never lost.
    """
    import sys

    import wardex_sdk._client as client_module

    target_line = _find_line(client_module.Client.capture_span, "self._buffer.append(span, size)")

    t = _Recording()
    cfg = WardexConfig(
        limits=LimitsConfig(max_buffer_bytes=5000), backend=BackendConfig(api_key="k")
    )
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
        # The injected drain fired *before* the append (self._buffer held only
        # the baseline at that point), so it exported just the baseline...
        assert len(t.envelopes) == 1
        assert len(t.envelopes[0].spans) == 1
        # ...and the append then landed in the fresh, post-drain buffer, so
        # the counter is exact -- not merely bounded -- against what's resident.
        actual = sum(len(s.output_data) + 512 for s in c._spans)
        assert c._buffered_bytes == actual
    finally:
        sys.settrace(None)
        c.close()


def test_eviction_subtraction_cannot_go_negative_across_a_reentrant_drain():
    """Reproduces (against the fixed code) the "negative" case found in the
    prior identity-gated design: a reentrant drain firing between
    `_SpanBuffer.evict_oldest`'s popleft() and its byte subtraction used to
    be able to reset the *separate* Client-level counter to 0 out from under
    a subtraction that had already passed an `if self._spans is spans:`
    check, driving it negative. With spans+bytes folded into one
    `_SpanBuffer` object, evict_oldest's `self` is fixed to whichever buffer
    it was called on for the method's whole duration -- a drain can only
    replace self._buffer for future callers, it can never reach into an
    already-resolved `self`'s own fields. So the subtraction always applies
    to the same object it popped from, and can't be reset by a swap it can't
    see.

    Injects a drain via sys.settrace right before the `self.bytes -=
    _span_size(evicted)` line inside _SpanBuffer.evict_oldest -- the exact
    analogue of the window that used to cause the negative-counter bug.
    """
    import sys

    import wardex_sdk._client as client_module

    target_line = _find_line(
        client_module._SpanBuffer.evict_oldest, "self.bytes -= _span_size(evicted)"
    )

    t = _Recording()
    cfg = WardexConfig(
        limits=LimitsConfig(max_buffer_bytes=2000, max_buffer_spans=1000),
        backend=BackendConfig(api_key="k"),
    )
    c = Client(cfg, t)
    fired = {"done": False}

    def line_tracer(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is client_module._SpanBuffer.evict_oldest.__code__
            and frame.f_lineno == target_line
            and not fired["done"]
        ):
            fired["done"] = True
            sys.settrace(None)
            c._drain(5.0)  # reentrant: same thread, same RLock capture_span already holds
            return None
        return line_tracer

    def call_tracer(frame, event, arg):
        if event == "call" and frame.f_code is client_module._SpanBuffer.evict_oldest.__code__:
            return line_tracer
        return call_tracer

    try:
        for _ in range(3):
            c.capture_span(_span(output_data=b"x" * 100))  # 3 * 612 = 1836 bytes resident

        sys.settrace(call_tracer)
        try:
            incoming = _span(output_data=b"y" * 300)  # 812 bytes; forces eviction to fit
            c.capture_span(incoming)  # the injected drain fires mid-eviction
        finally:
            sys.settrace(None)

        assert fired["done"], "trace hook never reached the target line -- test is stale"
        # The core assertion: never negative (the reproduced bug was -4488).
        assert c._buffered_bytes >= 0
        # Folding spans+bytes keeps the total exact, not merely non-negative.
        actual = sum(len(s.output_data) + 512 for s in c._spans)
        assert c._buffered_bytes == actual
        assert len(t.envelopes) == 1  # the mid-flight drain exported the survivors
    finally:
        c.close()


def test_append_increment_cannot_overstate_across_a_reentrant_drain():
    """Reproduces (against the fixed code) the "overstate" case found in the
    prior identity-gated design: a reentrant drain firing between
    `_SpanBuffer.append`'s deque append and its byte increment used to be
    able to export the just-appended span and reset the *separate*
    Client-level counter to 0, out from under an increment that had already
    passed its identity check -- leaving the counter overstated (the
    appended span's size) against an empty buffer. With spans+bytes folded
    into one object, append's `self` is fixed to whichever buffer it was
    called on, so the increment always applies to the same object the
    append landed in -- the swap can't retarget it.

    Injects a drain via sys.settrace right before the `self.bytes += size`
    line inside _SpanBuffer.append -- the exact analogue of the window that
    used to cause the overstated-counter bug.
    """
    import sys

    import wardex_sdk._client as client_module

    target_line = _find_line(client_module._SpanBuffer.append, "self.bytes += size")

    t = _Recording()
    cfg = WardexConfig(
        limits=LimitsConfig(max_buffer_bytes=5000), backend=BackendConfig(api_key="k")
    )
    c = Client(cfg, t)
    fired = {"done": False}

    def line_tracer(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is client_module._SpanBuffer.append.__code__
            and frame.f_lineno == target_line
            and not fired["done"]
        ):
            fired["done"] = True
            sys.settrace(None)
            c._drain(5.0)  # reentrant: same thread, same RLock capture_span already holds
            return None
        return line_tracer

    def call_tracer(frame, event, arg):
        if event == "call" and frame.f_code is client_module._SpanBuffer.append.__code__:
            return line_tracer
        return call_tracer

    try:
        c.capture_span(_span(output_data=b"x" * 100))  # baseline resident span, 612 bytes

        sys.settrace(call_tracer)
        try:
            new_span = _span(output_data=b"y" * 50)  # 562 bytes
            c.capture_span(new_span)  # the injected drain fires mid-append
        finally:
            sys.settrace(None)

        assert fired["done"], "trace hook never reached the target line -- test is stale"
        # The core assertion: never overstated (the reproduced bug was 612
        # resident against an empty buffer -- 0 actual spans).
        actual = sum(len(s.output_data) + 512 for s in c._spans)
        assert c._buffered_bytes == actual
        # The injected drain fired after the span was already appended to the
        # live buffer, so it was exported -- nothing is resident afterward.
        assert c._buffered_bytes == 0
        assert len(c._spans) == 0
        assert len(t.envelopes) == 1
        assert len(t.envelopes[0].spans) == 2  # baseline + new_span, both exported
    finally:
        c.close()
