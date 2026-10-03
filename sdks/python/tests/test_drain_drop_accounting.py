"""Every way `Client._drain` drops a batch leaves a count behind.

A batch that vanishes without a count is, to anyone reading the process
afterwards, indistinguishable from a quiet period: nothing was captured, or
nothing was sent. So each scenario below makes the drain drop in one specific
way, and the test checks conservation -- every span captured is delivered,
filtered on purpose by `before_send_envelope`, still waiting in the buffer, or
COUNTED:

  * on `_lost`, when close() could not ship it and nothing will retry it;
  * on `_dropped`, when it did not fit back into a full buffer;
  * in `counters` under `client.drain.*`, when a raise took it -- the hook's
    raise or the transport's.

A new drop path is added here with the counter it lands on; a drop path that
lands on none fails the conservation check rather than passing in silence.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import (
    Envelope,
    InternalSpan,
    InternalStateSnapshot,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import UNDELIVERED, Transport

pytestmark = pytest.mark.usefixtures("fresh_counters")

N = 6
EXPORT_RAISED = "client.drain.span_dropped.export_raised"
HOOK_RAISED = "client.drain.span_dropped.before_send_raised"


@pytest.fixture(autouse=True)
def _fresh_reports():
    """`report_once` is process-global: a key spent by an earlier test would
    hide the one line these tests count."""
    reset_reports_for_test()
    yield
    reset_reports_for_test()


def _span(name: str = "s") -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _snapshot() -> InternalStateSnapshot:
    return InternalStateSnapshot(
        trace_id=TraceId.generate(), span_id=SpanId.generate(), timestamp_ns=1
    )


def _client(transport: Transport, *, max_buffer_spans: int = 1000, **config) -> Client:
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
            limits=LimitsConfig(max_buffer_spans=max_buffer_spans),
            **config,
        ),
        transport,
    )
    c._worker.stop()  # only the drains each scenario names may run
    return c


def _capture(c: Client, n: int = N) -> None:
    for i in range(n):
        c.capture_span(_span(f"s{i}"))


class _Records(Transport):
    def __init__(self) -> None:
        self.spans: list[InternalSpan] = []

    def export(self, envelope: Envelope, *, timeout=None) -> None:
        self.spans.extend(envelope.spans)


class _Raises(_Records):
    def __init__(self, exc: Exception | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.exc = exc

    def export(self, envelope: Envelope, *, timeout=None) -> None:
        self.calls += 1
        raise self.exc if self.exc is not None else RuntimeError("backend exploded")


class _Declines(_Records):
    def __init__(self, on_export: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.on_export = on_export

    def export(self, envelope: Envelope, *, timeout=None) -> object:
        if self.on_export is not None:
            on_export, self.on_export = self.on_export, None
            on_export()
        return UNDELIVERED


@dataclass
class _Outcome:
    client: Client
    transport: _Records
    captured: int
    filtered: int = 0  # dropped by the host's own hook, on purpose: not a loss


def _hook_raises() -> _Outcome:
    def boom(envelope):
        raise RuntimeError("hook bug")

    t = _Records()
    c = _client(t, before_send_envelope=boom)
    _capture(c)
    c.flush()
    return _Outcome(c, t, N)


def _export_raises() -> _Outcome:
    t = _Raises()
    c = _client(t)
    _capture(c)
    c.flush()
    return _Outcome(c, t, N)


def _export_raises_on_close() -> _Outcome:
    t = _Raises()
    c = _client(t)
    _capture(c)
    c.close(1.0)
    return _Outcome(c, t, N)


def _export_raises_after_the_hook_filtered() -> _Outcome:
    def keep_half(envelope):
        return replace(envelope, spans=envelope.spans[: N // 2])

    t = _Raises()
    c = _client(t, before_send_envelope=keep_half)
    _capture(c)
    c.flush()
    return _Outcome(c, t, N, filtered=N - N // 2)


def _declined_on_close() -> _Outcome:
    t = _Declines()
    c = _client(t)
    _capture(c)
    c.close(1.0)
    return _Outcome(c, t, N)


def _declined_after_close() -> _Outcome:
    box: dict[str, Client] = {}

    def closes_wardex(envelope):
        box["c"].close(0.0)  # the host shuts wardex down from inside the drain
        return envelope

    t = _Declines()
    c = _client(t, before_send_envelope=closes_wardex)
    box["c"] = c
    _capture(c)
    c.flush()
    return _Outcome(c, t, N)


def _close_behind_an_export() -> _Outcome:
    t = _Records()
    c = _client(t)
    _capture(c)
    held, release = threading.Event(), threading.Event()

    def in_flight_export() -> None:
        with c._export_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=in_flight_export, daemon=True)
    holder.start()
    assert held.wait(5.0)
    try:
        c.close(0.05)  # the final drain never gets the slot
    finally:
        release.set()
        holder.join(5.0)
    return _Outcome(c, t, N)


def _declined_into_a_refilled_buffer() -> _Outcome:
    cap = 4
    box: dict[str, Client] = {}
    t = _Declines(on_export=lambda: _capture(box["c"], cap))  # refills while it is away
    c = _client(t, max_buffer_spans=cap)
    box["c"] = c
    _capture(c, cap)
    c.flush()
    return _Outcome(c, t, 2 * cap)


_SCENARIOS = {
    "hook_raises": (_hook_raises, f"counters:{HOOK_RAISED}"),
    "export_raises": (_export_raises, f"counters:{EXPORT_RAISED}"),
    "export_raises_on_close": (_export_raises_on_close, f"counters:{EXPORT_RAISED}"),
    "export_raises_after_the_hook_filtered": (
        _export_raises_after_the_hook_filtered,
        f"counters:{EXPORT_RAISED}",
    ),
    "declined_on_close": (_declined_on_close, "_lost"),
    "declined_after_close": (_declined_after_close, "_lost"),
    "close_behind_an_export": (_close_behind_an_export, "_lost"),
    "declined_into_a_refilled_buffer": (_declined_into_a_refilled_buffer, "_dropped"),
}


def _accounted(c: Client) -> dict[str, int]:
    out = {"_lost": c._lost, "_dropped": c._dropped}
    for key, n in counters.snapshot().items():
        if key.startswith("client.drain."):
            out[f"counters:{key}"] = n
    return out


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_every_drain_drop_path_leaves_a_count(scenario):
    make, where = _SCENARIOS[scenario]
    o = make()
    # Read before any cleanup: the next drain resets `_dropped`.
    accounted = _accounted(o.client)
    delivered = len(o.transport.spans)
    pending = len(o.client._buffer.spans)
    lost = o.captured - delivered - o.filtered - pending
    assert lost > 0, f"the scenario dropped nothing, so it guards nothing: {accounted}"
    assert accounted.get(where, 0) == lost, (
        f"{lost} span(s) left the drain uncounted on {where}; counted: {accounted}"
    )
    assert sum(accounted.values()) == lost, (
        f"captured {o.captured} = delivered {delivered} + filtered {o.filtered} + pending "
        f"{pending} + counted, but counted is {accounted} (expected {lost} in total)"
    )
    o.client.close(0.5)


def test_a_raising_export_is_said_once_per_process_and_counted_every_time(capsys):
    """Off-debug, the default every production process runs: one line however
    often the transport raises, and a count that keeps every dropped span."""
    t = _Raises()
    c = _client(t)
    assert c.config.debug is False, "the silence this guards against lived off-debug"
    capsys.readouterr()
    for _ in range(10):
        _capture(c)
        c.flush()  # must not raise into the host
    err = capsys.readouterr().err
    assert t.calls == 10
    assert counters.get(EXPORT_RAISED) == 10 * N
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert lines == [
        f"[wardex] transport export raised; a batch of {N} span(s) was dropped and is not "
        "retried, because the transport may have sent part of it (every such drop is counted "
        f"under {EXPORT_RAISED}; re-run with debug=True for the traceback)"
    ], f"expected exactly one line naming the counter, got {err!r}"
    assert "backend exploded" not in err, "the exception's text is debug-gated"
    c.close(0.5)


def test_a_raising_export_counts_dropped_state_snapshots_and_names_them(capsys):
    t = _Raises()
    c = _client(t)
    _capture(c, 2)
    c.capture_snapshot(_snapshot())
    c.capture_snapshot(_snapshot())
    c.capture_snapshot(_snapshot())
    capsys.readouterr()
    c.flush()
    assert counters.get(EXPORT_RAISED) == 2
    assert counters.get("client.drain.snapshot_dropped.export_raised") == 3
    assert "a batch of 2 span(s) and 3 state snapshot(s) was dropped" in capsys.readouterr().err
    c.close(0.5)


def test_a_raising_before_send_envelope_counts_snapshots_too(capsys):
    def boom(envelope):
        raise RuntimeError("hook bug")

    c = _client(_Records(), before_send_envelope=boom)
    c.capture_snapshot(_snapshot())
    c.flush()
    assert counters.get("client.drain.snapshot_dropped.before_send_raised") == 1
    assert counters.get(HOOK_RAISED) == 0
    c.close(0.5)


def test_a_raising_export_shows_its_traceback_under_debug_even_if_str_raises(capsys):
    """Under debug the traceback is printed, and printing it may not raise.
    An exception whose `__str__` raises is not exotic -- it is a lazily
    formatted error from state already torn down -- and a handler that
    formatted it into an f-string raised a second time, out of `flush()` and
    into the host."""

    class _Unprintable(Exception):
        def __str__(self) -> str:
            raise ValueError("cannot format")

    t = _Raises(_Unprintable())
    c = _client(t, debug=True)
    _capture(c)
    capsys.readouterr()
    c.flush()  # must not raise
    err = capsys.readouterr().err
    assert counters.get(EXPORT_RAISED) == N
    assert "transport export raised" in err
    assert "swallowed in client.export" in err and "_Unprintable" in err, err
    c.close(0.5)
