"""A hand-opened span still open when the process stops ships, marked as cut off.

The failure this pins shut: a `@wardex.workflow` (or any `wardex.span()` block)
open when the process received the default SIGTERM, or when `wardex.close()`
ran, never shipped. Its block never reached its end — the default SIGTERM ends
the process inside the signal handler, and `close()` refuses every capture after
it — so the backend got only the children that had already finished, each
pointing at a parent that never arrived and claiming full confidence in it.
Adapter units always had this guarantee; hand-opened spans now have the same,
with the same marker.

Two halves live beside their kind: the real SIGTERM in a child process is in
`test_batching_integration.py`, and the fork child's table in
`test_fork_semantics.py`.
"""

from __future__ import annotations

import signal
import threading

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport

_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)

    def spans(self):
        return [sp for env in self.envelopes for sp in env.spans]

    def named(self, name):
        return [sp for sp in self.spans() if sp.name == name]


def _markers(sp):
    return () if sp.capture_integrity is None else sp.capture_integrity.limitations


def _install(transport=None):
    transport = transport or _Recording()
    client = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            intercept=False,
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        transport,
    )
    _runtime.runtime().install(client, client.config)
    return transport, client


@pytest.fixture(autouse=True)
def clean_runtime():
    """Reset the runtime around every test and put the real signal table back:
    the SIG_DFL branch writes SIG_DFL into the table itself, and SIG_IGN left
    there would be inherited by every subprocess the session spawns later."""
    _runtime.runtime()._uninstall_signal_handlers()
    saved = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()
    for signum, handler in saved.items():
        if handler is not None:
            signal.signal(signum, handler)


def _signal_with(prev, monkeypatch):
    kills: list = []
    monkeypatch.setattr(_runtime.os, "kill", lambda *a: kills.append(a))
    _runtime.runtime()._prev_handlers[signal.SIGTERM] = prev
    _runtime._handler(signal.SIGTERM, None)
    return kills


# ==========================================================================
# the signal handler, driven in-process
# ==========================================================================


def test_the_default_sigterm_ships_an_open_span_once(monkeypatch):
    transport, _client = _install()
    with wardex.span("open"):
        assert _signal_with(signal.SIG_DFL, monkeypatch)  # the process would end here
        (shipped,) = transport.named("open")
        assert Limitation.UNIT_INTERRUPTED in _markers(shipped)
    _hub.get_client().flush()
    assert len(transport.named("open")) == 1, "the block's own exit shipped a second copy"


@pytest.mark.parametrize(
    "prev", [signal.SIG_IGN, lambda signum, frame: None], ids=["ignored", "app-handler"]
)
def test_a_signal_the_app_survives_leaves_open_spans_running(monkeypatch, prev):
    transport, client = _install()
    with wardex.span("still-running"):
        assert _signal_with(prev, monkeypatch) == []
        assert transport.named("still-running") == []
    client.flush()
    (shipped,) = transport.named("still-running")
    assert _markers(shipped) == ()


def test_one_span_that_fails_to_ship_cannot_cost_the_others_or_the_flush(monkeypatch):
    transport, client = _install()

    def broken(client, marker):
        raise RuntimeError("wardex bug")

    with wardex.span("good"):
        _runtime.runtime().track_open_span(-1, broken)
        _signal_with(signal.SIG_DFL, monkeypatch)
        (good,) = transport.named("good")
    assert Limitation.UNIT_INTERRUPTED in _markers(good)


# ==========================================================================
# close() and atexit
# ==========================================================================


def test_close_from_another_thread_ships_the_open_run_marked_once():
    transport, _client = _install()

    @wardex.workflow(name="run")
    def run():
        with wardex.span("child"):
            pass
        closer = threading.Thread(target=wardex.close)
        closer.start()
        closer.join(timeout=10)
        return "done"

    assert run() == "done"
    (root,) = transport.named("run")
    (child,) = transport.named("child")
    assert Limitation.UNIT_INTERRUPTED in _markers(root)
    assert child.parent_span_id.value == root.context.span_id.value


def test_close_inside_the_block_ships_it_and_a_later_init_gets_no_copy():
    transport, _client = _install()
    with wardex.span("cut"):
        wardex.close()
        later, _ = _install()
    _hub.get_client().flush()
    (cut,) = transport.named("cut")
    assert Limitation.UNIT_INTERRUPTED in _markers(cut)
    assert later.named("cut") == []


def test_re_init_leaves_an_open_span_to_finish_on_the_new_client():
    first, _ = _install()
    with wardex.span("spanning"):
        second, client = _install()
    client.flush()
    assert first.named("spanning") == []
    (shipped,) = second.named("spanning")
    assert _markers(shipped) == ()


def test_a_closed_client_interrupts_nothing():
    transport, client = _install()
    wardex.close()
    with wardex.span("after-close"):
        wardex.close()  # the client is already closed: nothing to ship to
        fresh, new_client = _install()
    new_client.flush()
    (shipped,) = fresh.named("after-close")
    assert _markers(shipped) == ()


def test_an_open_decorated_generator_ships_marked_at_close():
    transport, _client = _install()

    @wardex.tool(name="stream")
    def stream():
        yield 1
        yield 2

    items = stream()
    assert next(items) == 1
    wardex.close()
    (shipped,) = transport.named("stream")
    assert Limitation.UNIT_INTERRUPTED in _markers(shipped)
    assert list(items) == [2]  # the host's generator is not disturbed
    assert len(transport.named("stream")) == 1


# ==========================================================================
# the table is bounded, and its overflow is counted
# ==========================================================================


def test_a_full_table_turns_new_spans_away_and_counts_them(monkeypatch):
    monkeypatch.setattr(_runtime, "_MAX_OPEN_SPANS", 1)
    transport, _client = _install()
    before = counters.snapshot().get("_runtime.open_spans_full", 0)
    with wardex.span("root"):
        with wardex.span("overflow"):
            assert counters.snapshot().get("_runtime.open_spans_full", 0) == before + 1
            wardex.close()
    # The tracked root is the one shipped; the overflow had no guarantee.
    (root,) = transport.named("root")
    assert Limitation.UNIT_INTERRUPTED in _markers(root)
    assert transport.named("overflow") == []


def test_no_tracked_span_outlives_its_block():
    _install()
    with wardex.span("a"):
        with wardex.span("b"):
            assert len(_runtime.runtime()._open_spans) == 2
    assert _runtime.runtime()._open_spans == {}
