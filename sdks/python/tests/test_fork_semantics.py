"""`os.fork()` semantics — the child-side reset, end to end.

The SDK registers exactly one fork hook (`after_in_child`; never `before` or
`after_in_parent` — the host's `os.fork()` must not wait on wardex), and that
hook makes three promises these tests hold:

  * a span ships from exactly ONE process — the parent owns and exports its
    pre-fork buffer, the child discards its inherited copy without emitting
    (prefork servers used to ship every master-buffered span N+1 times);
  * the child hangs on nothing it inherited — every SDK lock is REPLACED, not
    acquired, so a fork landing mid-export or mid-teardown leaves no lock the
    child could wait on forever;
  * patches survive, state does not — the monkeypatches crossed the fork in
    the memory image and still work; only the per-process mutable state
    (buffers, tables, threads records) is reset.

Every test forks for real (`os.fork` or the explicit `multiprocessing` "fork"
context — macOS defaults to spawn, which never runs the hook and is out of
scope by design). Child work is kept minimal and results come back over a
pipe; the child never runs pytest assertions of its own.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.testing import RecordingTransport

fork_only = pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")

#: 3.12+ warns on fork() in a multi-threaded process — which is this suite's
#: exact subject: the batch worker IS alive when the host forks, and the
#: child-side reset is what makes that safe for wardex's own state. Filtered
#: HERE, per design, so a future `-W error` policy cannot break the suite for
#: doing the thing it exists to test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)

#: A drain interval no test outlives: the parent worker sits idle in `wait()`
#: holding no locks, so nothing drains until a test says so.
_IDLE = BatchingConfig(flush_interval=3600.0, flush_on_signals=False)


def _span(name: str) -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )


def _exported_names(transport: RecordingTransport) -> list[str]:
    return [span.name for envelope in transport.envelopes for span in envelope.spans]


def _run_in_child(fn, timeout: float = 30.0):
    """fork(), run `fn` in the child, return (exitcode, its JSON payload).

    The child reports through a pipe and leaves through `os._exit` — atexit
    must NOT run in these children unless a test is specifically about it, and
    pytest's own machinery must never run there. A child that raises reports
    the repr and exits 1 so the parent's assertion message says what happened.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        code = 0
        try:
            payload = fn()
            encoded = json.dumps(payload).encode()
        except BaseException as exc:  # noqa: BLE001 — reported to the parent
            encoded = json.dumps({"error": repr(exc)}).encode()
            code = 1
        try:
            os.write(w, encoded)
            os.close(w)
        finally:
            os._exit(code)
    os.close(w)
    chunks = []
    while True:
        chunk = os.read(r, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r)
    deadline = time.monotonic() + timeout
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail(f"forked child {pid} did not exit within {timeout}s (hang)")
        time.sleep(0.01)
    payload = json.loads(b"".join(chunks).decode()) if chunks else None
    return os.waitstatus_to_exitcode(status), payload


@pytest.fixture()
def installed_recording():
    """A full `init()` against a recording transport, torn down afterwards.

    `init()` is the path that registers the fork hook (it is `Runtime`-owned
    and irreversible), so these tests go through it rather than building a
    bare `Client` — a bare client is exactly the shape the hook must NOT
    touch, and `test_batching_integration` already covers its lazy-respawn
    backstop.
    """
    transport = RecordingTransport()
    wardex.init(transport=transport, intercept=False, batching=_IDLE)
    yield transport, _hub.get_client()
    wardex.close()


# --------------------------------------------------------------------------
# §6.1 — the duplication itself: one span, one process, once
# --------------------------------------------------------------------------


@fork_only
def test_parent_exports_its_prefork_spans_exactly_once_and_child_only_its_own(
    installed_recording,
):
    """The failure this whole design removes: gunicorn/uWSGI/celery `--preload`
    forks AFTER `init()`, every worker inherits the master's buffered spans,
    and every worker's lazily-respawned worker thread used to export them
    again — N+1 copies of each on the backend, read as a token spike the
    application never had."""
    transport, client = installed_recording
    for i in range(3):
        client.capture_span(_span(f"parent-{i}"))

    def child():
        child_client = _hub.get_client()
        for i in range(2):
            child_client.capture_span(_span(f"child-{i}"))
        child_client.flush()
        return {"exported": _exported_names(transport)}

    code, payload = _run_in_child(child)
    assert code == 0, payload
    assert sorted(payload["exported"]) == ["child-0", "child-1"], (
        "the child must export exactly what IT captured — a parent-* name here "
        "is the inherited buffer shipping twice"
    )
    # The parent still owns its three, and they ship exactly once, from here.
    client.flush()
    assert sorted(_exported_names(transport)) == ["parent-0", "parent-1", "parent-2"]


@fork_only
def test_child_reset_is_counted_and_bounded(installed_recording):
    """The reset announces itself in the child's diagnostics — and stays cheap.

    The counter is bumped AFTER the diag reset (step 6 after step 0), so its
    presence also proves the ordering: a hook that cleared its own tally was
    rev-1's bug. The duration bound is deliberately generous (a CI box under
    load is not a benchmark), but it turns "the reset went quadratic" into a
    red test instead of a production stall; the measured figure on this
    machine is tens of microseconds.
    """
    _transport, _client = installed_recording

    def child():
        from wardex_sdk._assembly import counters

        return {
            "fork_child_reinit": counters.get("_runtime.fork_child_reinit"),
            "reinit_us": _runtime.runtime()._fork_reinit_us,
        }

    code, payload = _run_in_child(child)
    assert code == 0, payload
    assert payload["fork_child_reinit"] == 1
    assert 0 < payload["reinit_us"] < 250_000, (
        f"child-side fork reset took {payload['reinit_us']}µs — the one-time "
        "cost bound exists so a regression here fails a test, not a fork"
    )


# --------------------------------------------------------------------------
# §6.3 — a fork landing mid-export, and the host fork that never waits
# --------------------------------------------------------------------------


class _PidGatedTransport(RecordingTransport):
    """Blocks `export()` in the OWNING process until released.

    Pid-aware so a forked child sailing through its inherited copy does not
    block on an Event only the parent's test body can set — the child's
    exports record immediately, which is exactly the behavior under test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.owner_pid = os.getpid()
        self.entered = threading.Event()
        self.release = threading.Event()

    def export(self, envelope, *, timeout=None):  # noqa: ANN001
        if os.getpid() == self.owner_pid:
            self.entered.set()
            assert self.release.wait(20), "the test forgot to release the export gate"
        return super().export(envelope, timeout=timeout)


@fork_only
def test_no_deadlock_when_fork_lands_mid_flush():
    """The child inherits `_export_lock` HELD by a thread that does not exist
    there — before the reset, its first flush (or its atexit) waited on it
    forever. And the batch mid-POST at fork time belongs to the parent: it
    ships exactly once, from the parent, when the POST completes."""
    transport = _PidGatedTransport()
    wardex.init(transport=transport, intercept=False, batching=_IDLE)
    try:
        client = _hub.get_client()
        client.capture_span(_span("stuck-behind-the-post"))
        flusher = threading.Thread(target=client.flush, name="test-flusher")
        flusher.start()
        assert transport.entered.wait(10), "the export never started"

        # The flusher owns _export_lock and sits mid-POST. Fork now.
        def child():
            c = _hub.get_client()
            c.capture_span(_span("child-0"))
            c.flush()  # hung on the inherited _export_lock before this work
            return {"exported": _exported_names(transport)}

        code, payload = _run_in_child(child)
        transport.release.set()
        flusher.join(10)
        assert not flusher.is_alive(), "the parent's own flush must also finish"
        assert code == 0, payload
        assert payload["exported"] == ["child-0"], (
            "the child ships its own capture and nothing of the parent's in-flight batch"
        )
        assert _exported_names(transport) == ["stuck-behind-the-post"], (
            "the mid-POST batch ships exactly once, from the process that owns it"
        )
    finally:
        wardex.close()


@fork_only
def test_fork_returns_promptly_whatever_locks_other_threads_hold():
    """I-fork-2, as a timing assertion: no before/after_in_parent hook exists,
    so `os.fork()` returns immediately even while one thread is mid-POST
    holding `_export_lock` and another holds `_buffer_lock`. A reintroduced
    before-hook that acquires either lock turns this into a hang — this test
    is the tripwire that fails first."""
    transport = _PidGatedTransport()
    wardex.init(transport=transport, intercept=False, batching=_IDLE)
    holder = flusher = None
    release_buffer = threading.Event()
    try:
        client = _hub.get_client()
        client.capture_span(_span("in-flight"))
        flusher = threading.Thread(target=client.flush, name="test-flusher")
        flusher.start()
        assert transport.entered.wait(10)

        buffer_held = threading.Event()

        def hold_buffer():
            with client._buffer_lock:
                buffer_held.set()
                release_buffer.wait(20)

        holder = threading.Thread(target=hold_buffer, name="test-buffer-holder")
        holder.start()
        assert buffer_held.wait(10)

        started = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            os._exit(0)  # the child's only job was to run the hook
        elapsed = time.perf_counter() - started
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert elapsed < 1.0, (
            f"os.fork() took {elapsed:.3f}s while SDK locks were held — the host's "
            "fork must never wait on wardex"
        )
    finally:
        release_buffer.set()
        transport.release.set()
        if holder is not None:
            holder.join(10)
        if flusher is not None:
            flusher.join(10)
        wardex.close()


@fork_only
def test_fork_while_runtime_lock_is_held_does_not_hang_the_child():
    """P row: `teardown()` holds `Runtime._lock` across the final drain — up
    to and including the POST. A fork in that window used to hand the child a
    lock nobody would ever release, and the child's own atexit (teardown →
    `with self._lock`) hung the interpreter's exit. Step 0 replaces it."""
    transport = RecordingTransport()
    wardex.init(transport=transport, intercept=False, batching=_IDLE)
    release = threading.Event()
    holder = None
    try:
        runtime = _runtime.runtime()
        held = threading.Event()

        def hold_runtime_lock():
            with runtime._lock:
                held.set()
                release.wait(20)

        holder = threading.Thread(target=hold_runtime_lock, name="test-runtime-holder")
        holder.start()
        assert held.wait(10)

        def child():
            c = _hub.get_client()
            c.capture_span(_span("child-teardown"))
            _runtime.runtime().teardown(timeout=5)  # the atexit path, on demand
            return {"exported": _exported_names(transport)}

        code, payload = _run_in_child(child)
        assert code == 0, payload
        assert payload["exported"] == ["child-teardown"], (
            "the child's teardown drains the child's own spans — nothing more, "
            "and without hanging on the inherited Runtime lock"
        )
    finally:
        release.set()
        if holder is not None:
            holder.join(10)
        wardex.close()


# --------------------------------------------------------------------------
# §6.5 — a grandchild resets again
# --------------------------------------------------------------------------


@fork_only
def test_grandchild_also_resets(installed_recording):
    """`register_at_fork` registrations are inherited, so the hook runs afresh
    in every generation: the grandchild discards what the child buffered,
    exactly as the child discarded what the parent buffered."""
    transport, client = installed_recording
    client.capture_span(_span("parent-0"))

    def child():
        c = _hub.get_client()
        c.capture_span(_span("child-0"))

        def grandchild():
            g = _hub.get_client()
            g.capture_span(_span("grandchild-0"))
            g.flush()
            return {"exported": _exported_names(transport)}

        gc_code, gc_payload = _run_in_child(grandchild)
        c.flush()
        return {
            "grandchild_code": gc_code,
            "grandchild": gc_payload,
            "child_exported": _exported_names(transport),
        }

    code, payload = _run_in_child(child)
    assert code == 0, payload
    assert payload["grandchild_code"] == 0, payload["grandchild"]
    assert payload["grandchild"]["exported"] == ["grandchild-0"]
    assert payload["child_exported"] == ["child-0"]
    client.flush()
    assert _exported_names(transport) == ["parent-0"]


# --------------------------------------------------------------------------
# the fork-held lock inventory: PatchSet and context._inject
# --------------------------------------------------------------------------


def test_patchset_fork_reinit_replaces_the_lock_and_keeps_the_records():
    from wardex_sdk._assembly import PatchSet

    class Target:
        def method(self) -> str:
            return "original"

    ps = PatchSet("test.fork")
    assert ps.patch(Target, "method", lambda self: "wrapped")
    old_lock = ps._lock
    ps._at_fork_reinit()
    assert ps._lock is not old_lock, "an inherited PatchSet lock may be held by a gone thread"
    assert len(ps) == 1, "the records survive — the child's teardown restores THROUGH them"
    ps.restore_all()
    assert Target().method() == "original", "the kept record is what makes the restore real"


def test_inject_fork_reinit_replaces_the_lock_and_keeps_the_install_record():
    from wardex_sdk.context import _inject

    old_lock = _inject._install_lock
    _inject._installed.add("sentinel-lib")
    try:
        _inject._at_fork_reinit()
        assert _inject._install_lock is not old_lock
        assert "sentinel-lib" in _inject._installed, (
            "the patch record is KEPT (I-fork-4): the patches crossed the fork "
            "and still work; forgetting them would strand the child's teardown"
        )
    finally:
        _inject._installed.discard("sentinel-lib")
        assert _inject._install_lock is not old_lock


# --------------------------------------------------------------------------
# §6.4 — the connection latch and the TRACKING_RESET_AT_FORK marker
# --------------------------------------------------------------------------

_HTTP_REQ = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\nContent-Length: 0\r\n\r\n"
_HTTP_RESP = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"


def _bare_seam():
    """An SSL seam driven directly (no monkeypatch), recording into a client
    whose mode admits plain transport spans."""
    from wardex_sdk._enums import CaptureMode
    from wardex_sdk._interceptors._ssl import SSLInterceptor

    class _SeamClient:
        def __init__(self) -> None:
            self.config = WardexConfig(
                capture_mode=CaptureMode.ALL, backend=BackendConfig(api_key="k")
            )
            self.spans: list = []

        def capture_span(self, span) -> None:  # noqa: ANN001
            self.spans.append(span)

    seam = SSLInterceptor()
    seam._client = _SeamClient()
    return seam


class _FakeTlsSocket:
    def __init__(self) -> None:
        self._alpn = "http/1.1"

    def selected_alpn_protocol(self) -> str:
        return self._alpn

    def getpeername(self) -> tuple[str, int]:
        return ("127.0.0.1", 443)

    def fileno(self) -> int:
        return -1


def _drive_txn(seam, sock) -> None:  # noqa: ANN001
    seam._on_request_bytes(sock, _HTTP_REQ)
    seam._on_response_bytes(sock, _HTTP_RESP)


def test_child_connection_tracking_is_its_own():
    """A NEW connection opened after the fork reset captures with its own
    tracker and carries no fork marker — the reset is about inherited state,
    not about the child's fresh traffic."""
    from wardex_sdk._assembly import Limitation

    seam = _bare_seam()
    inherited = _FakeTlsSocket()
    _drive_txn(seam, inherited)  # the parent's tracking, about to be inherited
    assert id(inherited) in seam._conns

    seam._at_fork_reinit()
    assert seam._conns == {}, "the child must not trust the parent's per-connection state"

    fresh = _FakeTlsSocket()
    _drive_txn(seam, fresh)
    spans = seam._client.spans
    assert [s.name for s in spans][-1] == "HTTP POST /v1/messages"
    assert Limitation.TRACKING_RESET_AT_FORK not in spans[-1].capture_integrity.limitations, (
        "a connection born in the child never crossed the fork"
    )


def test_inherited_socket_first_span_carries_the_fork_marker():
    """The host keeps using a pre-fork socket: its tracker restarted
    mid-stream, and exactly ONE span says so — every other path that drops a
    live connection state leaves a marker, and the fork path may not be the
    silent exception."""
    from wardex_sdk._assembly import Limitation, counters

    seam = _bare_seam()
    sock = _FakeTlsSocket()
    _drive_txn(seam, sock)  # tracked in the parent
    before = counters.get("interceptors.seam.tracking_reset_at_fork")

    seam._at_fork_reinit()
    _drive_txn(seam, sock)  # the child keeps using the inherited socket
    _drive_txn(seam, sock)  # and again — the marker must not repeat

    spans = seam._client.spans[1:]  # [0] is the parent's own capture
    assert len(spans) == 2
    assert Limitation.TRACKING_RESET_AT_FORK in spans[0].capture_integrity.limitations, (
        "the first span on a fork-crossing connection must say its tracking restarted"
    )
    assert Limitation.TRACKING_RESET_AT_FORK not in spans[1].capture_integrity.limitations, (
        "the marker is the reset event's, and the reset happened once"
    )
    assert counters.get("interceptors.seam.tracking_reset_at_fork") == before + 1


def test_fork_latch_is_bounded_by_max_connections():
    """The latch cannot outgrow the table it snapshots: at most
    `max_connections` ids survive, newest first — the same drop-oldest
    posture as the table's own cap."""
    seam = _bare_seam()
    socks = [_FakeTlsSocket() for _ in range(6)]
    for sock in socks:
        seam._on_request_bytes(sock, _HTTP_REQ)
    seam._limits = {**seam._limits, "max_connections": 4}
    seam._at_fork_reinit()
    assert seam._reset_at_fork_ids == {id(s) for s in socks[-4:]}, (
        "keep the newest cap-many ids; anything older was the table's own next eviction"
    )


@fork_only
def test_installed_seam_reset_is_wired_through_the_fork_hook():
    """The hook actually REACHES an installed seam: the child finds the
    connection table empty and the fork latch holding the inherited id.
    (The graph-walk guard generalizes this to every holder; this is the
    seam-specific behavior pinned end to end through a real fork.)"""
    wardex.init(transport=RecordingTransport(), intercept=True, batching=_IDLE)
    try:
        from wardex_sdk._interceptors._registry import get_registry

        seam = get_registry()._installed["ssl"]
        sock = _FakeTlsSocket()
        seam._on_request_bytes(sock, _HTTP_REQ)
        assert id(sock) in seam._conns

        def child():
            return {
                "conns": len(seam._conns),
                "latched": id(sock) in seam._reset_at_fork_ids,
            }

        code, payload = _run_in_child(child)
        assert code == 0, payload
        assert payload == {"conns": 0, "latched": True}
        assert id(sock) in seam._conns, "the PARENT's tracking is untouched by the child's reset"
    finally:
        wardex.close()


# --------------------------------------------------------------------------
# §6.9 — registration lifecycle
# --------------------------------------------------------------------------


def test_hooks_register_exactly_once_across_init_reset_cycles(monkeypatch):
    """`os.register_at_fork` has no unregister, so a flag that `reset()`
    cleared would stack one live registration per init/reset cycle for the
    life of the process — pytest being the process where that compounds."""
    calls: list[dict] = []
    monkeypatch.setattr(os, "register_at_fork", lambda **kwargs: calls.append(kwargs))
    runtime = _runtime.runtime()
    monkeypatch.setattr(runtime, "_fork_hooks_registered", False)
    for _ in range(3):
        wardex.init(transport=RecordingTransport(), intercept=False, batching=_IDLE)
        _hub.reset_for_test()
    assert len(calls) == 1, "one registration per process, however many init()s"
    assert runtime._fork_hooks_registered is True, (
        "reset() must keep the flag — the OS-level registration is irreversible"
    )


def test_only_the_child_hook_is_ever_registered(monkeypatch):
    """I-fork-2's registration half: no `before`, no `after_in_parent`.

    A `before` hook that touches any SDK lock can deadlock the host's own
    `os.fork()` against the existing buffer→spawn lock ordering, and uWSGI
    runs only the child hook anyway. If someone reintroduces one, this fails
    before the AB-BA does.
    """
    calls: list[dict] = []
    monkeypatch.setattr(os, "register_at_fork", lambda **kwargs: calls.append(kwargs))
    runtime = _runtime.runtime()
    monkeypatch.setattr(runtime, "_fork_hooks_registered", False)
    wardex.init(transport=RecordingTransport(), intercept=False, batching=_IDLE)
    _hub.reset_for_test()
    assert len(calls) == 1
    assert list(calls[0]) == ["after_in_child"], (
        f"registered {sorted(calls[0])} — after_in_child is the ONLY fork hook this SDK may own"
    )


# --------------------------------------------------------------------------
# the client's own reset, without a fork
# --------------------------------------------------------------------------


def _bare_client() -> tuple[Client, RecordingTransport]:
    transport = RecordingTransport()
    config = WardexConfig(
        limits=LimitsConfig(max_buffer_spans=8),
        backend=BackendConfig(api_key="k"),
        batching=_IDLE,
    )
    return Client(config, transport), transport


def test_client_fork_reinit_discards_the_buffer_and_replaces_every_lock():
    client, transport = _bare_client()
    old_thread, old_wake = client._worker._thread, client._worker._wake
    try:
        client.capture_span(_span("prefork"))
        before = (client._buffer_lock, client._export_lock, client._close_lock)
        client._at_fork_reinit()
        after = (client._buffer_lock, client._export_lock, client._close_lock)
        for old, new in zip(before, after, strict=True):
            assert old is not new, "an inherited lock may be held by a thread that is gone"
        assert list(client._spans) == [], "the parent owns the pre-fork buffer (I-fork-3)"
        assert client._buffered_bytes == 0
        client.flush()
        assert transport.envelopes == [], "a discarded buffer must not ship from the child"
    finally:
        # In-process simulation only: a REAL fork child has no parent worker
        # thread to orphan. Here the pre-reset thread still waits on the
        # pre-reset Event, which close() (poking the replacement) cannot reach.
        client.close()
        if old_thread is not None and old_thread.is_alive():
            old_wake.set()
            old_thread.join(5)


def test_client_fork_reinit_on_a_closed_client_stays_closed():
    client, transport = _bare_client()
    client.close()
    client._at_fork_reinit()
    assert client._closed, "the child of a closed parent is closed"
    client.capture_span(_span("late"))
    client.flush()
    assert transport.envelopes == [], "a closed child must not begin capturing"


def test_worker_fork_reinit_is_the_lazy_posture():
    client, _transport = _bare_client()
    worker = client._worker
    old_thread, old_wake = worker._thread, worker._wake
    try:
        old_lock = worker._spawn_lock
        worker._at_fork_reinit()
        assert worker._spawn_lock is not old_lock
        assert worker._wake is not old_wake
        assert worker._thread is None and worker._thread_for_pid is None
        assert not worker.is_alive(), "no eager thread — respawn stays lazy"
        # the next capture is what brings it back, exactly as before this work
        client.capture_span(_span("wakes-the-worker"))
        assert worker.is_alive()
    finally:
        # Same in-process-simulation cleanup as the client test above.
        client.close()
        if old_thread is not None and old_thread.is_alive():
            old_wake.set()
            old_thread.join(5)


def test_diag_reset_for_new_process_replaces_and_clears():
    from wardex_sdk._assembly import _diag

    _diag.counters.bump("somewhere.parent")
    _diag.report_once("a parent line", key="fork-test-key")
    old_counters_lock = _diag.counters._lock
    old_report_lock = _diag._REPORT_LOCK
    _diag.diag_reset_for_new_process()
    try:
        assert _diag.counters._lock is not old_counters_lock
        assert _diag._REPORT_LOCK is not old_report_lock
        assert _diag.counters.snapshot() == {}, "inherited tallies are the parent's"
        # the dedup table forgot the parent's keys: the child's first report prints
        assert "fork-test-key" not in _diag._REPORTED
    finally:
        _diag.reset_reports_for_test()
