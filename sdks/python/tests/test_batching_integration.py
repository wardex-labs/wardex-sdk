"""End-to-end batching & lifecycle — real signals, real fork, real HTTP."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import Envelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.transport._base import Transport
from wardex_sdk.transport._otlp_http import OtlpHttpTransport

#: The SIGTERM tests below need the POSIX default disposition: a child that
#: dies of the signal itself, with the exit code saying so.
posix_signals = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")


def _wait_for(predicate, timeout=5.0):
    """Poll a predicate with a generous upper bound — no bare-sleep asserts."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


def _span():
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )


class _CollectingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.server.received.append(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # keep test output quiet
        pass


def test_periodic_flush_posts_encoded_batch_without_manual_flush():
    """Spans reach the wire (native encode on the worker thread) with no flush()."""
    server = HTTPServer(("127.0.0.1", 0), _CollectingHandler)
    server.received = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        t = OtlpHttpTransport(f"http://127.0.0.1:{server.server_port}/v1/traces")
        c = Client(
            WardexConfig(
                backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=0.05)
            ),
            t,
        )
        c.capture_span(_span())
        assert _wait_for(lambda: len(server.received) >= 1)
        c.close()
    finally:
        server.shutdown()
        server.server_close()  # release the listening socket


_SIGTERM_CHILD = """
import sys, time
import wardex_sdk as wardex
from wardex_sdk import NoOpTransport
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId

marker = sys.argv[1]

def mark(envelope):
    with open(marker, "w") as f:
        f.write(str(len(envelope.spans)))
    return None  # drop after recording — no network needed

# interval 3600 + threshold 512: only the signal handler can flush this span
wardex.init(transport=NoOpTransport(), before_send_envelope=mark, intercept=False,
            backend=wardex.BackendConfig(api_key="k"),
            batching=wardex.BatchingConfig(flush_interval=3600.0))
_hub.get_client().capture_span(InternalSpan(
    context=SpanContext(TraceId.generate(), SpanId.generate()),
    parent_span_id=None, name="s", kind=SpanKind.INTERNAL,
    start_time_ns=1, end_time_ns=2))
print("ready", flush=True)
time.sleep(60)
"""


@posix_signals
def test_sigterm_flushes_and_preserves_exit_code(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_SIGTERM_CHILD)
    marker = tmp_path / "flushed.txt"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(marker)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()  # reap — no zombie on the failure path
    assert rc == -signal.SIGTERM  # default termination (exit code) preserved
    assert marker.read_text() == "1"  # our handler flushed the span first


# A run cut off mid-way by SIGTERM: the spans still open ship marked, once,
# with the children that had already finished under them.

_SIGTERM_OPEN_RUN_CHILD = r"""
import json, os, signal, sys, time
import wardex_sdk as wardex
from wardex_sdk.transport._base import Transport

out = sys.argv[1]

class FileTransport(Transport):
    def export(self, envelope):
        with open(out, "a") as f:
            for s in envelope.spans:
                integ = s.capture_integrity
                f.write(json.dumps({
                    "name": s.name,
                    "span": s.context.span_id.value.hex(),
                    "parent": None if s.parent_span_id is None else s.parent_span_id.value.hex(),
                    "start": s.start_time_ns,
                    "end": s.end_time_ns,
                    "markers": [] if integ is None else [m.value for m in integ.limitations],
                }) + "\n")

# interval 3600: only the signal handler can flush anything here
wardex.init(transport=FileTransport(), intercept=False,
            backend=wardex.BackendConfig(api_key="k"),
            batching=wardex.BatchingConfig(flush_interval=3600.0))

@wardex.tool(name="stream")
def stream():
    yield 1
    yield 2

@wardex.workflow(name="run")
def run():
    with wardex.span("chat"):
        pass
    items = stream()
    next(items)
    with wardex.span("inner"):
        print("ready", flush=True)
        time.sleep(30)

run()
"""


@posix_signals
def test_sigterm_ships_the_open_run_marked_with_its_finished_children_under_it(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_SIGTERM_OPEN_RUN_CHILD)
    out = tmp_path / "spans.jsonl"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(out)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert rc == -signal.SIGTERM  # the default termination is unchanged
    spans = [json.loads(line) for line in out.read_text().splitlines()]
    names = sorted(s["name"] for s in spans)
    assert names == ["chat", "inner", "run", "stream"]  # each exactly once
    by = {s["name"]: s for s in spans}
    run = by["run"]
    assert run["parent"] is None
    assert run["markers"] == ["unit_interrupted"]
    assert by["chat"]["parent"] == run["span"]
    assert by["chat"]["markers"] == []  # it finished on its own
    for still_open in ("inner", "stream"):
        assert by[still_open]["parent"] == run["span"]
        assert by[still_open]["markers"] == ["unit_interrupted"]
        assert by[still_open]["end"] <= run["end"]  # newest first: no child outlives its parent


_SIGTERM_PENDING_CHILD = """
import contextvars, sys, time
import wardex_sdk as wardex
from wardex_sdk import NoOpTransport
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId

marker = sys.argv[1]

def mark(envelope):
    with open(marker, "w") as f:
        f.write(",".join(sorted(s.name for s in envelope.spans)))
    return None

wardex.init(transport=NoOpTransport(), before_send_envelope=mark, intercept=False,
            backend=wardex.BackendConfig(api_key="k"),
            batching=wardex.BatchingConfig(flush_interval=3600.0))
client = _hub.get_client()
client.capture_span(InternalSpan(
    context=SpanContext(TraceId.generate(), SpanId.generate()),
    parent_span_id=None, name="buffered", kind=SpanKind.INTERNAL,
    start_time_ns=1, end_time_ns=2))

def _span(name):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None, name=name, kind=SpanKind.INTERNAL,
        start_time_ns=1, end_time_ns=2)

class SlowJob:
    size = 64
    def __init__(self):
        self.ctx = contextvars.copy_context()
    def run(self):
        time.sleep(1.5)  # slower than the signal path's half-budget
        return _span("pending-parsed")
    def fallback(self, marker_member):
        return _span("pending-fallback")

client._finalize.stop(0.1)  # keep the job PENDING until the signal drains it
client.capture_deferred(SlowJob())
print("ready", flush=True)
time.sleep(60)
"""


@posix_signals
def test_sigterm_ships_the_pending_parse_too(tmp_path):
    """The signal path is a flush with no second chance: a deferred job still
    pending when SIGTERM lands must leave — parsed inside the half-budget or
    as a PARSE_SKIPPED_AT_SHUTDOWN fallback after it — and the spans that
    were already buffered must leave WITH it (the export keeps its floor)."""
    script = tmp_path / "child.py"
    script.write_text(_SIGTERM_PENDING_CHILD)
    marker = tmp_path / "flushed.txt"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(marker)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert rc == -signal.SIGTERM
    names = marker.read_text().split(",")
    assert "buffered" in names
    assert "pending-parsed" in names or "pending-fallback" in names
    assert len(names) == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_fork_child_respawns_worker_and_flushes():
    t = _Recording()
    # interval 3600: parent worker sits idle in wait() holding no locks → fork-safe
    c = Client(
        WardexConfig(
            limits=LimitsConfig(max_buffer_spans=8),
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
    pid = os.fork()
    if pid == 0:
        # child: the worker thread did not survive the fork; ensure_alive respawns
        try:
            c.capture_span(_span())
            c.capture_span(_span())  # threshold max(1, 8//4)=2 → wake
            ok = _wait_for(lambda: sum(len(e.spans) for e in t.envelopes) >= 2)
            os._exit(0 if ok else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    c.close()
    assert os.waitstatus_to_exitcode(status) == 0
