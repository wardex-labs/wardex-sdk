"""End-to-end batching & lifecycle — real signals, real fork, real HTTP."""

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
from wardex_sdk._types import InternalEnvelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.transport._base import Transport
from wardex_sdk.transport._otlp_http import OtlpHttpTransport


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
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
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
wardex.init(transport=NoOpTransport(), before_send=mark, intercept=False,
            backend=wardex.BackendConfig(api_key="k"),
            batching=wardex.BatchingConfig(flush_interval=3600.0))
_hub.get_client().capture_span(InternalSpan(
    context=SpanContext(TraceId.generate(), SpanId.generate()),
    parent_span_id=None, name="s", kind=SpanKind.INTERNAL,
    start_time_ns=1, end_time_ns=2))
print("ready", flush=True)
time.sleep(60)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
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
