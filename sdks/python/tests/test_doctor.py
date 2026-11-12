"""`python -m wardex_sdk doctor` and `explain`: one command proves the install.

Run as a user runs them, in a fresh process with the environment as the only
configuration, and judged by the exit code and the text a person or a coding
agent reads.
"""

from __future__ import annotations

import http.server
import os
import subprocess
import sys
import threading
from collections.abc import Iterator

import pytest

_SECRET = "s3cret-header-value"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("WARDEX_", "OTEL_")) and k != "PYTEST_CURRENT_TEST"
    }
    clean.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "wardex_sdk", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean,
    )


class _Collector(http.server.BaseHTTPRequestHandler):
    bodies: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
        self.bodies.append(self.rfile.read(int(self.headers.get("content-length", 0))))
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def collector() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def test_doctor_says_ok_with_the_destination_when_the_collector_takes_the_span(collector):
    proc = _run(
        "doctor",
        env={
            "WARDEX_ENDPOINT": collector,
            "OTEL_EXPORTER_OTLP_HEADERS": f"authorization={_SECRET}",
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[-1] == f"OK: {collector}/v1/traces accepted the test span."
    assert "wardex diagnostics" in proc.stdout
    assert _SECRET not in proc.stdout + proc.stderr


def _answering(content_type: str, body: bytes) -> type[http.server.BaseHTTPRequestHandler]:
    class _Answering(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    return _Answering


def _doctor_against(handler: type[http.server.BaseHTTPRequestHandler]):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        return url, _run("doctor", "--timeout", "5", env={"WARDEX_ENDPOINT": url})
    finally:
        server.shutdown()


#: `ExportTraceServiceResponse{partial_success{rejected_spans: 1, error_message: "nope"}}`.
_REJECTED_PROTOBUF = b"\x0a\x08\x08\x01\x12\x04nope"


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("application/x-protobuf", _REJECTED_PROTOBUF),
        ("application/json", b'{"partialSuccess": {"rejectedSpans": "1", "errorMessage": "nope"}}'),
    ],
    ids=["protobuf", "json"],
)
def test_doctor_fails_when_the_collector_rejects_the_span_in_a_2xx(content_type, body):
    """OTLP lets a receiver answer 200 and still reject spans: that is not an install that works."""
    url, proc = _doctor_against(_answering(content_type, body))
    assert proc.returncode == 1, proc.stdout
    last = proc.stdout.splitlines()[-1]
    assert last.startswith(f"FAIL: {url}/v1/traces rejected the test span"), last
    assert "nope" in last


def test_doctor_fails_when_a_web_page_answers_instead_of_a_collector():
    """A 200 with an HTML page is a proxy, a login page or a web app at a wrong path."""
    url, proc = _doctor_against(_answering("text/html", b"<html>sign in</html>"))
    assert proc.returncode == 1, proc.stdout
    last = proc.stdout.splitlines()[-1]
    assert last.startswith(f"FAIL: {url}/v1/traces answered with a web page"), last
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in last


def test_doctor_does_not_claim_acceptance_from_an_answer_it_cannot_read():
    url, proc = _doctor_against(_answering("text/plain", b"ok"))
    assert proc.returncode == 0, proc.stdout
    last = proc.stdout.splitlines()[-1]
    assert last.startswith(f"OK: {url}/v1/traces answered HTTP 200"), last
    assert "accepted" not in last


def test_doctor_names_the_failure_and_its_fix_when_the_collector_is_down():
    proc = _run("doctor", "--timeout", "5", env={"WARDEX_ENDPOINT": "http://127.0.0.1:9"})
    assert proc.returncode == 1
    last = proc.stdout.splitlines()[-1]
    assert last.startswith("FAIL: network_error:"), proc.stdout
    assert "reachable" in last


def test_doctor_says_how_to_configure_when_nothing_is():
    proc = _run("doctor")
    assert proc.returncode == 1
    assert "FAIL: wardex has nowhere to send data" in proc.stdout
    assert "WARDEX_ENDPOINT=" in proc.stdout and "WARDEX_API_KEY=" in proc.stdout


def test_explain_prints_a_marker_and_fails_on_an_unknown_name():
    known = _run("explain", "unit_inferred_sole")
    assert known.returncode == 0
    assert known.stdout.startswith("unit_inferred_sole (a limitation marker")
    unknown = _run("explain", "no_such_marker")
    assert unknown.returncode == 1


def test_explain_all_is_the_markdown_list():
    proc = _run("explain", "--all")
    assert proc.returncode == 0
    assert proc.stdout.startswith("# wardex limitation markers and loss reasons")
    assert "### `otlp_attribute_truncated`" in proc.stdout


def test_the_debug_line_shows_the_config_with_its_secrets_masked(capsys):
    import wardex_sdk as wardex

    wardex.init(
        backend=wardex.BackendConfig(
            endpoint="http://127.0.0.1:9/v1/traces?api_key=SECRETQUERY#SECRETFRAG",
            headers={"Authorization": "Bearer SECRETHEADER"},
        ),
        adapters=wardex.AdaptersConfig(enabled=()),
        debug=True,
    )
    wardex.close()
    err = capsys.readouterr().err
    [line] = [ln for ln in err.splitlines() if "resolved config" in ln]
    assert "backend.endpoint=http://127.0.0.1:9/v1/traces" in line
    assert "Authorization: (masked)" in line
    assert "SECRET" not in err


def test_the_report_has_what_a_bug_report_needs_and_no_secret():
    import platform

    import wardex_sdk as wardex

    wardex.init(
        backend=wardex.BackendConfig(
            endpoint="http://127.0.0.1:9/v1/traces?token=q#frag", headers={"X-Key": _SECRET}
        ),
        adapters=wardex.AdaptersConfig(enabled=()),
    )
    try:
        got = wardex.diagnostics()
        text = got.report()
    finally:
        wardex.close()
    assert got.wardex_version == wardex.__version__
    assert got.python_version == platform.python_version()
    assert got.native == "loaded"
    assert got.transport == "OtlpHttpTransport"
    assert got.destination == "http://127.0.0.1:9/v1/traces"
    assert got.capture_policy == "agent"
    assert "ssl" in got.interceptors and "ssl.SSLSocket.recv" in got.patched
    assert "wardex-batch-worker" in got.worker_threads
    assert ("backend.headers", "{X-Key: (masked)}") in got.config
    for leak in (_SECRET, "token=q", "frag"):
        assert leak not in text
    assert text.startswith("wardex diagnostics\n")
    assert f"destination: {got.destination}\n" in text


class _ExportState:
    """The delivery state a built-in transport offers, as `export_state`."""

    def __init__(self, last_success_at=None, last_error=None, retry_queued=0):
        self.destination = "http://collector:4318/v1/traces"
        self.last_success_at = last_success_at
        self.last_error = last_error
        self.last_error_at = None
        self.retry_queued = retry_queued
        self.retry_queued_bytes = 0


class _TrackingTransport:
    def __init__(self, state):
        self.export_state = state


def _last_export_line(transport) -> str:
    from wardex_sdk._diagnostics import Diagnostics
    from wardex_sdk._environment import transport_facts

    report = Diagnostics(initialized=True, **transport_facts(transport)).report()
    return next(line for line in report.splitlines() if line.startswith("last export: "))


def test_the_report_reads_the_delivery_state_a_transport_offers():
    assert _last_export_line(_TrackingTransport(_ExportState())) == "last export: none yet"
    assert _last_export_line(_TrackingTransport(_ExportState(last_success_at=0.0))) == (
        "last export: succeeded at 1970-01-01T00:00:00Z"
    )
    assert _last_export_line(
        _TrackingTransport(_ExportState(last_error="HTTP 503", retry_queued=2))
    ) == ("last export: last failure: HTTP 503; 2 request(s) held for a retry")
    assert _last_export_line(object()) == "last export: not tracked by this transport"


def test_diagnostics_reads_a_real_exporters_delivery_state(collector):
    import wardex_sdk as wardex

    wardex.init(transport=wardex.OtlpHttpTransport(f"{collector}/v1/traces"), intercept=False)
    try:
        with wardex.span("delivered"):
            pass
        wardex.flush(5.0)
        got = wardex.diagnostics()
        assert got.export_tracked is True
        assert got.destination == f"{collector}/v1/traces"
        assert got.last_export_success is not None
        assert got.last_export_error is None
        assert got.retry_queued == 0
    finally:
        wardex.close()


def test_diagnostics_reads_a_real_exporters_failure_and_what_it_holds():
    import wardex_sdk as wardex

    wardex.init(transport=wardex.OtlpHttpTransport("http://127.0.0.1:9/v1/traces"), intercept=False)
    try:
        with wardex.span("refused"):
            pass
        wardex.flush(5.0)
        got = wardex.diagnostics()
        assert got.export_tracked is True
        assert got.destination == "http://127.0.0.1:9/v1/traces"
        assert got.last_export_success is None
        assert got.last_export_error == "ConnectionRefusedError"
        assert got.retry_queued == 1
        assert "1 request(s) held for a retry" in got.report()
    finally:
        wardex.close()
