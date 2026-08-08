"""Who gets blamed when an export is cut off -- asked through the public calls.

The unconfirmed-delivery report is a claim about the CALLER: "the budget you
passed was shorter than this transport's own timeout, so wardex cannot confirm
delivery". Only `wardex.flush(t)` / `wardex.close(t)` can make that true, and
only the client knows whether a number was passed at all -- by the time a budget
reaches the transport it is `deadline - now`, which on a bare `flush()` is
already a shade UNDER the transport's configured timeout even though nobody
chose it.

Which is why this file drives the real entry points into a real
`OtlpHttpTransport` against a real dead backend and reads stderr, instead of
handing the transport numbers by hand. The defect this replaces shipped green
under transport-level tests that passed `requested in (None, 10.0, 99.0)` --
values the client never produces on the default path -- so every arm of the
guard was exercised except the one the client actually reaches. The claim is
host-facing, so it is pinned where the host stands.

The failure that makes it matter is at the bottom: the report is one line per
key per process, so a line spent on a bare `flush()` against a down backend
BURNS the key, and the genuine report never prints again.
"""

from __future__ import annotations

import signal
import socket
import time

import pytest

from wardex_sdk import _runtime
from wardex_sdk._client import Client, _UnnamedTimeout
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.assembly._diag import reset_reports_for_test
from wardex_sdk.transport._otlp_http import OtlpHttpTransport

# Long enough that the socket really does block on it, short enough that a test
# waiting the whole thing out stays a test.
SHORT = 0.4


@pytest.fixture
def black_hole():
    """A URL whose backend accepts the connection and then never answers.

    A listening socket nobody ever `accept()`s: the kernel completes the
    handshake from the backlog and buffers the POST, so the connect and the send
    both succeed and `urlopen` blocks in the read -- which is the shape this
    report is about. A refused port would fail instantly instead, and an instant
    refusal is deliberately NOT reported (it is the backend's news, not the
    caller's), so it cannot exercise this path at all.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}/v1/traces"
    finally:
        sock.close()


@pytest.fixture(autouse=True)
def _fresh_reports():
    """`_REPORTED` is process-global; one line per key per process is the point."""
    reset_reports_for_test()
    yield
    reset_reports_for_test()


def _span(name="s"):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _client(endpoint: str, *, configured: float) -> Client:
    """A client whose only drains are the ones the test asks for."""
    transport = OtlpHttpTransport(endpoint=endpoint, timeout=configured)
    client = Client(WardexConfig(api_key="k", flush_interval=3600.0), transport)
    client._worker.stop()
    return client


def _cut_short_lines(err: str) -> list[str]:
    return [ln for ln in err.splitlines() if "cut off after" in ln]


def _timed(fn) -> float:
    started = time.monotonic()
    fn()
    return time.monotonic() - started


# -- the caller named nothing: the report must not fire ----------------------


def test_a_bare_flush_against_a_dead_backend_blames_nobody(black_hole, capsys):
    """THE DEFECT. A bare `flush()` follows the transport's own configured
    timeout, so the budget arriving at the transport is that same number minus
    the acquire, the swap, `before_send` and the encode -- always a little
    short of it, always, on the path every host takes by default.

    Reading "shorter than configured" as "the caller chose it" therefore fired
    on every bare `flush()` against a slow or down backend, and said so in a
    sentence that contradicted itself: cut off "by the 10.0s budget its caller
    passed", "which is shorter than this transport's own 10.0s timeout". Nobody
    passed anything. The backend is down; that is not news this channel carries.
    """
    client = _client(black_hole, configured=SHORT)
    client.capture_span(_span())
    capsys.readouterr()
    elapsed = _timed(client.flush)
    err = capsys.readouterr().err
    client.close(0.1)

    assert not _cut_short_lines(err), (
        f"a bare flush() against a down backend was blamed on its caller: {err!r}"
    )
    assert elapsed >= SHORT * 0.8, (
        f"the POST never reached the socket, so nothing was under test ({elapsed:.2f}s)"
    )


def test_a_bare_close_does_not_call_wardexs_own_default_a_budget_the_caller_passed(
    black_hole, capsys
):
    """The same defect through the other door, and the reason the fix could not
    be a tweak to the inequality: `close()`'s 5s is wardex's OWN shutdown
    default, not a number any caller passed, so under a transport configured for
    longer it was reported as one.

    Slow on purpose. The scenario needs a transport configured for MORE than the
    5s shutdown default -- that is the only way an unnamed close() budget can
    land strictly under `configured`, which is exactly the condition that used
    to fire -- and that default is not a knob a test may turn without testing
    something other than what hosts run.
    """
    client = _client(black_hole, configured=6.0)
    client.capture_span(_span())
    capsys.readouterr()
    elapsed = _timed(client.close)
    err = capsys.readouterr().err

    assert not _cut_short_lines(err), (
        f"wardex's own shutdown default was reported as a budget the caller passed: {err!r}"
    )
    assert 4.0 <= elapsed <= 12.0, (
        f"a bare close() did not spend its own 5s default ({elapsed:.2f}s)"
    )


def test_a_timeout_at_the_transports_own_configured_limit_stays_silent(black_hole, capsys):
    """Unchanged, and still the right silence: `flush(99.0)` narrows to the
    transport's own timeout, so what expired is the transport's number and not
    the caller's, however loudly the caller named a bigger one."""
    client = _client(black_hole, configured=SHORT)
    client.capture_span(_span())
    capsys.readouterr()
    elapsed = _timed(lambda: client.flush(99.0))
    err = capsys.readouterr().err
    client.close(0.1)

    assert not _cut_short_lines(err), (
        f"the transport's own limit was blamed on flush(99.0): {err!r}"
    )
    assert elapsed >= SHORT * 0.8, f"the POST never reached the socket ({elapsed:.2f}s)"


# -- wardex calling its own flush() is not a caller either --------------------


@pytest.fixture
def signal_handler_only(monkeypatch):
    """`_runtime._handler` with everything but the flush taken out of the way.

    Driven through the handler FUNCTION rather than through `Client.flush`,
    because the defect is precisely that the client cannot see who called it:
    the handler passed a bare `2.0` and, from inside `flush`, that is
    byte-identical to a host writing `flush(2.0)`. A test that called `flush`
    directly would have to name the number itself and would therefore be testing
    the other case.

    `_prev_handlers` empty means the handler chains to nothing -- no `os.kill`,
    no re-raise -- so what runs is the flush and only the flush. `_close_units`
    is None for the same reason.
    """
    runtime = _runtime.runtime()
    monkeypatch.setattr(runtime, "_prev_handlers", {})
    monkeypatch.setattr(runtime, "_close_units", None)

    def deliver(client: Client) -> None:
        monkeypatch.setattr(runtime, "_client", client)
        _runtime._handler(signal.SIGTERM, None)

    return deliver


def test_the_signal_handlers_own_budget_is_not_a_number_any_caller_passed():
    """Stated directly, because the two tests below run against a real socket
    and this is the property they depend on: the handler's 2s is wardex's, and
    it says so in its type rather than leaving `flush` to guess from the value.
    """
    assert isinstance(_runtime._SIGNAL_FLUSH_TIMEOUT, _UnnamedTimeout), (
        f"_SIGNAL_FLUSH_TIMEOUT is a plain {type(_runtime._SIGNAL_FLUSH_TIMEOUT).__name__}, "
        f"so a flush from the signal handler is indistinguishable from a host's flush(2.0) "
        f"and gets the host blamed for it"
    )


def test_a_signal_flush_against_a_dead_backend_does_not_blame_the_host(
    black_hole, capsys, signal_handler_only
):
    """THE DEFECT, through the door the fix did not close first time.

    The handler spends 2 seconds, and 2 is under any transport configured for
    more -- which is the whole firing condition. So every host that Ctrl-Cs
    against a slow backend was told that the budget IT passed had cut an export
    short, and advised to "pass a larger timeout to confirm delivery". No host
    passed anything: `_SIGNAL_FLUSH_TIMEOUT` is wardex's, and there is no knob
    for it, so the advice is unactionable as well as false.
    """
    client = _client(black_hole, configured=6.0)
    client.capture_span(_span())
    capsys.readouterr()
    elapsed = _timed(lambda: signal_handler_only(client))
    err = capsys.readouterr().err
    client.close(0.1)

    assert not _cut_short_lines(err), (
        f"the signal handler's own 2s budget was reported as the host's: {err!r}"
    )
    assert 1.0 <= elapsed <= 5.0, (
        f"the signal flush did not spend its own 2s bound against the socket ({elapsed:.2f}s)"
    )


def test_a_signal_flush_does_not_burn_the_key_the_real_report_needs(
    black_hole, capsys, signal_handler_only
):
    """The consequence that makes it a MAJOR rather than a wrong sentence.

    One line per key per PROCESS. A host that Ctrl-Cs once and carries on --
    SIGINT with a `KeyboardInterrupt` handler above us is exactly that -- spent
    the key on a line about a number it never chose, and the genuine report was
    silent for the rest of the process. Both halves here, in that order, with no
    reset in between.
    """
    client = _client(black_hole, configured=6.0)
    try:
        client.capture_span(_span("first"))
        capsys.readouterr()
        signal_handler_only(client)
        assert not _cut_short_lines(capsys.readouterr().err), "the signal flush reported"

        client.capture_span(_span("second"))
        client.flush(SHORT)  # named by the host: the report it can act on
        lines = _cut_short_lines(capsys.readouterr().err)
        assert len(lines) == 1, (
            "the signal handler's flush burned the one-line-per-process key and "
            f"silenced the real report: {lines!r}"
        )
        assert f"{SHORT:.1f}s budget" in lines[0], lines[0]
    finally:
        client.close(0.1)


# -- the caller named a number: the report must fire, and name it ------------


def test_an_explicit_flush_budget_that_cuts_an_export_short_is_reported(black_hole, capsys):
    """The event the channel exists for. `flush(0.4)` under a transport
    configured for 5s: the POST went out, the caller stopped waiting, and the
    spans are deliberately NOT re-queued because the backend may already hold
    them. Off-debug this was silence byte-identical to a successful export, so
    the line is the only thing the caller ever gets -- and it has to name the
    number the caller would recognize, not the remainder left after the encode.
    """
    client = _client(black_hole, configured=5.0)
    client.capture_span(_span())
    capsys.readouterr()
    elapsed = _timed(lambda: client.flush(SHORT))
    err = capsys.readouterr().err
    client.close(0.1)

    lines = _cut_short_lines(err)
    assert len(lines) == 1, f"an export the caller cut short said nothing off-debug: {err!r}"
    line = lines[0]
    assert f"{SHORT:.1f}s budget" in line, f"the report did not name the caller's number: {line}"
    assert "5.0s timeout" in line, f"the report did not name the transport's own number: {line}"
    assert "cannot CONFIRM" in line, line
    assert "lost" not in line.lower(), f"the report claimed a loss it cannot know about: {line}"
    assert elapsed < 3.0, f"flush({SHORT}) was not a real bound ({elapsed:.2f}s)"


def test_an_explicit_close_budget_that_cuts_an_export_short_is_reported(black_hole, capsys):
    """`close(t)` is a number the caller named just as much as `flush(t)` is.
    The distinction the fix draws is passed-vs-derived, not flush-vs-close, and
    a fix that keyed on the operation instead would go silent here."""
    client = _client(black_hole, configured=5.0)
    client.capture_span(_span())
    capsys.readouterr()
    elapsed = _timed(lambda: client.close(SHORT))
    err = capsys.readouterr().err

    lines = _cut_short_lines(err)
    assert len(lines) == 1, f"an export close({SHORT}) cut short said nothing: {err!r}"
    assert f"{SHORT:.1f}s budget" in lines[0], lines[0]
    assert elapsed < 6.0, f"close({SHORT}) was not a real bound ({elapsed:.2f}s)"


# -- and the consequence that made the false report a MAJOR ------------------


def test_a_bare_flush_does_not_burn_the_key_the_real_report_needs(black_hole, capsys):
    """Why a false line is worse than a noisy one.

    `report_once` is one line per key per PROCESS. So the bare `flush()` that
    used to over-fire did not merely say something wrong -- it consumed the key,
    and every genuinely caller-cut-short export afterwards printed nothing at
    all for the rest of the process. A host that flushes on a timer against a
    backend having a bad minute lost the diagnostic before it ever needed it.

    Both halves in one process, in the order that used to break: the default
    path first, the real one second. Deliberately NOT resetting in between --
    the reset would be the thing under test doing the test's job.
    """
    client = _client(black_hole, configured=1.0)
    try:
        client.capture_span(_span("first"))
        capsys.readouterr()
        client.flush()  # bare: derived budget, must stay silent AND spend nothing
        assert not _cut_short_lines(capsys.readouterr().err), "the bare flush() reported"

        client.capture_span(_span("second"))
        client.flush(SHORT)  # named: the report the caller can act on
        lines = _cut_short_lines(capsys.readouterr().err)
        assert len(lines) == 1, (
            "the bare flush() burned the one-line-per-process key and silenced the "
            f"real report: {lines!r}"
        )
        assert f"{SHORT:.1f}s budget" in lines[0], lines[0]
    finally:
        client.close(0.1)
