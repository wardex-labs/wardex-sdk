"""`python -m wardex_sdk`: check an install, and look up what a marker means.

    python -m wardex_sdk doctor          initialize from the environment, print
                                         the diagnostics report, send one test
                                         span, and say OK or what failed
    python -m wardex_sdk explain NAME    what a limitation marker or loss
                                         reason means, and what to do
    python -m wardex_sdk explain --all   every marker and reason, as Markdown

`doctor` exits 0 when the configured exporter took the test span and 1
otherwise, so a script or a coding agent can check an install by the exit
code alone. A 2xx is not taken on trust: an OTLP answer that rejects the span
(`partial_success`) or a web page answering in a collector's place is a
failure, and an answer the doctor cannot read is said as such. Its output is
plain text in a fixed order and never holds a key, a header value or a
captured payload.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TextIO

import wardex_sdk as wardex

from . import _runtime
from ._assembly._explain import markdown
from ._environment import transport_facts
from ._native import NATIVE_OK
from .transport._noop import NoOpTransport
from .transport._otlp_answer import MAX_ANSWER_BYTES, Answer, read_answer

#: The variables `init()` reads a destination from.
_DESTINATION_VARIABLES = (
    "WARDEX_API_KEY",
    "WARDEX_BASE_URL",
    "WARDEX_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)

TEST_SPAN_NAME = "wardex doctor test span"

_HOW_TO_CONFIGURE = (
    "wardex has nowhere to send data. Set one of these and run this again:\n"
    "  WARDEX_API_KEY=<project key>                  send to a wardex project\n"
    "  WARDEX_ENDPOINT=http://<collector>:4318       send OTLP to a collector\n"
    "  OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318\n"
    "or pass a destination to wardex.init() in the application."
)


def doctor(out: TextIO, *, environ: Mapping[str, str], deadline: float = 10.0) -> int:
    """Initialize from the environment as an application would, print the
    report, send one test span, and judge. Returns the exit code: 0 when the
    exporter took the span, 1 when it did not or there was nowhere to send it."""

    def say(line: str = "") -> None:
        out.write(line + "\n")
        out.flush()

    say("wardex doctor")
    configured = [name for name in _DESTINATION_VARIABLES if environ.get(name)]
    say(f"destination variables set: {', '.join(configured) or 'none'}")
    wardex.init()
    say()
    say(wardex.diagnostics().report().rstrip("\n"))
    say()
    if not NATIVE_OK:
        say(
            "FAIL: the native module did not load, so nothing is captured or exported. "
            "Reinstall wardex-sdk for this platform and Python "
            "(pip install --force-reinstall wardex-sdk)."
        )
        return 1
    transport = getattr(_runtime.runtime().client, "_transport", None)
    if transport is None or isinstance(transport, NoOpTransport):
        # What init() said names the variable to fix; the how-to is for a
        # process where nothing was set at all.
        said = [p for p in wardex.diagnostics().config_problems if "not running" not in p]
        say(f"FAIL: {' '.join(said)}" if said else f"FAIL: {_HOW_TO_CONFIGURE}")
        wardex.close()
        return 1
    where = transport_facts(transport).get("destination") or type(transport).__name__
    outcomes = _observe_exports(transport)
    before = {(x.reason, x.unit): x.count for x in wardex.diagnostics().losses}
    started = time.time()
    say(f"sending one span named {TEST_SPAN_NAME!r} to {where}")
    answers, restore = _observe_answers()
    try:
        with wardex.span(TEST_SPAN_NAME):
            pass
        wardex.flush(timeout=deadline)
    finally:
        restore()
    if getattr(getattr(transport, "export_state", None), "retry_queued", 0):
        # This process ends here, so a request held for a retry is never sent:
        # close() drops it under the reason it failed for, which is the verdict.
        wardex.close()
    code, line = _verdict(transport, outcomes, answers, before, started, where)
    say(line)
    wardex.close()
    return code


def _observe_exports(transport: Any) -> list[object]:
    """Wrap this transport instance's `export` to keep what each call returned
    or raised. The doctor owns its process, so it may."""
    seen: list[object] = []
    original: Callable[..., Any] = transport.export

    def export(envelope: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = original(envelope, *args, **kwargs)
        except BaseException as exc:
            seen.append(exc)
            raise
        seen.append(result)
        return result

    transport.export = export
    return seen


class _Replayed:
    """A 2xx answer the doctor read first, handed to the exporter to read again."""

    def __init__(self, response: Any, body: bytes) -> None:
        self.headers, self.status = response.headers, response.status
        self._response, self._body = response, io.BytesIO(body)

    def read(self, amt: int = -1) -> bytes:
        return self._body.read(amt)

    def __enter__(self) -> _Replayed:
        return self

    def __exit__(self, *exc: object) -> None:
        self._response.close()


def _observe_answers() -> tuple[list[tuple[int, Answer]], Callable[[], None]]:
    """Keep the status and the classified body of every 2xx answer the
    exporter reads, by wrapping `urllib.request.urlopen` for the send, and
    return them with the call that puts urlopen back. The exporter reads the
    same bytes after the doctor, so what it records is what it would have. The
    doctor owns its process, so it may."""
    answers: list[tuple[int, Answer]] = []
    real = urllib.request.urlopen

    def urlopen(*args: Any, **kwargs: Any) -> Any:
        response = real(*args, **kwargs)
        try:
            body = response.read(MAX_ANSWER_BYTES)
            headers = response.headers
            answer = read_answer(
                headers.get("content-type", ""), headers.get("content-encoding", ""), body
            )
            answers.append((int(response.status), answer))
        except Exception:  # noqa: BLE001 — an unreadable answer is judged as no answer
            return response
        return _Replayed(response, body)

    def restore() -> None:
        urllib.request.urlopen = real

    urllib.request.urlopen = urlopen
    return answers, restore


def _judge_answer(status: int, answer: Answer, where: str) -> tuple[int, str] | None:
    """A 2xx that is not a clean OTLP acceptance: a failure, or an OK that
    says it could not confirm. None when the answer accepted the span."""
    if answer.kind == "html":
        return 1, (
            f"FAIL: {where} answered with a web page (HTTP {status}, {answer.content_type}), "
            "not a telemetry receiver: a proxy, a login page or a web app is at that address. "
            "Point WARDEX_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT at the collector's OTLP/HTTP "
            "address (usually port 4318)."
        )
    if answer.kind == "otlp" and answer.rejected > 0:
        said = " ".join(answer.message.split())[:200] or "no reason given"
        return 1, (
            f"FAIL: {where} rejected the test span ({answer.rejected} span(s) rejected in an "
            f"HTTP {status} answer: {said}). Read the collector's logs for why it refuses "
            "these spans."
        )
    if answer.kind == "other":
        return 0, (
            f"OK: {where} answered HTTP {status} to the test span, but not with an OTLP "
            f"response ({answer.content_type or 'no content type'}), so wardex cannot confirm "
            f"the span was kept. Check that the receiver shows a span named {TEST_SPAN_NAME!r}."
        )
    return None


def _refused(error: object, where: str) -> tuple[int, str] | None:
    """A receiver's HTTP refusal, said with the fix its status calls for: a
    loss reason's own fix names a project key, which a collector has not."""
    if not isinstance(error, str) or not error.startswith("HTTP "):
        return None
    status = error[len("HTTP ") :]
    if status == "404":
        fix = "Nothing answers at that path: OTLP/HTTP collectors take traces at /v1/traces."
    elif status in ("401", "403"):
        fix = (
            "It refused the credentials: check OTEL_EXPORTER_OTLP_HEADERS (or "
            "backend.headers) for a collector, WARDEX_API_KEY for a wardex receiver."
        )
    else:
        fix = "Its own logs say why it refused the span."
    return 1, f"FAIL: {where} answered {error} to the test span. {fix}"


def _verdict(
    transport: Any,
    outcomes: list[object],
    answers: list[tuple[int, Answer]],
    before: Mapping[tuple[str, str], int],
    started: float,
    where: str,
) -> tuple[int, str]:
    """OK only on evidence the exporter took the span; otherwise what the
    receiver answered (its own words name the cause best), the first loss
    recorded since `before` with its fix, or what the exporter did."""
    unconfirmed = None
    for status, answer in answers:
        judged = _judge_answer(status, answer, where)
        if judged is not None and judged[0]:
            return judged
        unconfirmed = unconfirmed or judged
    refused = _refused(getattr(getattr(transport, "export_state", None), "last_error", None), where)
    if refused is not None:
        return refused
    new = [
        loss
        for loss in wardex.diagnostics().losses
        if loss.nature != "none" and loss.count > before.get((loss.reason, loss.unit), 0)
    ]
    if new:
        loss = new[0]
        here = " Here, run the doctor with a larger --timeout." * ("timeout" in loss.next_step)
        return 1, f"FAIL: {loss.reason}: {loss.meaning} {loss.next_step}{here}"
    if unconfirmed is not None:
        return unconfirmed
    state = getattr(transport, "export_state", None)
    if state is not None:
        success = getattr(state, "last_success_at", None)
        if success is not None and success >= started:
            return 0, f"OK: {where} accepted the test span."
        error = getattr(state, "last_error", None)
        if error:
            return 1, f"FAIL: the export to {where} did not succeed ({error})."
    if not outcomes:
        return 1, (
            f"FAIL: the test span was not exported to {where} within the deadline. "
            "Check that the address is reachable from this process."
        )
    last = outcomes[-1]
    if isinstance(last, BaseException):
        return 1, f"FAIL: the exporter raised {type(last).__name__} while sending to {where}."
    if last is not None:
        return 1, (
            f"FAIL: the export to {where} did not finish within the deadline; the receiver "
            "may be slow or unreachable."
        )
    return 0, f"OK: {where} accepted the test span."


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = sys.stdout if out is None else out
    parser = argparse.ArgumentParser(
        prog="python -m wardex_sdk", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="check the install end to end")
    check.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the export")
    look = commands.add_parser("explain", help="what a limitation marker or loss reason means")
    which = look.add_mutually_exclusive_group(required=True)
    which.add_argument("name", nargs="?", help="a marker or reason, e.g. unit_inferred_sole")
    which.add_argument("--all", action="store_true", help="every marker and reason, as Markdown")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(out, environ=os.environ, deadline=args.timeout)
    if args.all:
        out.write(markdown())
        return 0
    said = wardex.explain(args.name)
    out.write(said + "\n")
    return 1 if "is not a wardex limitation marker" in said else 0


if __name__ == "__main__":
    sys.exit(main())
