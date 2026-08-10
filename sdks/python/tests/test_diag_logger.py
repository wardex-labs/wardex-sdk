"""The diagnostic channel: the stdlib logger `wardex_sdk`.

Diagnostics moved from bare `print(..., file=sys.stderr)` onto
`logging.getLogger("wardex_sdk")` so a host can route or silence them with the
tools it already has. Three properties carry the migration, and each has a
test here:

* ZERO-CONFIG BEHAVIOR IS UNCHANGED: the pre-attached handler writes the same
  bytes to stderr the prints did — `[wardex] ` prefix included, added by the
  handler's formatter, resolved against the CURRENT `sys.stderr` at emit time
  so capsys, `redirect_stderr` and a swapped fd 2 all keep working.

* A HOST THAT TAKES THE LOGGER OWNS IT: its handler receives the clean
  message (no prefix — that belongs to wardex's own formatter), and nothing
  double-prints through the root logger (`propagate` is False).

* EMISSION NEVER RAISES INTO THE HOST (I6): a raising host handler and an
  unwritable stderr are both swallowed and counted under `LOG_FAILED`, and
  the emission runs under the exporter's self-capture suppression so a log
  handler that POSTs cannot have its traffic captured by wardex's own seams.
"""

from __future__ import annotations

import io
import logging
import sys
import uuid

import pytest

from wardex_sdk._assembly import diag_info, diag_warning
from wardex_sdk._assembly._diag import (
    LOG_FAILED,
    _StderrAtEmitTime,
    counters,
    logger,
    report_once,
)
from wardex_sdk._suppress import is_suppressed


@pytest.fixture
def wardex_logger():
    """The `wardex_sdk` logger, with its configuration restored afterwards.

    Tests below replace handlers and levels; leaking that would silence the
    stderr assertions of every capsys test that runs after this module.
    """
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    try:
        yield logger
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate


class _Recording(logging.Handler):
    """A host's own handler: stores the clean messages it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.levels: list[int] = []
        self.suppressed_during_emit: list[bool] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())
        self.levels.append(record.levelno)
        self.suppressed_during_emit.append(is_suppressed())


# --- zero-config behavior ----------------------------------------------------


def test_default_output_is_byte_identical_to_the_print_it_replaced(capsys):
    diag_warning("dropped 3 spans (buffer full)")
    assert capsys.readouterr().err == "[wardex] dropped 3 spans (buffer full)\n"

    diag_info("no transport or backend.endpoint configured: capturing, exporting nothing")
    assert capsys.readouterr().err == (
        "[wardex] no transport or backend.endpoint configured: capturing, exporting nothing\n"
    )


def test_both_severities_pass_the_default_level():
    """INFO announcements must reach stderr out of the box; a logger left at
    the root's WARNING default would silently eat the NoOp-transport line."""
    assert logger.isEnabledFor(logging.INFO)
    assert logger.isEnabledFor(logging.WARNING)


def test_propagate_is_false_so_nothing_double_prints_through_root():
    assert logger.propagate is False


def test_stderr_is_resolved_at_emit_time_not_at_import(monkeypatch):
    """A host that redirects stderr AFTER `import wardex_sdk` still receives
    the diagnostics — the handler stores no stream at construction."""
    target = io.StringIO()
    monkeypatch.setattr(sys, "stderr", target)
    diag_warning("redirected")
    assert target.getvalue() == "[wardex] redirected\n"


def test_report_once_emits_through_the_logger_with_the_prefix(capsys):
    key = f"test.diag.{uuid.uuid4()}"
    report_once("adapter x failed to load", key=key)
    report_once("adapter x failed to load", key=key)
    err = capsys.readouterr().err
    assert err == "[wardex] adapter x failed to load\n"  # once, prefixed, clean call site


# --- a host that configures the logger owns it -------------------------------


def test_a_host_handler_receives_clean_messages_and_stderr_goes_silent(wardex_logger, capsys):
    recording = _Recording()
    wardex_logger.handlers[:] = [recording]

    diag_warning("envelope dropped (boom)")
    diag_info("resolved config: WardexConfig(...)")
    report_once("could not ship 2 buffered span(s)", key=f"test.diag.{uuid.uuid4()}")

    assert capsys.readouterr().err == ""  # the host handler replaced stderr entirely
    assert recording.messages == [
        "envelope dropped (boom)",  # no "[wardex] " — the prefix is wardex's formatter's
        "resolved config: WardexConfig(...)",
        "could not ship 2 buffered span(s)",
    ]
    assert recording.levels == [logging.WARNING, logging.INFO, logging.WARNING]


def test_install_default_handler_defers_to_a_host_configured_logger(wardex_logger):
    """The attach-if-empty rule: a logger that already carries handlers is the
    host's, and re-running the installer must not add wardex's."""
    from wardex_sdk._assembly._diag import _install_default_handler

    recording = _Recording()
    wardex_logger.handlers[:] = [recording]
    _install_default_handler()
    assert wardex_logger.handlers == [recording]


# --- emission never raises into the host -------------------------------------


def test_a_raising_host_handler_does_not_raise_into_the_host(wardex_logger):
    class Raising(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            raise RuntimeError("host handler bug")

    wardex_logger.handlers[:] = [Raising()]
    before = counters.get(LOG_FAILED)
    diag_warning("must not raise")  # the assertion is that this line returns
    assert counters.get(LOG_FAILED) == before + 1  # ...and the failure is counted


def test_the_default_handler_survives_an_unwritable_stderr(monkeypatch):
    class Closed:
        def write(self, _s):
            raise ValueError("I/O operation on closed file")

        def flush(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stderr", Closed())
    before = counters.get(LOG_FAILED)
    report_once("into a closed stderr", key=f"test.diag.{uuid.uuid4()}")
    assert counters.get(LOG_FAILED) == before + 1


def test_emission_runs_under_self_capture_suppression(wardex_logger):
    """A host log handler that POSTs its records over HTTP must not have that
    traffic captured by wardex's own byte seams: the suppression guard the
    exporter uses covers every handler invocation too."""
    recording = _Recording()
    wardex_logger.handlers[:] = [recording]
    assert not is_suppressed()
    diag_warning("suppressed while emitting")
    assert recording.suppressed_during_emit == [True]
    assert not is_suppressed()


# --- the default handler's own shape -----------------------------------------


def test_the_default_handler_is_attached_and_formats_with_the_prefix():
    handlers = [h for h in logger.handlers if isinstance(h, _StderrAtEmitTime)]
    assert len(handlers) == 1
    record = logging.LogRecord(
        name="wardex_sdk",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="plain message",
        args=(),
        exc_info=None,
    )
    assert handlers[0].format(record) == "[wardex] plain message"
