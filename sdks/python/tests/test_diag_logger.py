"""The diagnostic channel: the stdlib logger `wardex_sdk`, with library manners.

Four properties, each with a test here:

* A PROCESS WITH NO LOGGING CONFIGURED SEES THE STDERR LINE: `[wardex] ` plus
  the message, INFO included, resolved against the CURRENT `sys.stderr` at
  emit time so capsys, `redirect_stderr` and a swapped fd 2 all keep working.

* A HOST THAT CONFIGURED LOGGING GETS THE LINE THROUGH LOGGING, decided per
  line, not at import: its levels decide what shows (INFO is hidden under
  `basicConfig(level=WARNING)`), its handlers decide where, the message is
  clean (no prefix), and nothing is also written to stderr. Importing wardex
  leaves the logging tree as it was.

* EVERY LINE CARRIES STRUCTURED FIELDS (`wardex_reason`, `wardex_count`,
  `wardex_subject`) for JSON pipelines, and a condition that keeps happening
  is said again, with its count, at most once per interval.

* EMISSION NEVER RAISES INTO THE HOST (I6): a raising host handler and an
  unwritable stderr are both swallowed and counted under `LOG_FAILED`, and
  the emission runs under the exporter's self-capture suppression so a log
  handler that POSTs cannot have its traffic captured by wardex's own seams.
"""

from __future__ import annotations

import io
import logging
import subprocess
import sys
import uuid

import pytest

from wardex_sdk._assembly import _diag, diag_info, diag_warning
from wardex_sdk._assembly._diag import (
    _STDERR,
    LOG_FAILED,
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


def test_both_severities_reach_stderr_when_nothing_is_configured(capsys):
    """INFO announcements must reach a bare process's stderr: the root's
    WARNING default would otherwise silently eat the NoOp-transport line."""
    diag_info("an announcement")
    diag_warning("a loss")
    assert capsys.readouterr().err == "[wardex] an announcement\n[wardex] a loss\n"


def test_a_level_set_on_the_wardex_logger_still_silences_the_stderr_line(wardex_logger, capsys):
    wardex_logger.setLevel(logging.ERROR)
    diag_warning("below the host's level")
    assert capsys.readouterr().err == ""


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
    wardex_logger.setLevel(logging.INFO)

    diag_warning("envelope dropped (boom)")
    diag_info("resolved config: WardexConfig(...)")
    report_once("could not ship 2 buffered span(s)", key=f"test.diag.{uuid.uuid4()}")

    assert capsys.readouterr().err == ""  # the host handler replaced stderr entirely
    assert recording.messages == [
        "envelope dropped (boom)",  # no "[wardex] " — the prefix is the stderr handler's
        "resolved config: WardexConfig(...)",
        "could not ship 2 buffered span(s)",
    ]
    assert recording.levels == [logging.WARNING, logging.INFO, logging.WARNING]


def test_every_line_carries_its_structured_fields(wardex_logger):
    class Fields(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[object, object, object]] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.seen.append((record.wardex_reason, record.wardex_count, record.wardex_subject))

    fields = Fields()
    wardex_logger.handlers[:] = [fields]
    key = f"test.diag.{uuid.uuid4()}"
    report_once("a loss", key=key, subject="http://collector")
    diag_warning("plain")
    diag_warning("counted", reason="queue_overflow", count=3, subject="span")
    assert fields.seen == [
        (key, 1, "http://collector"),
        (None, None, None),
        ("queue_overflow", 3, "span"),
    ]


def test_a_condition_that_keeps_happening_is_said_again_with_its_count(monkeypatch, capsys):
    now = [1000.0]
    monkeypatch.setattr(_diag, "_monotonic", lambda: now[0])
    key = f"test.diag.{uuid.uuid4()}"
    for _ in range(5):
        report_once("exports are failing", key=key)
    assert capsys.readouterr().err == "[wardex] exports are failing\n"  # once, inside the interval
    now[0] += _diag.RESURFACE_S
    report_once("exports are failing", key=key)
    err = capsys.readouterr().err
    assert err.startswith("[wardex] exports are failing (still happening: 6 times")
    report_once("exports are failing", key=key)
    assert capsys.readouterr().err == ""  # the interval starts again


_HOST_SCRIPT = """
import json, logging, sys
import wardex_sdk
from wardex_sdk._assembly import diag_info, diag_warning
tree = (logging.getLogger("wardex_sdk").handlers, logging.getLogger("wardex_sdk").propagate,
        logging.getLogger("wardex_sdk").level)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout,
                    format="HOST %(levelname)s %(name)s %(wardex_reason)s %(message)s")
diag_info("an announcement the host's level hides")
diag_warning("a loss", reason="network_error")
print("TREE", json.dumps([len(tree[0]), tree[1], tree[2]]))
"""


def test_a_host_that_configures_logging_after_import_gets_the_lines_its_way():
    """In a fresh process, because pytest owns this one's root logger."""
    proc = subprocess.run(
        [sys.executable, "-c", _HOST_SCRIPT], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "[wardex]" not in proc.stderr, "a configured host must not also get the stderr line"
    assert proc.stdout.splitlines() == [
        "HOST WARNING wardex_sdk network_error a loss",
        "TREE [0, true, 0]",  # importing wardex left the logging tree untouched
    ]


_DICTCONFIG_SCRIPT = """
import logging.config, sys
from wardex_sdk._assembly import diag_warning
logging.config.dictConfig({
    "version": 1,
    "formatters": {"f": {"format": "HOST %(levelname)s %(name)s %(wardex_reason)s %(message)s"}},
    "handlers": {"o": {"class": "logging.StreamHandler", "stream": "ext://sys.stdout",
                       "formatter": "f"}},
    "root": {"handlers": ["o"], "level": "INFO"},
})
diag_warning("a loss", reason="network_error")
"""


def test_a_host_that_configures_logging_with_dictconfig_after_import_still_gets_the_lines():
    """`dictConfig`'s default `disable_existing_loggers=True` disables every
    logger that exists when it runs, wardex's among them. That is how Django
    and `uvicorn --log-config` configure logging, after their imports, and it
    is not the host choosing to silence wardex: a loss line must still reach
    the host's handlers."""
    proc = subprocess.run(
        [sys.executable, "-c", _DICTCONFIG_SCRIPT], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["HOST WARNING wardex_sdk network_error a loss"]
    assert "[wardex]" not in proc.stderr


def test_a_test_runner_capture_still_sees_the_line(caplog, capsys):
    """pytest's own capture handlers are not a host pipeline: the stderr line is
    still owed, and `caplog` still sees the record."""
    diag_warning("seen by both")
    assert "[wardex] seen by both" in capsys.readouterr().err
    assert "seen by both" in caplog.messages


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


def test_the_stderr_handler_formats_with_the_prefix():
    record = logging.LogRecord(
        name="wardex_sdk",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="plain message",
        args=(),
        exc_info=None,
    )
    assert _STDERR.format(record) == "[wardex] plain message"


def test_a_refused_call_is_said_under_debug_by_host_and_path_only_and_rate_limited(
    monkeypatch, capsys
):
    from wardex_sdk._assembly import debug_refused_call

    now = [5000.0]
    monkeypatch.setattr(_diag, "_monotonic", lambda: now[0])
    _diag.reset_reports_for_test()
    debug_refused_call("api.example.com", "/v1/items?token=abc", "not a model call")
    debug_refused_call("api.example.com", "/v1/items?token=def", "not a model call")
    for i in range(_diag._REFUSED_PER_WINDOW + 5):
        debug_refused_call("api.example.com", f"/other/{i}", "not a model call")
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "[wardex] debug: not recorded: api.example.com/v1/items (not a model call)"
    assert len(lines) == _diag._REFUSED_PER_WINDOW
    assert "token" not in "".join(lines)
    now[0] += _diag._REFUSED_WINDOW_S
    debug_refused_call("api.example.com", "/v1/items", "not a model call")
    assert "(6 more refused since the last such line)" in capsys.readouterr().err


_WARNING_HOST_SCRIPT = """
import logging, os, sys
logging.basicConfig(stream=sys.stdout, format="HOST %(levelname)s %(message)s")  # WARNING
for name in [k for k in os.environ if k.startswith(("WARDEX_", "OTEL_"))]:
    del os.environ[name]
case = sys.argv[1]
if case == "disabled":
    os.environ["WARDEX_DISABLED"] = "1"
import wardex_sdk as wardex
if case == "pii-off":
    os.environ["WARDEX_PII"] = "off"
    wardex.init(transport=wardex.NoOpTransport())
else:
    wardex.init()
"""


@pytest.mark.parametrize(
    ("case", "said"),
    [
        ("no-destination", "wardex is not running"),
        ("disabled", "WARDEX_DISABLED is set"),
        ("pii-off", "PII masking is off"),
    ],
)
def test_a_default_basicconfig_host_still_sees_that_wardex_is_not_doing_its_job(case, said):
    """`logging.basicConfig()` leaves the root at WARNING. A line that changes
    what the app gets (nothing installed, nothing masked) must pass it."""
    proc = subprocess.run(
        [sys.executable, "-c", _WARNING_HOST_SCRIPT, case],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert any(
        line.startswith("HOST WARNING") and said in line for line in proc.stdout.splitlines()
    ), proc.stdout
