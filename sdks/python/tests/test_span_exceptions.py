"""An exception that leaves a span is recorded on it, the way OpenTelemetry records one.

A `@wardex.tool` that raised used to ship `status=UNSET` with no trace of the failure,
so the record of a run said nothing about why a branch died and the answer had to be
found somewhere else. Now the span says `status=ERROR`, names the failure in
`error.type`, and carries one `exception` event with `exception.type`,
`exception.message` and `exception.stacktrace` — and the host still receives the very
exception it raised, with the traceback it had.

The message and the stack trace are host text, so they go through the encoder's
masking on both wires, every frame's file is placed by the call-site rule (never an
absolute path), and this process's home folder is written as `~` wherever else the
text starts a path with it, and nowhere it might be naming a different folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import os
import re
import sys
import traceback
from pathlib import Path

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _source_paths, _tracing
from wardex_sdk._assembly import counters
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import StatusCode
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._native import native
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport
from wardex_sdk.transport._codec import decode

EMAIL = "alice.kim@example.com"
API_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


@pytest.fixture
def recording() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    return t


class ToolFailed(Exception):
    pass


def _spans(t: _Recording) -> dict:
    _hub.get_client().flush()
    return {sp.name: sp for env in t.envelopes for sp in env.spans}


def _exception_events(sp) -> list[dict]:
    return [dict(ev.attributes) for ev in sp.events if ev.name == "exception"]


def _the_exception(sp) -> dict:
    (event,) = _exception_events(sp)
    return event


def _otlp_events(t: _Recording, name: str) -> list[dict]:
    """The span's events as the OTLP encoder ships them, under the stored policy."""
    out = []
    for env in t.envelopes:
        for body in t.encode(env):
            decoded = native.codec.decode_otlp_traces(gzip.decompress(body))
            for rs in decoded["resource_spans"]:
                for ss in rs["scope_spans"]:
                    out += [ev for sp in ss["spans"] if sp["name"] == name for ev in sp["events"]]
    return out


def _envelope_events(t: _Recording, name: str) -> list[dict]:
    """The span's events as the wardex envelope encoder ships them."""
    out = []
    for env in t.envelopes:
        wire, _ = native.codec.encode_envelope_export(
            env, t._pii_mode, list(t._pii_disabled), t._limits, **t._pii_names()
        )
        for item in decode(wire)["items"]:
            sp = item.get("span")
            if sp and sp["name"] == name:
                out += [
                    {**ev, "attributes": {a["key"]: a["value"] for a in ev["attributes"]}}
                    for ev in sp["events"]
                ]
    return out


def _frame_files(stacktrace: str) -> list[str]:
    return re.findall(r'File "(.+?)", line \d', stacktrace)


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def test_a_raising_sync_decorated_function_ships_a_failed_span(recording):
    @wardex.tool
    def search():
        raise ValueError("index unavailable")

    with pytest.raises(ValueError):
        search()

    sp = _spans(recording)["search"]
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "ValueError"
    event = _the_exception(sp)
    assert event["exception.type"] == "ValueError"
    assert event["exception.message"] == "index unavailable"
    assert event["exception.stacktrace"].startswith("Traceback (most recent call last):\n")
    assert event["exception.stacktrace"].endswith("ValueError: index unavailable\n")
    assert "in search" in event["exception.stacktrace"]


def test_a_raising_async_decorated_function_ships_a_failed_span(recording):
    @wardex.agent
    async def researcher():
        await asyncio.sleep(0)
        raise ValueError("no sources")

    with pytest.raises(ValueError):
        asyncio.run(researcher())

    sp = _spans(recording)["researcher"]
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "ValueError"
    assert _the_exception(sp)["exception.message"] == "no sources"


@pytest.mark.parametrize("deco", [wardex.workflow, wardex.step])
def test_every_decorator_records_the_failure(recording, deco):
    @deco
    def unit():
        raise ValueError("x")

    with pytest.raises(ValueError):
        unit()

    sp = _spans(recording)["unit"]
    assert sp.status is StatusCode.ERROR
    assert len(_exception_events(sp)) == 1


def test_an_exception_leaving_a_span_block_ships_a_failed_span(recording):
    with pytest.raises(ValueError), wardex.span("rank-results"):
        raise ValueError("empty candidate list")

    sp = _spans(recording)["rank-results"]
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "ValueError"
    assert _the_exception(sp)["exception.message"] == "empty candidate list"


def test_an_exception_leaving_a_conversation_block_ships_a_failed_span(recording):
    with pytest.raises(ValueError), wardex.conversation("chat"):
        raise ValueError("turn failed")

    sp = _spans(recording)["chat"]
    assert sp.status is StatusCode.ERROR
    assert len(_exception_events(sp)) == 1


def test_a_user_defined_type_is_named_fully_qualified(recording):
    """OTel's `exception.type` is the fully qualified class name; a builtin's is bare."""
    with pytest.raises(ToolFailed), wardex.span("call"):
        raise ToolFailed("quota")

    sp = _spans(recording)["call"]
    expected = f"{ToolFailed.__module__}.ToolFailed"
    assert sp.error_type == expected
    assert _the_exception(sp)["exception.type"] == expected


def test_each_span_the_exception_leaves_records_it_and_a_caught_one_marks_nothing(recording):
    @wardex.tool
    def fetch():
        raise ValueError("timeout")

    @wardex.agent
    def planner():
        fetch()

    @wardex.workflow
    def handled():
        with contextlib.suppress(ValueError):
            fetch()
        return "fallback"

    with pytest.raises(ValueError):
        planner()
    assert handled() == "fallback"

    spans = _spans(recording)
    assert spans["planner"].status is StatusCode.ERROR
    assert len(_exception_events(spans["planner"])) == 1
    assert spans["handled"].status is StatusCode.UNSET
    assert spans["handled"].events == ()


def test_a_message_whose_str_raises_still_records_the_failure(recording):
    class Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no str for you")

    with pytest.raises(Unprintable), wardex.span("odd"):
        raise Unprintable()

    sp = _spans(recording)["odd"]
    assert sp.status is StatusCode.ERROR
    assert _the_exception(sp)["exception.message"] == "<exception str() failed>"


# --------------------------------------------------------------------------
# the host's side: same object, same traceback, its own status wins
# --------------------------------------------------------------------------


def _raise_inside(cm, exc: BaseException) -> None:
    with cm:
        raise exc


def _catch(cm, exc: BaseException) -> BaseException:
    try:
        _raise_inside(cm, exc)
    except BaseException as caught:  # noqa: BLE001 — the test inspects what arrived
        return caught
    raise AssertionError("nothing was raised")


def test_the_host_receives_the_same_exception_with_the_same_traceback(recording):
    bare = ValueError("bare")
    seen_bare = _catch(contextlib.nullcontext(), bare)
    wrapped = ValueError("wrapped")
    seen_wrapped = _catch(wardex.span("block"), wrapped)

    assert seen_wrapped is wrapped
    assert traceback.extract_tb(seen_wrapped.__traceback__) == traceback.extract_tb(
        seen_bare.__traceback__
    )
    assert wrapped.__context__ is None and wrapped.__cause__ is None
    assert getattr(wrapped, "__notes__", None) is None
    assert _spans(recording)["block"].status is StatusCode.ERROR


def test_a_block_that_does_not_raise_keeps_its_status(recording):
    with wardex.span("quiet"):
        pass
    with wardex.span("fine") as s:
        s.set_status(StatusCode.OK)

    spans = _spans(recording)
    assert spans["quiet"].status is StatusCode.UNSET
    assert spans["quiet"].events == ()
    assert spans["fine"].status is StatusCode.OK
    assert spans["fine"].error_type is None


def test_a_status_the_host_set_wins_and_the_event_still_records(recording):
    with pytest.raises(ValueError), wardex.span("expected-miss") as s:
        s.set_status(StatusCode.OK)
        raise ValueError("cache miss")

    sp = _spans(recording)["expected-miss"]
    assert sp.status is StatusCode.OK
    assert sp.error_type is None
    assert len(_exception_events(sp)) == 1


def test_an_error_type_the_host_named_wins(recording):
    with pytest.raises(ValueError), wardex.span("named") as s:
        s.set_status(StatusCode.ERROR)
        s.set_error("rate_limited")
        raise ValueError("429")

    assert _spans(recording)["named"].error_type == "rate_limited"


def test_an_error_status_without_a_type_takes_the_exception_type(recording):
    """Before, a host-set ERROR with no type shipped OTel's `_OTHER` fallback even
    though the exception that ended the block names the failure exactly."""
    with pytest.raises(ValueError), wardex.span("untyped") as s:
        s.set_status(StatusCode.ERROR)
        raise ValueError("bad input")

    assert _spans(recording)["untyped"].error_type == "ValueError"


@pytest.mark.parametrize(
    "exc",
    [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(3), GeneratorExit()],
    ids=lambda e: type(e).__name__,
)
def test_a_cancellation_or_exit_signal_is_not_recorded_as_a_failure(recording, exc):
    seen = _catch(wardex.span("stopped"), exc)

    assert seen is exc
    sp = _spans(recording)["stopped"]
    assert sp.status is StatusCode.UNSET
    assert sp.error_type is None
    assert sp.events == ()


def test_a_wardex_failure_while_recording_never_reaches_the_host(recording, monkeypatch):
    def broken(*_args):
        raise RuntimeError("formatter defect")

    monkeypatch.setattr(_tracing, "_stacktrace", broken)
    before = counters.get("tracing.exception_stacktrace")

    with pytest.raises(ValueError, match="host failure"), wardex.span("degraded"):
        raise ValueError("host failure")

    sp = _spans(recording)["degraded"]
    assert sp.status is StatusCode.ERROR
    event = _the_exception(sp)
    assert event["exception.message"] == "host failure"
    assert event["exception.stacktrace"] == ""
    assert counters.get("tracing.exception_stacktrace") == before + 1


def test_a_signal_while_recording_still_finishes_the_span(recording, monkeypatch):
    """A KeyboardInterrupt landing mid-record must not leave the span unshipped and
    installed as the parent of everything opened after it."""

    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(_tracing, "_stacktrace", interrupted)

    with pytest.raises(KeyboardInterrupt), wardex.span("interrupted"):
        raise ValueError("x")
    with wardex.span("after"):
        pass

    spans = _spans(recording)
    assert spans["interrupted"].status is StatusCode.ERROR
    assert spans["after"].parent_span_id is None


class _ModuleLookupRaises(type):
    """A host metaclass whose `__module__` is a property that raises."""

    @property
    def __module__(cls):
        raise RuntimeError("the host's own metaclass failed")


class _EveryLookupRaises(type):
    """A host metaclass that refuses every attribute lookup on its classes."""

    def __getattribute__(cls, name: str):
        raise RuntimeError("the host's own metaclass failed")


@pytest.mark.parametrize("meta", [_ModuleLookupRaises, _EveryLookupRaises])
def test_an_exception_class_whose_name_lookup_raises_still_reaches_the_host(recording, meta):
    """Naming the class used to run the host's metaclass outside any guard, so what
    it raised left the block in place of the host's own exception, and the span
    shipped UNSET with no event."""
    odd = meta("Odd", (Exception,), {"__module__": "support_bot.errors", "__qualname__": "Odd"})
    host = odd("the host's own failure")

    seen = _catch(wardex.span("odd-class"), host)

    assert seen is host
    sp = _spans(recording)["odd-class"]
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "support_bot.errors.Odd"
    event = _the_exception(sp)
    assert event["exception.type"] == "support_bot.errors.Odd"
    assert event["exception.message"] == "the host's own failure"


def test_a_signal_whose_class_lookup_raises_still_reaches_the_host(recording):
    """Telling a failure from a signal used to ask the exception for `__class__`,
    which runs host code for anything that is not an `Exception`."""

    class Stop(BaseException):
        @property
        def __class__(self):
            raise RuntimeError("the host's own property failed")

    host = Stop()
    seen = _catch(wardex.span("odd-signal"), host)
    with wardex.span("after"):
        pass

    assert seen is host
    spans = _spans(recording)
    assert spans["odd-signal"].status is StatusCode.UNSET
    assert spans["odd-signal"].events == ()
    assert spans["after"].parent_span_id is None


# --------------------------------------------------------------------------
# what leaves the process
# --------------------------------------------------------------------------


@pytest.mark.parametrize("read", [_otlp_events, _envelope_events], ids=["otlp", "envelope"])
def test_both_wires_carry_the_event_under_the_standard_keys(recording, read):
    with pytest.raises(ValueError), wardex.span("wired"):
        raise ValueError("boom")

    _spans(recording)
    (event,) = [ev for ev in read(recording, "wired") if ev["name"] == "exception"]
    assert set(event["attributes"]) == {
        "exception.type",
        "exception.message",
        "exception.stacktrace",
    }
    assert event["attributes"]["exception.type"] == "ValueError"


@pytest.mark.parametrize("read", [_otlp_events, _envelope_events], ids=["otlp", "envelope"])
def test_the_message_and_the_stack_trace_are_masked_on_both_wires(recording, read):
    @wardex.tool
    def lookup():
        raise ValueError(f"no account for {EMAIL} using {API_KEY}")

    with pytest.raises(ValueError):
        lookup()

    _spans(recording)
    (event,) = read(recording, "lookup")
    text = "\n".join(str(v) for v in event["attributes"].values())
    assert EMAIL not in text
    assert API_KEY not in text
    assert "[EMAIL]" in event["attributes"]["exception.message"]
    assert "[EMAIL]" in event["attributes"]["exception.stacktrace"]


_AGENT_SOURCE = """\
import wardex_sdk as wardex


@wardex.tool
def open_report(path):
    with open(path) as f:
        return f.read()


def explode():
    raise ValueError("inner")


@wardex.tool
def chained():
    try:
        explode()
    except ValueError as e:
        raise RuntimeError("outer") from e
"""


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A package imported from a project under a fake home folder; the fake home is
    this process's home for the test (`HOME` on POSIX, `USERPROFILE` on Windows)."""
    home = tmp_path / "Users" / "alice.kim"
    root = home / "work" / "acme-acquisition-2027"
    (root / "support_bot").mkdir(parents=True)
    (root / "support_bot" / "__init__.py").write_text("")
    (root / "support_bot" / "agent.py").write_text(_AGENT_SOURCE)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.syspath_prepend(str(root))
    before = set(sys.modules)
    yield home
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _assert_no_absolute_frame(stacktrace: str) -> None:
    files = _frame_files(stacktrace)
    assert files
    assert not [f for f in files if os.path.isabs(f)], files


@pytest.mark.parametrize("read", [_otlp_events, _envelope_events], ids=["otlp", "envelope"])
def test_frame_files_leave_package_relative_and_the_home_folder_as_tilde(
    recording, fake_home, read
):
    from support_bot import agent

    missing = fake_home / "reports" / "q3.txt"
    with pytest.raises(FileNotFoundError):
        agent.open_report(str(missing))

    _spans(recording)
    (event,) = read(recording, "open_report")
    stacktrace = event["attributes"]["exception.stacktrace"]
    message = event["attributes"]["exception.message"]
    assert os.path.join("support_bot", "agent.py") in _frame_files(stacktrace)
    _assert_no_absolute_frame(stacktrace)
    for text in (stacktrace, message):
        assert str(fake_home) not in text
        assert "alice.kim" not in text
    assert os.path.join("~", "reports", "q3.txt") in message


def test_a_chained_traceback_places_every_frame(recording, fake_home):
    from support_bot import agent

    with pytest.raises(RuntimeError):
        agent.chained()

    stacktrace = _the_exception(_spans(recording)["chained"])["exception.stacktrace"]
    assert "The above exception was the direct cause of the following exception" in stacktrace
    assert "ValueError: inner" in stacktrace
    assert "in explode" in stacktrace
    _assert_no_absolute_frame(stacktrace)
    assert str(fake_home) not in stacktrace


def _message_through_a_span(recording: _Recording, text: str) -> tuple[str, str]:
    with pytest.raises(ValueError), wardex.span("scrubbed"):
        raise ValueError(text)
    event = _the_exception(_spans(recording)["scrubbed"])
    return event["exception.message"], event["exception.stacktrace"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No such file: '{home}{sep}q3.txt'", "No such file: '~{sep}q3.txt'"),
        ("File exists: '{home}'", "File exists: '~'"),
        ('open("{home}{sep}notes.md")', 'open("~{sep}notes.md")'),
        ("PATH=/usr/bin:{home}{sep}bin", "PATH=/usr/bin:~{sep}bin"),
        ("cd {home}", "cd ~"),
        ("root is {home}\nnext line", "root is ~\nnext line"),
        ("see file://{home}{sep}a.txt", "see file://~{sep}a.txt"),
    ],
)
def test_the_home_folder_is_written_as_tilde_where_it_starts_a_path(
    recording, fake_home, text, expected
):
    home, sep = str(fake_home), os.sep
    message, stacktrace = _message_through_a_span(recording, text.format(home=home, sep=sep))

    assert message == expected.format(sep=sep)
    assert expected.format(sep=sep) in stacktrace


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("permission denied for {home}.", "permission denied for ~."),
        ("cannot write to {home} (read-only)", "cannot write to ~ (read-only)"),
        ("searched {home}, then /tmp", "searched ~, then /tmp"),
        ("{home}: Permission denied", "~: Permission denied"),
        ("no config in ({home})", "no config in (~)"),
        ("PATH={home}{sep}bin:{home}:/usr/bin", "PATH=~{sep}bin:~:/usr/bin"),
        # A space ends the path even where a folder's name went on with one:
        # the cost of never sending the user name out of a sentence.
        ("copy {home} 2{sep}report.txt", "copy ~ 2{sep}report.txt"),
    ],
)
@pytest.mark.parametrize("read", [_otlp_events, _envelope_events], ids=["otlp", "envelope"])
def test_the_home_folder_a_sentence_names_leaves_as_tilde_on_both_wires(
    recording, fake_home, read, text, expected
):
    """The home folder used to be rewritten only before a separator, a quote or a
    line end, so a message naming it in a sentence (`permission denied for
    /Users/alice.`, `cannot write to /Users/alice (read-only)`) sent the OS user
    name out in the message and in the stack trace's last line, on both wires."""
    home, sep = str(fake_home), os.sep
    with pytest.raises(ValueError), wardex.span("prose"):
        raise ValueError(text.format(home=home, sep=sep))

    _spans(recording)
    (event,) = read(recording, "prose")
    message = event["attributes"]["exception.message"]
    stacktrace = event["attributes"]["exception.stacktrace"]
    assert message == expected.format(sep=sep)
    assert stacktrace.endswith(f"ValueError: {message}\n")
    for value in (message, stacktrace):
        assert home not in value
        assert "alice.kim" not in value


@pytest.mark.parametrize(
    "text",
    [
        "copy {home}-old{sep}report.txt",
        "copy {home}.bak{sep}report.txt",
        "copy {home}+test{sep}report.txt",
        "copy {home}berly{sep}report.txt",
        "restore {sep}backup{home}{sep}report.txt",
        "restore .{home}{sep}report.txt",
    ],
)
def test_a_different_folder_that_contains_the_home_path_is_left_as_written(
    recording, fake_home, text
):
    """The home folder used to be matched wherever its string appeared, so a sibling
    whose name merely begins with it (`alice.kim-old`) and a copy of it under another
    folder (`/backup/Users/alice.kim`) both left as `~`: a path the SDK never saw,
    claiming a different folder was the host's home."""
    written = text.format(home=str(fake_home), sep=os.sep)
    message, stacktrace = _message_through_a_span(recording, written)

    assert message == written
    assert written in stacktrace


def test_the_otlp_value_cap_cuts_a_long_stack_trace_and_says_so(recording):
    recording._set_limits(LimitsConfig(max_otlp_attribute_bytes=256).to_native())

    with pytest.raises(ValueError), wardex.span("long"):
        raise ValueError("x" * 4096)

    _spans(recording)
    (event,) = _otlp_events(recording, "long")
    for value in event["attributes"].values():
        assert len(value.encode()) <= 256
    for env in recording.envelopes:
        for body in recording.encode(env):
            decoded = native.codec.decode_otlp_traces(gzip.decompress(body))
            (sp,) = decoded["resource_spans"][0]["scope_spans"][0]["spans"]
            assert "otlp_attribute_truncated" in str(sp["attributes"].get("wardex.limitations"))


# --------------------------------------------------------------------------
# what recording costs the host
# --------------------------------------------------------------------------


def _frame_count(stacktrace: str) -> int:
    return len(_frame_files(stacktrace))


def _left_out(stacktrace: str) -> list[int]:
    found = re.findall(r"^ *\[(\d+) earlier frames? not recorded\]$", stacktrace, re.M)
    return [int(n) for n in found]


def test_a_deep_recursion_records_a_bounded_stack_trace_on_every_span(recording):
    """Every span an exception leaves records it, and each used to format the whole
    traceback below it. A recursion through a decorated function, N spans deep,
    formatted N traces of up to 2N frames while the host unwound: about 10 s and
    20 MB of text for one RecursionError that unwinds in 10 ms without recording.
    Each traceback now keeps its frames nearest the raise and says how many came
    before."""

    @wardex.tool
    def descend(n):
        return descend(n + 1)

    with pytest.raises(RecursionError) as caught:
        descend(0)

    _hub.get_client().flush()
    spans = [sp for env in recording.envelopes for sp in env.spans]
    traces = {
        sp.context.span_id: ev["exception.stacktrace"]
        for sp in spans
        for ev in _exception_events(sp)
        if ev["exception.stacktrace"]
    }
    assert len(traces) > _source_paths._MAX_FRAMES  # deep enough for the cut to bite
    for trace in traces.values():
        assert _frame_count(trace) <= _source_paths._MAX_FRAMES
        assert trace.splitlines()[-1].startswith("RecursionError: maximum recursion depth")

    (outermost,) = [sp for sp in spans if sp.parent_span_id is None]
    trace = traces[outermost.context.span_id]
    # The test's own frame is the one frame of the caught traceback above the span.
    depth = sum(1 for _ in traceback.walk_tb(caught.tb)) - 1
    assert _frame_count(trace) == _source_paths._MAX_FRAMES
    assert _left_out(trace) == [depth - _source_paths._MAX_FRAMES]


def _sink(n: int) -> None:
    if n == 0:
        raise ValueError("bottom")
    _sink(n - 1)


def test_a_chained_traceback_is_cut_the_same_way(recording):
    with pytest.raises(RuntimeError), wardex.span("wrap"):
        try:
            _sink(100)
        except ValueError as e:
            cause_depth = sum(1 for _ in traceback.walk_tb(e.__traceback__))
            raise RuntimeError("outer") from e

    trace = _the_exception(_spans(recording)["wrap"])["exception.stacktrace"]
    cause, _, outer = trace.partition("The above exception was the direct cause")
    assert _left_out(cause) == [cause_depth - _source_paths._MAX_FRAMES]
    assert _frame_count(cause) <= _source_paths._MAX_FRAMES
    assert "ValueError: bottom" in cause
    assert _left_out(outer) == []
    assert outer.endswith("RuntimeError: outer\n")


def _placed_as_python_prints(exc: BaseException) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return re.sub(r'File "(.+?)", line ', lambda m: f'File "{os.path.basename(m[1])}", line ', text)


def test_a_stack_trace_under_the_cut_is_what_python_prints(recording):
    """The cut and the per-frame formatting reuse change no frame Python prints:
    under the cap the stack trace is the stdlib's own, files placed, every time."""
    try:
        try:
            _sink(3)
        except ValueError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as e:
        first = _source_paths._stacktrace(e, e.__traceback__)
        again = _source_paths._stacktrace(e, e.__traceback__)
        expected = _placed_as_python_prints(e)

    assert first == expected
    assert again == expected


#: 3.10 formats each frame inline in `StackSummary.format`: no per-frame hook, so
#: nothing is formatted through `_Frames.format_frame_summary` and nothing is held.
_HOOKED = hasattr(traceback.StackSummary, "format_frame_summary")


def test_a_frame_is_formatted_once_and_a_changed_line_afresh(monkeypatch):
    monkeypatch.setattr(_source_paths, "_FORMATTED", {})
    calls = []
    stdlib = getattr(traceback.StackSummary, "format_frame_summary", None)

    def counting(self, frame_summary, **kwargs):
        calls.append(frame_summary.line)
        return stdlib(self, frame_summary, **kwargs)

    monkeypatch.setattr(traceback.StackSummary, "format_frame_summary", counting, raising=False)

    def frames(*lines: str) -> list[str]:
        summaries = [traceback.FrameSummary("agent.py", 7, "run", line=line) for line in lines]
        return _source_paths._Frames(summaries).format()

    assert frames("x = 1", "x = 1") == frames("x = 1", "x = 1")
    assert calls == (["x = 1"] if _HOOKED else [])
    assert frames("x = 2") != frames("x = 1")
    assert calls == (["x = 1", "x = 2"] if _HOOKED else [])


def test_the_formatted_frames_held_are_bounded(monkeypatch):
    monkeypatch.setattr(_source_paths, "_FORMATTED", {})
    for i in range(_source_paths._FORMATTED_MAX + 10):
        summary = traceback.FrameSummary("agent.py", i + 1, "run", line="pass")
        _source_paths._Frames([summary]).format()
        assert len(_source_paths._FORMATTED) <= _source_paths._FORMATTED_MAX
    assert bool(_source_paths._FORMATTED) is _HOOKED
