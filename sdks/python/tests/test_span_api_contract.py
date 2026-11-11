"""The span API as a developer meets it: what each call accepts, refuses and exports.

Every check here reads what leaves the process (the OTLP span a backend
receives, decoded from the exporter's own bytes) or what the developer's own
line does (raise, return), never the SDK's internals. The cases:

* a wrong type fails at the developer's call, naming what is accepted, and an
  enum field takes its value string (`kind="client"`); a span that still
  reaches the exporter with a field it cannot encode costs that span alone,
  never the batch it rides in with other requests' spans;
* the decorators export the identity an adapter's span carries
  (`gen_ai.agent.name`, `gen_ai.tool.name`, ...) under the same name shape,
  `invoke_agent researcher`, from the qualified function name when no name is
  given; `@tool` says the body ran in-process;
* attribute values: lists and dicts as JSON text, None removes, a non-str key
  raises;
* the code inside a span reaches it (`wardex.current_span()`), records its
  input, output and events, and the decorators can capture arguments and the
  return value;
* `async with wardex.span(...)` works, and bare `@wardex.span` fails where it
  is written;
* the decorators keep the wrapped function's signature and annotations;
* `testing.SpanNode` shows a span's status, attributes and events.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._native import native
from wardex_sdk.testing import RecordingTransport


class _Wire(wardex.Transport):
    """Exports through the real encoder, as a backend-bound transport does, and
    keeps each span as the OTLP receiver decodes it."""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    def export(self, envelope, *, timeout=None):
        for body in self.encode(envelope, compress=False):
            decoded = native.codec.decode_otlp_traces(body)
            for resource in decoded["resource_spans"]:
                for scope in resource["scope_spans"]:
                    self.spans += scope["spans"]


@pytest.fixture
def wire() -> _Wire:
    _hub.reset_for_test()
    transport = _Wire()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), transport))
    return transport


def _shipped(wire: _Wire) -> dict[str, dict]:
    _hub.get_client().flush()
    return {sp["name"]: sp for sp in wire.spans}


def _losses() -> list[tuple[str, int]]:
    return [(loss.reason, loss.count) for loss in wardex.diagnostics().losses]


# -- wrong types fail at the call; one bad span never costs its batch ---------


def test_an_enum_field_takes_its_value_string(wire):
    with wardex.span("call", kind="client") as s:
        s.status = "error"
        assert s.kind is wardex.SpanKind.CLIENT
        assert s.status is wardex.StatusCode.ERROR
    with wardex.span("other"):
        pass
    shipped = _shipped(wire)
    assert set(shipped) == {"call", "other"}
    assert shipped["call"]["kind"] == 3  # OTLP SPAN_KIND_CLIENT
    assert shipped["call"]["status"]["code"] == 2  # OTLP STATUS_CODE_ERROR
    assert _losses() == []


def test_text_given_for_bytes_is_sent_as_utf8(wire):
    with wardex.span("io") as s:
        s.input_data = "héllo"
        s.output_data = bytearray(b"done")
    attributes = _shipped(wire)["io"]["attributes"]
    assert attributes["wardex.input_data"] == "héllo"
    assert attributes["wardex.output_data"] == "done"


@pytest.mark.parametrize(
    ("call", "error", "names"),
    [
        (lambda: wardex.span(123), TypeError, "str"),
        (lambda: wardex.span("x", kind=5), TypeError, "'client'"),
        (lambda: wardex.span("x", kind="planner"), ValueError, "'client'"),
        (lambda: wardex.span("x", agent="researcher"), TypeError, "AgentAttributes"),
        (lambda: wardex.AgentAttributes(name="a", agent_type="planner"), ValueError, "sub_agent"),
        (lambda: wardex.ToolAttributes(name="t", execution_type="local"), ValueError, "in_process"),
        (lambda: wardex.ToolAttributes(name=7), TypeError, "str"),
        (
            lambda: wardex.GenAIAttributes(operation="chat", input_tokens="12"),
            TypeError,
            "int",
        ),
        (
            lambda: wardex.GenAIAttributes(operation="chat", finish_reasons="stop"),
            TypeError,
            "('stop',)",
        ),
    ],
    ids=[
        "span-name",
        "span-kind-type",
        "span-kind-value",
        "span-agent",
        "agent-type",
        "tool-execution-type",
        "tool-name",
        "gen-ai-int",
        "gen-ai-tuple",
    ],
)
def test_a_wrong_argument_raises_at_the_call_naming_what_is_accepted(call, error, names):
    with pytest.raises(error) as raised:
        call()
    assert names in str(raised.value)


@pytest.mark.parametrize(
    ("field", "value", "names"),
    [
        ("status", 5, "'error'"),
        ("kind", None, "'client'"),
        ("name", 123, "str"),
        ("status_message", b"x", "str"),
        ("input_data", 5, "set_input"),
        ("end_time_ns", "now", "time.time_ns()"),
        ("tool", "search", "ToolAttributes"),
    ],
)
def test_a_wrong_type_set_on_a_span_raises_on_that_line(wire, field, value, names):
    with wardex.span("s") as s:
        with pytest.raises((TypeError, ValueError)) as raised:
            setattr(s, field, value)
    assert names in str(raised.value)
    assert set(_shipped(wire)) == {"s"}


def test_a_span_the_exporter_cannot_encode_is_left_out_alone(wire, capsys):
    """The setters refuse wrong types, so this reaches past them the one way
    left: writing the span's backing field directly."""
    with wardex.span("other-request"):
        pass
    with wardex.span("broken") as s:
        s._draft._kind = "client"
    with wardex.span("after"):
        pass
    assert set(_shipped(wire)) == {"other-request", "after"}
    assert _losses() == [("unmarshallable", 1)]
    err = capsys.readouterr().err
    assert "left out of the export" in err
    assert "kind" in err
    assert "transport export raised" not in err


@pytest.mark.parametrize("field", ["start_time_ns", "end_time_ns"])
@pytest.mark.parametrize("value", [2**64, 1_700_000_000_000_000_000_000], ids=["2**64", "ns*1000"])
def test_a_timestamp_past_the_wire_range_raises_on_that_line(wire, field, value):
    """The wire's timestamp is an unsigned 64-bit count; a nanosecond count
    multiplied by 1000 by mistake is past it."""
    with wardex.span("other-request"):
        pass
    with wardex.span("s") as s:
        with pytest.raises(ValueError, match="time.time_ns()"):
            setattr(s, field, value)
    assert set(_shipped(wire)) == {"other-request", "s"}
    assert _losses() == []


def test_a_timestamp_written_past_the_setter_costs_its_span_alone(wire, capsys):
    with wardex.span("other-request"):
        pass
    with wardex.span("broken") as s:
        s._end_time_ns = 2**64
    assert set(_shipped(wire)) == {"other-request"}
    assert _losses() == [("unmarshallable", 1)]
    assert "transport export raised" not in capsys.readouterr().err


# -- decorators: the identity an adapter span carries ------------------------


def test_agent_name_is_exported_as_the_agent_identity(wire):
    @wardex.agent(name="researcher")
    def research(question):
        return question

    research("q")
    span = _shipped(wire)["invoke_agent researcher"]
    assert span["attributes"]["gen_ai.agent.name"] == "researcher"
    assert span["attributes"]["gen_ai.operation.name"] == "invoke_agent"


def test_tool_name_is_exported_as_the_tool_identity_run_in_process(wire):
    @wardex.tool(name="search")
    def search(query):
        return query

    search("q")
    attributes = _shipped(wire)["execute_tool search"]["attributes"]
    assert attributes["gen_ai.tool.name"] == "search"
    assert attributes["gen_ai.operation.name"] == "execute_tool"
    assert attributes["wardex.tool.execution_type"] == "in_process"


def test_workflow_and_step_carry_the_keys_their_adapter_spans_do(wire):
    @wardex.step(name="summarize")
    def summarize():
        return 1

    @wardex.workflow(name="pipeline")
    def pipeline():
        return summarize()

    pipeline()
    shipped = _shipped(wire)
    assert shipped["invoke_workflow pipeline"]["attributes"]["gen_ai.workflow.name"] == "pipeline"
    assert shipped["execute_step summarize"]["attributes"]["wardex.step.name"] == "summarize"


def test_the_attributes_name_names_the_span_when_no_name_is_given(wire):
    @wardex.tool(attributes=wardex.ToolAttributes(name="web_search", description="d"))
    def lookup():
        return 1

    lookup()
    attributes = _shipped(wire)["execute_tool web_search"]["attributes"]
    assert attributes["gen_ai.tool.name"] == "web_search"
    assert attributes["gen_ai.tool.description"] == "d"
    assert attributes["wardex.tool.execution_type"] == "in_process"


def test_a_name_that_disagrees_with_the_attributes_fails_at_decoration():
    with pytest.raises(ValueError, match="disagrees"):
        wardex.agent(name="researcher", attributes=wardex.AgentAttributes(name="other"))


def test_a_hand_built_tool_block_does_not_claim_a_network_call():
    assert wardex.ToolAttributes(name="t").execution_type is wardex.ToolExecutionType.UNKNOWN


def test_two_classes_methods_of_one_name_are_two_span_names(wire):
    class Researcher:
        @wardex.agent
        def run(self):
            return 1

    class Writer:
        @wardex.agent
        def run(self):
            return 2

    Researcher().run()
    Writer().run()
    shipped = _shipped(wire)
    assert {"invoke_agent Researcher.run", "invoke_agent Writer.run"} <= set(shipped)


def test_a_method_exports_its_own_name_as_the_identity(wire):
    """The span name is qualified so two classes' methods differ; the identity
    key is the name an adapter would report for the same tool, the function's."""

    class Toolbox:
        @wardex.tool
        def search(self, query):
            return query

    Toolbox().search("q")
    span = _shipped(wire)["execute_tool Toolbox.search"]
    assert span["attributes"]["gen_ai.tool.name"] == "search"


def test_a_decorated_agent_states_no_agent_type_it_did_not_observe(wire):
    @wardex.agent(name="helper")
    def helper():
        return 1

    @wardex.agent(name="orchestrator")
    def orchestrate():
        return helper()

    orchestrate()
    shipped = _shipped(wire)
    for name in ("invoke_agent helper", "invoke_agent orchestrator"):
        assert "wardex.agent.type" not in shipped[name]["attributes"], name
    assert shipped["invoke_agent helper"]["attributes"]["gen_ai.agent.name"] == "helper"


# -- attribute values --------------------------------------------------------


def test_lists_and_dicts_are_sent_as_their_json_text(wire):
    with wardex.span("s") as s:
        s.set_attribute("ids", [3, 1, 2])
        s.set_attribute("filters", {"b": 1, "a": ["x"]})
        s.set_attribute("price", Decimal("1.5"))
    attributes = _shipped(wire)["s"]["attributes"]
    assert attributes["ids"] == "[3,1,2]"
    assert attributes["filters"] == '{"a":["x"],"b":1}'
    assert attributes["price"] == 1.5
    assert "wardex.codec.unmarshalled" not in attributes


def test_none_removes_an_attribute(wire):
    with wardex.span("s") as s:
        s.set_attribute("draft", "x")
        s.set_attribute("draft", None)
        s.set_attribute("kept", None)
    attributes = _shipped(wire)["s"]["attributes"]
    assert "draft" not in attributes
    assert "kept" not in attributes
    assert "wardex.codec.unmarshalled" not in attributes


@pytest.mark.parametrize(
    ("key", "value", "names"),
    [
        (5, "v", "key must be a str"),
        ("k", object(), "list, tuple or dict"),
        ("k", b"raw", "list, tuple or dict"),
        ("k", [object()], "JSON-serialisable"),
    ],
    ids=["int-key", "object", "bytes", "unserialisable-list"],
)
def test_an_attribute_the_wire_cannot_hold_raises_at_the_call(wire, key, value, names):
    with wardex.span("s") as s:
        with pytest.raises(TypeError, match=names):
            s.set_attribute(key, value)


class _BrokenNumber:
    def __index__(self) -> int:
        raise RuntimeError("no value")


@pytest.mark.parametrize(
    ("value", "error", "names"),
    [
        (2**63, ValueError, "as text with str"),
        (-(2**63) - 1, ValueError, "as text with str"),
        (_BrokenNumber(), TypeError, "_BrokenNumber"),
    ],
    ids=["past-int64", "below-int64", "failing-index"],
)
def test_a_number_the_wire_cannot_hold_raises_at_the_call(wire, value, error, names):
    with wardex.span("s") as s:
        with pytest.raises(error, match=names):
            s.set_attribute("n", value)
        s.set_attribute("edge", 2**63 - 1)
    attributes = _shipped(wire)["s"]["attributes"]
    assert attributes["edge"] == 2**63 - 1
    assert "wardex.codec.unmarshalled" not in attributes


# -- set_tag shares the attribute rule -----------------------------------------


def _tag_writer(scope, write: str):
    return wardex.set_tag if write == "module" else scope.set_tag


@pytest.mark.parametrize("write", ["module", "scope"])
def test_a_tag_value_follows_the_attribute_rule(wire, write):
    with wardex.isolation_scope() as scope:
        set_tag = _tag_writer(scope, write)
        set_tag("count", 3)
        set_tag("ids", [3, 1, 2])
        set_tag("draft", "x")
        set_tag("draft", None)
        set_tag("never", None)
        with wardex.span("tagged"):
            pass
    attributes = _shipped(wire)["tagged"]["attributes"]
    assert attributes["count"] == 3
    assert attributes["ids"] == "[3,1,2]"
    assert "draft" not in attributes
    assert "never" not in attributes
    assert "wardex.codec.unmarshalled" not in attributes


@pytest.mark.parametrize("write", ["module", "scope"])
@pytest.mark.parametrize(
    ("key", "value", "error", "names"),
    [
        (5, "v", TypeError, "key must be a str"),
        ("n", 2**63, ValueError, "as text with str"),
        ("k", object(), TypeError, "list, tuple or dict"),
    ],
    ids=["int-key", "past-int64", "object"],
)
def test_a_tag_the_wire_cannot_hold_raises_at_the_call(wire, write, key, value, error, names):
    with wardex.isolation_scope() as scope:
        with pytest.raises(error, match=names):
            _tag_writer(scope, write)(key, value)
        with wardex.span("after"):
            pass
    attributes = _shipped(wire)["after"]["attributes"]
    assert "n" not in attributes and "k" not in attributes
    assert "wardex.codec.unmarshalled" not in attributes


# -- reaching the span from inside it ----------------------------------------


def test_current_span_is_none_outside_every_span():
    assert wardex.current_span() is None


def test_a_decorated_tool_records_its_input_and_output_as_tool_arguments(wire):
    @wardex.tool
    def search(query, limit=3):
        span = wardex.current_span()
        span.set_input({"query": query, "limit": limit})
        span.set_output(["a", "b"])
        span.add_event("cache_miss", {"key": query, "shards": [1, 2]})
        return "ok"

    search("cats")
    span = _shipped(wire)["execute_tool search"]
    assert json.loads(span["attributes"]["gen_ai.tool.call.arguments"]) == {
        "query": "cats",
        "limit": 3,
    }
    assert json.loads(span["attributes"]["gen_ai.tool.call.result"]) == ["a", "b"]
    (event,) = span["events"]
    assert event["name"] == "cache_miss"
    assert event["attributes"] == {"key": "cats", "shards": "[1,2]"}


def test_current_span_returns_the_innermost_open_span():
    with wardex.span("outer") as outer:
        with wardex.span("inner") as inner:
            assert wardex.current_span() is inner
        assert wardex.current_span() is outer
    assert wardex.current_span() is None


def test_current_span_follows_a_decorated_generator_and_an_async_function():
    seen = []

    @wardex.step
    def produce() -> Iterator[int]:
        seen.append(wardex.current_span())
        yield 1
        seen.append(wardex.current_span())

    @wardex.agent
    async def act() -> str:
        await asyncio.sleep(0)
        seen.append(wardex.current_span())
        return "done"

    assert list(produce()) == [1]
    assert wardex.current_span() is None
    assert asyncio.run(act()) == "done"
    assert [s.name for s in seen] == ["execute_step produce"] * 2 + ["invoke_agent act"]


def test_current_span_after_a_yield_is_the_block_left_open_across_it(wire):
    @wardex.step
    def stream() -> Iterator[int]:
        with wardex.span("fetch"):
            yield 1
            wardex.current_span().set_attribute("marker", "after-yield")
            yield 2
        wardex.current_span().set_attribute("closed", "fetch")

    assert list(stream()) == [1, 2]
    assert wardex.current_span() is None
    shipped = _shipped(wire)
    assert shipped["fetch"]["attributes"]["marker"] == "after-yield"
    assert "marker" not in shipped["execute_step stream"]["attributes"]
    assert shipped["execute_step stream"]["attributes"]["closed"] == "fetch"


def test_abandoned_streams_in_one_long_lived_task_do_not_pile_up_their_spans(wire):
    """An async generator holding a span across a yield, abandoned after its
    first item, is closed on another task: the span it leaves behind must not
    hold the one before it, or a worker task keeps every one it ever opened."""
    import gc
    import weakref

    async def stream():
        with wardex.span("chunk") as s:
            s.set_input("p" * 1000)
            yield 1
            yield 2

    async def worker() -> list:
        alive = []
        for _ in range(30):
            async for _ in stream():
                alive.append(weakref.ref(wardex.current_span()))
                break
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        gc.collect()
        assert wardex.current_span() is None
        return [ref for ref in alive if ref() is not None]

    assert len(asyncio.run(worker())) <= 2


def test_a_value_set_input_cannot_encode_raises_at_the_call(wire):
    with wardex.span("s") as s:
        with pytest.raises(TypeError, match="JSON-serialisable"):
            s.set_input({"when": object()})


def test_capture_io_records_the_arguments_and_the_return_value(wire):
    class Planner:
        @wardex.agent(capture_io=True)
        def plan(self, goal, *, steps=2, context=None):
            return {"goal": goal, "steps": steps}

    marker = object()
    Planner().plan("ship", steps=3, context=marker)
    attributes = _shipped(wire)["invoke_agent Planner.plan"]["attributes"]
    arguments = json.loads(attributes["wardex.input_data"])
    assert arguments == {"goal": "ship", "steps": 3, "context": repr(marker)}
    assert json.loads(attributes["wardex.output_data"]) == {"goal": "ship", "steps": 3}


def _client_with_attribute_cap(transport, cap: int) -> None:
    _hub.reset_for_test()
    config = WardexConfig(
        backend=BackendConfig(api_key="k"),
        limits=LimitsConfig(max_otlp_attribute_bytes=cap),
    )
    transport._set_limits(config.limits.to_native())  # what init() does
    _hub.set_client(Client(config, transport))


def test_capture_io_keeps_no_more_of_the_arguments_than_can_ship():
    """Encoding stops just past the export cap, so a huge argument costs the
    host thread and the buffer what one attribute can carry, not its size."""
    recording = RecordingTransport()
    _client_with_attribute_cap(recording, 1000)

    @wardex.tool(capture_io=True)
    def load(rows):
        return list(rows)

    load(list(range(200_000)))
    _hub.get_client().flush()
    (span,) = [span for envelope in recording.envelopes for span in envelope.spans]
    assert 1000 < len(span.input_data) <= 1004
    assert span.input_data.startswith(b'{"rows": [0, 1, 2')
    assert 1000 < len(span.output_data) <= 1004


def test_capture_io_cut_is_marked_as_truncated_on_the_wire():
    transport = _Wire()
    _client_with_attribute_cap(transport, 1000)

    @wardex.tool(capture_io=True)
    def load(rows):
        return len(rows)

    load(list(range(200_000)))
    attributes = _shipped(transport)["execute_tool load"]["attributes"]
    assert len(attributes["gen_ai.tool.call.arguments"]) <= 1000
    assert attributes["wardex.limitations"] == ["otlp_attribute_truncated"]


class _Unprintable:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def test_capture_io_records_what_json_cannot_hold_by_its_repr(wire):
    @wardex.tool(capture_io=True)
    def lookup(table, extra=None):
        return {(1, 2): "pair"}

    lookup({(1, 2): "tuple-key"}, extra=_Unprintable())
    attributes = _shipped(wire)["execute_tool lookup"]["attributes"]
    arguments = json.loads(attributes["gen_ai.tool.call.arguments"])
    assert arguments == {"table": "{(1, 2): 'tuple-key'}", "extra": "<unprintable _Unprintable>"}
    assert json.loads(attributes["gen_ai.tool.call.result"]) == "{(1, 2): 'pair'}"
    assert _losses() == []


def test_capture_io_is_off_by_default(wire):
    @wardex.tool
    def search(query):
        return "secret result"

    search("secret query")
    attributes = _shipped(wire)["execute_tool search"]["attributes"]
    assert "gen_ai.tool.call.arguments" not in attributes
    assert "gen_ai.tool.call.result" not in attributes


# -- async with, and bare @wardex.span ----------------------------------------


def test_async_with_opens_the_span_over_the_awaits_inside(wire):
    async def main():
        async with wardex.span("fetch") as s:
            await asyncio.sleep(0)
            with wardex.span("parse"):
                pass
            return s

    opened = asyncio.run(main())
    shipped = _shipped(wire)
    assert shipped["parse"]["parent_span_id"] == shipped["fetch"]["span_id"]
    assert opened.name == "fetch"


def test_bare_span_as_a_decorator_fails_where_it_is_written():
    with pytest.raises(TypeError, match="not a decorator") as raised:

        @wardex.span
        def work():
            return 1

    assert "@wardex.workflow" in str(raised.value)


# -- the decorators keep the wrapped signature ---------------------------------


def _plain(query: str, limit: int = 3) -> list[str]:
    return [query] * limit


async def _async(query: str) -> str:
    return query


def _gen(count: int) -> Iterator[int]:
    yield from range(count)


async def _agen(count: int) -> AsyncIterator[int]:
    for i in range(count):
        yield i


@pytest.mark.parametrize("decorator", ["workflow", "agent", "tool", "step"])
@pytest.mark.parametrize("form", ["bare", "called"])
@pytest.mark.parametrize("target", [_plain, _async, _gen, _agen], ids=lambda f: f.__name__)
def test_a_decorated_function_keeps_its_signature_and_annotations(decorator, form, target):
    deco = getattr(wardex, decorator)
    decorated = deco(target) if form == "bare" else deco(name="n")(target)
    assert inspect.signature(decorated) == inspect.signature(target)
    assert typing.get_type_hints(decorated) == typing.get_type_hints(target)


# -- the test harness reads status, attributes and events -----------------------


def test_span_node_shows_status_attributes_and_events():
    _hub.reset_for_test()
    recording = RecordingTransport()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), recording))
    wardex.set_user(wardex.UserInfo(id="u-1"))

    @wardex.tool
    def failing():
        wardex.current_span().add_event("retry", {"attempt": 2})
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        failing()
    _hub.get_client().flush()
    (node,) = recording.spans
    assert node.status is wardex.StatusCode.ERROR
    assert node.attributes["user.id"] == "u-1"
    assert node.attributes["gen_ai.tool.name"] == "failing"
    assert node.attributes["error.type"] == "RuntimeError"
    assert [event.name for event in node.events] == ["retry", "exception"]
    assert node.events[0].attributes == {"attempt": 2}


def test_each_decorator_declares_a_signature_preserving_overload_pair():
    """What a type checker reads: `@x` and `@x(...)` both typed `Callable[P, R]`
    to `Callable[P, R]`. Read from the source, so the claim holds on every
    interpreter (`typing.get_overloads` is 3.11+); when mypy is installed, it
    also checks a sample that calls decorated functions wrongly."""
    import ast
    import importlib.util
    import pathlib
    import subprocess
    import sys

    source = pathlib.Path(wardex.__file__).with_name("_decorators.py").read_text()
    overloads: dict[str, list[str]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Name) and d.id == "overload" for d in node.decorator_list
        ):
            overloads.setdefault(node.name, []).append(ast.unparse(node.returns))
    for name in ("workflow", "agent", "tool", "step"):
        assert overloads.get(name) == [
            "Callable[P, R]",
            "Callable[[Callable[P, R]], Callable[P, R]]",
        ], name

    if importlib.util.find_spec("mypy") is None:
        return  # the dev environment declares no type checker
    sample = pathlib.Path(__file__).with_name("typing_decorators_sample.py")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--warn-unused-ignores",
            "--follow-imports=silent",
            "--no-incremental",
            str(sample),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
