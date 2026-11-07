"""The two OpenTelemetry surfaces pydantic-ai's spans reach wardex through.

Driven with a recording fake `Bridge` and called the way pydantic-ai 2.66.0
calls a tracer (measured: `start_as_current_span(name, attributes=...,
kind=..., record_exception=False, set_status_on_exception=False)` for runs,
chats and tools; `start_span(name, context=set_span_in_context(chat),
attributes=..., start_time=t0)` then `end(t1)` for a failed attempt). No
pydantic-ai import and no wardex registry: what is under test is that each
surface delivers the three calls faithfully, and never anything that quotes
content. Mode U runs on a real `opentelemetry-sdk` provider from the test
group, because the processor is only right if the SDK calls it the way this
file assumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import opentelemetry.metrics as otel_metrics
import opentelemetry.trace as otel_trace
import pytest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.trace import SpanKind, Status, StatusCode

from wardex_sdk._adapters import _pydantic_ai_otel as surface


class FakeBridge:
    """Records every call; `attrs` is snapshotted, and the live mapping kept
    beside it so a test can check it IS the span's own."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict[str, Any]]] = []

    def started(self, span: object, **kw: Any) -> None:
        kw["live"] = kw["attrs"]
        kw["attrs"] = dict(kw["attrs"])
        self.calls.append(("started", span, kw))

    def attributes_set(self, span: object, attrs: Mapping[str, object]) -> None:
        self.calls.append(("attributes_set", span, {"attrs": dict(attrs), "live": attrs}))

    def ended(self, span: object, **kw: Any) -> None:
        kw["attrs"] = dict(kw["attrs"])
        self.calls.append(("ended", span, kw))

    def of(self, method: str) -> list[tuple[object, dict[str, Any]]]:
        return [(span, kw) for m, span, kw in self.calls if m == method]


class Switch:
    """The adapter's `_installed` flag, as the surfaces read it."""

    def __init__(self) -> None:
        self.on = True

    def __call__(self) -> bool:
        return self.on


@pytest.fixture
def bridge() -> FakeBridge:
    return FakeBridge()


@pytest.fixture
def installed() -> Switch:
    return Switch()


@pytest.fixture
def tracer(bridge, installed):
    return surface.Provider(bridge, installed).get_tracer("pydantic-ai", "2.66.0")


def _pydantic_span(tracer, name, **attrs):
    """The exact call pydantic-ai makes for a run, chat or tool span."""
    return tracer.start_as_current_span(
        name,
        attributes=attrs,
        kind=SpanKind.CLIENT if name.startswith("chat") else SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
    )


# --------------------------------------------------------------------------
# Mode W — wardex's own opentelemetry-api implementation
# --------------------------------------------------------------------------


def test_the_provider_is_an_api_tracer_provider_handing_out_one_tracer(bridge, installed):
    provider = surface.Provider(bridge, installed)
    assert isinstance(provider, otel_trace.TracerProvider)
    first = provider.get_tracer("pydantic-ai", "2.66.0")
    assert isinstance(first, otel_trace.Tracer)
    assert provider.get_tracer("pydantic-ai") is first


def test_start_as_current_span_makes_the_span_current_for_the_body(tracer, bridge):
    before = otel_trace.get_current_span()
    with _pydantic_span(tracer, "invoke_agent a", **{"gen_ai.operation.name": "invoke_agent"}) as s:
        assert otel_trace.get_current_span() is s
        assert isinstance(s, otel_trace.Span)
    assert otel_trace.get_current_span() is before
    [(span, kw)] = bridge.of("started")
    assert span is s
    assert kw["name"] == "invoke_agent a"
    assert kw["kind"] is SpanKind.INTERNAL
    assert kw["parent"] is None
    assert kw["attrs_follow"] is True
    assert kw["attrs"] == {"gen_ai.operation.name": "invoke_agent"}
    [(ended, _)] = bridge.of("ended")
    assert ended is s  # end_on_exit=True by default


def test_a_child_span_is_delivered_with_its_parent_span_object(tracer, bridge):
    with _pydantic_span(tracer, "invoke_agent a") as run:
        with _pydantic_span(tracer, "chat m") as chat:
            pass
    starts = dict((kw["name"], (span, kw)) for span, kw in bridge.of("started"))
    assert starts["chat m"][1]["parent"] is run
    assert starts["chat m"][0] is chat
    assert starts["chat m"][1]["kind"] is SpanKind.CLIENT
    run_ctx, chat_ctx = run.get_span_context(), chat.get_span_context()
    assert run_ctx.is_valid and chat_ctx.is_valid
    assert chat_ctx.trace_id == run_ctx.trace_id
    assert chat_ctx.span_id != run_ctx.span_id
    assert [kw["name"] for _, kw in bridge.of("ended")] == ["chat m", "invoke_agent a"]


def test_attributes_are_readable_live_and_read_only(tracer, bridge):
    with _pydantic_span(tracer, "chat m", **{"gen_ai.operation.name": "chat"}) as span:
        attrs = span.attributes
        assert isinstance(attrs, Mapping)
        span.set_attributes({"gen_ai.request.model": "gpt-4o-mini", "gen_ai.provider.name": "x"})
        # pydantic's FallbackModel reads exactly this to decide on a refresh.
        assert span.attributes.get("gen_ai.request.model") == "gpt-4o-mini"
        assert attrs["gen_ai.provider.name"] == "x"
        with pytest.raises(TypeError):
            attrs["k"] = "v"  # type: ignore[index]
        span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
    sets = bridge.of("attributes_set")
    assert [s is span for s, _ in sets] == [True, True]
    assert sets[0][1]["attrs"]["gen_ai.request.model"] == "gpt-4o-mini"
    # The live mapping, the same one `started` was handed.
    [(_, started)] = bridge.of("started")
    assert sets[0][1]["live"] is started["live"]
    [(_, ended)] = bridge.of("ended")
    assert ended["attrs"]["gen_ai.response.finish_reasons"] == ["stop"]


def test_is_recording_is_true_while_installed_and_false_after(tracer, installed):
    with _pydantic_span(tracer, "invoke_agent a") as span:
        assert span.is_recording() is True
        installed.on = False
        # pydantic gates its content serialization on this: a span wardex
        # will never read costs nothing once the adapter is down.
        assert span.is_recording() is False
        installed.on = True
    assert span.is_recording() is False  # ended


def test_an_attempt_span_honours_context_and_both_timestamps(tracer, bridge):
    with _pydantic_span(tracer, "chat m") as chat:
        pass
    t0, t1 = 1_000_000_000, 1_250_000_000
    attempt = tracer.start_span(
        "model request attempt m",
        context=otel_trace.set_span_in_context(chat),
        attributes={"pydantic_ai.model_request.attempt": 0},
        start_time=t0,
    )
    assert otel_trace.get_current_span() is not attempt  # start_span does not make it current
    attempt.set_status(Status(StatusCode.ERROR, "ConnectError: upstream said secret"))
    attempt.end(t1)
    started = [kw for span, kw in bridge.of("started") if span is attempt]
    ended = [kw for span, kw in bridge.of("ended") if span is attempt]
    assert started[0]["parent"] is chat
    assert started[0]["start_ns"] == t0
    assert ended[0]["end_ns"] == t1
    assert attempt.get_span_context().trace_id == chat.get_span_context().trace_id


def test_a_root_span_without_timestamps_is_stamped_now(tracer, bridge):
    import time

    before = time.time_ns()
    span = tracer.start_span("invoke_agent a")
    span.end()
    after = time.time_ns()
    [(_, s)] = bridge.of("started")
    [(_, e)] = bridge.of("ended")
    assert before <= s["start_ns"] <= e["end_ns"] <= after


@pytest.mark.parametrize(
    "status",
    [StatusCode.ERROR, Status(StatusCode.ERROR, "RuntimeError: tool exploded secret=abc")],
    ids=["bare-code", "status-with-description"],
)
def test_an_error_status_reaches_the_bridge_without_its_description(tracer, bridge, status):
    with _pydantic_span(tracer, "invoke_agent a") as span:
        span.set_status(status)
    [(_, kw)] = bridge.of("ended")
    assert kw["error"] is True
    assert "secret=abc" not in repr(bridge.calls)


def test_an_unset_or_ok_status_is_not_an_error(tracer, bridge):
    with _pydantic_span(tracer, "invoke_agent a") as span:
        span.set_status(Status(StatusCode.OK))
    with _pydantic_span(tracer, "invoke_agent b"):
        pass
    assert [kw["error"] for _, kw in bridge.of("ended")] == [False, False]


def test_an_exception_reaches_the_bridge_as_its_bare_type_name_only(tracer, bridge):
    with _pydantic_span(tracer, "execute_tool t") as span:
        span.record_exception(RuntimeError("tool exploded secret=abc"), escaped=True)
        span.set_status(StatusCode.ERROR)
    with _pydantic_span(tracer, "execute_tool u") as span:
        # With content off pydantic records the exception as this event.
        span.add_event(
            "exception",
            {
                "exception.type": "pydantic_ai.exceptions.ToolRetryError",
                "exception.message": "secret=abc",
                "exception.escaped": "True",
            },
        )
        span.add_event("pydantic_ai.cache.not_enabled", {"model_name": "m"})
        span.set_status(StatusCode.ERROR)
    types = [kw["error_type"] for _, kw in bridge.of("ended")]
    assert types == ["RuntimeError", "ToolRetryError"]
    assert "secret=abc" not in repr(bridge.calls)


def test_update_name_calls_nothing_and_the_end_carries_the_current_name(tracer, bridge):
    with _pydantic_span(tracer, "chat fallback:a,b") as span:
        n = len(bridge.calls)
        span.update_name("chat gpt-4o-mini")
        assert len(bridge.calls) == n
    [(_, kw)] = bridge.of("ended")
    assert kw["name"] == "chat gpt-4o-mini"  # what an SDK span's `.name` says at end, too


def test_end_twice_delivers_one_end_and_nothing_after_it(tracer, bridge):
    span = tracer.start_span("invoke_agent a")
    span.end()
    span.end()
    span.set_attribute("late", 1)
    span.set_attributes({"later": 2})
    assert len(bridge.of("ended")) == 1
    assert bridge.of("attributes_set") == []


def test_a_span_built_while_uninstalled_is_inert(tracer, bridge, installed):
    installed.on = False
    with _pydantic_span(tracer, "invoke_agent late") as span:
        assert span.is_recording() is False
        span.set_attribute("k", "v")
    # Told once at start (so the bridge counts it), then nothing at all.
    assert [m for m, _, _ in bridge.calls] == ["started"]
    installed.on = True
    span.end()
    assert [m for m, _, _ in bridge.calls] == ["started"]


def test_a_live_span_ending_after_uninstall_still_reaches_the_bridge(tracer, bridge, installed):
    span = tracer.start_span("invoke_agent a")
    installed.on = False
    span.end()
    # Delivered, so the bridge can count it as late rather than lose it.
    assert len(bridge.of("ended")) == 1


def test_an_exception_in_the_body_ends_the_span_and_sets_no_status(tracer, bridge):
    with pytest.raises(ValueError):
        with _pydantic_span(tracer, "invoke_agent a"):
            raise ValueError("boom")
    [(_, kw)] = bridge.of("ended")
    # pydantic passes set_status_on_exception=False and records errors itself.
    assert kw["error"] is False
    assert otel_trace.get_current_span() is otel_trace.INVALID_SPAN


def test_end_on_exit_false_leaves_the_span_open(tracer, bridge):
    with tracer.start_as_current_span("invoke_agent a", end_on_exit=False) as span:
        pass
    assert bridge.of("ended") == []
    assert span.is_recording() is True
    span.end()
    assert len(bridge.of("ended")) == 1


def test_mode_w_never_touches_the_global_tracer_or_meter_provider(bridge, installed):
    tracer_provider = otel_trace.get_tracer_provider()
    meter_provider = otel_metrics.get_meter_provider()
    done = otel_trace._TRACER_PROVIDER_SET_ONCE._done
    tracer = surface.Provider(bridge, installed).get_tracer("pydantic-ai")
    with _pydantic_span(tracer, "invoke_agent a"):
        pass
    meter = surface.meter_provider()
    assert isinstance(meter, otel_metrics.NoOpMeterProvider)
    meter.get_meter("pydantic-ai", "2.66.0").create_histogram("gen_ai.client.token.usage").record(1)
    assert otel_trace.get_tracer_provider() is tracer_provider
    assert otel_metrics.get_meter_provider() is meter_provider
    assert otel_trace._TRACER_PROVIDER_SET_ONCE._done is done


# --------------------------------------------------------------------------
# Mode U — a duck-typed processor on the host's SDK provider
# --------------------------------------------------------------------------


@pytest.fixture
def sdk(bridge, installed):
    provider = SdkTracerProvider()
    processor = surface.Processor(bridge, installed)
    provider.add_span_processor(processor)
    yield SimpleNamespace(provider=provider, processor=processor)
    provider.shutdown()


def test_the_processor_has_every_method_the_sdk_calls(bridge, installed):
    processor = surface.Processor(bridge, installed)
    for method in ("on_start", "on_end", "_on_ending", "shutdown", "force_flush"):
        assert callable(getattr(processor, method))
    assert processor.shutdown() is None
    assert processor.force_flush() is True
    assert processor.force_flush(timeout_millis=5) is True


def test_on_start_delivers_the_live_span_its_parent_and_its_start_time(sdk, bridge):
    tracer = sdk.provider.get_tracer("pydantic-ai", "2.66.0")
    with _pydantic_span(
        tracer, "invoke_agent a", **{"gen_ai.operation.name": "invoke_agent"}
    ) as run:
        with _pydantic_span(tracer, "chat m") as chat:
            chat.set_attribute("gen_ai.request.model", "m")
    starts = {kw["name"]: (span, kw) for span, kw in bridge.of("started")}
    assert starts["invoke_agent a"][0] is run
    assert starts["invoke_agent a"][1]["parent"] is None
    assert starts["invoke_agent a"][1]["attrs"] == {"gen_ai.operation.name": "invoke_agent"}
    assert starts["chat m"][0] is chat
    assert starts["chat m"][1]["parent"] is run  # the object `on_start` was handed
    assert starts["chat m"][1]["start_ns"] == chat.start_time
    assert starts["chat m"][1]["attrs_follow"] is False
    # No attribute callback exists on an SDK processor.
    assert bridge.of("attributes_set") == []


def test_the_end_is_delivered_for_the_live_span_with_final_state(sdk, bridge):
    tracer = sdk.provider.get_tracer("pydantic-ai", "2.66.0")
    with _pydantic_span(tracer, "execute_tool t") as span:
        span.add_event("exception", {"exception.type": "builtins.ValueError"})
        span.add_event(
            "exception",
            {"exception.type": "pydantic_ai.exceptions.ToolRetryError", "exception.message": "s3"},
        )
        span.set_attribute("gen_ai.tool.call.result", "r")
        span.set_status(Status(StatusCode.ERROR, "ToolRetryError: secret=abc"))
    [(ended, kw)] = bridge.of("ended")
    [(started, _)] = bridge.of("started")
    assert ended is span is started  # identity: the bridge's state is keyed on it
    assert kw["end_ns"] == span.end_time
    assert kw["error"] is True
    assert kw["error_type"] == "ToolRetryError"  # the LAST exception event
    assert kw["attrs"]["gen_ai.tool.call.result"] == "r"
    assert kw["name"] == "execute_tool t"
    assert "secret=abc" not in repr(bridge.calls)
    assert "s3" not in repr(bridge.calls)  # the exception message is never read


def test_a_failed_attempt_reaches_the_processor_back_dated(sdk, bridge):
    tracer = sdk.provider.get_tracer("pydantic-ai", "2.66.0")
    with _pydantic_span(tracer, "chat m") as chat:
        pass
    attempt = tracer.start_span(
        "model request attempt m",
        context=otel_trace.set_span_in_context(chat),
        attributes={"pydantic_ai.model_request.attempt": 0},
        start_time=1_000,
    )
    attempt.end(2_000)
    s = [kw for span, kw in bridge.of("started") if span is attempt][0]
    e = [kw for span, kw in bridge.of("ended") if span is attempt][0]
    assert (s["parent"], s["start_ns"], e["end_ns"]) == (chat, 1_000, 2_000)


def test_spans_of_another_scope_are_ignored(sdk, bridge):
    for scope in ("fastmcp", "mcp-python-sdk", "my-app"):
        with sdk.provider.get_tracer(scope).start_as_current_span("tools/call add"):
            pass
    assert bridge.calls == []


def test_the_processor_delivers_no_start_while_uninstalled(sdk, bridge, installed):
    tracer = sdk.provider.get_tracer("pydantic-ai", "2.66.0")
    early = tracer.start_span("invoke_agent early")
    installed.on = False
    with tracer.start_as_current_span("invoke_agent late"):
        pass
    early.end()
    assert [kw["name"] for _, kw in bridge.of("started")] == ["invoke_agent early"]
    # A span whose start was delivered still has its end delivered, so the
    # bridge counts it as late instead of it vanishing.
    assert [kw["name"] for _, kw in bridge.of("ended")] == ["invoke_agent early"]


def test_an_sdk_without_on_ending_still_ends_the_live_span(bridge, installed):
    """`on_end` gets a fresh ReadableSpan copy; an SDK that never calls
    `_on_ending` must still have its end matched to the span `on_start` saw."""
    processor = surface.Processor(bridge, installed)
    provider = SdkTracerProvider()
    span = provider.get_tracer("pydantic-ai", "2.66.0").start_span("invoke_agent a")
    processor.on_start(span)
    span.end()  # the provider has no processor of ours; drive the old-SDK path by hand
    readable = span._readable_span()
    assert readable is not span
    processor.on_end(readable)
    processor.on_end(readable)  # a second end of the same span is nothing
    [(ended, kw)] = bridge.of("ended")
    assert ended is span
    assert kw["end_ns"] == span.end_time


def test_a_span_whose_start_was_never_seen_has_its_end_delivered_once(bridge, installed):
    """Attached while a pydantic span was open: its end reaches the bridge once (from
    `_on_ending`, never again from `on_end`), so the bridge can count it. Another scope's span,
    or one ending while uninstalled, is not delivered."""
    provider = SdkTracerProvider()
    early = provider.get_tracer("pydantic-ai", "2.66.0").start_span("invoke_agent early")
    mine = provider.get_tracer("my-app").start_span("mine")
    gone = provider.get_tracer("pydantic-ai", "2.66.0").start_span("invoke_agent gone")
    provider.add_span_processor(surface.Processor(bridge, installed))
    early.end()
    mine.end()
    installed.on = False
    gone.end()
    assert bridge.of("started") == []
    [(ended, kw)] = bridge.of("ended")
    assert ended is early and kw["name"] == "invoke_agent early"
    provider.shutdown()


def test_on_end_after_on_ending_delivers_one_end(sdk, bridge):
    with sdk.provider.get_tracer("pydantic-ai").start_as_current_span("invoke_agent a"):
        pass
    assert len(bridge.of("ended")) == 1


def test_a_processor_patched_into_the_tuple_route_receives_spans(bridge, installed):
    provider = SdkTracerProvider()
    settings = SimpleNamespace(tracer=provider.get_tracer("pydantic-ai", "2.66.0"))
    route, target = surface.attach_target(settings, otel_trace.get_tracer_provider())
    assert (route, target) == ("tuple", provider._active_span_processor)
    before = target._span_processors
    target._span_processors = before + (surface.Processor(bridge, installed),)
    with settings.tracer.start_as_current_span("invoke_agent a"):
        pass
    target._span_processors = before
    with settings.tracer.start_as_current_span("invoke_agent b"):
        pass
    assert [kw["name"] for _, kw in bridge.of("ended")] == ["invoke_agent a"]


class _LogfireShapedProxy:
    """Logfire's provider: `add_span_processor`, and no processor tuple."""

    def __init__(self) -> None:
        self.added: list[object] = []

    def add_span_processor(self, processor: object) -> None:
        self.added.append(processor)


def test_attach_target_over_every_shape_instrumentation_can_take():
    sdk_provider = SdkTracerProvider()
    proxy = _LogfireShapedProxy()
    cases = {
        "sdk settings": (SimpleNamespace(tracer=sdk_provider.get_tracer("pydantic-ai")), None),
        "logfire settings": (SimpleNamespace(tracer=SimpleNamespace(provider=proxy)), None),
        "true, api default": (True, otel_trace.ProxyTracerProvider()),
        "true, sdk global": (True, sdk_provider),
        "true, logfire global": (True, proxy),
        "true, nothing to attach to": (True, object()),
        "settings, proxy tracer": (SimpleNamespace(tracer=otel_trace.ProxyTracer("p", None)), None),
        "settings, no tracer": (SimpleNamespace(), None),
    }
    found = {name: surface.attach_target(*args) for name, args in cases.items()}
    assert found == {
        "sdk settings": ("tuple", sdk_provider._active_span_processor),
        "logfire settings": ("public", proxy),
        "true, api default": ("not_enabled", None),
        "true, sdk global": ("tuple", sdk_provider._active_span_processor),
        "true, logfire global": ("public", proxy),
        "true, nothing to attach to": ("unreachable", None),
        "settings, proxy tracer": ("unreachable", None),
        "settings, no tracer": ("unreachable", None),
    }
