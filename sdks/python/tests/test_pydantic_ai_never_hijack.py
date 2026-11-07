"""wardex never takes pydantic-ai's instrumentation or OpenTelemetry's globals from the host.

Mode W (instrumentation off): wardex patches `Agent._instrument_default` to its own settings and
hands the class its own value back at `close()`. The global tracer and meter providers are never
set — `set_tracer_provider` is once-only per process, so taking it would silently break a later
`logfire.configure()` — and a host `instrument_all(x)` made after `init()` is the host's and is
left in place.

Mode U (instrumentation on): wardex adds a processor to the provider the host's settings resolve
to and takes it off again; the host's own exporter sees exactly what it saw without wardex.

The shapes that need the GLOBAL provider set (`instrument_all(True)` reads it) run in a child
process: setting it is once-only, and this process must keep the API's default for every other
test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from typing import Any

import opentelemetry.metrics as otel_metrics
import opentelemetry.trace as otel_trace
import pytest
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.trace import SynchronousMultiSpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

import wardex_sdk as wardex
from _pydantic_ai_support import host_provider, user_owned, wardex_owned
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, AdapterState
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_ONLY = AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,))


def _run(name: str = "probe") -> None:
    Agent(TestModel(), name=name).run_sync("hi")


def _names(transport: RecordingTransport) -> list[str]:
    wardex.flush()
    return sorted(s.name for e in transport.envelopes for s in e.spans)


@pytest.fixture
def transport() -> Iterator[RecordingTransport]:
    reset_reports_for_test()  # each test reads its own once-said lines
    recording = RecordingTransport()
    yield recording
    wardex.close()
    reset_reports_for_test()


# -- Mode W ------------------------------------------------------------------------


def test_wardex_owned_instrumentation_touches_no_global_and_is_handed_back(transport):
    with wardex_owned():
        tracer_provider = otel_trace.get_tracer_provider()
        meter_provider = otel_metrics.get_meter_provider()
        wardex.init(transport=transport, adapters=_ONLY)
        assert vars(Agent)["_instrument_default"] is not False, "wardex turned it on"
        _run()
        assert _names(transport) == ["chat test", "invoke_agent probe"]
        wardex.close()
        assert vars(Agent)["_instrument_default"] is False
        assert otel_trace.get_tracer_provider() is tracer_provider
        assert otel_metrics.get_meter_provider() is meter_provider
        assert otel_trace._TRACER_PROVIDER_SET_ONCE._done is False


def test_a_host_instrument_all_after_init_is_the_hosts_and_survives_close(transport, capsys):
    """The host's later settings replace wardex's, so the adapter receives nothing from then on;
    that is said once at close, and the host's settings are left in place."""
    with wardex_owned():
        wardex.init(transport=transport, adapters=_ONLY)
        provider, exporter = host_provider()
        mine = InstrumentationSettings(tracer_provider=provider, meter_provider=NoOpMeterProvider())
        Agent.instrument_all(mine)
        wardex.close()
        assert vars(Agent)["_instrument_default"] is mine
        assert counters.get("adapters.pydantic_ai.patch_superseded") == 1
        assert counters.get("adapters.pydantic_ai.instrumentation_replaced") == 1
        err = capsys.readouterr().err
        assert err.count("instrumentation was replaced after wardex.init()") == 1
        _run()
        assert sorted(s.name for s in exporter.get_finished_spans()) == [
            "chat test",
            "invoke_agent probe",
        ]


# -- Mode U ------------------------------------------------------------------------


def test_the_hosts_exporter_sees_the_same_spans_with_and_without_wardex(transport):
    with user_owned() as setup:
        before = setup.processors()
        _run()
        alone = sorted(s.name for s in setup.exporter.get_finished_spans())
        setup.exporter.clear()

        wardex.init(transport=transport, adapters=_ONLY)
        assert vars(Agent)["_instrument_default"] is setup.settings
        assert setup.processors() is not before
        _run()
        assert sorted(s.name for s in setup.exporter.get_finished_spans()) == alone
        assert _names(transport) == alone, "wardex recorded the same three spans as units"
        wardex.close()

        assert setup.processors() is before
        assert vars(Agent)["_instrument_default"] is setup.settings
        assert counters.get("adapters.pydantic_ai.instrumentation_replaced") == 0


def test_a_processor_the_host_added_after_init_leaves_wardexs_inert_and_says_so(transport, capsys):
    """The host's `add_span_processor` rebuilds the tuple wardex patched, so the patch is
    superseded and wardex's processor stays in it. Said once, like a provider with no remove;
    the host's processors, the late one included, keep receiving every span."""
    with user_owned() as setup:
        wardex.init(transport=transport, adapters=_ONLY)
        late = InMemorySpanExporter()
        setup.provider.add_span_processor(SimpleSpanProcessor(late))
        wardex.close()
        assert counters.get("adapters.pydantic_ai.processor_left_inert") == 1
        assert "stays registered but inert" in capsys.readouterr().err
        shipped = len(transport.envelopes)
        _run("after")
        assert len(transport.envelopes) == shipped
        assert sorted(s.name for s in late.get_finished_spans()) == [
            "chat test",
            "invoke_agent after",
        ]


class _ReadOnlyTuple(SynchronousMultiSpanProcessor):
    """A multi-processor whose `_span_processors` cannot be assigned from outside once frozen,
    while its own `add_span_processor` still works."""

    @property
    def _span_processors(self) -> Any:
        return self.__dict__.get("_inner", ())

    @_span_processors.setter
    def _span_processors(self, value: Any) -> None:
        if self.__dict__.get("_frozen"):
            raise AttributeError("read-only")
        self.__dict__["_inner"] = value

    def add_span_processor(self, span_processor: Any) -> None:
        with self._lock:
            self.__dict__["_inner"] = (*self._span_processors, span_processor)


def test_a_processor_tuple_wardex_may_not_replace_falls_back_to_the_public_method(
    transport, capsys
):
    multi = _ReadOnlyTuple()
    provider = TracerProvider(active_span_processor=multi)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    multi.__dict__["_frozen"] = True
    settings = InstrumentationSettings(tracer_provider=provider, meter_provider=NoOpMeterProvider())
    with wardex_owned():
        Agent.instrument_all(settings)
        wardex.init(transport=transport, adapters=_ONLY)
        (status,) = [s for s in wardex.diagnostics().adapters if s.name == "pydantic_ai"]
        assert status.state is AdapterState.INSTALLED
        _run()
        assert _names(transport) == ["chat test", "invoke_agent probe"]
        assert sorted(s.name for s in exporter.get_finished_spans()) == [
            "chat test",
            "invoke_agent probe",
        ]
        wardex.close()
        assert counters.get("adapters.pydantic_ai.patch_refused") == 1
        assert counters.get("adapters.pydantic_ai.processor_left_inert") == 1
        assert "Reinstall the framework" not in capsys.readouterr().err
        provider.shutdown()


class _ProxyTracer(otel_trace.Tracer):
    """Logfire's shape: a tracer with no `span_processor`, whose `.provider` is a proxy."""

    def __init__(self, inner: Any, provider: Any) -> None:
        self._inner = inner
        self.provider = provider

    def start_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.start_span(*args, **kwargs)

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.start_as_current_span(*args, **kwargs)


class _ProxyProvider(otel_trace.TracerProvider):
    """Logfire's `ProxyTracerProvider` shape: a public `add_span_processor` and no reachable
    processor tuple. Its inner SDK provider is its own business."""

    def __init__(self) -> None:
        self._inner, self.exporter = host_provider()
        self.added: list[Any] = []

    def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
        return _ProxyTracer(self._inner.get_tracer(*args, **kwargs), self)

    def add_span_processor(self, processor: Any) -> None:
        self.added.append(processor)
        self._inner.add_span_processor(processor)


def test_a_proxy_provider_gets_one_processor_and_it_is_inert_after_close(transport, capsys):
    proxy = _ProxyProvider()
    settings = InstrumentationSettings(tracer_provider=proxy, meter_provider=NoOpMeterProvider())
    with wardex_owned():
        Agent.instrument_all(settings)
        wardex.init(transport=transport, adapters=_ONLY)
        assert len(proxy.added) == 1
        _run()
        assert _names(transport) == ["chat test", "invoke_agent probe"]
        wardex.close()
        assert counters.get("adapters.pydantic_ai.processor_left_inert") == 1
        assert "stays registered but inert" in capsys.readouterr().err

        shipped = len(transport.envelopes)
        _run("after")
        assert len(transport.envelopes) == shipped, "the processor left behind records nothing"
        names = sorted(s.name for s in proxy.exporter.get_finished_spans())
        assert names.count("invoke_agent after") == 1, "the host's own pipeline is untouched"


# -- the global provider, in a child -------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import json, sys
    import opentelemetry.trace as t
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    import wardex_sdk as wardex
    from wardex_sdk._assembly import counters
    from wardex_sdk._config import AdaptersConfig
    from wardex_sdk._enums import AdapterName, AdapterState
    from wardex_sdk.testing import RecordingTransport

    scenario = sys.argv[1]
    exporter = InMemorySpanExporter()
    sdk = TracerProvider()
    sdk.add_span_processor(SimpleSpanProcessor(exporter))
    transport = RecordingTransport()
    out = {}

    class Bare(t.TracerProvider):
        def get_tracer(self, *args, **kwargs):
            return t.NoOpTracer()

    if scenario == "true_sdk_global":
        t.set_tracer_provider(sdk)
        before = sdk._active_span_processor._span_processors
    elif scenario == "true_bare_global":
        t.set_tracer_provider(Bare())
    if scenario != "later_global":
        Agent.instrument_all(True)
    wardex.init(transport=transport, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    out["state"] = {s.name: s.state.value for s in wardex.diagnostics().adapters}["pydantic_ai"]
    out["default_is_true"] = vars(Agent)["_instrument_default"] is True
    if scenario == "later_global":
        t.set_tracer_provider(sdk)
        out["took_effect"] = t.get_tracer_provider() is sdk
    Agent(TestModel(), name="probe").run_sync("hi")
    wardex.flush()
    out["wardex"] = sorted(s.name for e in transport.envelopes for s in e.spans)
    out["host"] = sorted(s.name for s in exporter.get_finished_spans())
    out["unreachable"] = counters.get("adapters.pydantic_ai.instrumentation_unreachable")
    status = {s.name: s for s in wardex.diagnostics().adapters}["pydantic_ai"]
    out["detail"], out["measured"] = status.detail, status.measured
    wardex.close()
    out["default_is_true_after"] = vars(Agent)["_instrument_default"] is True
    if scenario == "true_sdk_global":
        out["tuple_restored"] = sdk._active_span_processor._span_processors is before
    print(json.dumps(out))
    """
)


def _child(tmp_path: Any, scenario: str) -> tuple[dict[str, Any], str]:
    script = tmp_path / "global_child.py"
    script.write_text(_CHILD)
    proc = subprocess.run(
        [sys.executable, str(script), scenario], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def test_a_later_global_set_tracer_provider_still_takes_effect(tmp_path):
    out, _ = _child(tmp_path, "later_global")
    assert out["took_effect"] is True
    assert out["wardex"] == ["chat test", "invoke_agent probe"]


def test_instrument_all_true_with_the_api_default_provider_is_wardex_owned(tmp_path):
    """`True` with nothing behind the global provider sends pydantic's spans nowhere: wardex
    takes it over (Mode W) and hands `True` back at `close()`."""
    out, _ = _child(tmp_path, "true_api_default")
    assert out["state"] == "installed"
    assert out["default_is_true"] is False and out["default_is_true_after"] is True
    assert out["wardex"] == ["chat test", "invoke_agent probe"]


def test_instrument_all_true_with_an_sdk_global_provider_is_user_owned(tmp_path):
    out, _ = _child(tmp_path, "true_sdk_global")
    assert out["state"] == "installed"
    assert out["default_is_true"] is True
    assert out["wardex"] == out["host"] == ["chat test", "invoke_agent probe"]
    assert out["tuple_restored"] is True


def test_instrument_all_true_with_a_provider_wardex_cannot_attach_to_is_unsupported(tmp_path):
    out, err = _child(tmp_path, "true_bare_global")
    assert out["state"] == "unsupported"
    assert out["unreachable"] == 1
    assert out["wardex"] == []
    assert err.count("accepts no span processor wardex can attach to") == 1
    # One line, and the status carries the adapter's own reason, not "reinstall the framework".
    assert len([line for line in err.splitlines() if "pydantic_ai" in line]) == 1
    assert "accepts no span processor wardex can attach to" in out["detail"]
    assert "Reinstall" not in out["detail"]
