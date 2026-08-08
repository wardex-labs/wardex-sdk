"""Design §5.1 capture-policy matrix, unit-tested at the _should_capture hook."""

from types import SimpleNamespace

import pytest

from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import CaptureMode
from wardex_sdk._types import InternalEnvelope, SpanContext, SpanId, TraceId
from wardex_sdk.interceptors import _seam
from wardex_sdk.interceptors._seam import ByteSeamInterceptor
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


class _Seam(ByteSeamInterceptor):
    def _select_tracker(self, obj):
        raise NotImplementedError

    def _resolve_timing(self, obj, st):
        raise NotImplementedError

    def name(self):
        return "test-seam"

    def install(self, client, ctx=None):
        self._client = client

    def uninstall(self):
        pass


def _seam_with(mode: CaptureMode) -> _Seam:
    _hub.reset_for_test()
    client = Client(
        WardexConfig(capture_mode=mode, backend=BackendConfig(api_key="k")), _Recording()
    )
    _hub.set_client(client)
    s = _Seam()
    s._client = client
    return s


def _ctx(remote: bool) -> SpanContext:
    return SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), is_remote=remote)


LOCAL = _ctx(remote=False)
REMOTE = _ctx(remote=True)


@pytest.fixture(autouse=True)
def _llm_semantics_by_marker(monkeypatch):
    """sem=='LLM' stands for parsed core semantics; anything else is generic."""
    monkeypatch.setattr(_seam, "has_core_semantics", lambda sem: sem == "LLM")


@pytest.mark.parametrize(
    ("sem", "parent", "expected"),
    [
        ("LLM", LOCAL, True),
        ("LLM", REMOTE, True),
        ("LLM", None, True),
        (None, LOCAL, True),
        (None, REMOTE, False),
        (None, None, False),
        ("partial", None, False),  # sem parsed but no core semantics -> generic
    ],
)
def test_agent_mode_matrix(sem, parent, expected):
    seam = _seam_with(CaptureMode.AGENT)
    txn = SimpleNamespace(parent=parent)
    assert seam._should_capture(SimpleNamespace(), txn, sem) is expected


@pytest.mark.parametrize(
    ("sem", "parent"),
    [(None, None), (None, REMOTE), ("LLM", None)],
)
def test_all_mode_captures_everything(sem, parent):
    seam = _seam_with(CaptureMode.ALL)
    txn = SimpleNamespace(parent=parent)
    assert seam._should_capture(SimpleNamespace(), txn, sem) is True


def test_gate_fails_open_on_internal_error():
    seam = _seam_with(CaptureMode.AGENT)
    txn = SimpleNamespace()  # no .parent attribute at all

    class Boom:
        def __eq__(self, other):
            raise RuntimeError("boom")

    assert seam._should_capture(SimpleNamespace(), txn, Boom()) is True


def test_no_client_captures():
    s = _Seam()
    assert s._should_capture(SimpleNamespace(), SimpleNamespace(parent=None), None) is True
