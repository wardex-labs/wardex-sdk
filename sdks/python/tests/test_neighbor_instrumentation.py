"""What wardex says at `init()` about other instrumentation in the same process.

Another OpenTelemetry instrumentation of model calls (OpenInference, OpenLLMetry,
the OpenTelemetry contrib OpenAI instrumentation, Logfire) records every call a
second time, in a trace of its own, and gevent's monkey-patched sockets are a
runtime wardex does not support. Neither used to be mentioned. Each is now said
once, with what will happen and what to do, found from what is installed or
already imported and without importing any of it.
"""

from __future__ import annotations

import importlib
import logging
import socket
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

import wardex_sdk as wardex
from wardex_sdk._assembly._diag import reset_reports_for_test


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def lines() -> Iterator[_Lines]:
    reset_reports_for_test()
    handler = _Lines()
    logger = logging.getLogger("wardex_sdk")
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        wardex.close()


def _package(root: Path, dotted: str, body: str = "") -> None:
    path = root.joinpath(*dotted.split("."))
    path.mkdir(parents=True)
    (path / "__init__.py").write_text(body)


_EXPLODES = "raise RuntimeError('wardex imported a package it must only look for')\n"


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield tmp_path
    importlib.invalidate_caches()


def _init(**kwargs) -> None:  # noqa: ANN003
    wardex.init(
        transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()), **kwargs
    )


def _said(lines: _Lines, needle: str) -> list[str]:
    return [m for m in lines.messages if needle in m]


def test_an_installed_openinference_instrumentation_is_said_once_without_importing_it(site, lines):
    (site / "openinference" / "instrumentation").mkdir(parents=True)
    (site / "openinference" / "instrumentation" / "__init__.py").write_text(_EXPLODES)
    _package(site, "openinference.instrumentation.openai", _EXPLODES)
    _package(site, "openinference.instrumentation.langchain", _EXPLODES)

    _init()
    wardex.close()
    _init()

    said = _said(lines, "openinference-instrumentation-")
    assert len(said) == 1, lines.messages
    line = said[0]
    assert "openinference-instrumentation-langchain" in line
    assert "openinference-instrumentation-openai" in line
    assert "recorded twice" in line
    assert "intercept=False" in line
    # The one trace needs a span to join: an instrumentation that opens its
    # own root span per model call is not joined by the context manager alone.
    assert "inside an active OpenTelemetry span" in line
    assert "continue_from_otel" in line
    assert not [m for m in sys.modules if m.startswith("openinference")]


def test_an_installed_opentelemetry_llm_instrumentation_is_said(site, lines):
    _package(site, "opentelemetry.instrumentation.openai_v2", _EXPLODES)
    _package(site, "opentelemetry.instrumentation.anthropic", _EXPLODES)
    _init()
    (line,) = _said(lines, "opentelemetry-instrumentation-")
    assert "opentelemetry-instrumentation-anthropic" in line
    assert "opentelemetry-instrumentation-openai-v2" in line
    assert "opentelemetry.instrumentation.openai_v2" not in sys.modules


def test_an_opentelemetry_instrumentation_of_something_else_is_not_a_duplicate(site, lines):
    _package(site, "opentelemetry.instrumentation.redis", _EXPLODES)
    _init()
    assert _said(lines, "opentelemetry-instrumentation-") == []


@pytest.mark.parametrize(
    ("module", "distribution"), [("logfire", "logfire"), ("traceloop", "traceloop-sdk")]
)
def test_logfire_and_traceloop_are_said_without_importing_them(site, lines, module, distribution):
    _package(site, module, _EXPLODES)
    _init()
    (line,) = _said(lines, f"{distribution} is installed")
    assert "recorded twice" in line
    assert module not in sys.modules


def test_nothing_is_said_when_wardex_records_no_model_calls(site, lines):
    _package(site, "logfire", _EXPLODES)
    _init(intercept=False)
    assert _said(lines, "logfire is installed") == []


def test_gevent_imported_before_init_is_said(lines, monkeypatch):
    monkeypatch.setitem(sys.modules, "gevent", types.ModuleType("gevent"))
    _init()
    (line,) = _said(lines, "gevent")
    assert "not supported" in line
    assert "after wardex.init()" in line


def test_gevent_patched_sockets_are_said(lines, monkeypatch):
    class GeventSocket(socket.socket):
        __module__ = "gevent._socket3"

    monkeypatch.setitem(sys.modules, "gevent", types.ModuleType("gevent"))
    monkeypatch.setattr(socket, "socket", GeventSocket)
    _init()
    (line,) = _said(lines, "gevent")
    assert "has monkey-patched" in line
    assert "not supported" in line


def test_gevent_imported_after_init_is_said_when_it_is_imported(site, lines, monkeypatch):
    # The gunicorn shape: the app (and init()) loads first, the worker then
    # imports gevent and patches. Said on that import, once, not before.
    _package(site, "gevent", "")
    _init()
    assert _said(lines, "gevent") == []
    try:
        importlib.import_module("gevent")
        (line,) = _said(lines, "gevent")
        assert "not supported" in line
        importlib.import_module("gevent")
        assert len(_said(lines, "gevent")) == 1
    finally:
        sys.modules.pop("gevent", None)


def test_gevent_imported_after_close_is_not_said(site, lines):
    _package(site, "gevent", "")
    meta_path_before = list(sys.meta_path)
    _init()
    wardex.close()
    assert sys.meta_path == meta_path_before, "wardex left an import hook behind"
    try:
        importlib.import_module("gevent")
        assert _said(lines, "gevent") == []
    finally:
        sys.modules.pop("gevent", None)


def test_a_process_without_any_of_them_hears_nothing(lines):
    _init()
    assert not [m for m in lines.messages if "installed" in m or "gevent" in m], lines.messages
