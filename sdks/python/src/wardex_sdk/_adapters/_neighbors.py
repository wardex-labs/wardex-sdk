"""What else in this process changes what wardex records, found without importing it.

Two kinds, each said once at `init()`:

- another instrumentation of model calls. It records every call wardex
  records, as a span in a trace of its own, so each call shows up twice.
  Found by the package being installed: a top-level `find_spec` reads the
  import system's directory caches and a namespace directory listing names an
  instrumentation package, and neither runs any of its code. Installed is not
  enabled, so the line says what happens IF it instruments.
- gevent. Its monkey-patched `socket` and `ssl` modules are a runtime wardex
  does not support, and a patch applied after `init()` replaces the very
  classes wardex patched, so nothing made through them is seen. Imported
  after `init()`, it is said when that import finishes.

Read off `sys.modules` and the file system only; about 0.2 ms, once, at `init()`.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from typing import TYPE_CHECKING

from .._assembly import guard, report_once
from ._registry import get_registry

if TYPE_CHECKING:
    from .._config import WardexConfig

__all__ = ["say_what_shares_the_process"]

#: `opentelemetry.instrumentation.<name>` packages that instrument model calls:
#: the OpenTelemetry contrib OpenAI instrumentation (`openai_v2`) and the
#: OpenLLMetry provider and framework instrumentations. An HTTP client or
#: database instrumentation records no model call and is not listed.
_OTEL_MODEL_CALLS = frozenset(
    {
        "alephalpha",
        "anthropic",
        "bedrock",
        "cohere",
        "crewai",
        "google_generativeai",
        "groq",
        "haystack",
        "langchain",
        "llamaindex",
        "mistralai",
        "ollama",
        "openai",
        "openai_agents",
        "openai_v2",
        "replicate",
        "sagemaker",
        "together",
        "transformers",
        "vertexai",
        "watsonx",
    }
)


def _children(top: str, sub: str) -> list[str]:
    """The package names under `<top>/<sub>/` on every path the import system
    would search for `top`, read from the directory and never imported."""
    spec = importlib.util.find_spec(top)
    locations = getattr(spec, "submodule_search_locations", None) or ()
    found = set()
    for location in locations:
        folder = os.path.join(location, sub)
        if os.path.isdir(folder):
            found.update(n for n in os.listdir(folder) if n.isidentifier() and n[0] != "_")
    return sorted(found)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _duplicates() -> list[tuple[str, list[str]]]:
    """(family, distribution names) for each model-call instrumentation installed."""
    found = []
    openinference = [
        f"openinference-instrumentation-{n.replace('_', '-')}"
        for n in _children("openinference", "instrumentation")
    ]
    if openinference:
        found.append(("openinference", openinference))
    otel = [
        f"opentelemetry-instrumentation-{n.replace('_', '-')}"
        for n in _children("opentelemetry", "instrumentation")
        if n in _OTEL_MODEL_CALLS
    ]
    if otel:
        found.append(("opentelemetry", otel))
    for module, distribution in (("traceloop", "traceloop-sdk"), ("logfire", "logfire")):
        if _installed(module):
            found.append((module, [distribution]))
    return found


def _duplicate_line(names: list[str]) -> str:
    one = len(names) == 1
    said = ", ".join(names)
    they, instrument = ("it", "instruments") if one else ("they", "instrument")
    return (
        f"{said} {'is' if one else 'are'} installed. If {they} {instrument} "
        "model calls in this process, each call is recorded twice, by "
        f"{'it' if one else 'them'} and by wardex, as spans in separate traces. To keep one "
        "record, turn one side off: leave that instrumentation disabled, or stop wardex "
        "recording model calls with wardex.init(intercept=False, "
        "adapters=wardex.AdaptersConfig(enabled=())). To keep both in one trace instead, "
        "make the calls inside an active OpenTelemetry span that your code opens around "
        "them, and inside `with wardex.continue_from_otel():`"
    )


_GEVENT_FIX = (
    "Run wardex in a process that does not monkey-patch with gevent (for gunicorn, a sync or "
    "gthread worker)"
)


def _gevent_line() -> str | None:
    if "gevent" not in sys.modules:
        return None
    if getattr(socket.socket, "__module__", "").partition(".")[0] == "gevent":
        return (
            "gevent has monkey-patched the socket module. Capture under gevent is not "
            "supported: wardex is not tested with gevent's sockets, so model calls may go "
            f"unrecorded without further notice. {_GEVENT_FIX}"
        )
    return (
        "gevent is imported. Capture under gevent is not supported: if gevent monkey-patches "
        "the socket and ssl modules after wardex.init(), connections made through them bypass "
        f"wardex and their model calls are not recorded. {_GEVENT_FIX}"
    )


def _say_gevent() -> None:
    line = _gevent_line()
    if line is not None:
        report_once(line, key="environment.gevent")


def say_what_shares_the_process(config: WardexConfig) -> None:
    """Say once each instrumentation that would record wardex's model calls a
    second time, and gevent, when wardex records anything they would affect.
    gevent not imported yet is said when the application imports it, which is
    where a gunicorn gevent worker that loaded the application first patches."""
    records_calls = config.intercept or config.adapters.enabled != ()
    with guard("adapters.neighbors"):
        if records_calls:
            for family, names in _duplicates():
                report_once(_duplicate_line(names), key=f"environment.duplicate.{family}")
        if not config.intercept:
            return
        if "gevent" in sys.modules:
            _say_gevent()
        else:
            get_registry().install_on_import("gevent", _say_gevent)
