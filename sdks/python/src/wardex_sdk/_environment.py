"""What a bug report needs about this process, beside the adapters and losses.

`environment()` reads the facts `wardex.diagnostics()` adds: versions, the
native module, the resolved configuration with every secret masked, the
exporter and where it sends, the seams and patches wardex installed, and its
threads. Every read is defensive, because the reader is a person filing a
report about a process that may already be in a bad state: a fact that cannot
be read is left at its empty default, never raised.
"""

from __future__ import annotations

import dataclasses
import enum
import platform
import sys
import threading
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import _runtime
from ._assembly import PatchSet, capture_mode_of, guard
from ._native import NATIVE_OK, unavailable_reason
from ._version import __version__

#: A setting whose name holds one of these words is shown as set or unset only.
_SECRET_WORDS = ("key", "token", "secret", "password", "auth", "credential", "cookie")

#: The settings that hold a URL, shown without credentials, query or fragment.
_URL_SETTINGS = frozenset({"endpoint", "base_url"})


def destination(url: object) -> str:
    """`url` as it may be printed: scheme, host, port and path only. Userinfo,
    query and fragment can carry a credential, so they never appear."""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port is not None else ""
        return urlunsplit((parts.scheme, host + port, parts.path, "", ""))
    except ValueError:
        return "<an unparseable URL>"


def _shown(name: str, value: Any) -> str:
    """One setting's value, as a report may show it."""
    lowered = name.lower()
    if value is None:
        return "unset"
    if any(word in lowered for word in _SECRET_WORDS):
        return "set (masked)"
    if lowered == "headers" and isinstance(value, Mapping):
        return "{" + ", ".join(f"{k}: (masked)" for k in sorted(value)) + "}"
    if lowered in _URL_SETTINGS and isinstance(value, str):
        return destination(value)
    if isinstance(value, enum.Enum):
        return str(value.value)
    if callable(value):
        return "set (a callable)"
    if isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, Iterable):
        items = [_shown(name, v) for v in value]
        return "[" + ", ".join(sorted(items)) + "]"
    return type(value).__name__


def _flatten(config: object, prefix: str = "") -> list[tuple[str, str]]:
    """Every leaf setting of a config dataclass, dotted, with its shown value."""
    out: list[tuple[str, str]] = []
    for field in dataclasses.fields(config):  # type: ignore[arg-type]
        value = getattr(config, field.name, None)
        name = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out += _flatten(value, f"{name}.")
        else:
            out.append((name, _shown(field.name, value)))
    return out


def _symbol(target: object, name: str) -> str:
    if isinstance(target, type):
        return f"{target.__module__}.{target.__qualname__}.{name}"
    owner = getattr(target, "__name__", None)
    return f"{owner}.{name}" if isinstance(owner, str) else f"{type(target).__qualname__}.{name}"


def _patch_sets(holder: object) -> list[PatchSet]:
    """The patch sets `holder` keeps, as attributes or as its `patches`."""
    found = [v for v in getattr(holder, "__dict__", {}).values() if isinstance(v, PatchSet)]
    for name in ("patches", "_patches"):
        value = getattr(holder, name, None)
        if isinstance(value, PatchSet) and value not in found:
            found.append(value)
    return found


def _patched(holders: Iterable[object]) -> tuple[str, ...]:
    """Each function the patch sets reachable from `holders` replaced."""
    found: set[str] = set()
    for holder in holders:
        for patch_set in _patch_sets(holder):
            for record in list(getattr(patch_set, "_patches", ())):
                target = record.target()
                if target is not None:
                    found.add(_symbol(target, record.name))
    return tuple(sorted(found))


def environment() -> dict[str, Any]:
    """The bug-report fields of `Diagnostics`, as keyword arguments."""
    facts: dict[str, Any] = {
        "wardex_version": __version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "native": "loaded" if NATIVE_OK else f"unavailable: {unavailable_reason()}",
    }
    with guard("diagnostics.threads"):
        facts["worker_threads"] = tuple(
            sorted(t.name for t in threading.enumerate() if t.name.startswith("wardex"))
        )
    runtime = _runtime.runtime()
    client = runtime.client
    if client is None:
        return facts
    with guard("diagnostics.config"):
        facts["config"] = tuple(_flatten(client.config))
        facts["capture_policy"] = str(capture_mode_of(client).value)
    with guard("diagnostics.transport"):
        facts.update(transport_facts(getattr(client, "_transport", None)))
    with guard("diagnostics.seams"):
        interceptors = getattr(runtime, "_interceptors", None)
        installed = dict(getattr(interceptors, "_installed", {}) or {})
        adapters = getattr(runtime, "_adapters", None)
        holders = list(installed.values())
        holders += list((getattr(adapters, "_installed", {}) or {}).values())
        holders += list((getattr(adapters, "_contexts", {}) or {}).values())
        inject = sys.modules.get("wardex_sdk.context._inject")
        holders += [inject] if inject is not None else []
        facts["interceptors"] = tuple(sorted(installed))
        facts["patched"] = _patched(holders)
    return facts


def transport_facts(transport: object) -> dict[str, Any]:
    """The exporter's class, destination and delivery state. The delivery state
    is read only where the transport offers it (`export_state`); its fields are
    read by name, each optional, so a transport that tracks less still reports
    what it has."""
    if transport is None:
        return {}
    facts: dict[str, Any] = {"transport": type(transport).__name__}
    state = getattr(transport, "export_state", None)
    url = getattr(state, "destination", None) or next(
        (
            v
            for v in (getattr(transport, n, None) for n in ("_endpoint", "_url", "endpoint", "url"))
            if isinstance(v, str)
        ),
        None,
    )
    if url:
        facts["destination"] = destination(url)
    if state is not None:
        success = getattr(state, "last_success_at", None)
        error = getattr(state, "last_error", None)
        queued = getattr(state, "retry_queued", None)
        facts["export_tracked"] = True
        facts["last_export_success"] = float(success) if success is not None else None
        facts["last_export_error"] = str(error) if error is not None else None
        facts["retry_queued"] = queued if isinstance(queued, int) else None
    return facts
