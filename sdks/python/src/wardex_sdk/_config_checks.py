"""What configuration wardex refuses, or questions, before it builds anything.

Three mistakes used to surface only as "no spans arrive" -- a symptom
indistinguishable from a healthy backend that simply received nothing:

  * a misspelled keyword on a config group. The generated dataclass `__init__`
    names the right spelling only on CPython 3.12+, and the declared floor is
    3.10, so on the older interpreters the refusal said what was wrong and not
    what was meant;
  * a misspelled `WARDEX_*` environment variable. Nothing enumerated them, so
    `WARDEX_ENDPONT` was read by nobody and mentioned by nobody;
  * an endpoint with no scheme. `collector:4318` parses with `collector` as its
    scheme, so it passed `init()` and then every POST to it failed.

Pure functions and one class decorator, importing nothing from the package:
`_limits.py` and `_config.py` both build on it, so it sits below both. It only
ANSWERS. The warnings its answers lead to are raised by `init()`, the one place
that knows a configuration is being installed rather than merely constructed.
It reads the environment only when called, never at import: an explicit
argument to `init()` must be able to outrank whatever the shell exported.
"""

from __future__ import annotations

import difflib
import functools
import os
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any, TypeVar
from urllib.parse import urlsplit

_C = TypeVar("_C", bound=type)

#: Every environment variable wardex reads under its own prefix: the env
#: contract `init()` documents. `tests/test_config.py` holds this tuple equal
#: to the `WARDEX_*` names the SDK's source actually reads, so a variable cannot
#: be read without being listed here, and the typo check below cannot call a
#: real variable a typo.
WARDEX_ENV_NAMES = (
    "WARDEX_API_KEY",
    "WARDEX_BASE_URL",
    "WARDEX_ENDPOINT",
    "WARDEX_SERVICE_NAME",
    "WARDEX_RELEASE",
    "WARDEX_ENVIRONMENT",
    "WARDEX_DEBUG",
)

#: `WARDEX_*` names the SDK never reads that are nonetheless not mistakes: this
#: repository's own end-to-end drivers and fixture recorders set them around an
#: `init()` call, and a warning there would be noise in every such run.
_RESERVED_ENV_PREFIXES = ("WARDEX_E2E_",)
_RESERVED_ENV_NAMES = frozenset(
    {"WARDEX_RECORD", "WARDEX_REGEN_ENVELOPES", "WARDEX_SMOKE_SCENARIO"}
)

#: The OpenTelemetry spellings `backend.endpoint` falls back to after
#: `WARDEX_ENDPOINT`, specific before generic.
_OTEL_ENDPOINT_NAMES = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")


def closest(name: str, candidates: Iterable[str]) -> str | None:
    """The candidate `name` was most likely meant to be, or None."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def refuses_unknown_keywords(cls: _C) -> _C:
    """Make a config dataclass refuse an unknown keyword by naming the right one.

    Applied ABOVE `@dataclass`, because it wraps the `__init__` the dataclass
    generated. `functools.wraps` keeps that signature visible to
    `inspect.signature`, `help()` and editors on every interpreter -- which is
    why this is not a shared `__new__`: 3.10's `inspect` reports a class's
    `__new__` as its signature, and every group would read `(*args, **kwargs)`.

    The message is the same everywhere: on 3.12+ it replaces CPython's own
    suggestion, and on 3.10 and 3.11 it is the only one there is.
    """
    init = cls.__init__
    accepted = tuple(f.name for f in fields(cls) if f.init)

    @functools.wraps(init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
        unknown = [name for name in kwargs if name not in accepted]
        if unknown:
            raise TypeError(_unknown_keywords(cls.__name__, unknown, accepted))
        init(self, *args, **kwargs)

    cls.__init__ = __init__  # type: ignore[misc]
    return cls


def _unknown_keywords(owner: str, unknown: list[str], accepted: tuple[str, ...]) -> str:
    named = []
    for name in unknown:
        hint = closest(name, accepted)
        named.append(f"{name!r}" + (f" (did you mean {hint!r}?)" if hint else ""))
    noun = "an unexpected keyword argument" if len(named) == 1 else "unexpected keyword arguments"
    return f"{owner} got {noun} {', '.join(named)}; its fields are: {', '.join(accepted)}"


def is_http_url(value: object) -> bool:
    """Whether `value` is an absolute `http://` or `https://` URL with a host:
    something a POST can actually be sent to."""
    try:
        parts = urlsplit(value)  # type: ignore[arg-type]
        scheme = parts.scheme
        return (
            isinstance(scheme, str) and scheme.lower() in ("http", "https") and bool(parts.hostname)
        )
    except (TypeError, ValueError, AttributeError):
        return False


def require_http_url(endpoint: str | None) -> None:
    """Refuse a `backend.endpoint` no export could reach.

    The value is not quoted back: an endpoint can carry a credential in its
    query or its userinfo, and this message ends up in tracebacks and logs.
    An empty string is left alone, as unset: `init()` already reads it so.
    """
    if endpoint and not is_http_url(endpoint):
        raise ValueError(
            "backend.endpoint must be an absolute http:// or https:// URL with a host, such as "
            "'http://collector:4318'; the configured value has no such scheme or no host, so "
            "every export to it would fail. Write the scheme and the host (when the argument "
            "is unset, init() reads this value from WARDEX_ENDPOINT)."
        )


def endpoint_from_env() -> str | None:
    """The exporter address the environment names, in precedence order.

    `WARDEX_ENDPOINT` first, returned as read: it is wardex's own variable, so a
    value no export could reach is a mistake made about wardex, and
    `BackendConfig` refuses it with a `ValueError` at configuration time.

    Then the OTel spellings, specific before generic, so a host already
    exporting OTLP elsewhere points wardex at the same collector with zero new
    variables -- but only a value that IS an http(s) URL. Those variables are
    shared by every OTel SDK in the process, and a scheme-less `collector:4317`
    is a legal value for a gRPC exporter; refusing it would let a setting meant
    for some other exporter stop `init()` in a process that had told wardex
    nothing. The first OTel variable that is SET decides: a specific one wardex
    cannot use does not fall through to the generic one it overrides.
    `unusable_otel_endpoint()` names it for the one case where it would have
    been wardex's destination.

    The value is stored as read -- the `/v1/traces` default path is the
    transport builder's to append (see `BackendConfig.endpoint`).
    """
    endpoint = os.environ.get("WARDEX_ENDPOINT")
    if endpoint:
        return endpoint
    otel = _first_otel_endpoint()
    return otel[1] if otel is not None and is_http_url(otel[1]) else None


def unusable_otel_endpoint() -> str | None:
    """The OTel variable `endpoint_from_env` passed over as not an http(s) URL."""
    otel = _first_otel_endpoint()
    return otel[0] if otel is not None and not is_http_url(otel[1]) else None


def _first_otel_endpoint() -> tuple[str, str] | None:
    for name in _OTEL_ENDPOINT_NAMES:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def endpoint_named_for_wardex(argument: str | None) -> str | None:
    """The endpoint the host gave WARDEX: the argument, else `WARDEX_ENDPOINT`.

    Not the OTel spellings. Every OTel SDK in the process reads those, so an
    endpoint inherited from them is not a setting anyone gave wardex, and it
    losing to an explicit `transport=` or project key is the expected case, not
    the conflict `init()` announces.
    """
    return argument or os.environ.get("WARDEX_ENDPOINT") or None


def env_typo_messages(environ: Mapping[str, str] | None = None) -> list[str]:
    """One message per set variable that looks like wardex's and is not one it reads.

    The prefix is matched case-insensitively, because `wardex_endpoint` is the
    same mistake as `WARDEX_ENDPONT`: variable names are case-sensitive, and
    neither is read.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    messages = []
    for name in sorted(env):
        upper = name.upper()
        if not upper.startswith("WARDEX_") or name in WARDEX_ENV_NAMES:
            continue
        if upper in _RESERVED_ENV_NAMES or upper.startswith(_RESERVED_ENV_PREFIXES):
            continue
        hint = closest(upper, WARDEX_ENV_NAMES)
        messages.append(
            f"{name} is set, but wardex reads no environment variable by that name, so its "
            "value is ignored. "
            + (f"Did you mean {hint}? " if hint else "")
            + f"(wardex reads: {', '.join(WARDEX_ENV_NAMES)})"
        )
    return messages
