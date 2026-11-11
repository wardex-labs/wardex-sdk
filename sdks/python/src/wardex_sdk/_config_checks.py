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
ANSWERS. What its answers lead to is said by `init()`, the one place that knows
a configuration is being installed rather than merely constructed.
It reads the environment only when called, never at import: an explicit
argument to `init()` must be able to outrank whatever the shell exported.

THE ENVIRONMENT RULE: a value read from an environment variable never raises.
A deployment's environment is not code its owner can fix by editing the call,
and a variable another tool set must not be able to stop the host from
starting. Helpers that read one append a sentence to the `problems` list they
are handed -- the variable, what is wrong with it, the fix -- and `init()` says
each on stderr and keeps it for `diagnostics()`.
"""

from __future__ import annotations

import difflib
import functools
import inspect
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields
from typing import Any, TypeVar
from urllib.parse import unquote, urlsplit

_C = TypeVar("_C", bound=type)
_F = TypeVar("_F", bound=Callable[..., Any])

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
    "WARDEX_PII",
    "WARDEX_DISABLED",
)

#: The spellings every boolean variable wardex reads accepts, case ignored.
#: Anything else is said once on stderr and the default applies.
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

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


#: The spellings `WARDEX_PII` accepts, compared stripped and lowercased.
_PII_ON = frozenset({"on", "1", "true", "yes"})
_PII_OFF = frozenset({"off", "0", "false", "no"})


def pii_off_from_env() -> bool | None:
    """`WARDEX_PII` read as a switch: True for `off` (`init()` says so once on stderr), False for
    `on`, None when it is unset, empty or neither. None keeps the default, masking on: a typo
    must not export unmasked text."""
    raw = os.environ.get("WARDEX_PII", "").strip().lower()
    if raw in _PII_OFF:
        return True
    if raw in _PII_ON:
        return False
    return None


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


def refuses_unknown_init_keywords(
    *,
    moved: Mapping[str, str],
    removed: Mapping[str, str],
    groups: tuple[tuple[str, type], ...],
) -> Callable[[_F], _F]:
    """Make `init()` refuse a keyword it does not take by naming where it lives.

    A moved or removed name says where it went (`moved`) or why it is gone
    (`removed`); a field of a config group names the group, because
    `init(endpoint=...)` is the call a reader of another SDK writes first;
    anything else gets the closest keyword `init()` does take. A wrapper rather
    than `**kwargs` on `init()`: `functools.wraps` keeps the exact signature for
    `inspect` and editors, and the refusal runs before anything is resolved.
    """

    def decorate(func: _F) -> _F:
        accepted = tuple(inspect.signature(func).parameters)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            unknown = [name for name in kwargs if name not in accepted]
            if not unknown:
                return func(*args, **kwargs)
            named = []
            for name in unknown:
                homes = [
                    f"{arg}={cls.__name__}({name}=...)"
                    for arg, cls in groups
                    if name in {f.name for f in fields(cls)}
                ]
                if name in moved or homes:
                    named.append(f"{name!r} -> write init({moved.get(name) or ' or '.join(homes)})")
                elif name in removed:
                    named.append(f"{name!r}: {removed[name]}")
                else:
                    hint = closest(name, accepted)
                    named.append(f"{name!r}" + (f" (did you mean {hint!r}?)" if hint else ""))
            noun = (
                "an unexpected keyword argument"
                if len(named) == 1
                else "unexpected keyword arguments"
            )
            raise TypeError(f"{func.__name__}() got {noun} " + "; ".join(named))

        return wrapper  # type: ignore[return-value]

    return decorate


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


def require_http_url(endpoint: str | None, *, field: str = "backend.endpoint") -> None:
    """Refuse a `backend.endpoint` (or `field`) no export could reach.

    For the value a host WROTE -- a bad argument raises like any other. One read
    from the environment never reaches here; see `endpoint_from_env`.

    The value is not quoted back: an endpoint can carry a credential in its
    query or its userinfo, and this message ends up in tracebacks and logs.
    An empty string is left alone, as unset: `init()` already reads it so.
    """
    if endpoint and not is_http_url(endpoint):
        raise ValueError(
            f"{field} must be an absolute http:// or https:// URL with a host, such as "
            "'http://collector:4318'; the configured value has no such scheme or no host, so "
            "every export to it would fail. Write the scheme and the host."
        )


def env_http_url(name: str) -> str | None:
    """The variable `name` when it holds an http(s) URL with a host, else None.
    One that does not is left unused, never refused: `init()` names it on
    stderr when it would have been the destination."""
    value = os.environ.get(name)
    return value if value and is_http_url(value) else None


def endpoint_from_env() -> str | None:
    """The exporter address the environment names, in precedence order.

    `WARDEX_ENDPOINT` first, then the OTel spellings, specific before generic,
    so a host already exporting OTLP elsewhere points wardex at the same
    collector with zero new variables. The first variable that is SET decides,
    and its value is used only when it IS an http(s) URL; one that is not does
    not fall through to a less specific variable it overrides.

    A value no export could reach is left unused here rather than refused,
    because whether it matters depends on what `init()` was also given. Under a
    `transport=` or a project key it was never going to be the destination; for
    the OTel spellings, which every OTel SDK in the process shares, a
    scheme-less `collector:4317` is a legal value for a gRPC exporter.
    `unusable_env_endpoint()` names the variable, and `init()` says it on
    stderr when it would have been the destination.

    The value is returned as read -- the `/v1/traces` default path is the
    transport builder's to append (see `BackendConfig.endpoint`).
    """
    found = _first_env_endpoint()
    return found[1] if found is not None and is_http_url(found[1]) else None


def unusable_env_endpoint() -> str | None:
    """The variable `endpoint_from_env` passed over as not an http(s) URL, or None."""
    found = _first_env_endpoint()
    return found[0] if found is not None and not is_http_url(found[1]) else None


def _first_env_endpoint() -> tuple[str, str] | None:
    for name in ("WARDEX_ENDPOINT", *_OTEL_ENDPOINT_NAMES):
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def env_base_url_problem(has_key: bool, has_endpoint: bool) -> tuple[str | None, bool]:
    """What to say about `WARDEX_BASE_URL`, and whether it leaves wardex with
    no destination. Read here, not off the config: the config keeps only a
    usable URL, and the one it dropped still has to be named."""
    value = os.environ.get("WARDEX_BASE_URL")
    if not value:
        return None, False
    if has_key and not is_http_url(value):
        # Falling back to the key's region would send a self-hosted
        # receiver's data to a different receiver.
        return (
            "WARDEX_BASE_URL is set but is not an http:// or https:// URL with a host, so wardex "
            "exports nothing rather than send to another receiver. Set WARDEX_BASE_URL to the "
            "receiver's address, such as http://receiver:8080, or unset it."
        ), True
    if has_key:
        return None, False
    if not has_endpoint:
        return (
            "WARDEX_BASE_URL is set but WARDEX_API_KEY is not, so wardex exports nothing: a "
            "wardex receiver rejects a request without a project key. Set WARDEX_API_KEY, or "
            "unset WARDEX_BASE_URL."
        ), True
    # A receiver address with no key means nothing, so it does not veto the
    # endpoint the host did configure.
    return (
        "WARDEX_BASE_URL is set but WARDEX_API_KEY is not, so wardex ignores it (a wardex "
        "receiver rejects a request without a project key) and exports to the configured "
        "endpoint. Set WARDEX_API_KEY, or unset WARDEX_BASE_URL."
    ), False


def endpoint_named_for_wardex(argument: str | None) -> str | None:
    """The endpoint the host gave WARDEX: the argument, else `WARDEX_ENDPOINT`.

    Not the OTel spellings. Every OTel SDK in the process reads those, so an
    endpoint inherited from them is not a setting anyone gave wardex, and it
    losing to an explicit `transport=` or project key is the expected case, not
    the conflict `init()` announces.
    """
    return argument or os.environ.get("WARDEX_ENDPOINT") or None


def env_typo_messages(environ: Mapping[str, str] | None = None) -> list[str]:
    """One message per set variable that looks like wardex's and nothing reads.

    ONE rule, applied the same way to every name: a variable passes only when
    its name is EXACTLY one that something reads -- wardex's own, or the
    repository tooling's -- because variable names are case-sensitive and so
    is everything that reads them. Only the prefix is matched without regard to
    case, which is what makes `wardex_endpoint` and `wardex_record` the same
    mistake as `WARDEX_ENDPONT`: none of the three is read. A name that differs
    from a tooling name only in case is pointed at that name; any other at the
    closest name wardex reads.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    messages = []
    for name in sorted(env):
        upper = name.upper()
        if not upper.startswith("WARDEX_") or name in WARDEX_ENV_NAMES:
            continue
        if name in _RESERVED_ENV_NAMES or name.startswith(_RESERVED_ENV_PREFIXES):
            continue
        tooling = upper in _RESERVED_ENV_NAMES or upper.startswith(_RESERVED_ENV_PREFIXES)
        hint = upper if tooling else closest(upper, WARDEX_ENV_NAMES)
        messages.append(
            f"{name} is set, but wardex reads no environment variable by that name, so its "
            "value is ignored. "
            + (f"Did you mean {hint}? " if hint else "")
            + f"(wardex reads: {', '.join(WARDEX_ENV_NAMES)})"
        )
    return messages


def env_flag(name: str, raw: str | None, problems: list[str]) -> bool | None:
    """The boolean variable `name`, whose value is `raw`: True or False, or None
    when it is unset, empty, or not a boolean -- the last with a sentence in
    `problems`, because a switch that silently stays at its default is how
    `WARDEX_DEBUG=1` read as "debug is broken". One parser for every boolean
    variable, so no two of them disagree about what "on" is spelled."""
    raw = raw or ""
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    if value:
        problems.append(
            f"{name}={raw[:40]!r} is not a boolean, so wardex ignores it and uses the "
            f"default. Set {name} to 1, true, yes or on (or 0, false, no or off)."
        )
    return None


def disabled_by(problems: list[str]) -> str | None:
    """The variable that switches wardex off, or None.

    `WARDEX_DISABLED` decides whenever it is a boolean -- `0` keeps wardex on in
    a process whose other OTel SDKs `OTEL_SDK_DISABLED` turned off -- and
    `OTEL_SDK_DISABLED`, the switch the OpenTelemetry specification gives every
    SDK, decides otherwise."""
    own = env_flag("WARDEX_DISABLED", os.environ.get("WARDEX_DISABLED"), problems)
    if own is not None:
        return "WARDEX_DISABLED" if own else None
    # Not `env_flag`: the specification reads only `true` (case ignored) as
    # true, and every other OTel SDK in the process does, so wardex turning off
    # on `1` would leave it the one SDK that did.
    raw = os.environ.get("OTEL_SDK_DISABLED") or ""
    value = raw.strip().lower()
    if value == "true":
        return "OTEL_SDK_DISABLED"
    if value not in ("", "false"):
        problems.append(
            f"OTEL_SDK_DISABLED={raw[:40]!r} is neither true nor false, and OpenTelemetry "
            "reads only true as true, so wardex stays on like every other OTel SDK in this "
            "process. Set OTEL_SDK_DISABLED=true to turn them all off, or WARDEX_DISABLED=1 "
            "to turn off wardex alone."
        )
    return None


def otel_pairs(raw: str) -> dict[str, str] | None:
    """`raw` in the OpenTelemetry list form, `key=value,key2=value2`: values
    percent-decoded, whitespace around each pair ignored, empty pairs skipped.
    None when any entry is not `key=value`, because the specification discards
    the whole value then -- for headers, a partial set would authenticate as
    nobody."""
    pairs: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            return None
        pairs[key.strip()] = unquote(value.strip())
    return pairs


def otel_resource(problems: list[str]) -> tuple[str | None, str | None, str | None]:
    """`service.name`, `service.version` and the deployment environment as the
    OpenTelemetry variables give them: `OTEL_SERVICE_NAME` over the
    `service.name` in `OTEL_RESOURCE_ATTRIBUTES` (the specification's order),
    and `deployment.environment.name` over its older spelling
    `deployment.environment`. These are the fallbacks below wardex's own
    variables, so a host already configured for OTel is named the same here."""
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    attributes = otel_pairs(raw) if raw.strip() else {}
    if attributes is None:
        problems.append(
            "OTEL_RESOURCE_ATTRIBUTES has an entry that is not key=value, so wardex ignores "
            "the whole variable. Write it as service.name=checkout,service.version=1.4 with "
            "percent-encoded values."
        )
        attributes = {}
    service = os.environ.get("OTEL_SERVICE_NAME") or attributes.get("service.name") or None
    environment = attributes.get("deployment.environment.name") or attributes.get(
        "deployment.environment"
    )
    return service, attributes.get("service.version") or None, environment or None


def grpc_protocol_variable() -> str | None:
    """The OTel protocol variable that asks for gRPC, or None: the
    traces-specific spelling over the generic one, as for the endpoint. wardex
    exports only OTLP over HTTP, so an endpoint inherited next to `grpc` is a
    gRPC port that would answer every POST with an error."""
    for name in ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL"):
        value = os.environ.get(name, "").strip()
        if value:
            return name if value.lower() == "grpc" else None
    return None


def require_instance(name: str, value: object, cls: type, *, example: str | None = None) -> None:
    """Refuse a value for the `name=` argument that is not a `cls` instance,
    naming the spelling that is. The common mistakes are a dict where a config
    group belongs and the class itself where an instance belongs; both used to
    fail later with an internal attribute or argument error."""
    if value is None or isinstance(value, cls):
        return
    if isinstance(value, type) and issubclass(value, cls):
        got, fix = f"the class {value.__name__}", f"pass an instance: {name}={value.__name__}()"
    elif isinstance(value, Mapping):
        got, fix = "a dict", f"build the group: {name}={cls.__name__}(**the_dict)"
    else:
        got, fix = type(value).__name__, f"write {name}={example or cls.__name__ + '(...)'}"
    raise TypeError(f"{name}= takes a {cls.__name__} instance, got {got}; {fix}")


def require_optional_str(name: str, value: object) -> None:
    """Refuse a non-string `name=` (`service_name=123`): it would reach the
    wire as the wrong type or not at all, with nothing said."""
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name}= takes a str, got {type(value).__name__}: write {name}=str(...)")


def require_exact_hosts(hosts: object) -> tuple[str, ...] | None:
    """`intercept_hosts` as a tuple, refusing the two shapes that match nothing.

    A bare string would canonicalize into its characters, and a pattern such
    as `*.corp.com` would be compared as the literal name `*.corp.com`: the
    field takes exact names on purpose (it widens what wardex READS off the
    wire, where a pattern that matched more than meant is expensive), so a
    pattern is refused with the fix rather than accepted and never matched."""
    if hosts is None:
        return None
    if isinstance(hosts, str):
        raise ValueError(
            f"intercept_hosts must be an iterable of host names, got the "
            f"bare string {hosts!r} — write ({hosts!r},)"
        )
    out = tuple(_unbracketed(h) for h in hosts)  # type: ignore[attr-defined]  # any iterable
    for host in out:
        if isinstance(host, str) and any(c in host for c in "*?["):
            raise ValueError(
                f"intercept_hosts takes exact host names (host or host:port), not patterns; "
                f"{host!r} would match no peer. List each host, such as 'api.corp.com'."
            )
    return out


def _unbracketed(host: object) -> object:
    """`[::1]:8443` as the connection reports its peer, `::1:8443`; `[::1]`
    as `::1`. A URL brackets an IPv6 literal, and the peer address has none."""
    if not isinstance(host, str) or not host.startswith("["):
        return host
    inside, sep, rest = host[1:].partition("]")
    port_ok = rest == "" or (rest.startswith(":") and rest[1:].isdigit())
    if not sep or ":" not in inside or not port_ok:
        return host
    return inside + rest
