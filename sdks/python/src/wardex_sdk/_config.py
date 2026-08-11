"""Configuration object — spec §2.3.

`WardexConfig` is the immutable configuration for SDK initialization. All
fields have sensible defaults.

THE GROUPS ARE A CROSS-LANGUAGE CONTRACT. wardex ships one SDK per language
against one core, and a user who has configured the Python SDK must be able to
read the Node or Java one without relearning it — so the group names below are
part of the wire the SDKs share, not a Python spelling choice. They are
recorded here, in the module that defines them, because a contract kept in a
design document is a contract nobody edits when the code moves:

    backend=BackendConfig(...)          WHERE the data goes and whose it is.
                                        The endpoint and the project key that
                                        travels on every envelope header.

    retention=RetentionPolicy(...)      HOW LONG a captured payload is worth
                                        keeping, and what promotes one run past
                                        the default.

    pii=PIIPolicy(...)                  WHAT LEAVES THE PROCESS. The masking
                                        mode and the categories exempted from
                                        it.

    batching=BatchingPolicy(...)        WHEN buffered spans are sent — the
                                        periodic flush, and whether a shutdown
                                        signal triggers one.

    limits=CaptureLimits(...)           HOW MUCH is captured. Resource bounds,
                                        owned by the core.

    propagation=PropagationPolicy(...)  WHETHER wardex MUTATES outbound traffic
                                        by injecting W3C trace headers, and
                                        into which hosts.

What stays top-level is what belongs to no group or to the SDK as a whole:
`debug`, `before_send`, `capture_mode`, `release`, `environment`, `tags`,
`adapters`, and the interception trio (`intercept`, `intercept_hosts`,
`interceptors`). The trio stays flat deliberately — `intercept` is the switch,
`interceptors` refines it and `intercept_hosts` scopes it, so filing one of the
three under a group would split a single concern across two levels, which is
the inconsistency grouping exists to remove.

There are no compatibility shims for the old flat spelling. `WardexConfig` is
in beta and a silent ignore is the one outcome a config change may not have, so
every moved name is refused in `__new__` with a message naming its new group.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ._enums import (
    AdapterName,
    CaptureMode,
    CaptureTrigger,
    InterceptorName,
    PIICategory,
    PIIMode,
    RetentionClass,
)
from ._limits import CaptureLimits
from ._types import BeforeSendCallback

#: Old flat field -> the spelling that replaces it. Every entry is refused in
#: `WardexConfig.__new__` so a caller gets a message naming the new home instead
#: of a bare unknown-keyword `TypeError` from the generated dataclass
#: `__init__` — or, worse, silence, which is what a dataclass would give a
#: field that had merely been renamed inside a group.
_MOVED: dict[str, str] = {
    "max_buffer_spans": "limits=CaptureLimits(max_buffer_spans=...)",
    "replay_buffer_size": "limits=CaptureLimits(replay_buffer_size=...)",
    "api_key": "backend=BackendConfig(api_key=...)",
    "endpoint": "backend=BackendConfig(endpoint=...)",
    "default_retention": "retention=RetentionPolicy(default=...)",
    "retention_triggers": "retention=RetentionPolicy(triggers=...)",
    "pii_mode": "pii=PIIPolicy(mode=...)",
    "pii_disabled_categories": "pii=PIIPolicy(disabled_categories=...)",
    "flush_interval": "batching=BatchingPolicy(flush_interval=...)",
    "flush_on_signals": "batching=BatchingPolicy(flush_on_signals=...)",
    "propagate_trace": "propagation=PropagationPolicy(enabled=...)",
    "propagate_targets": "propagation=PropagationPolicy(targets=...)",
}


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Where captured data goes, and whose project it belongs to.

    Named for the destination rather than for the `Transport` that reaches it,
    and the difference is not cosmetic: `init(transport=...)` already takes the
    Transport OBJECT, so a group called `transport=` could never be spelled
    through the SDK's only entry point, and a user who tried would hand `init()`
    a config group where a Transport was expected.
    """

    api_key: str | None = field(default=None, repr=False)
    """Identifies the project on every envelope header.

    `repr=False` because a config object's string form ends up in logs, crash
    reports and debugger output, none of which is a place for a credential:
    string forms of config objects never contain secret material.
    """

    endpoint: str | None = None
    """Where to send. `init()` without a `transport=` builds the default
    OTLP/HTTP exporter against this address; with neither, it installs
    `NoOpTransport` and captures into nothing.

    An explicit `transport=` wins over this field — a `Transport` carries its
    own address (`OtlpHttpTransport(endpoint=...)`), and under `debug` the
    losing endpoint is announced on stderr rather than ignored in silence."""


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long a captured payload is worth keeping."""

    default: RetentionClass = RetentionClass.SUMMARY_ONLY
    triggers: frozenset[CaptureTrigger] = frozenset(
        {CaptureTrigger.ERROR, CaptureTrigger.MANUAL_MARK}
    )
    """What promotes a run past `default` — an error, a manual mark."""


@dataclass(frozen=True, slots=True)
class PIIPolicy:
    """What leaves the process, and in what shape."""

    mode: PIIMode = PIIMode.MASK
    disabled_categories: frozenset[PIICategory] = frozenset()
    """Categories exempted from masking. Has no effect when `mode` is OFF, and
    `init()` says so on stderr under `debug` rather than leaving the caller to
    conclude the exemption was honoured."""

    def __post_init__(self) -> None:
        if self.mode in (PIIMode.REDACT, PIIMode.HASH):
            raise NotImplementedError(
                f"PIIMode.{self.mode.name} is not implemented yet "
                "(v1 supports MASK/OFF; see the PII masking design doc)"
            )


@dataclass(frozen=True, slots=True)
class BatchingPolicy:
    """When buffered spans are sent."""

    flush_interval: float = 5.0
    flush_on_signals: bool = True
    """Whether SIGINT/SIGTERM trigger a flush before the app's own handler runs.
    Turning it off is the only lever a host has over the 2s budget that flush
    takes; see `_runtime._SIGNAL_FLUSH_TIMEOUT`."""

    def __post_init__(self) -> None:
        if self.flush_interval <= 0:
            raise ValueError(f"flush_interval must be > 0, got {self.flush_interval}")


@dataclass(frozen=True, slots=True)
class PropagationPolicy:
    """Whether wardex mutates outbound traffic, and into which hosts.

    The only group that can change what the host application SENDS, which is
    why it is off by default and why it has a group of its own rather than a
    flag among flags.
    """

    enabled: bool = False
    targets: tuple[str, ...] | None = None
    """Glob patterns matched CASE-INSENSITIVELY against the outbound host.
    `None` means every host the patched clients reach.

    Hostnames are case-insensitive, so both sides of the match are folded and
    `*.MyCorp.com` admits `api.mycorp.com`. The patterns are folded here, once,
    which is also why they read back lowercased — that is the answer to "did
    the capitals I typed do anything".
    """

    def __post_init__(self) -> None:
        if self.targets is None:
            return
        for pattern in self.targets:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(
                    f"propagation targets must be non-empty glob strings, got {pattern!r}"
                )
        # Folded at construction and not per request. The config is built once
        # and consulted on every outbound call the allowlist admits, so the
        # side of the comparison that cannot change belongs here; the injector
        # is then left folding only the host. It also puts the case rule where
        # a user can see it, instead of in a private matcher nobody reads.
        object.__setattr__(self, "targets", tuple(pattern.lower() for pattern in self.targets))


@dataclass(frozen=True, slots=True)
class WardexConfig:
    """Wardex SDK configuration. An immutable dataclass."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    pii: PIIPolicy = field(default_factory=PIIPolicy)
    batching: BatchingPolicy = field(default_factory=BatchingPolicy)
    limits: CaptureLimits = field(default_factory=CaptureLimits)
    propagation: PropagationPolicy = field(default_factory=PropagationPolicy)

    adapters: tuple[AdapterName, ...] | None = None

    interceptors: tuple[InterceptorName, ...] | None = None
    """Which byte seams `intercept=True` installs. `None` means all of them.

    `intercept` is the SWITCH and this is the refinement: with `intercept=False`
    a selection installs nothing and says so under `debug`, because a refinement
    of a switch that is off is not an error but silence about it is how a user
    concludes it was honoured. `()` is a choice — install none — and `None` is
    the absence of one; they may not collapse into each other, the same
    distinction `adapters` already draws.

    ORDER IS NOT A CALLER'S TO SET. `interceptors._INTERCEPTORS` is walked and
    filtered by this tuple rather than the other way round, so a reordered
    selection installs in the same order as an unordered one: SSL patches
    `ssl.SSLSocket` and the raw socket seam patches `socket.socket` underneath
    it, which is a fact about the stack rather than a preference.

    EVERY MEMBER OF `InterceptorName` IS INSTALLABLE, which is why nothing here
    can be selected and quietly do nothing. `GRPC`, `WEBSOCKET` and `SSE` were
    members until selection went live and named no unit at all — they are
    protocols the byte seams parse, and `Protocol` is their home — so they were
    removed rather than left to be rejected: a name that cannot be spelled needs
    no validation. What IS rejected, in `__post_init__`, is a value that is not
    a member: `interceptors=("ssl",)` is the mistake a user actually makes, and
    matched against nothing it would install nothing in silence.
    """

    debug: bool = False
    before_send: BeforeSendCallback | None = None
    intercept: bool = False

    intercept_hosts: tuple[str, ...] | None = None
    """Plaintext peers — `host` or `host:port` — captured whatever the capture
    mode says. Matched CASE-INSENSITIVELY. `None` leaves every connection to
    the shared policy.

    Naming a host by hand is a more specific opt-in than a global mode, so it
    bypasses rather than composes (`interceptors._socket`). Exact names and not
    globs, unlike `propagation.targets`: that one scopes a header wardex
    WRITES, where a user has to be able to name a whole domain, while this one
    widens what wardex READS off the wire — the direction in which a pattern
    that matched more than its author meant is expensive.
    """

    capture_mode: CaptureMode = CaptureMode.AGENT

    release: str | None = None
    environment: str | None = None
    tags: tuple[tuple[str, str], ...] = ()

    def __new__(cls, *args: object, **kwargs: object) -> WardexConfig:
        moved = [name for name in _MOVED if name in kwargs]
        if moved:
            raise TypeError(
                "WardexConfig groups its fields by concern; these moved:\n"
                + "\n".join(f"  {name} -> {_MOVED[name]}" for name in moved)
                + "\nThe groups are backend, retention, pii, batching, limits and"
                " propagation — see the Configuration section of the README:"
                " https://github.com/wardex-labs/wardex-sdk#configuration"
            )
        # object.__new__, not super().__new__: @dataclass(slots=True) rebuilds
        # the class to attach __slots__, which invalidates the zero-arg
        # super() closure cell on some interpreters (observed on CPython
        # 3.13; not 3.14). object.__new__(cls) sidesteps that entirely.
        return object.__new__(cls)

    def __post_init__(self) -> None:
        """Validate the fields that belong to no group.

        Each group validates its own where they are declared — that is half the
        point of having them — so all that is left here is `interceptors`, and
        it is refused HERE, where the mistake was made, rather than skipped at
        install time. `install_configured_interceptors` walks its own table and
        keeps what was asked for, so a value it does not recognize —
        `interceptors=("ssl",)`, the string, is the one a user actually writes —
        matches nothing and installs nothing, in silence, which is
        indistinguishable from `intercept=False`.
        """
        if self.interceptors is not None:
            for name in self.interceptors:
                if not isinstance(name, InterceptorName):
                    raise ValueError(
                        f"interceptors entries must be InterceptorName members, got {name!r}"
                    )

    @property
    def effective_retention(self) -> RetentionClass:
        """Return the effective Retention class based on the environment.

        local/staging/development environments are upgraded to REPLAYABLE.
        """
        if self.environment in ("local", "staging", "development"):
            return RetentionClass.REPLAYABLE
        return self.retention.default

    @classmethod
    def from_env(cls, **overrides: object) -> WardexConfig:
        """Read configuration from environment variables. Can be overridden via overrides.

        Environment variables read:
        - WARDEX_API_KEY
        - WARDEX_ENDPOINT
        - WARDEX_ENVIRONMENT
        - WARDEX_DEBUG (true/false, case-insensitive)

        The overrides are spelled as `WardexConfig` FIELDS and are forwarded
        whole, so the two backend values arrive as one `backend=BackendConfig(
        ...)` and `from_env(api_key=...)` gets the same message naming the new
        group that the constructor gives. This path used to keep its own list of
        four names and drop every other override on the floor — a config builder
        that silently ignores half of what it is handed is the shape of bug this
        whole change is about.

        A GROUP IS NOT AN ALL-OR-NOTHING OVERRIDE. Every variable above is
        resolved PER FIELD, `backend`'s two included: a caller who sets only
        `backend=BackendConfig(endpoint=...)` still gets `WARDEX_API_KEY`. The
        alternative — skipping both env reads as soon as a `backend=` arrives —
        is the same silent ignore one level down, and grouping made it the
        normal spelling rather than an edge case, because overriding one backend
        value now means constructing the whole group. `None` is the absence of a
        value here, exactly as it was when these were flat fields.
        """
        resolved: dict[str, object] = dict(overrides)
        given: BackendConfig = resolved.get("backend") or BackendConfig()  # type: ignore[assignment]
        resolved["backend"] = BackendConfig(
            api_key=given.api_key or os.environ.get("WARDEX_API_KEY"),
            endpoint=given.endpoint or os.environ.get("WARDEX_ENDPOINT"),
        )
        if resolved.get("environment") is None:
            resolved["environment"] = os.environ.get("WARDEX_ENVIRONMENT")
        resolved["debug"] = (
            bool(resolved["debug"])
            if "debug" in resolved
            else os.environ.get("WARDEX_DEBUG", "").lower() == "true"
        )
        return cls(**resolved)  # type: ignore[arg-type]
