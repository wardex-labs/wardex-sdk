"""Configuration object — spec §2.3.

`WardexConfig` is the immutable configuration for SDK initialization. All
fields have sensible defaults, every field is keyword-only, and `init()`'s
keyword parameters mirror these fields one for one (a drift test holds the two
signatures together).

THE GROUPS ARE A CROSS-LANGUAGE CONTRACT. wardex ships one SDK per language
against one core, and a user who has configured the Python SDK must be able to
read the Node or Java one without relearning it — so the group names below are
part of the wire the SDKs share, not a Python spelling choice. They are
recorded here, in the module that defines them, because a contract kept in a
design document is a contract nobody edits when the code moves:

    backend=BackendConfig(...)          WHERE the data goes and whose it is.
                                        The endpoint and the project key that
                                        travels on every envelope header.

    pii=PIIConfig(...)                  WHAT LEAVES THE PROCESS. The masking
                                        mode and the categories exempted from
                                        it.

    batching=BatchingConfig(...)        WHEN buffered spans are sent — the
                                        periodic flush, whether a shutdown
                                        signal triggers one, and the budget a
                                        bare `close()` spends.

    limits=LimitsConfig(...)            HOW MUCH is captured. Resource bounds,
                                        owned by the core.

    propagation=PropagationConfig(...)  WHETHER wardex MUTATES outbound traffic
                                        by injecting W3C trace headers, and
                                        into which hosts.

    adapters=AdaptersConfig(...)        WHICH framework adapters install, and
                                        each adapter's own options.

What stays top-level is what belongs to no group or to the SDK as a whole:
`debug`, `before_send_envelope`, `capture_mode`, `service_name`, `release`,
`environment`, and the
interception trio (`intercept`, `intercept_hosts`, `interceptors`).
The trio stays flat deliberately — `intercept` is the switch, `interceptors`
refines it and `intercept_hosts` scopes it, so filing one of the three under a
group would split a single concern across two levels, which is the
inconsistency grouping exists to remove. That is the GROUPING BOUNDARY RULE,
and it decides both directions: a capture surface gets a group when its
members carry option payloads — which is why `adapters` is a group, its
per-adapter options being payloads no flat tuple can hold — while a bare
switch/refinement/scope trio stays flat.

CONFIG ROUND-TRIPS AS WRITTEN. Lossless bijective canonicalization is
permitted — every collection field accepts any iterable and is canonicalized
in `__post_init__` (list→tuple, set→frozenset) so two configs built from
different container types compare equal. Value mutation — case folding,
trimming, path or default substitution — is not: what a user typed is what
they read back. Derived values (the endpoint path append, env fallbacks) are
computed at the consumer and never written back into the config object; the
one exception is `init()`'s environment resolution, whose whole job is to
build the RESOLVED config, so what it read from `WARDEX_*` is exactly what its
config carries.

There are no compatibility shims for an old spelling. `WardexConfig` is in
beta and a silent ignore is the one outcome a config change may not have, so
every moved name is refused in `__new__` with a message naming its new home,
and every removed name with a message saying why it is gone.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields

from ._enums import (
    AdapterName,
    CaptureMode,
    InterceptorName,
    PIICategory,
    PIIMode,
)
from ._limits import LimitsConfig
from ._types import BeforeSendEnvelopeCallback


class WardexConfigWarning(UserWarning):
    """A configuration that is legal but conflicts with itself.

    Raised-as-a-warning (never an exception) by `init()` when one setting
    silently disables another: an explicit `transport=` next to a
    `backend.endpoint`, PII category exemptions under `PIIMode.OFF`, or an
    `interceptors=` selection with `intercept=False`. Each is a valid program
    — the warning exists because the losing setting would otherwise be
    indistinguishable from one that was honoured. Filter it like any warning
    category (`warnings.simplefilter("ignore", WardexConfigWarning)`).
    """


#: Old flat field -> the spelling that replaces it. Every entry is refused in
#: `WardexConfig.__new__` so a caller gets a message naming the new home instead
#: of a bare unknown-keyword `TypeError` from the generated dataclass
#: `__init__` — or, worse, silence, which is what a dataclass would give a
#: field that had merely been renamed inside a group.
_MOVED: dict[str, str] = {
    "max_buffer_spans": "limits=LimitsConfig(max_buffer_spans=...)",
    "api_key": "backend=BackendConfig(api_key=...)",
    "endpoint": "backend=BackendConfig(endpoint=...)",
    "pii_mode": "pii=PIIConfig(mode=...)",
    "pii_disabled_categories": "pii=PIIConfig(disabled_categories=...)",
    "flush_interval": "batching=BatchingConfig(flush_interval=...)",
    "flush_on_signals": "batching=BatchingConfig(flush_on_signals=...)",
    "propagate_trace": "propagation=PropagationConfig(enabled=...)",
    "propagate_targets": "propagation=PropagationConfig(targets=...)",
    # Renamed rather than regrouped: the hook vetoes a whole BATCH, and a bare
    # `before_send` would import a per-event mental model onto it. Refused like
    # every other moved name — silence is the one outcome it may not have.
    "before_send": "before_send_envelope=... (renamed: the hook receives one whole batch)",
}

#: Removed field -> why it is gone and what to do instead. Same mechanism as
#: `_MOVED`, different message shape: these names have NO new spelling, and a
#: caller who wrote one must hear that the setting is gone rather than be sent
#: hunting for a group it never moved into. Every entry here shipped in a
#: published beta, which is why each is refused by name instead of falling
#: through to the dataclass's bare unknown-keyword TypeError.
_REMOVED: dict[str, str] = {
    "retention": (
        "retention is reserved and returns when its backend consumer ships; "
        "nothing in-process reads it today"
    ),
    "default_retention": (
        "the retention group is reserved and returns when its backend consumer "
        "ships; nothing in-process reads it today"
    ),
    "retention_triggers": (
        "the retention group is reserved and returns when its backend consumer "
        "ships; nothing in-process reads it today"
    ),
    "tags": (
        "tags was cut: config tags had no reader; use wardex.set_tag() "
        "(scope tags ship with this batch)"
    ),
    "replay_buffer_size": (
        "replay_buffer_size was cut: nothing reads it; it returns with its "
        "consumer when a replay buffer ships"
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
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

    Unset, `init()` reads `WARDEX_API_KEY` from the environment.
    """

    endpoint: str | None = None
    """Where to send. `init()` without a `transport=` builds the default
    OTLP/HTTP exporter against this address; with neither, it installs
    `NoOpTransport`, captures into nothing, and says so once on stderr.

    Unset, `init()` reads `WARDEX_ENDPOINT`, then
    `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, then `OTEL_EXPORTER_OTLP_ENDPOINT`.

    THE ENDPOINT RULE: a URL whose path is empty or `/` gets `/v1/traces`
    appended when the default transport is built — `http://collector:4318`
    exports to `http://collector:4318/v1/traces` — while a URL with an explicit
    path is used verbatim. The append happens at transport construction and is
    never written back here: this field reads back exactly as configured.

    An explicit `transport=` wins over this field — a `Transport` carries its
    own address (`OtlpHttpTransport(endpoint=...)`) — and the losing endpoint
    is announced with a `WardexConfigWarning` rather than ignored in silence.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class PIIConfig:
    """What leaves the process, and in what shape."""

    mode: PIIMode = PIIMode.MASK
    disabled_categories: AbstractSet[PIICategory] = frozenset()
    """Categories exempted from masking. Accepts any iterable of
    `PIICategory`; reads back as a `frozenset`. Has no effect when `mode` is
    OFF, and `init()` says so with a `WardexConfigWarning` rather than leaving
    the caller to conclude the exemption was honoured."""

    def __post_init__(self) -> None:
        # Validate BEFORE canonicalizing: `frozenset("email")` is a frozenset
        # of five characters, and the error a wrong element type produces after
        # conversion names the converted shape instead of the mistake.
        for category in self.disabled_categories:
            if not isinstance(category, PIICategory):
                raise ValueError(
                    f"pii disabled_categories entries must be PIICategory members, got {category!r}"
                )
        object.__setattr__(self, "disabled_categories", frozenset(self.disabled_categories))


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchingConfig:
    """When buffered spans are sent, and how long a shutdown may spend on them."""

    flush_interval: float = 5.0
    flush_on_signals: bool = True
    """Whether SIGINT/SIGTERM trigger a flush before the app's own handler runs.
    Turning it off is the only lever a host has over the 2s budget that flush
    takes; see `_runtime._SIGNAL_FLUSH_TIMEOUT`."""

    shutdown_timeout: float = 5.0
    """The budget a bare `close()` spends — the atexit hook, re-init's teardown
    of the previous client, and `wardex.close()` with no argument all resolve
    their default from this one number. An explicit `close(30.0)` is a
    caller-owned budget and ignores it. One number, one home: the field's own
    default matches `transport._base.DEFAULT_TIMEOUT`."""

    def __post_init__(self) -> None:
        if self.flush_interval <= 0:
            raise ValueError(f"flush_interval must be > 0, got {self.flush_interval}")
        if self.shutdown_timeout <= 0:
            raise ValueError(f"shutdown_timeout must be > 0, got {self.shutdown_timeout}")


@dataclass(frozen=True, slots=True, kw_only=True)
class PropagationConfig:
    """Whether wardex mutates outbound traffic, and into which hosts.

    The only group that can change what the host application SENDS, which is
    why it is off by default and why it has a group of its own rather than a
    flag among flags.
    """

    enabled: bool = False
    targets: Iterable[str] | None = None
    """Glob patterns matched against the outbound host. Accepts any iterable
    of strings; reads back as a tuple. `None` means every host the patched
    clients reach.

    MATCHING IS CASE-INSENSITIVE — hostnames are, so `*.MyCorp.com` admits
    `api.mycorp.com` — but the field reads back EXACTLY as written: config
    round-trips as written, and the folding both sides of the match need lives
    in the injector (`context._inject`), computed once per configured
    allowlist rather than per request.
    """

    def __post_init__(self) -> None:
        if self.targets is None:
            return
        # A bare string IS an iterable of strings, so `targets="*.corp"` would
        # canonicalize into a tuple of one-character patterns that each pass
        # the per-entry check and match nothing. Refused by name before the
        # conversion can make the mistake unrecognizable.
        if isinstance(self.targets, str):
            raise ValueError(
                f"propagation targets must be an iterable of glob strings, "
                f"got the bare string {self.targets!r} — write ({self.targets!r},)"
            )
        targets = tuple(self.targets)
        for pattern in targets:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(
                    f"propagation targets must be non-empty glob strings, got {pattern!r}"
                )
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnthropicAgentSdkConfig:
    """The Anthropic Agent SDK adapter's own options.

    BOTH FIELDS ARE AHEAD OF THEIR CONSUMER, and that is stated rather than
    hidden: the thing that reads them is the bridge implementation they were
    designed for — the one that merges the Claude CLI's own OTel telemetry
    into wardex's tree — and it has not landed. Until it does they gate
    nothing, and the release that ships these fields must contain it: a config
    field never ships before its consumer.
    """

    otel_bridge: bool = False
    """The opt-in that will merge the Claude CLI's own OTel telemetry into
    wardex's tree. `False` — the default — must reproduce today's tree
    byte-for-byte: the bridge adds spans, never rearranges the ones wardex
    already builds. Consumed by the bridge implementation this field was
    designed for; until it lands the flag gates nothing."""

    otel_bridge_drain: float = 0.2
    """How long, in SECONDS (every duration field is), a session's end waits
    for the CLI's final telemetry batch before the bridge stops listening.
    Must be >= 0; `0` means no wait at all. Skipped on the atexit and signal
    paths, whose budgets are the shutdown's to spend. Same consumer as
    `otel_bridge`, and the same not-yet-landed caveat."""

    def __post_init__(self) -> None:
        if self.otel_bridge_drain < 0:
            raise ValueError(
                f"otel_bridge_drain must be >= 0 (seconds), got {self.otel_bridge_drain}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AdaptersConfig:
    """WHICH framework adapters install, and each adapter's own options.

    SELECTION AND OPTIONS ARE SEPARATE FIELDS OF ONE GROUP, deliberately:
    configuring an adapter's option never touches `enabled`, so auto-detection
    survives — `AdaptersConfig(anthropic_agent_sdk=AnthropicAgentSdkConfig(
    otel_bridge=True))` still leaves every other installed framework
    auto-detected. The alternative shapes both collapse that: a per-adapter
    top-level kwarg couples selection to spelling, and an instance list makes
    naming an option a de-selection of everything unnamed.

    ONE IDENTIFIER PER ADAPTER: the per-adapter field name equals the
    `AdapterName` value equals `adapter.name()` — `anthropic_agent_sdk`, never
    a second spelling — so selection, options and the registry's install table
    key one adapter the same way.

    A PER-ADAPTER CONFIG CLASS IS CREATED WITH ITS FIRST REAL OPTION, never
    ahead of it — which is why `langgraph` has no field here. An empty options
    class is a name a user can spell that configures nothing, the config-shaped
    twin of a selectable no-op.
    """

    enabled: tuple[AdapterName, ...] | None = None
    """WHICH adapters install — exactly the flat field's old semantics. `None`
    means auto-detect installed frameworks; `()` means install none; a tuple
    means exactly these. A choice and the absence of one, and they may not
    collapse into each other. Accepts any iterable; reads back as a tuple.
    Every entry must be an `AdapterName` member, refused here where the
    mistake was made (the mirror of `interceptors`): a bare string matched
    against nothing would install nothing, in silence."""

    anthropic_agent_sdk: AnthropicAgentSdkConfig = field(default_factory=AnthropicAgentSdkConfig)
    """The Anthropic Agent SDK adapter's options. Setting them selects
    nothing and disturbs no auto-detection; see the class docstring."""

    def __post_init__(self) -> None:
        if self.enabled is None:
            return
        enabled = tuple(self.enabled)
        for name in enabled:
            if not isinstance(name, AdapterName):
                raise ValueError(
                    f"adapters enabled entries must be AdapterName members, got {name!r}"
                )
        object.__setattr__(self, "enabled", enabled)


def _non_default_adapter_options(adapters: AdaptersConfig) -> tuple[AdapterName, ...]:
    """The adapters whose options differ from a default-constructed group.

    "Differs from default" IS dataclass equality against the default-constructed
    instance — the frozen dataclasses exist to make value equality the one
    definition of sameness, and a field-by-field diff here would be a second.
    Consumed by the two configured-but-not-installed announcements: `init()`'s
    unconditional `WardexConfigWarning` for an adapter the user's own `enabled=`
    excludes, and `_adapters/`'s install-time debug line for one the
    environment did not detect.
    """
    default = AdaptersConfig()
    return tuple(
        AdapterName(f.name)
        for f in fields(AdaptersConfig)
        if f.name != "enabled" and getattr(adapters, f.name) != getattr(default, f.name)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class WardexConfig:
    """Wardex SDK configuration. An immutable, keyword-only dataclass.

    `init()` is the way in: its keyword parameters are these fields plus
    `transport=`, and it resolves the `WARDEX_*` environment variables into
    the instance it builds. Constructing one directly skips that resolution.
    """

    backend: BackendConfig = field(default_factory=BackendConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)

    adapters: AdaptersConfig = field(default_factory=AdaptersConfig)
    """WHICH framework adapters install, and each adapter's own options. The
    selection lives at `adapters.enabled` (same `None`/`()`/tuple semantics
    the flat field had); the flat tuple spelling is refused in `__post_init__`
    with the new one, because a tuple read as a group would be ignored in
    silence — the one outcome a config change may not have."""

    interceptors: Iterable[InterceptorName] | None = None
    """Which byte seams `intercept=True` installs. Accepts any iterable; reads
    back as a tuple. `None` means all of them.

    `intercept` is the SWITCH and this is the refinement: with `intercept=False`
    a selection installs nothing, and `init()` says so with a
    `WardexConfigWarning`, because a refinement of a switch that is off is not
    an error but silence about it is how a user concludes it was honoured.
    `()` is a choice — install none — and `None` is the absence of one; they
    may not collapse into each other, the same distinction `adapters` draws.

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
    before_send_envelope: BeforeSendEnvelopeCallback | None = None

    intercept: bool = True
    """The interception switch, ON by default: `init()` is the consent, the
    adapters already auto-install, and zero-instrumentation capture is the
    product. Mutation (`propagation`) stays off; PII masking stays on.
    `intercept=False` is the documented opt-out."""

    intercept_hosts: Iterable[str] | None = None
    """Plaintext peers — `host` or `host:port` — captured whatever the capture
    mode says. Accepts any iterable of strings; reads back as a tuple. Matched
    CASE-INSENSITIVELY. `None` leaves every connection to the shared policy.

    Naming a host by hand is a more specific opt-in than a global mode, so it
    bypasses rather than composes (`interceptors._socket`). Exact names and not
    globs, unlike `propagation.targets`: that one scopes a header wardex
    WRITES, where a user has to be able to name a whole domain, while this one
    widens what wardex READS off the wire — the direction in which a pattern
    that matched more than its author meant is expensive.
    """

    capture_mode: CaptureMode = CaptureMode.AGENT

    service_name: str | None = None
    """Maps to the OTLP resource's `service.name` — the app's identity in any
    OTel backend, the axis every one of them groups by. Unset, `init()` reads
    `WARDEX_SERVICE_NAME`; still unset, the wire falls back to semconv's
    `unknown_service:python` (never the SDK's own name — the SDK travels in
    `telemetry.sdk.*`)."""

    release: str | None = None
    """Maps to the OTLP resource's `service.version`. Unset, `init()` reads
    `WARDEX_RELEASE`; still unset, no key is emitted."""

    environment: str | None = None
    """Maps to the OTLP resource's `deployment.environment.name`. Unset,
    `init()` reads `WARDEX_ENVIRONMENT`; still unset, no key is emitted."""

    def __new__(cls, *args: object, **kwargs: object) -> WardexConfig:
        moved = [name for name in _MOVED if name in kwargs]
        removed = [name for name in _REMOVED if name in kwargs]
        if moved or removed:
            lines = [f"  {name} -> {_MOVED[name]}" for name in moved]
            lines += [f"  {name}: {_REMOVED[name]}" for name in removed]
            raise TypeError(
                "WardexConfig groups its fields by concern; these are not fields:\n"
                + "\n".join(lines)
                + "\nThe groups are backend, pii, batching, limits,"
                " propagation and adapters — see the Configuration section of"
                " the README:"
                " https://github.com/wardex-labs/wardex-sdk#configuration"
            )
        # object.__new__, not super().__new__: @dataclass(slots=True) rebuilds
        # the class to attach __slots__, which invalidates the zero-arg
        # super() closure cell on some interpreters (observed on CPython
        # 3.13; not 3.14). object.__new__(cls) sidesteps that entirely.
        return object.__new__(cls)

    def __post_init__(self) -> None:
        """Validate and canonicalize the fields that belong to no group.

        Each group validates its own where they are declared — that is half the
        point of having them — so what is left here is the flat collections.
        Every one accepts any iterable and is stored as a tuple, so a config
        built from a list compares equal to one built from a tuple, and `()` is
        preserved as the choice it is rather than collapsing into `None`.

        `interceptors` entries are refused HERE, where the mistake was made,
        rather than skipped at install time. `install_configured_interceptors`
        walks its own table and keeps what was asked for, so a value it does
        not recognize — `interceptors=("ssl",)`, the string, is the one a user
        actually writes — matches nothing and installs nothing, in silence,
        which is indistinguishable from `intercept=False`.

        `adapters` is the one RESHAPED field: it was the flat selection tuple
        and is a group now, so the old spelling arrives through a keyword that
        still exists and `_MOVED` cannot see it. It is refused here instead,
        by type, with the new spelling in the message.
        """
        if not isinstance(self.adapters, AdaptersConfig):
            raise TypeError(
                "adapters= now takes AdaptersConfig; the selection tuple moved: "
                "adapters=(AdapterName.X,) -> adapters=AdaptersConfig(enabled=(AdapterName.X,))"
            )
        if self.interceptors is not None:
            interceptors = tuple(self.interceptors)
            for name in interceptors:
                if not isinstance(name, InterceptorName):
                    raise ValueError(
                        f"interceptors entries must be InterceptorName members, got {name!r}"
                    )
            object.__setattr__(self, "interceptors", interceptors)
        if self.intercept_hosts is not None:
            # Same bare-string hazard as `PropagationConfig.targets`: a lone
            # host name would canonicalize into its characters and match no
            # peer at all, in silence.
            if isinstance(self.intercept_hosts, str):
                raise ValueError(
                    f"intercept_hosts must be an iterable of host names, got the "
                    f"bare string {self.intercept_hosts!r} — write ({self.intercept_hosts!r},)"
                )
            object.__setattr__(self, "intercept_hosts", tuple(self.intercept_hosts))


def _resolve_config(
    *,
    backend: BackendConfig | None = None,
    pii: PIIConfig | None = None,
    batching: BatchingConfig | None = None,
    limits: LimitsConfig | None = None,
    propagation: PropagationConfig | None = None,
    adapters: AdaptersConfig | None = None,
    interceptors: tuple[InterceptorName, ...] | None = None,
    intercept: bool = True,
    intercept_hosts: Sequence[str] | None = None,
    capture_mode: CaptureMode = CaptureMode.AGENT,
    service_name: str | None = None,
    release: str | None = None,
    environment: str | None = None,
    before_send_envelope: BeforeSendEnvelopeCallback | None = None,
    debug: bool = False,
) -> WardexConfig:
    """Fold the environment into `init()`'s arguments and build the config.

    THE RESOLUTION ORDER, PER FIELD: an explicit argument wins; an unset one
    falls back to its `WARDEX_*` variable; `backend.endpoint` additionally
    falls through to the OTel spellings. The result IS the resolved config —
    what `init()` installs and what `client.config` answers with — so an env
    value that won is readable there rather than applied invisibly.

    A GROUP IS NOT AN ALL-OR-NOTHING OVERRIDE. `backend`'s two fields resolve
    independently: a caller who sets only `backend=BackendConfig(endpoint=...)`
    still gets `WARDEX_API_KEY`. The alternative — skipping both env reads as
    soon as a `backend=` arrives — is a silent ignore one level down, and
    grouping made the whole-group spelling the normal one.

    `debug` is the one field the environment can only turn ON: `debug=False`
    is the parameter's default, so it cannot be read as a veto, and
    `WARDEX_DEBUG=true` (case-insensitive) must be able to switch diagnostics
    on for a program nobody can edit. Same shape as Sentry's.

    `None` for a whole group means that group's defaults. `intercept_hosts`
    keeps `()` distinct from `None`: an empty tuple is a choice, not the
    absence of one.
    """
    given = backend if backend is not None else BackendConfig()
    resolved_backend = BackendConfig(
        api_key=given.api_key or os.environ.get("WARDEX_API_KEY"),
        endpoint=given.endpoint or _endpoint_from_env(),
    )
    return WardexConfig(
        backend=resolved_backend,
        pii=pii if pii is not None else PIIConfig(),
        batching=batching if batching is not None else BatchingConfig(),
        limits=limits if limits is not None else LimitsConfig(),
        propagation=propagation if propagation is not None else PropagationConfig(),
        adapters=adapters if adapters is not None else AdaptersConfig(),
        interceptors=interceptors,
        intercept=intercept,
        intercept_hosts=tuple(intercept_hosts) if intercept_hosts is not None else None,
        capture_mode=capture_mode,
        service_name=(
            service_name if service_name is not None else os.environ.get("WARDEX_SERVICE_NAME")
        ),
        release=release if release is not None else os.environ.get("WARDEX_RELEASE"),
        environment=(
            environment if environment is not None else os.environ.get("WARDEX_ENVIRONMENT")
        ),
        before_send_envelope=before_send_envelope,
        debug=bool(debug) or os.environ.get("WARDEX_DEBUG", "").lower() == "true",
    )


def _endpoint_from_env() -> str | None:
    """The exporter address the environment names, in precedence order.

    `WARDEX_ENDPOINT` first; then the OTel spellings, specific before generic,
    so a host already exporting OTLP elsewhere points wardex at the same
    collector with zero new variables. The value is stored as read — the
    `/v1/traces` default path is the transport builder's to append (see
    `BackendConfig.endpoint`), never this function's to bake in.
    """
    for name in (
        "WARDEX_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        endpoint = os.environ.get(name)
        if endpoint:
            return endpoint
    return None
