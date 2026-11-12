"""Wardex SDK — observability for AI agents."""

from __future__ import annotations

import warnings as _warnings
from collections.abc import Sequence as _Sequence
from contextlib import AbstractContextManager as _AbstractContextManager
from typing import Any as _Any
from uuid import UUID as _UUID

from . import _hub, _runtime
from ._assembly import _ledger
from ._assembly import diag_info as _diag_info
from ._assembly import report_once as _report_once
from ._assembly._explain import explain
from ._client import _FOLLOW_TRANSPORT_TIMEOUT
from ._client import Client as _Client
from ._config import (
    _MOVED,
    _REMOVED,
    AdaptersConfig,
    AnthropicAgentSdkConfig,
    BackendConfig,
    BatchingConfig,
    CodexExecConfig,
    PIIConfig,
    PropagationConfig,
    WardexConfigWarning,
    _non_default_adapter_options,
)
from ._config import _resolve_config as _resolve_config_from_env
from ._config_checks import disabled_by as _disabled_by
from ._config_checks import endpoint_named_for_wardex as _endpoint_named_for_wardex
from ._config_checks import env_typo_messages as _env_typo_messages
from ._config_checks import refuses_unknown_init_keywords as _refuses_unknown_init_keywords
from ._config_checks import require_instance as _require_instance
from ._config_checks import require_optional_str as _require_optional_str
from ._decorators import agent, step, tool, workflow
from ._diagnostics import AdapterStatus, Diagnostics, Loss
from ._enums import (
    AdapterName,
    AdapterState,
    AgentType,
    CaptureMode,
    InterceptorName,
    OperationName,
    OutputType,
    PIICategory,
    PIIMode,
    ProviderName,
    SnapshotType,
    SpanKind,
    StatusCode,
    ToolExecutionType,
    ToolType,
)
from ._limits import LimitsConfig

# NOT public: imported for use by `init()` and `close()` below, and the
# underscore aliases are what make the non-export literal — `wardex_sdk.
# NATIVE_OK` no longer resolves at all, instead of resolving while being
# deliberately absent from `__all__`. Said here rather than left to be guessed,
# because the pair looks like a supported feature probe and is not one: the
# degraded mode it describes is ANNOUNCED, once, on stderr by `init()`, and a
# host does not have to ask. Their home is `wardex_sdk._native`, private, and
# they are free to change shape there.
from ._native import NATIVE_OK as _NATIVE_OK
from ._native import unavailable_reason as _unavailable_reason
from ._scope import ACCOUNT_KEY as _ACCOUNT_KEY
from ._scope import NO_USER as _NO_USER
from ._scope import Scope, UserInfo
from ._scope import user_from_call as _user_from_call
from ._snapshot_api import capture_state_snapshot
from ._tracing import Span, conversation, current_span, span
from ._types import (
    AgentAttributes,
    BeforeSendEnvelopeCallback,
    CallSite,
    ConversationContext,
    Envelope,
    GenAIAttributes,
    InputRef,
    ToolAttributes,
    ToolDefinition,
    ToolDefinitionSet,
)
from ._values import AttributeValue as _AttributeValue
from ._version import __version__
from .context._asgi import WardexAsgiMiddleware
from .context._contextvar import bind_context
from .context._propagate import (
    continue_from_otel,
    continue_trace,
    get_trace_headers,
    get_traceparent,
)
from .context._with_only import WithOrSyncDecorator as _WithOrSyncDecorator
from .context._wsgi import WardexWsgiMiddleware
from .transport._base import Transport
from .transport._console import ConsoleTransport
from .transport._noop import NoOpTransport
from .transport._otlp_http import OtlpHttpTransport
from .transport._resolve import announce_pii_env as _announce_pii_env
from .transport._resolve import default_transport as _default_transport
from .transport._resolve import said as _said
from .transport._resolve import say as _say
from .transport._wardex import WardexTransport

__all__ = [
    "__version__",
    "init",
    "conversation",
    "span",
    "current_span",
    "workflow",
    "agent",
    "step",
    "tool",
    "capture_state_snapshot",
    "set_tag",
    "set_user",
    "set_account",
    "set_context",
    "isolation_scope",
    "new_scope",
    "flush",
    "close",
    "diagnostics",
    "explain",
    "bind_context",
    "continue_trace",
    "continue_from_otel",
    "get_traceparent",
    "get_trace_headers",
    "WardexAsgiMiddleware",
    "WardexWsgiMiddleware",
    # Config groups — every one of them is passed to `init()` by name
    "AdaptersConfig",
    "AnthropicAgentSdkConfig",
    "BackendConfig",
    "BatchingConfig",
    "CodexExecConfig",
    "LimitsConfig",
    "PIIConfig",
    "PropagationConfig",
    "WardexConfigWarning",
    # Enums — importable directly from user code
    "AdapterName",
    "AdapterState",
    "AgentType",
    "CaptureMode",
    "InterceptorName",
    "OperationName",
    "OutputType",
    "PIICategory",
    "PIIMode",
    "ProviderName",
    "SnapshotType",
    "SpanKind",
    "StatusCode",
    "ToolExecutionType",
    "ToolType",
    # Types
    "AdapterStatus",
    "Diagnostics",
    "Loss",
    "Span",
    "UserInfo",
    "Scope",
    "AgentAttributes",
    "BeforeSendEnvelopeCallback",
    "CallSite",
    "ConversationContext",
    "Envelope",
    "GenAIAttributes",
    "InputRef",
    "ToolAttributes",
    "ToolDefinition",
    "ToolDefinitionSet",
    # Transport
    "Transport",
    "NoOpTransport",
    "ConsoleTransport",
    "OtlpHttpTransport",
    "WardexTransport",
]


def _install_nothing() -> None:
    """End `init()` with nothing installed, so every public call stays the no-op
    it is before `init()`; a previous `init()`'s client is closed."""
    if _runtime.runtime().client is not None:
        _runtime.runtime().teardown()


@_refuses_unknown_init_keywords(
    moved=_MOVED,
    removed=_REMOVED,
    groups=(
        ("backend", BackendConfig),
        ("pii", PIIConfig),
        ("batching", BatchingConfig),
        ("limits", LimitsConfig),
        ("propagation", PropagationConfig),
        ("adapters", AdaptersConfig),
    ),
)
def init(
    *,
    transport: Transport | None = None,
    backend: BackendConfig | None = None,
    pii: PIIConfig | None = None,
    batching: BatchingConfig | None = None,
    limits: LimitsConfig | None = None,
    propagation: PropagationConfig | None = None,
    adapters: AdaptersConfig | None = None,
    interceptors: tuple[InterceptorName, ...] | None = None,
    intercept: bool = True,
    intercept_hosts: _Sequence[str] | None = None,
    capture_mode: CaptureMode = CaptureMode.AGENT,
    service_name: str | None = None,
    release: str | None = None,
    environment: str | None = None,
    before_send_envelope: BeforeSendEnvelopeCallback | None = None,
    debug: bool = False,
) -> None:
    """Initialize wardex: resolve the config, build the client, install it.

    The keyword parameters are `WardexConfig`'s fields plus `transport=` — a
    drift test holds the two signatures together — and `None` for a config
    group means that group's defaults. A keyword that belongs to a group
    (`endpoint=`, `api_key=`) is refused with the spelling that works:
    `backend=BackendConfig(endpoint=...)`.

    RESOLUTION ORDER, per field: an explicit argument wins; an unset one falls
    back to its environment variable; only then does the default apply. The
    frozen env contract:

        backend.api_key      WARDEX_API_KEY
        backend.base_url     WARDEX_BASE_URL
        backend.endpoint     WARDEX_ENDPOINT, else
                             OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, else
                             OTEL_EXPORTER_OTLP_ENDPOINT
        backend.headers      OTEL_EXPORTER_OTLP_HEADERS (OTLP destination only)
        service_name         WARDEX_SERVICE_NAME, else OTEL_SERVICE_NAME, else
                             service.name in OTEL_RESOURCE_ATTRIBUTES
        release              WARDEX_RELEASE, else service.version in
                             OTEL_RESOURCE_ATTRIBUTES
        environment          WARDEX_ENVIRONMENT, else deployment.environment
                             (.name) in OTEL_RESOURCE_ATTRIBUTES
        debug                WARDEX_DEBUG
        pii.mode             WARDEX_PII=off (on, the default, masks)
        (kill switch)        WARDEX_DISABLED, else OTEL_SDK_DISABLED

    Booleans accept 1/true/yes/on and 0/false/no/off, case ignored.
    `WARDEX_DEBUG` can only turn `debug` ON (`debug=False` is the default, so
    it cannot veto). The client's config carries what the environment resolved.

    THE ENVIRONMENT NEVER RAISES. A variable wardex cannot use is said in one
    stderr line naming it and the fix, and kept in
    `diagnostics().config_problems`; when it was the destination, wardex
    installs nothing. The same value as an ARGUMENT raises
    `TypeError`/`ValueError`. THE KILL SWITCH, `WARDEX_DISABLED=1` (or
    `OTEL_SDK_DISABLED=true`), installs nothing and says so once.

    TRANSPORT RESOLUTION, in precedence order, decided by what was configured
    and nothing else:

      1. an explicit `transport=` instance carries its own address and wins
         outright (`ConsoleTransport()` prints to stdout, for a first look);
      2. `backend.api_key` names a wardex project, so `WardexTransport` is
         built against the receiver the key's region names — or against
         `backend.base_url` for a self-hosted receiver. A key that fails its
         checksum, or a `base_url` that is not an http(s) URL, raises as an
         argument and, read from the environment, is said on stderr and
         exports nothing;
      3. `backend.endpoint` builds the OTLP/HTTP exporter against a
         third-party collector (`/v1/traces` is appended unless the path
         already ends with it; `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is used as
         written — see `BackendConfig.endpoint`), with `backend.headers` as
         its request headers;
      4. with none of those, wardex installs nothing and says so once on
         stderr, naming `WARDEX_ENDPOINT` and `WARDEX_API_KEY`.

    When a setting loses a precedence fight — the endpoint under an explicit
    `transport=` or under a project key, PII exemptions under `PIIMode.OFF`,
    an `interceptors=` selection under `intercept=False` — it is announced,
    because a config value that loses in silence is indistinguishable from one
    that was honoured: as a `WardexConfigWarning` when every side of it is an
    argument, on stderr when a side came from the environment. An endpoint
    inherited from the OTel variables, which every OTel SDK in the process
    shares, was never set for wardex and is not announced.

    `intercept=True` is the default: `init()` is the consent and
    zero-instrumentation capture is the product. Mutation of outbound traffic
    (`propagation`) stays opt-in; PII masking stays on.

    DIAGNOSTICS go to the stdlib logger `wardex_sdk` (default: one-line stderr
    messages prefixed `[wardex] `).
    """
    for name, value, cls in (
        ("backend", backend, BackendConfig),
        ("pii", pii, PIIConfig),
        ("batching", batching, BatchingConfig),
        ("limits", limits, LimitsConfig),
        ("propagation", propagation, PropagationConfig),
    ):
        # `adapters` is judged by `WardexConfig`, whose refusal names the
        # selection-tuple spelling it replaced.
        _require_instance(name, value, cls)
    _require_instance("transport", transport, Transport, example="ConsoleTransport()")
    for name, value in (
        ("service_name", service_name),
        ("release", release),
        ("environment", environment),
    ):
        _require_optional_str(name, value)
    problems: list[str] = []
    config = _resolve_config_from_env(
        backend=backend,
        pii=pii,
        batching=batching,
        limits=limits,
        propagation=propagation,
        adapters=adapters,
        interceptors=interceptors,
        intercept=intercept,
        intercept_hosts=intercept_hosts,
        capture_mode=capture_mode,
        service_name=service_name,
        release=release,
        environment=environment,
        before_send_envelope=before_send_envelope,
        debug=debug,
        problems=problems,
    )
    # Every argument has been judged: from here on nothing raises for a value
    # the environment supplied, and what is said replaces the last call's.
    _said.clear()
    before_switch = len(problems)
    switch = _disabled_by(problems)
    if switch is not None:
        # A bad value of the switch variables themselves is still said: it is
        # why the other one decided. The rest is moot with nothing installed.
        for message in problems[before_switch:]:
            _say(message)
        _say(
            f"{switch} is set, so wardex installed nothing and captures and exports nothing. "
            "To turn wardex on, unset it or set WARDEX_DISABLED=0."
        )
        _install_nothing()
        return
    # A stray `WARDEX_ENDPONT` reads as "no spans", like a healthy backend
    # receiving nothing. Said, not warned: `-W error` would stop the host.
    for message in (*_env_typo_messages(), *problems):
        _say(message)
    if not _NATIVE_OK:
        # Without the core there is nothing to capture WITH, and the failure is
        # one nobody can fix from Python. Returning here is what makes degraded
        # mode complete rather than half-applied: no client is built, so no
        # interceptor, adapter, propagation patch, atexit hook or signal handler
        # is ever installed, and every one of those modules -- each of which
        # still reaches the extension -- stays unreachable by construction.
        _say(
            f"native extension unavailable, wardex is disabled: "
            f"nothing will be captured or exported ({_unavailable_reason()})"
        )
        return
    given = backend if backend is not None else BackendConfig()
    if given.base_url and not config.backend.api_key:
        # A receiver address with nothing to authenticate with, written in
        # code: every POST would be a 401 the transport is fail-silent about.
        raise ValueError(
            "backend.base_url names a wardex receiver but backend.api_key is unset; "
            "a receiver rejects unauthenticated requests. Set WARDEX_API_KEY or "
            "BackendConfig(api_key=...)"
        )
    # Judged on the endpoint the host gave WARDEX -- the argument, else
    # `WARDEX_ENDPOINT` -- not on the resolved one: an endpoint inherited from
    # the process-wide OTel variables was never passed to wardex.
    endpoint_loser = "backend endpoint" if given.endpoint else "WARDEX_ENDPOINT"
    if transport is not None and _endpoint_named_for_wardex(given.endpoint):
        reason = "transport= carries its own address"
        if given.endpoint:
            _warnings.warn(f"backend endpoint ignored: {reason}", WardexConfigWarning, stacklevel=3)
        else:
            _say(f"WARDEX_ENDPOINT ignored: {reason}")
    if config.pii.mode is PIIMode.OFF and config.pii.disabled_categories:
        _warnings.warn(
            "pii disabled_categories has no effect when pii mode is OFF",
            WardexConfigWarning,
            stacklevel=3,
        )
    if config.pii.mode is PIIMode.OFF and (
        config.pii.extra_secret_names or config.pii.reveal_names
    ):
        _warnings.warn(
            "pii extra_secret_names and reveal_names have no effect when pii mode is OFF",
            WardexConfigWarning,
            stacklevel=3,
        )
    if config.interceptors is not None and not config.intercept:
        # Detected here rather than at install time: `_interceptors/` keeps the
        # matching behavior (a selection under intercept=False installs
        # nothing) and this is its one announcement.
        _warnings.warn(
            "interceptors=... has no effect without intercept=True",
            WardexConfigWarning,
            stacklevel=3,
        )
    if config.adapters.enabled is not None:
        # Adapter options set while the SAME config object's `enabled=`
        # excludes the adapter: a contradiction the user wrote, announced
        # unconditionally — the mirror of interceptors under intercept=False.
        # Its twin — options set while the adapter is merely NOT DETECTED — is
        # environment-dependent and legitimate in a config shared across
        # services, so that one is a debug-only stderr line at install time
        # (`_adapters/install_configured_adapters`), never a warning.
        for name in _non_default_adapter_options(config.adapters):
            if name not in config.adapters.enabled:
                _warnings.warn(
                    f"adapters.{name.value} options set but {name.value} is "
                    "excluded by adapters.enabled",
                    WardexConfigWarning,
                    stacklevel=3,
                )
    if pii is None:
        _announce_pii_env(config.pii.mode)
    if config.debug:
        # One line, at install time, with the RESOLVED config, masked as the
        # report masks it: a URL's query and a header value can hold a secret.
        from . import _environment

        flat = ", ".join(f"{name}={value}" for name, value in _environment.flatten(config))
        _diag_info(f"resolved config: {flat}")
    if transport is not None:
        resolved_transport = transport
    else:
        default, config = _default_transport(config, given)
        if default is None:
            _install_nothing()
            return
        if config.backend.api_key and _endpoint_named_for_wardex(given.endpoint):
            # Announced only now that the key produced a transport: a key that
            # did not would make this line contradict the one it said. The key
            # routes to the wardex receiver and is never sent to a collector.
            key = "api_key" if given.api_key else "WARDEX_API_KEY"
            fix = "drop api_key" if given.api_key else "unset WARDEX_API_KEY"
            reason = (
                f"{key} routes to the wardex receiver "
                f"({fix} to export OTLP to that collector instead)"
            )
            if given.endpoint and given.api_key:
                _warnings.warn(
                    f"backend endpoint ignored: {reason}", WardexConfigWarning, stacklevel=3
                )
            else:
                _say(f"{endpoint_loser} ignored: {reason}")
        resolved_transport = default
    # Plumbing, not subclass hooks: the private setters install what
    # `Transport.encode()` — the sanctioned, masked path to wire bytes — reads.
    resolved_transport._set_pii_policy(config.pii)
    # The transport encodes, so the encoder's ceilings are its business too --
    # `max_otlp_attribute_bytes` and `max_otlp_request_bytes` are configured
    # here and enforced there, and a transport that never received them would
    # advertise both knobs and honour neither.
    resolved_transport._set_limits(config.limits.to_native())
    client = _Client(config, resolved_transport)
    # One call, because there is one install order and the `Runtime` owns it:
    # the client slot, atexit, the signal handlers, the interceptor and adapter
    # registries and the propagation patches, in the order `close()` undoes them
    # in. Spelling the sequence out here is what let this function and the
    # teardown paths drift apart about what "installed" means.
    _runtime.runtime().install(client, config)


def set_tag(key: str, value: _AttributeValue | list | tuple | dict | None) -> None:
    """Set a tag that exported spans in this scope carry as an attribute.

    The value follows `Span.set_attribute`'s rule: a list, tuple or dict is
    sent as its JSON text, None removes the tag, and a non-str key or a value
    the wire cannot hold raises here (`ValueError` for an int past 64 bits).

    The tag lands on the running context's isolation scope (Sentry 2.x
    semantics), and these rules decide which spans see it:

    * Work started from this context afterwards (an asyncio task, a thread
      started through `wardex.bind_context`) reads it. A later write here
      does not reach work already started, so a loop that sets each
      request's user and then starts that request gives each its own.
    * A tag set inside a task, a bound thread, or an `isolation_scope()` /
      `new_scope()` block stays there: siblings, the code that started the
      task and the code after the block never see it. That includes a
      function run through `asyncio.to_thread()`, which is how FastAPI and
      Starlette run a plain `def` dependency: a `set_user()` there does not
      reach the request. Set identity in the `async def` code, or return the
      value and set it there.
    * A new `threading.Thread` starts with NO tags, because Python gives a
      new thread an empty context. Wrap its target in
      `wardex.bind_context(fn)` to carry tags set at startup into it.
    * A pool thread (a `ThreadPoolExecutor` worker, `loop.run_in_executor`)
      is not new for the jobs after its first: it keeps what earlier jobs
      set on it. Submit through `wardex.bind_context(fn)`, or open
      `wardex.isolation_scope()` in each job.

    A span-local attribute with the same key wins over the scope tag. A span
    reads the scope when it ends, so one held open across a `yield` that
    another request's code resumes carries what that request set.
    """
    _hub.write_scope(_hub._isolation_scope, Scope.set_tag, key, value)


def set_user(
    user: UserInfo | str | int | None = _NO_USER,
    /,
    *,
    id: str | int | None = None,
    email: str | None = None,
    username: str | None = None,
    ip_address: str | None = None,
) -> None:
    """Attach the end user to the scope; exported spans carry `user.*`.

    Accepted forms, anything else raises `TypeError` here::

        wardex.set_user("user-42")                       # the user id
        wardex.set_user(42)                              # an int id is sent as "42"
        wardex.set_user(id="user-42", email="a@b.example")
        wardex.set_user(wardex.UserInfo(id="user-42"))
        wardex.set_user(None)                            # clear it again

    `id`/`email`/`username` map to the `user.id`/`user.email`/`user.name`
    span attributes and `ip_address` to `client.address`; None or empty
    fields are skipped, and a call whose fields are all None or empty (an
    anonymous user) clears the user. Scoped exactly like `set_tag`: see its
    docstring.
    """
    resolved = _user_from_call(
        user, {"id": id, "email": email, "username": username, "ip_address": ip_address}
    )
    _hub.write_scope(_hub._isolation_scope, Scope.set_user, resolved)


def set_account(id: str | int | _UUID | None) -> None:
    """Attribute the work in this scope to one of YOUR customers: `wardex.account.id`.

    An account is the customer's customer: the company or workspace your
    agent serves, the unit you compare behaviour across. It is not your
    wardex login or project. Scoped exactly like `set_user`, so set it per
    request::

        async def handle(request):
            with wardex.continue_trace(request.headers):
                wardex.set_account(request.state.workspace_id)
                await run_agent(request)

    `id` is a `str`, an `int` or a `uuid.UUID` (sent as its text); None or
    "" clears the account, so a request with no workspace goes on. Anything
    else raises `TypeError` here. The id is masked like any value: one that
    reads as an e-mail, an IP address or a phone number ships as a
    placeholder, so use an opaque id.
    """
    if isinstance(id, _UUID) or type(id) is int:
        id = str(id)
    elif id is not None and not isinstance(id, str):
        raise TypeError(
            "set_account() takes the account id as a str, an int or a uuid.UUID (None clears "
            f"it), got {type(id).__name__}"
        )
    _hub.write_scope(_hub._isolation_scope, Scope.set_tag, _ACCOUNT_KEY, id or None)


def set_context(key: str, value: dict[str, _Any]) -> None:
    """Attach a named payload to the scope; its scalars reach exported spans.

    Each `str`/`int`/`float`/`bool` value is stamped as the attribute
    `wardex.context.<key>.<name>` (masked like every attribute), at most 32
    keys per payload. Other values stay on the scope and are
    not exported; that is said once on the `wardex_sdk` logger. A second call
    with the same `key` replaces the first payload. Scoped exactly like
    `set_tag`. A non-str `key` or a non-mapping `value` raises `TypeError`.
    """
    skipped, over = _hub.write_scope(_hub._isolation_scope, Scope.set_context, key, value)
    if skipped:
        _report_once(
            f"wardex.set_context({key!r}): {sorted(skipped)[:5]} hold values that are not "
            "str, int, float or bool, so they are kept on the scope and not recorded on "
            "spans. Pass scalar values (for example str(value)) to record them.",
            key="wardex.set_context.non_scalar",
        )
    if over:
        _report_once(
            f"wardex.set_context({key!r}): {over} keys past the first 32 are kept on the "
            "scope and not recorded on spans. Split the payload into several named contexts.",
            key="wardex.set_context.too_many_keys",
        )


def isolation_scope() -> _AbstractContextManager[Scope]:
    """A request or job boundary: identity set inside stays inside.

    Use it around one unit of work that has its own user, account or tags,
    when nothing has opened one for you (`continue_trace`, the ASGI/WSGI
    middleware and the framework adapters already do)::

        for job in queue:
            with wardex.isolation_scope():
                wardex.set_account(job.account_id)
                process(job)

    The block starts from a copy of the enclosing tags, user and contexts;
    whatever is set inside is discarded at exit. The yielded `Scope` reads and
    writes the block, as the module-level setters inside it do. The trace is
    untouched: spans inside keep the parent and conversation they would have
    had. As a decorator it serves a plain function only (see
    `WithOrSyncDecorator`).
    """
    return _WithOrSyncDecorator(
        "isolation_scope",
        _hub.public_isolation_scope(),
        "Open it inside the function: `with wardex.isolation_scope():`.",
    )


def new_scope() -> _AbstractContextManager[Scope]:
    """A temporary override inside a request: everything set in it ends with it.

    Use it to label a few calls differently without changing the request's
    identity::

        wardex.set_user("user-42")            # the whole request
        with wardex.new_scope():
            wardex.set_tag("phase", "retry")  # only the calls in here
            call_model()

    `set_tag`/`set_user`/`set_account`/`set_context` inside the block write to
    the block and are gone at exit; nested blocks compose; a task started
    inside reads the block and keeps its own writes. The yielded `Scope` reads
    and writes the block too. As a decorator it serves a plain function only
    (see `WithOrSyncDecorator`).
    """
    return _WithOrSyncDecorator(
        "new_scope",
        _hub.public_new_scope(),
        "Open it inside the function: `with wardex.new_scope():`.",
    )


def flush(timeout: float | None = None) -> None:
    """Send everything buffered and wait for it.

    `timeout=None` — the default — means FOLLOW THE TRANSPORT'S CONFIGURED
    TIMEOUT: a bare `flush()` is "send what you have, I will wait", so it does
    not cap the POST below what the transport was configured for (an
    `OtlpHttpTransport(timeout=10.0)` gets its 10 seconds). Pass a number for a
    real wall-clock bound: `flush(2.0)` returns within about two seconds
    whatever the transport was configured for. `close()` is the other operation
    and follows `batching.shutdown_timeout` instead; see below.

    `None` is mapped to an internal sentinel here, at the public boundary, and
    every layer below asks that budget what it IS rather than comparing it to
    a known object: any budget wardex picked for itself is an
    `_UnnamedTimeout`, and the one that means "follow the transport" says so
    in a field. So the distinction this signature draws between "no argument"
    and an explicit number survives every layer it passes through -- and
    survives being copied, deepcopied or pickled on the way, which an identity
    check could not have.
    """
    client = _hub.get_client()
    if client is not None:
        client.flush(_FOLLOW_TRANSPORT_TIMEOUT if timeout is None else timeout)


def close(timeout: float | None = None) -> None:
    """Uninstall everything, drain what is buffered, and close the transport.

    `timeout` bounds each shutdown step. `None` — the default — means FOLLOW
    `batching.shutdown_timeout` (5 seconds unless configured): the number a
    bare `close()` spends has one home, on the config, shared with the atexit
    hook and re-init's teardown of the previous client. Unlike `flush()` this
    default does NOT follow the transport, deliberately: close() runs when the
    process is going away, and an unbounded one ate the whole termination
    grace period on the way out. Pass a number (`close(30.0)`) for a
    caller-owned budget when keeping the tail matters more than exiting
    promptly.

    `None` is expressed downward by NOT passing a budget: the runtime hands
    the client its own default, a sentinel that resolves to the client's
    `batching.shutdown_timeout`, and `Client.close` asks that budget's TYPE
    rather than comparing it to any one object — so it can tell "wardex picked
    this number" from "the host asked for this number". Only the second is a
    number anyone chose, and only the second can be blamed for an export it
    cuts short. Nothing on this path compares budgets by identity, which is
    what lets a default be copied and still mean what it says.
    """
    if not _NATIVE_OK:
        # `init()` returned before installing anything, so there is nothing to
        # uninstall and no client to drain. The return still comes first rather
        # than being left to the runtime to discover: `_adapters/` reaches the
        # extension at IMPORT time, so a `close()` in a host's shutdown path --
        # an atexit hook, a `finally`, a test teardown -- would raise ImportError
        # out of a teardown that cannot handle it, and the process would die on
        # the way out instead of on the way in.
        return
    # The uninstall ORDER is the runtime's, not this function's. It used to be
    # written out here and again in the re-init/atexit path, with the same
    # order-sensitive reasoning copied onto both -- and the copies could not see
    # each other's state, so `close()` dropped the propagation patches and the
    # atexit path did not.
    _runtime.runtime().teardown(timeout=timeout)


def diagnostics() -> Diagnostics:
    """wardex's own account of this process: what each adapter is doing, and
    what wardex has lost.

    One entry per adapter this release ships, with its state and, where it
    found its framework, the installed version and whether this release was
    tested against it. An adapter that is installed on an untested version is
    `INSTALLED` with `measured=False`; one whose framework is here but whose
    surface it did not recognize is `UNSUPPORTED`. The line wardex printed
    about an adapter, if any, is its `detail`.

    `losses` is every kind of thing wardex did not record since the process
    started, each with its count and the sentences that say what it means and
    what to do. They are the counts wardex reports to a wardex receiver, kept
    here as running totals that sending them does not reset.

    Reads state and changes none, so it is safe to call at any time and from
    any thread. `initialized` is False before `init()`, after `close()`, and
    when the native extension is unavailable. After `close()` the adapters are
    still described as the last `init()` left them, so a check made once the
    run is over reads what happened; before the first `init()` there are none.

    `config_problems` is every line the most recent `init()` said about its
    configuration -- a variable it could not use, the kill switch, no
    destination -- which is where to look when `initialized` is False.

    It also carries what a bug report needs (versions, the native module, the
    resolved configuration with secrets masked, the exporter and where it
    sends, what wardex patched, its threads); `.report()` renders all of it as
    text safe to paste. `python -m wardex_sdk doctor` prints that report and
    sends one test span to say whether the install works.
    """
    # Imported here, not at the top: the package root's imports run through
    # the cycle `_hub` settles, and this module's would enter it from the side.
    from . import _environment

    live, adapters = _runtime.runtime().describe()
    return Diagnostics(
        initialized=live,
        adapters=adapters,
        losses=_losses(),
        config_problems=tuple(_said),
        **_environment.environment(),
    )


def _losses() -> tuple[Loss, ...]:
    """The drop ledger's totals as `Loss` entries, in the vocabulary's order."""
    taken = _ledger.snapshot()
    order = {reason: i for i, reason in enumerate(_ledger.DropReason)}
    where: dict[_ledger.DropReason, list[tuple[str, int]]] = {}
    for (reason, label), n in taken.where.items():
        where.setdefault(reason, []).append((label, n))
    found = sorted(taken.counts.items(), key=lambda item: (order[item[0][0]], item[0][1].value))
    return tuple(
        Loss(
            reason.value,
            unit.value,
            n,
            _ledger.REASONS[reason].loss,
            _ledger.REASONS[reason].meaning,
            _ledger.REASONS[reason].next_step,
            tuple(sorted(where.get(reason, ()))),
            taken.undeliverable.get((reason, unit), 0),
        )
        for (reason, unit), n in found
        if n > 0
    )


# Root-namespace hygiene: the stdlib names above are imported under
# underscore aliases rather than deleted after use, because `typing.
# get_type_hints` resolves this module's string annotations against its
# globals — a deleted `_Any` would make `init`'s signature unresolvable to
# every consumer that asks. `annotations` is different: it is the
# future-feature object the first import statement binds, nothing resolves
# through it, and it is a leak like any other. `tests/test_public_surface.py`
# holds the resulting invariant: the public namespace is `__all__`, the
# dunders, underscore-private names, and the three public subpackages
# (`transport`, `context`, `testing`) — nothing else.
del annotations
