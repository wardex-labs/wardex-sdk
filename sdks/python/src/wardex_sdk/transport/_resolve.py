"""The transport `init()` builds when it is handed none, and the lines `init()`
says about the configuration it read.

A value read from an environment variable never raises: it is said in one
stderr line naming the variable and the fix, kept in `said` for
`wardex.diagnostics().config_problems`, and when it was the destination
nothing is installed. The same value given as an argument raises.
"""

from __future__ import annotations

import os
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

from .._assembly import diag_info, diag_warning
from .._config import WARDEX_INGEST_HOSTS, BackendConfig, WardexConfig, region_of_key
from .._config_checks import (
    endpoint_named_for_wardex,
    env_base_url_problem,
    grpc_protocol_variable,
    otel_pairs,
    pii_off_from_env,
    unusable_env_endpoint,
)
from .._enums import PIIMode
from ._base import Transport
from ._otlp_http import OtlpHttpTransport
from ._wardex import WardexTransport, receiver_config_problem

#: Every line the most recent `init()` said about its configuration, in order:
#: what `diagnostics().config_problems` answers with. Cleared, never appended
#: to across calls, so it describes the configuration in force.
said: list[str] = []


def say(message: str, *, problem: bool = True) -> None:
    """One line about the configuration on stderr, kept for `diagnostics()`."""
    said.append(message)
    (diag_warning if problem else diag_info)(message)


def announce_pii_env(mode: PIIMode) -> None:
    """Say on stderr what `WARDEX_PII` did to an `init()` with no `pii=` argument: masking off, or
    a value that is neither on nor off and so left masking on."""
    raw = os.environ.get("WARDEX_PII", "")
    if mode is PIIMode.OFF:
        diag_warning(
            f"PII masking is off (WARDEX_PII={raw.strip()}): captured text is exported as "
            "written, URL credentials excepted. Unset WARDEX_PII or set it to on to mask it."
        )
    elif raw.strip() and pii_off_from_env() is None:
        diag_warning(
            f"WARDEX_PII={raw[:40]!r} is not on or off, so masking stays on. "
            "Set WARDEX_PII=off to turn masking off, or on to keep it."
        )


def traces_endpoint(endpoint: str, *, argument: str | None = None) -> str:
    """The URL the default OTLP/HTTP transport actually POSTs to.

    THE ENDPOINT RULE (documented on `BackendConfig.endpoint`), the OTel one:
    an endpoint names a collector BASE, so `/v1/traces` is appended to its
    path -- unless the path already ends with `/v1/traces`, which is the full
    address written out. That covers `endpoint=`, `WARDEX_ENDPOINT` and the
    generic `OTEL_EXPORTER_OTLP_ENDPOINT` (Langfuse documents
    `https://host/api/public/otel`, which is posted to at
    `.../api/public/otel/v1/traces`). `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is
    the signal's own address and is used as written. `argument` is the
    endpoint given to `init()`, which wins over every variable.

    Applied here, at transport construction, and NEVER written back into
    `config.backend.endpoint` — the config reads back as written.
    """
    if (
        not argument
        and not os.environ.get("WARDEX_ENDPOINT")
        and os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    ):
        return endpoint
    parts = urlsplit(endpoint)
    path = parts.path.rstrip("/")
    if path.endswith("/v1/traces"):
        return endpoint
    return urlunsplit(parts._replace(path=path + "/v1/traces"))


def default_transport(
    config: WardexConfig, given: BackendConfig
) -> tuple[Transport | None, WardexConfig]:
    """The transport `init()` builds when handed none, and the config with what
    that choice read written in; `(None, config)`, after saying why, when
    nothing usable names a destination. An unusable ARGUMENT raises; an
    unusable VARIABLE exports nothing rather than export somewhere else."""
    backend = config.backend
    if given.base_url is None:
        problem_line, veto = env_base_url_problem(bool(backend.api_key), bool(backend.endpoint))
        if problem_line is not None:
            say(problem_line)
            if veto:
                return None, config
    if backend.api_key:
        problem = receiver_config_problem(backend.api_key, backend.base_url)
        if problem is not None:
            message, setting = problem
            if getattr(given, setting):
                raise ValueError(message)
            variable = "WARDEX_API_KEY" if setting == "api_key" else "WARDEX_BASE_URL"
            say(f"{message} The value came from {variable}, so wardex exports nothing.")
            return None, config
        # The project key names a wardex project, and it rides only in the
        # wardex receiver's Authorization header, never in a line or a repr.
        # `base_url` wins over the region table: a self-hosted receiver issues
        # keys in the same form, and its operator names it. Without one the
        # key's region tag is the whole of the routing, and an unknown tag
        # never falls back -- a batch sent to the wrong receiver is a 401.
        base_url = backend.base_url
        if base_url is None:
            try:
                region = region_of_key(backend.api_key)
            except ValueError:
                if given.api_key:
                    raise
                say(
                    "WARDEX_API_KEY is not a wardex project key (expected the form "
                    "wdx_<region>_<secret>), so wardex exports nothing. Set WARDEX_API_KEY to "
                    "the key your project's settings show."
                )
                return None, config
            base_url = WARDEX_INGEST_HOSTS.get(region)
            if base_url is None:
                unknown = (
                    f"names region {region!r}, which this SDK version has no receiver for "
                    f"(known: {sorted(WARDEX_INGEST_HOSTS)})"
                )
                fix = "Set WARDEX_BASE_URL to the receiver's address, or upgrade wardex-sdk."
                if given.api_key:
                    raise ValueError(f"api_key {unknown}. {fix}")
                say(f"WARDEX_API_KEY {unknown}, so wardex exports nothing. {fix}")
                return None, config
        return WardexTransport(base_url, backend.api_key, debug=config.debug), config
    if backend.endpoint:
        grpc = None if endpoint_named_for_wardex(given.endpoint) else grpc_protocol_variable()
        if grpc is not None:
            say(
                f"{grpc} is grpc, and wardex exports OTLP over HTTP only, so it does not send "
                "to the OTel endpoint (a gRPC port answers every HTTP request with an error) and "
                "exports nothing. Set WARDEX_ENDPOINT to the collector's OTLP/HTTP address, such "
                "as http://collector:4318."
            )
            return None, config
        headers = backend.headers
        raw = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers is None and raw.strip():
            headers = otel_pairs(raw)
            if headers is None:
                # The value is not quoted: headers are where collectors keep
                # their credentials, and this line reaches logs.
                say(
                    "OTEL_EXPORTER_OTLP_HEADERS has an entry that is not key=value, so wardex "
                    "exports nothing rather than send requests without the collector's "
                    "credentials. Write it as key=value,key2=value2 with percent-encoded values."
                )
                return None, config
            config = replace(config, backend=replace(backend, headers=headers))
        # A third-party collector: the project key is never among its headers.
        transport = OtlpHttpTransport(
            traces_endpoint(backend.endpoint, argument=given.endpoint),
            headers=dict(headers) if headers else None,
            debug=config.debug,
        )
        return transport, config
    unusable = unusable_env_endpoint()
    if unusable is not None:
        say(
            f"{unusable} is set but is not an http:// or https:// URL with a host, so wardex "
            "exports nothing (a scheme-less host:port is the form gRPC exporters take). Set "
            "WARDEX_ENDPOINT to the collector's OTLP/HTTP address, such as "
            "http://collector:4318."
        )
        return None, config
    # Unconditional, like the degraded-mode line: a wardex that exports nothing
    # looks exactly like a backend that is up and receiving no traffic. A
    # warning, so a host logging at the stdlib's default level still sees it.
    say(
        "wardex is not running: no destination is configured. Set WARDEX_ENDPOINT to your "
        "collector's OTLP/HTTP URL (such as http://localhost:4318) or WARDEX_API_KEY to your "
        "wardex project key. To capture without exporting (a test, or before_send_envelope as "
        "the only consumer), pass transport=wardex_sdk.NoOpTransport()."
    )
    return None, config
