"""`init()` against the environment it starts in, and against the keywords a
reader of the README types first.

The promise under test: a value that came from an ENVIRONMENT VARIABLE never
stops the host from starting -- it is said in one stderr line naming the
variable and the fix, kept in `diagnostics().config_problems`, and, when it
was the destination, nothing is installed. The same value passed as an
ARGUMENT still raises. Each test reads what a host can see: whether `init()`
raised, what stderr said, what `diagnostics()` answers, and the config the
installed client carries.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import wardex_sdk as wardex
from conftest import project_key
from wardex_sdk import BackendConfig, ConsoleTransport, NoOpTransport, _hub


def _not_running(capsys) -> str:
    """Assert `init()` installed nothing and return what it said on stderr."""
    err = capsys.readouterr().err
    state = wardex.diagnostics()
    assert state.initialized is False
    assert state.config_problems, "init() installed nothing and said nothing about why"
    for line in state.config_problems:
        assert line in err
    return err


# --------------------------------------------------------------------------
# a bad environment value never raises
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "named"),
    [
        (
            {"OTEL_EXPORTER_OTLP_HEADERS": "foo", "WARDEX_ENDPOINT": "http://c.invalid:4318"},
            "OTEL_EXPORTER_OTLP_HEADERS",
        ),
        ({"WARDEX_ENDPOINT": "collector:4318"}, "WARDEX_ENDPOINT"),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "collector:4317"}, "OTEL_EXPORTER_OTLP_ENDPOINT"),
        ({"WARDEX_API_KEY": "sk-not-a-wardex-key"}, "WARDEX_API_KEY"),
        ({"WARDEX_API_KEY": project_key("mars")}, "WARDEX_API_KEY"),
        ({"WARDEX_BASE_URL": "http://127.0.0.1:1"}, "WARDEX_BASE_URL"),
        (
            {
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://c.invalid:4317",
            },
            "OTEL_EXPORTER_OTLP_PROTOCOL",
        ),
    ],
    ids=[
        "headers-not-key-value",
        "wardex-endpoint-without-scheme",
        "otel-endpoint-without-scheme",
        "key-of-the-wrong-form",
        "key-of-an-unknown-region",
        "base-url-without-a-key",
        "grpc-protocol",
    ],
)
def test_a_bad_variable_is_said_once_and_installs_nothing(monkeypatch, capsys, env, named):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    wardex.init(intercept=False)  # must not raise
    err = _not_running(capsys)
    [line] = [ln for ln in err.splitlines() if named in ln]
    assert "exports nothing" in line


def test_a_bad_key_from_the_environment_is_never_quoted_back(monkeypatch, capsys):
    """The line reaches logs; the key does not."""
    monkeypatch.setenv("WARDEX_API_KEY", "sk-live-0123456789")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer t0k3n,broken")
    monkeypatch.setenv("WARDEX_ENDPOINT", "http://c.invalid:4318")
    wardex.init(intercept=False)
    err = capsys.readouterr().err
    assert "0123456789" not in err
    assert "t0k3n" not in err
    assert "0123456789" not in repr(wardex.diagnostics())


def test_malformed_otel_headers_matter_only_when_otlp_is_the_destination(monkeypatch, capsys):
    """Every OTel SDK in the process reads this variable. Under an explicit
    transport= or a project key wardex never sends those headers, so a value
    another tool owns is not wardex's to judge."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "foo")
    monkeypatch.setenv("WARDEX_ENDPOINT", "http://c.invalid:4318")
    wardex.init(intercept=False, transport=NoOpTransport())
    assert wardex.diagnostics().initialized is True
    wardex.close()
    wardex.init(intercept=False, backend=BackendConfig(api_key=project_key()))
    assert wardex.diagnostics().initialized is True
    wardex.close()
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in capsys.readouterr().err


def test_otel_headers_reach_the_resolved_config_when_otlp_is_the_destination(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", " a=1 , b=x%20y ,,")
    monkeypatch.setenv("WARDEX_ENDPOINT", "http://c.invalid:4318")
    wardex.init(intercept=False)
    client = _hub.get_client()
    assert dict(client.config.backend.headers) == {"a": "1", "b": "x y"}
    assert client._transport._headers == {"a": "1", "b": "x y"}


@pytest.mark.parametrize(
    "backend",
    [
        BackendConfig(api_key="sk-not-a-wardex-key"),
        BackendConfig(api_key=project_key("mars")),
        BackendConfig(base_url="http://127.0.0.1:1"),
    ],
    ids=["key-of-the-wrong-form", "key-of-an-unknown-region", "base-url-without-a-key"],
)
def test_the_same_bad_value_as_an_argument_still_raises(backend):
    with pytest.raises(ValueError):
        wardex.init(intercept=False, backend=backend)
    assert wardex.diagnostics().initialized is False


def test_a_grpc_protocol_is_not_judged_for_an_endpoint_given_to_wardex(monkeypatch):
    """`OTEL_EXPORTER_OTLP_PROTOCOL` describes the OTel endpoint variables; an
    address set for wardex itself is OTLP/HTTP by definition."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("WARDEX_ENDPOINT", "http://c.invalid:4318")
    wardex.init(intercept=False)
    assert wardex.diagnostics().initialized is True


# --------------------------------------------------------------------------
# -W error cannot turn the environment into a crash
# --------------------------------------------------------------------------


def test_a_stray_wardex_variable_is_said_on_stderr_even_under_warnings_as_errors(
    monkeypatch, capsys
):
    monkeypatch.setenv("WARDEX_ENDPONT", "http://c.invalid:4318")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wardex.init(intercept=False, transport=NoOpTransport())
    err = capsys.readouterr().err
    assert "WARDEX_ENDPONT is set" in err
    assert "Did you mean WARDEX_ENDPOINT?" in err
    assert any("WARDEX_ENDPONT" in line for line in wardex.diagnostics().config_problems)
    assert wardex.diagnostics().initialized is True


def test_a_wardex_endpoint_variable_losing_to_a_transport_is_said_not_warned(monkeypatch, capsys):
    monkeypatch.setenv("WARDEX_ENDPOINT", "http://c.invalid:4318")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wardex.init(intercept=False, transport=NoOpTransport())
    assert "WARDEX_ENDPOINT ignored: transport= carries its own address" in (
        capsys.readouterr().err
    )


def test_an_endpoint_argument_losing_to_a_key_from_the_environment_is_said_not_warned(
    monkeypatch, capsys
):
    monkeypatch.setenv("WARDEX_API_KEY", project_key())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wardex.init(intercept=False, backend=BackendConfig(endpoint="http://c.invalid:4318"))
    err = capsys.readouterr().err
    assert "backend endpoint ignored: WARDEX_API_KEY routes" in err
    # The key came from the environment: the fix is there, not in the call.
    assert "unset WARDEX_API_KEY" in err
    assert "drop api_key" not in err


def test_a_key_read_from_a_secret_file_with_its_newline_is_the_key(monkeypatch, capsys):
    """A Kubernetes secret mounted as a file ends with a newline; the variable
    holds it too. The whitespace is no part of a key."""
    monkeypatch.setenv("WARDEX_API_KEY", project_key() + "\r\n")
    wardex.init(intercept=False)
    try:
        assert "mistyped" not in capsys.readouterr().err
        assert wardex.diagnostics().initialized
    finally:
        wardex.close(0.1)


def test_an_unusable_key_from_the_environment_does_not_claim_it_won(monkeypatch, capsys):
    """The key produced no transport, so "endpoint ignored ... drop api_key"
    would contradict the next line and name an argument nobody passed."""
    monkeypatch.setenv("WARDEX_API_KEY", "sk-123")
    wardex.init(intercept=False, backend=BackendConfig(endpoint="http://c.invalid:4318"))
    err = _not_running(capsys)
    assert "WARDEX_API_KEY is not a wardex project key" in err
    assert "endpoint ignored" not in err
    assert "drop api_key" not in err


@pytest.mark.parametrize(
    "endpoint_env",
    [{}, {"WARDEX_ENDPOINT": "http://c.invalid:4318"}],
    ids=["endpoint-argument", "endpoint-variable"],
)
def test_a_stale_base_url_variable_without_a_key_does_not_veto_an_endpoint(
    monkeypatch, capsys, endpoint_env
):
    """A receiver address with no key means nothing; the endpoint the host
    did configure stays the destination, and the stale variable is named."""
    monkeypatch.setenv("WARDEX_BASE_URL", "http://127.0.0.1:8")
    for name, value in endpoint_env.items():
        monkeypatch.setenv(name, value)
    backend = None if endpoint_env else BackendConfig(endpoint="http://c.invalid:4318")
    wardex.init(intercept=False, backend=backend)
    assert wardex.diagnostics().initialized is True
    assert isinstance(_hub.get_client()._transport, wardex.OtlpHttpTransport)
    assert "WARDEX_BASE_URL is set but WARDEX_API_KEY is not" in capsys.readouterr().err


def test_a_base_url_variable_without_a_scheme_installs_nothing(monkeypatch, capsys):
    key = project_key()
    monkeypatch.setenv("WARDEX_BASE_URL", "collector:8080")
    monkeypatch.setenv("WARDEX_API_KEY", key)
    wardex.init(intercept=False)
    err = _not_running(capsys)
    assert "WARDEX_BASE_URL is set but is not an http://" in err
    assert key not in err


def test_a_base_url_argument_without_a_scheme_raises():
    with pytest.raises(ValueError, match="base_url must be an absolute http"):
        BackendConfig(base_url="collector:8080", api_key=project_key())


# --------------------------------------------------------------------------
# init() keywords: the README's names reach a refusal that names the fix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keyword", "fix"),
    [
        ("endpoint", "backend=BackendConfig(endpoint=...)"),
        ("api_key", "backend=BackendConfig(api_key=...)"),
        ("base_url", "backend=BackendConfig(base_url=...)"),
        ("flush_interval", "batching=BatchingConfig(flush_interval=...)"),
        ("max_buffer_bytes", "limits=LimitsConfig(max_buffer_bytes=...)"),
    ],
)
def test_a_group_field_passed_to_init_names_its_group(keyword, fix):
    with pytest.raises(TypeError) as excinfo:
        wardex.init(**{keyword: "x"})
    assert f"unexpected keyword argument {keyword!r}" in str(excinfo.value)
    assert fix in str(excinfo.value)
    assert wardex.diagnostics().initialized is False


def test_a_misspelled_init_keyword_gets_the_closest_one_on_every_interpreter():
    with pytest.raises(TypeError, match="did you mean 'service_name'"):
        wardex.init(servce_name="x")


def test_init_keeps_its_exact_signature():
    params = inspect.signature(wardex.init).parameters
    assert "endpoint" not in params
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({"backend": {"endpoint": "http://c.invalid:4318"}}, "backend=BackendConfig(**the_dict)"),
        ({"transport": NoOpTransport}, "transport=NoOpTransport()"),
        ({"transport": "console"}, "transport= takes a Transport instance, got str"),
        ({"service_name": 123}, "service_name= takes a str"),
        ({"release": 1.2}, "release= takes a str"),
    ],
    ids=[
        "dict-for-a-group",
        "transport-class",
        "transport-string",
        "int-service-name",
        "float-release",
    ],
)
def test_an_argument_of_the_wrong_type_is_refused_with_the_fix(kwargs, says):
    with pytest.raises(TypeError) as excinfo:
        wardex.init(**kwargs)
    assert says in str(excinfo.value)


def test_a_host_pattern_in_intercept_hosts_is_refused_with_the_fix():
    """The field takes exact names; a glob would be compared literally and
    match no peer, so it is refused rather than accepted and never used."""
    with pytest.raises(ValueError, match="not patterns"):
        wardex.init(transport=NoOpTransport(), intercept_hosts=["*.corp.com"])


def test_a_bracketed_ipv6_host_in_intercept_hosts_is_kept_in_the_form_peers_match():
    """`[::1]:8443` is how a URL writes it; the connection reports `::1`."""
    wardex.init(transport=NoOpTransport(), intercept_hosts=["[::1]:8443", "[::1]"])
    assert _hub.get_client().config.intercept_hosts == ("::1:8443", "::1")


# --------------------------------------------------------------------------
# boolean variables: one parser
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "on"),
    [("1", True), ("true", True), ("YES", True), (" on ", True), ("0", False), ("off", False)],
)
def test_wardex_debug_accepts_every_boolean_spelling(monkeypatch, raw, on):
    monkeypatch.setenv("WARDEX_DEBUG", raw)
    wardex.init(intercept=False, transport=NoOpTransport())
    assert _hub.get_client().config.debug is on


def test_a_wardex_debug_value_that_is_not_a_boolean_is_said_and_ignored(monkeypatch, capsys):
    monkeypatch.setenv("WARDEX_DEBUG", "maybe")
    wardex.init(intercept=False, transport=NoOpTransport())
    assert _hub.get_client().config.debug is False
    assert "WARDEX_DEBUG='maybe' is not a boolean" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the kill switch, and no destination
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [{"WARDEX_DISABLED": "1"}, {"WARDEX_DISABLED": "true"}, {"OTEL_SDK_DISABLED": "TRUE"}],
    ids=["wardex-1", "wardex-true", "otel"],
)
def test_the_kill_switch_installs_nothing_and_says_so_once(monkeypatch, capsys, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WARDEX_ENDPOINT", "http://c.invalid:4318")
    wardex.init(transport=ConsoleTransport())
    err = _not_running(capsys)
    assert err.count("[wardex] ") == 1
    assert f"{next(iter(env))} is set, so wardex installed nothing" in err
    # Every public call stays a safe no-op.
    with wardex.span("work"):
        wardex.set_tag("k", "v")
    wardex.flush()
    wardex.close()


def test_the_kill_switch_still_says_its_own_bad_value(monkeypatch, capsys):
    monkeypatch.setenv("WARDEX_DISABLED", "bogus")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    wardex.init(transport=NoOpTransport())
    err = _not_running(capsys)
    assert "WARDEX_DISABLED='bogus' is not a boolean" in err
    assert "OTEL_SDK_DISABLED is set, so wardex installed nothing" in err


@pytest.mark.parametrize("raw", ["1", "yes", "on"])
def test_otel_sdk_disabled_is_true_only_as_the_otel_specification_spells_it(
    monkeypatch, capsys, raw
):
    """Every other OTel SDK in the process reads only `true` as true, so wardex
    turning off on `1` would leave it the one SDK that did. The value is said."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", raw)
    wardex.init(intercept=False, transport=NoOpTransport())
    assert wardex.diagnostics().initialized is True
    assert f"OTEL_SDK_DISABLED={raw!r}" in capsys.readouterr().err


def test_otel_sdk_disabled_false_is_quiet(monkeypatch, capsys):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    wardex.init(intercept=False, transport=NoOpTransport())
    assert wardex.diagnostics().initialized is True
    assert "OTEL_SDK_DISABLED" not in capsys.readouterr().err


def test_the_kill_switch_closes_a_client_a_previous_init_installed(monkeypatch):
    wardex.init(intercept=False, transport=NoOpTransport())
    assert wardex.diagnostics().initialized is True
    monkeypatch.setenv("WARDEX_DISABLED", "yes")
    wardex.init(intercept=False, transport=NoOpTransport())
    assert wardex.diagnostics().initialized is False


def test_wardex_disabled_zero_keeps_wardex_on_under_otel_sdk_disabled(monkeypatch):
    """The OTel switch turns every OTel SDK off; wardex's own variable decides
    for wardex when it is set."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("WARDEX_DISABLED", "0")
    wardex.init(intercept=False, transport=NoOpTransport())
    assert wardex.diagnostics().initialized is True


@pytest.mark.no_destination
def test_no_destination_installs_nothing_and_names_the_variables_that_fix_it(capsys):
    wardex.init()
    err = _not_running(capsys)
    assert "Set WARDEX_ENDPOINT" in err
    assert "WARDEX_API_KEY" in err
    # A before_send_envelope hook used as the only sink is not a destination.
    assert "transport=wardex_sdk.NoOpTransport()" in err


@pytest.mark.parametrize("transport", [ConsoleTransport, NoOpTransport])
def test_an_explicit_transport_counts_as_a_destination(transport):
    wardex.init(intercept=False, transport=transport())
    assert wardex.diagnostics().initialized is True
    assert wardex.diagnostics().config_problems == ()


# --------------------------------------------------------------------------
# service identity from the OTel variables
# --------------------------------------------------------------------------


def _identity() -> tuple[str | None, str | None, str | None]:
    config = _hub.get_client().config
    return config.service_name, config.release, config.environment


def test_the_otel_resource_variables_name_the_service(monkeypatch):
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.name=from-attrs,service.version=1.4,deployment.environment.name=prod",
    )
    wardex.init(intercept=False, transport=NoOpTransport())
    assert _identity() == ("from-attrs", "1.4", "prod")


def test_otel_service_name_wins_over_the_resource_attribute(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-otel")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attrs,deployment.environment=staging"
    )
    wardex.init(intercept=False, transport=NoOpTransport())
    assert _identity() == ("from-otel", None, "staging")


def test_wardex_variables_and_arguments_win_over_the_otel_ones(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-otel")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.version=9,deployment.environment=x")
    monkeypatch.setenv("WARDEX_SERVICE_NAME", "from-wardex")
    monkeypatch.setenv("WARDEX_RELEASE", "2.0")
    wardex.init(intercept=False, transport=NoOpTransport(), environment="arg")
    assert _identity() == ("from-wardex", "2.0", "arg")


def test_malformed_resource_attributes_are_said_and_ignored(monkeypatch, capsys):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=x,broken")
    wardex.init(intercept=False, transport=NoOpTransport())
    assert _identity() == (None, None, None)
    assert "OTEL_RESOURCE_ATTRIBUTES has an entry that is not key=value" in (
        capsys.readouterr().err
    )
    assert wardex.diagnostics().initialized is True
