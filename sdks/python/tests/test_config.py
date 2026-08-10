"""WardexConfig — the groups, their defaults, and what a moved name now says."""

import dataclasses
import inspect

import pytest

import wardex_sdk
from wardex_sdk._config import (
    _MOVED,
    _REMOVED,
    AdaptersConfig,
    AnthropicAgentSdkConfig,
    BackendConfig,
    BatchingConfig,
    PIIConfig,
    PropagationConfig,
    WardexConfig,
    _resolve_config,
)
from wardex_sdk._enums import AdapterName, InterceptorName, PIICategory, PIIMode
from wardex_sdk._limits import LimitsConfig


def test_defaults():
    c = WardexConfig()
    assert c.pii.mode == PIIMode.MASK
    assert c.intercept is True  # init() is the consent; capture is the product
    assert c.propagation.enabled is False  # mutation stays opt-in
    assert c.batching.shutdown_timeout == 5.0


def test_new_accepts_zero_arguments():
    # Regression: a hand-written __new__ guarding the moved kwargs must not
    # break the plain no-argument construction path.
    c = WardexConfig()
    assert c.backend.api_key is None


def test_construction_is_keyword_only():
    """`kw_only=True` on every config dataclass: a positional argument has no
    stable meaning across languages or releases, so there is none to pass."""
    with pytest.raises(TypeError):
        WardexConfig(BackendConfig(api_key="my-key"))  # type: ignore[misc]
    with pytest.raises(TypeError):
        BackendConfig("my-key")  # type: ignore[misc]
    with pytest.raises(TypeError):
        PIIConfig(PIIMode.OFF)  # type: ignore[misc]
    with pytest.raises(TypeError):
        BatchingConfig(1.0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        PropagationConfig(True)  # type: ignore[misc]
    with pytest.raises(TypeError):
        LimitsConfig(4)  # type: ignore[misc]
    with pytest.raises(TypeError):
        AdaptersConfig((AdapterName.LANGGRAPH,))  # type: ignore[misc]
    with pytest.raises(TypeError):
        AnthropicAgentSdkConfig(True)  # type: ignore[misc]


def test_every_group_round_trips_what_it_was_given():
    """Each group reaches the config intact, and none of them collides.

    Written as one construction rather than five, because the failure a
    per-group test cannot see is a field landing in the wrong group: with
    `pii=` alone under test, `PIIConfig` holding what `batching=` was handed
    would still read back correctly.
    """
    config = WardexConfig(
        backend=BackendConfig(api_key="k", endpoint="https://collector.example/v1/traces"),
        pii=PIIConfig(mode=PIIMode.OFF, disabled_categories=frozenset({PIICategory.EMAIL})),
        batching=BatchingConfig(flush_interval=0.25, flush_on_signals=False, shutdown_timeout=9.0),
        limits=LimitsConfig(max_headers=4),
        propagation=PropagationConfig(enabled=True, targets=("*.mycorp.com",)),
        adapters=AdaptersConfig(
            enabled=(AdapterName.ANTHROPIC_AGENT_SDK,),
            anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=True, otel_bridge_drain=1.5),
        ),
    )

    assert config.backend.api_key == "k"
    assert config.backend.endpoint == "https://collector.example/v1/traces"
    assert config.pii.mode is PIIMode.OFF
    assert config.pii.disabled_categories == frozenset({PIICategory.EMAIL})
    assert config.batching.flush_interval == 0.25
    assert config.batching.flush_on_signals is False
    assert config.batching.shutdown_timeout == 9.0
    assert config.limits.max_headers == 4
    assert config.propagation.enabled is True
    assert config.propagation.targets == ("*.mycorp.com",)
    assert config.adapters.enabled == (AdapterName.ANTHROPIC_AGENT_SDK,)
    assert config.adapters.anthropic_agent_sdk.otel_bridge is True
    assert config.adapters.anthropic_agent_sdk.otel_bridge_drain == 1.5


def test_a_default_config_matches_a_config_of_default_groups():
    """The defaults did not move when the fields did.

    A `field(default_factory=...)` that named the wrong group, or a group whose
    own default drifted from the flat field it replaced, changes what every
    host that configures nothing gets — silently, because nothing in the SDK
    reads a default and complains about it.
    """
    assert WardexConfig() == WardexConfig(
        backend=BackendConfig(),
        pii=PIIConfig(),
        batching=BatchingConfig(),
        limits=LimitsConfig(),
        propagation=PropagationConfig(),
        adapters=AdaptersConfig(),
    )


# --------------------------------------------------------------------------
# init() and WardexConfig cannot drift apart
# --------------------------------------------------------------------------


def test_init_parameters_are_wardex_config_fields_plus_transport():
    """The drift test: `init()`'s explicit signature mirrors the config.

    Written as exact set equality over an explicit mapping so that a future
    field must be added to BOTH — a parameter `init()` takes that the config
    cannot hold is silently dropped, and a field the config holds that
    `init()` cannot spell is unreachable through the SDK's only entry point.
    """
    #: init() parameters that are deliberately NOT WardexConfig fields. Adding
    #: a name here requires the same justification `transport` has: it is
    #: resolved INTO the client rather than stored on the config.
    init_only = {"transport"}
    #: WardexConfig fields that init() deliberately does not take. Empty on
    #: purpose — every field is spellable through init() today.
    config_only: set[str] = set()

    init_params = set(inspect.signature(wardex_sdk.init).parameters)
    config_fields = {f.name for f in dataclasses.fields(WardexConfig)}

    assert init_params - config_fields == init_only
    assert config_fields - init_params == config_only


# --------------------------------------------------------------------------
# the old flat names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_MOVED))
def test_every_moved_field_is_refused_by_name(name):
    """A silent ignore is the one outcome a config change may not have.

    `WardexConfig` is a frozen dataclass, so an unknown keyword raises — but
    the message it raises on its own is "unexpected keyword argument", which
    tells a user their spelling is wrong and nothing about where the setting
    went. Every moved name is refused here with its replacement in the text.
    """
    with pytest.raises(TypeError, match=_MOVED[name].split("(")[0]):
        WardexConfig(**{name: None})


@pytest.mark.parametrize("name", sorted(_REMOVED))
def test_every_removed_field_is_refused_with_its_reason(name):
    """A cut setting shipped in a published beta, so its spelling refuses
    loudly — with why it is gone, not with a bare unknown-keyword error."""
    with pytest.raises(TypeError, match=name):
        WardexConfig(**{name: None})


def test_retention_and_tags_refusals_say_what_to_do_instead():
    with pytest.raises(TypeError, match="reserved") as excinfo:
        WardexConfig(retention=None)
    assert "backend consumer" in str(excinfo.value)
    with pytest.raises(TypeError, match="set_tag") as excinfo:
        WardexConfig(tags=())
    assert "cut" in str(excinfo.value)


def test_the_error_names_every_moved_field_at_once():
    """One pass, not one round trip per field. A user who moved four settings
    should not have to run their program four times to be told about them."""
    with pytest.raises(TypeError) as excinfo:
        WardexConfig(api_key="k", flush_interval=1.0, propagate_trace=True)
    message = str(excinfo.value)
    assert "backend=BackendConfig(api_key=...)" in message
    assert "batching=BatchingConfig(flush_interval=...)" in message
    assert "propagation=PropagationConfig(enabled=...)" in message


def test_an_unknown_keyword_is_still_an_ordinary_type_error():
    with pytest.raises(TypeError):
        WardexConfig(no_such_option=1)


# --------------------------------------------------------------------------
# env resolution — init()'s half, unit-tested through _resolve_config
# --------------------------------------------------------------------------


def test_resolve_config_reads_the_environment(monkeypatch):
    monkeypatch.setenv("WARDEX_API_KEY", "sk-test")
    monkeypatch.setenv("WARDEX_ENDPOINT", "https://collector.example")
    monkeypatch.setenv("WARDEX_SERVICE_NAME", "checkout-api")
    monkeypatch.setenv("WARDEX_ENVIRONMENT", "local")
    monkeypatch.setenv("WARDEX_RELEASE", "1.2.3")
    c = _resolve_config()
    assert c.backend.api_key == "sk-test"
    assert c.backend.endpoint == "https://collector.example"
    assert c.service_name == "checkout-api"
    assert c.environment == "local"
    assert c.release == "1.2.3"


def test_resolve_config_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("WARDEX_API_KEY", "from-env")
    monkeypatch.setenv("WARDEX_SERVICE_NAME", "from-env")
    monkeypatch.setenv("WARDEX_ENVIRONMENT", "from-env")
    monkeypatch.setenv("WARDEX_RELEASE", "from-env")
    c = _resolve_config(
        backend=BackendConfig(api_key="explicit"),
        service_name="explicit",
        environment="explicit",
        release="explicit",
    )
    assert c.backend.api_key == "explicit"
    assert c.service_name == "explicit"
    assert c.environment == "explicit"
    assert c.release == "explicit"


def test_resolve_config_merges_the_backend_group_field_by_field(monkeypatch):
    """A partial `backend=` overrides one field; the sibling still comes from env.

    Grouping made the whole-group override the NORMAL spelling — there is no
    `init(endpoint=...)` — so resolving the group all-or-nothing would make
    `WARDEX_API_KEY` vanish for anyone who set only the endpoint, and an
    envelope header carrying an empty project key is exactly the silent ignore
    this config shape exists to remove.
    """
    monkeypatch.setenv("WARDEX_API_KEY", "from-env")
    monkeypatch.setenv("WARDEX_ENDPOINT", "https://from-env.example")

    c = _resolve_config(backend=BackendConfig(endpoint="https://explicit.example"))

    assert c.backend.endpoint == "https://explicit.example"
    assert c.backend.api_key == "from-env"


def test_resolve_config_endpoint_falls_back_to_the_otel_spellings(monkeypatch):
    """WARDEX_ENDPOINT, else the traces-specific OTel variable, else the
    generic one — stored as read; the `/v1/traces` append belongs to the
    transport builder, never to the config (round-trips as written)."""
    monkeypatch.delenv("WARDEX_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://otel.example/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://generic.example")
    assert _resolve_config().backend.endpoint == "http://otel.example/v1/traces"

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    assert _resolve_config().backend.endpoint == "http://generic.example"

    monkeypatch.setenv("WARDEX_ENDPOINT", "http://wardex.example")
    assert _resolve_config().backend.endpoint == "http://wardex.example"


def test_debug_env_var_can_only_turn_debug_on(monkeypatch):
    """`WARDEX_DEBUG=true` switches diagnostics on for a program nobody can
    edit; `debug=False` is the signature's default and therefore cannot veto
    it. Same shape as Sentry's."""
    monkeypatch.setenv("WARDEX_DEBUG", "TRUE")
    assert _resolve_config().debug is True
    assert _resolve_config(debug=False).debug is True  # the default cannot veto
    monkeypatch.delenv("WARDEX_DEBUG")
    assert _resolve_config().debug is False
    assert _resolve_config(debug=True).debug is True


def test_from_env_is_gone():
    """Env resolution folded into init(); the second way in was deleted."""
    assert not hasattr(WardexConfig, "from_env")


# --------------------------------------------------------------------------
# per-group validation and canonicalization
# --------------------------------------------------------------------------


class TestPiiConfig:
    def test_pii_mode_defaults_to_mask(self):
        assert WardexConfig().pii.mode is PIIMode.MASK

    def test_disabled_categories_default_empty(self):
        assert WardexConfig().pii.disabled_categories == frozenset()

    def test_pii_mode_has_no_selectable_no_ops(self):
        """REDACT and HASH raised `NotImplementedError` from the constructor —
        selectable names whose only behavior was to refuse. They are gone
        until implemented, so the raise is now unspellable."""
        assert {m.name for m in PIIMode} == {"MASK", "OFF"}

    def test_disabled_categories_canonicalize_to_a_frozenset(self):
        assert PIIConfig(disabled_categories=[PIICategory.EMAIL]).disabled_categories == frozenset(
            {PIICategory.EMAIL}
        )
        assert isinstance(
            PIIConfig(disabled_categories={PIICategory.EMAIL}).disabled_categories, frozenset
        )

    def test_disabled_categories_entries_are_validated_before_conversion(self):
        with pytest.raises(ValueError, match="PIICategory"):
            PIIConfig(disabled_categories={"email"})  # type: ignore[arg-type]

    def test_category_values_match_ffi_contract(self):
        assert {c.value for c in PIICategory} == {
            "email",
            "phone_number",
            "credit_card",
            "us_ssn",
            "ip_address",
            "us_bank_routing",
            "iban",
            "secret",
        }


def test_batching_defaults():
    c = WardexConfig()
    assert c.batching.flush_interval == 5.0
    assert c.limits.resolved()["max_buffer_spans"] == 2048
    assert c.batching.flush_on_signals is True
    assert c.batching.shutdown_timeout == 5.0


def test_post_init_rejects_nonpositive_flush_interval():
    with pytest.raises(ValueError):
        BatchingConfig(flush_interval=0)
    with pytest.raises(ValueError):
        WardexConfig(batching=BatchingConfig(flush_interval=-1.0))


def test_post_init_rejects_nonpositive_shutdown_timeout():
    with pytest.raises(ValueError):
        BatchingConfig(shutdown_timeout=0)
    with pytest.raises(ValueError):
        BatchingConfig(shutdown_timeout=-2.0)


def test_post_init_rejects_bad_max_buffer_spans():
    with pytest.raises(ValueError):
        WardexConfig(limits=LimitsConfig(max_buffer_spans=0))


def test_propagation_defaults_off():
    from wardex_sdk._enums import CaptureMode

    cfg = WardexConfig(backend=BackendConfig(api_key="k"))
    assert cfg.propagation.enabled is False
    assert cfg.propagation.targets is None
    assert cfg.capture_mode is CaptureMode.AGENT


def test_propagate_targets_validated_at_init():
    with pytest.raises(ValueError):
        PropagationConfig(targets=("",))
    with pytest.raises(ValueError):
        PropagationConfig(targets=(123,))  # type: ignore[arg-type]
    # a bare string is an iterable of strings and would canonicalize into
    # one-character patterns that match nothing — refused by name
    with pytest.raises(ValueError, match="bare string"):
        PropagationConfig(targets="*.mycorp.com")  # type: ignore[arg-type]
    # valid globs pass
    WardexConfig(
        backend=BackendConfig(api_key="k"),
        propagation=PropagationConfig(targets=("*.mycorp.com", "api.internal")),
    )


def test_propagate_targets_round_trip_as_written():
    """Config round-trips as written — the capitals a user typed are the
    capitals they read back. Matching stays case-insensitive; the fold lives
    in the injector (see test_inject.py), not in the config."""
    assert PropagationConfig(targets=("*.MyCorp.com", "API.internal")).targets == (
        "*.MyCorp.com",
        "API.internal",
    )


# --------------------------------------------------------------------------
# collection fields accept any iterable and canonicalize
# --------------------------------------------------------------------------


def test_collection_fields_accept_any_iterable_and_read_back_canonical():
    """§1.3: lossless bijective canonicalization — list→tuple, set→frozenset —
    so two configs built from different container types compare equal."""
    from_lists = WardexConfig(
        interceptors=[InterceptorName.SSL],
        intercept_hosts=["db.internal:5432"],
        propagation=PropagationConfig(enabled=True, targets=["*.mycorp.com"]),
    )
    from_tuples = WardexConfig(
        interceptors=(InterceptorName.SSL,),
        intercept_hosts=("db.internal:5432",),
        propagation=PropagationConfig(enabled=True, targets=("*.mycorp.com",)),
    )
    assert from_lists == from_tuples
    assert from_lists.interceptors == (InterceptorName.SSL,)
    assert from_lists.intercept_hosts == ("db.internal:5432",)
    assert from_lists.propagation.targets == ("*.mycorp.com",)


def test_a_generator_is_an_iterable_too():
    cfg = WardexConfig(interceptors=(n for n in (InterceptorName.SSL,)))
    assert cfg.interceptors == (InterceptorName.SSL,)


def test_an_empty_selection_does_not_collapse_into_none():
    """`()` is a choice — install none — and `None` is the absence of one."""
    cfg = WardexConfig(interceptors=(), adapters=AdaptersConfig(enabled=()), intercept_hosts=())
    assert cfg.interceptors == ()
    assert cfg.adapters.enabled == ()
    assert cfg.intercept_hosts == ()


def test_intercept_hosts_refuses_a_bare_string():
    with pytest.raises(ValueError, match="bare string"):
        WardexConfig(intercept_hosts="db.internal")  # type: ignore[arg-type]


def test_interceptor_entries_are_validated_before_conversion():
    with pytest.raises(ValueError, match="InterceptorName"):
        WardexConfig(interceptors=("ssl",))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the adapters group — selection and per-adapter options in one place
# --------------------------------------------------------------------------


class TestAdaptersConfig:
    def test_the_default_group_selects_auto_detection_and_default_options(self):
        cfg = WardexConfig()
        assert cfg.adapters == AdaptersConfig()
        assert cfg.adapters.enabled is None  # auto-detect, exactly the flat field's old None
        assert cfg.adapters.anthropic_agent_sdk == AnthropicAgentSdkConfig()
        assert cfg.adapters.anthropic_agent_sdk.otel_bridge is False
        assert cfg.adapters.anthropic_agent_sdk.otel_bridge_drain == 0.2

    def test_the_old_tuple_spelling_is_refused_with_the_new_one(self):
        """Refuse, don't ignore: the keyword still exists, so `_MOVED` cannot
        see the old shape — it is refused by TYPE, with the new spelling."""
        with pytest.raises(TypeError, match=r"AdaptersConfig\(enabled=") as excinfo:
            WardexConfig(adapters=(AdapterName.LANGGRAPH,))  # type: ignore[arg-type]
        assert "adapters= now takes AdaptersConfig" in str(excinfo.value)
        with pytest.raises(TypeError, match="AdaptersConfig"):
            WardexConfig(adapters=[AdapterName.LANGGRAPH])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="AdaptersConfig"):
            WardexConfig(adapters=None)  # type: ignore[arg-type]

    def test_enabled_accepts_any_iterable_and_reads_back_as_a_tuple(self):
        assert AdaptersConfig(enabled=[AdapterName.LANGGRAPH]).enabled == (AdapterName.LANGGRAPH,)
        assert AdaptersConfig(enabled=(n for n in (AdapterName.ANTHROPIC_AGENT_SDK,))).enabled == (
            AdapterName.ANTHROPIC_AGENT_SDK,
        )

    def test_enabled_entries_are_validated_like_interceptors(self):
        """`enabled=("langgraph",)` is the mistake a user actually makes, and
        matched against nothing it would install nothing in silence."""
        with pytest.raises(ValueError, match="AdapterName"):
            AdaptersConfig(enabled=("langgraph",))  # type: ignore[arg-type]

    def test_enabled_refuses_a_bare_string(self):
        """A bare string IS an iterable, and its characters all fail the
        member check — refused loudly rather than matched against nothing."""
        with pytest.raises(ValueError, match="AdapterName"):
            AdaptersConfig(enabled="langgraph")  # type: ignore[arg-type]

    def test_configuring_an_option_never_touches_selection(self):
        """The design the group exists for: options and selection are separate
        fields, so setting one leaves auto-detection (`enabled=None`) alive."""
        cfg = AdaptersConfig(anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=True))
        assert cfg.enabled is None

    def test_otel_bridge_drain_must_be_a_nonnegative_number_of_seconds(self):
        with pytest.raises(ValueError, match="otel_bridge_drain"):
            AnthropicAgentSdkConfig(otel_bridge_drain=-0.1)
        assert AnthropicAgentSdkConfig(otel_bridge_drain=0).otel_bridge_drain == 0

    def test_per_adapter_fields_are_named_by_adapter_name_values(self):
        """One identifier per adapter: the per-adapter field name equals the
        `AdapterName` value (which a registry test holds equal to
        `adapter.name()`). A field named anything else is options nothing can
        ever pick up."""
        option_fields = {f.name for f in dataclasses.fields(AdaptersConfig) if f.name != "enabled"}
        assert option_fields <= {name.value for name in AdapterName}


# --------------------------------------------------------------------------
# secret hygiene
# --------------------------------------------------------------------------


def test_the_api_key_never_appears_in_a_repr():
    """String forms of config objects never contain secret material.

    A config's repr ends up in logs, crash reports and debugger output — none
    of which is a place for a credential. `repr=False` on the field is the
    mechanism; this asserts the OUTCOME on both the group and the whole
    config, so a refactor that rebuilds either dataclass has to keep it.
    """
    secret = "wk-secret-123"
    assert secret not in repr(BackendConfig(api_key=secret))
    assert secret not in repr(WardexConfig(backend=BackendConfig(api_key=secret)))
