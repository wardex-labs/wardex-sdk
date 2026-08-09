"""WardexConfig — the groups, their defaults, and what a moved name now says."""

import pytest

from wardex_sdk._config import (
    _MOVED,
    BackendConfig,
    BatchingPolicy,
    PIIPolicy,
    PropagationPolicy,
    RetentionPolicy,
    WardexConfig,
)
from wardex_sdk._enums import CaptureTrigger, PIICategory, PIIMode, RetentionClass
from wardex_sdk._limits import CaptureLimits


def test_defaults():
    c = WardexConfig()
    assert c.retention.default == RetentionClass.SUMMARY_ONLY
    assert c.pii.mode == PIIMode.MASK
    assert c.limits.resolved()["replay_buffer_size"] == 100
    assert CaptureTrigger.ERROR in c.retention.triggers


def test_new_accepts_zero_arguments():
    # Regression: a hand-written __new__ guarding the moved kwargs must not
    # break the plain no-argument construction path.
    c = WardexConfig()
    assert c.backend.api_key is None


def test_new_accepts_positional_construction():
    # Regression: __new__ must accept *args — positional construction worked
    # via the generated __init__ before the moved-field guard was added and
    # must keep working for every field, groups included.
    c = WardexConfig(BackendConfig(api_key="my-key"))
    assert c.backend.api_key == "my-key"


def test_every_group_round_trips_what_it_was_given():
    """Each group reaches the config intact, and none of them collides.

    Written as one construction rather than six, because the failure a
    per-group test cannot see is a field landing in the wrong group: with
    `pii=` alone under test, `PIIPolicy` holding what `batching=` was handed
    would still read back correctly.
    """
    config = WardexConfig(
        backend=BackendConfig(api_key="k", endpoint="https://collector.example/v1/traces"),
        retention=RetentionPolicy(
            default=RetentionClass.REPLAYABLE, triggers=frozenset({CaptureTrigger.ERROR})
        ),
        pii=PIIPolicy(mode=PIIMode.OFF, disabled_categories=frozenset({PIICategory.EMAIL})),
        batching=BatchingPolicy(flush_interval=0.25, flush_on_signals=False),
        limits=CaptureLimits(max_headers=4),
        propagation=PropagationPolicy(enabled=True, targets=("*.mycorp.com",)),
    )

    assert config.backend.api_key == "k"
    assert config.backend.endpoint == "https://collector.example/v1/traces"
    assert config.retention.default is RetentionClass.REPLAYABLE
    assert config.retention.triggers == frozenset({CaptureTrigger.ERROR})
    assert config.pii.mode is PIIMode.OFF
    assert config.pii.disabled_categories == frozenset({PIICategory.EMAIL})
    assert config.batching.flush_interval == 0.25
    assert config.batching.flush_on_signals is False
    assert config.limits.max_headers == 4
    assert config.propagation.enabled is True
    assert config.propagation.targets == ("*.mycorp.com",)


def test_a_default_config_matches_a_config_of_default_groups():
    """The defaults did not move when the fields did.

    A `field(default_factory=...)` that named the wrong group, or a group whose
    own default drifted from the flat field it replaced, changes what every
    host that configures nothing gets — silently, because nothing in the SDK
    reads a default and complains about it.
    """
    assert WardexConfig() == WardexConfig(
        backend=BackendConfig(),
        retention=RetentionPolicy(),
        pii=PIIPolicy(),
        batching=BatchingPolicy(),
        limits=CaptureLimits(),
        propagation=PropagationPolicy(),
    )


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


def test_the_error_names_every_moved_field_at_once():
    """One pass, not one round trip per field. A user who moved four settings
    should not have to run their program four times to be told about them."""
    with pytest.raises(TypeError) as excinfo:
        WardexConfig(api_key="k", flush_interval=1.0, propagate_trace=True)
    message = str(excinfo.value)
    assert "backend=BackendConfig(api_key=...)" in message
    assert "batching=BatchingPolicy(flush_interval=...)" in message
    assert "propagation=PropagationPolicy(enabled=...)" in message


def test_an_unknown_keyword_is_still_an_ordinary_type_error():
    with pytest.raises(TypeError):
        WardexConfig(no_such_option=1)


# --------------------------------------------------------------------------
# retention / env
# --------------------------------------------------------------------------


def test_effective_retention_upgrades_local():
    assert WardexConfig(environment="local").effective_retention == RetentionClass.REPLAYABLE
    assert WardexConfig(environment="staging").effective_retention == RetentionClass.REPLAYABLE
    assert WardexConfig(environment="development").effective_retention == RetentionClass.REPLAYABLE


def test_effective_retention_production_keeps_default():
    assert WardexConfig(environment="production").effective_retention == RetentionClass.SUMMARY_ONLY


def test_post_init_rejects_bad_buffer_size():
    with pytest.raises(ValueError):
        WardexConfig(limits=CaptureLimits(replay_buffer_size=0))


def test_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("WARDEX_API_KEY", "sk-test")
    monkeypatch.setenv("WARDEX_ENDPOINT", "https://collector.example")
    monkeypatch.setenv("WARDEX_ENVIRONMENT", "local")
    c = WardexConfig.from_env()
    assert c.backend.api_key == "sk-test"
    assert c.backend.endpoint == "https://collector.example"
    assert c.environment == "local"


def test_from_env_backend_override_wins(monkeypatch):
    monkeypatch.setenv("WARDEX_API_KEY", "from-env")
    c = WardexConfig.from_env(backend=BackendConfig(api_key="explicit"))
    assert c.backend.api_key == "explicit"


def test_from_env_merges_the_backend_group_field_by_field(monkeypatch):
    """A partial `backend=` overrides one field; the sibling still comes from env.

    Grouping made the whole-group override the NORMAL spelling — there is no
    `from_env(endpoint=...)` any more — so resolving the group all-or-nothing
    would make `WARDEX_API_KEY` vanish for anyone who set only the endpoint, and
    an envelope header carrying an empty project key is exactly the silent
    ignore this config change exists to remove.
    """
    monkeypatch.setenv("WARDEX_API_KEY", "from-env")
    monkeypatch.setenv("WARDEX_ENDPOINT", "https://from-env.example")

    c = WardexConfig.from_env(backend=BackendConfig(endpoint="https://explicit.example"))

    assert c.backend.endpoint == "https://explicit.example"
    assert c.backend.api_key == "from-env"


def test_from_env_refuses_a_moved_name_like_the_constructor_does():
    """It forwards its overrides whole, so the guard covers this path too.

    It used to keep its own list of four field names and drop every other
    override on the floor, which meant `from_env(api_key=...)` would have been
    honoured here and refused one line away in `WardexConfig(...)`.
    """
    with pytest.raises(TypeError, match="backend=BackendConfig"):
        WardexConfig.from_env(api_key="k")


def test_from_env_forwards_any_other_override():
    assert WardexConfig.from_env(release="1.2.3").release == "1.2.3"


def test_from_env_explicit_debug_override_wins(monkeypatch):
    monkeypatch.setenv("WARDEX_DEBUG", "true")
    # an explicit override wins over the environment variable
    assert WardexConfig.from_env(debug=False).debug is False
    # without an override, the environment variable is read
    assert WardexConfig.from_env().debug is True


# --------------------------------------------------------------------------
# per-group validation
# --------------------------------------------------------------------------


class TestPiiConfig:
    def test_pii_mode_defaults_to_mask(self):
        assert WardexConfig().pii.mode is PIIMode.MASK

    def test_disabled_categories_default_empty(self):
        assert WardexConfig().pii.disabled_categories == frozenset()

    def test_redact_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="REDACT"):
            WardexConfig(pii=PIIPolicy(mode=PIIMode.REDACT))

    def test_the_group_validates_itself_without_a_config(self):
        """Validation lives with the field, which is what makes a group a group.

        Left on `WardexConfig.__post_init__`, a `PIIPolicy` built by hand and
        held for later would pass every check until the config was constructed
        — arbitrarily far from the line that got it wrong.
        """
        with pytest.raises(NotImplementedError, match="HASH"):
            PIIPolicy(mode=PIIMode.HASH)

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


def test_post_init_rejects_nonpositive_flush_interval():
    with pytest.raises(ValueError):
        BatchingPolicy(flush_interval=0)
    with pytest.raises(ValueError):
        WardexConfig(batching=BatchingPolicy(flush_interval=-1.0))


def test_post_init_rejects_bad_max_buffer_spans():
    with pytest.raises(ValueError):
        WardexConfig(limits=CaptureLimits(max_buffer_spans=0))


def test_propagation_defaults_off():
    from wardex_sdk._enums import CaptureMode

    cfg = WardexConfig(backend=BackendConfig(api_key="k"))
    assert cfg.propagation.enabled is False
    assert cfg.propagation.targets is None
    assert cfg.capture_mode is CaptureMode.AGENT


def test_propagate_targets_validated_at_init():
    with pytest.raises(ValueError):
        PropagationPolicy(targets=("",))
    with pytest.raises(ValueError):
        PropagationPolicy(targets=(123,))  # type: ignore[arg-type]
    # valid globs pass
    WardexConfig(
        backend=BackendConfig(api_key="k"),
        propagation=PropagationPolicy(targets=("*.mycorp.com", "api.internal")),
    )
