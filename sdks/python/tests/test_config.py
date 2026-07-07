import pytest

from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureTrigger, PIICategory, PIIMode, RetentionClass


def test_defaults():
    c = WardexConfig()
    assert c.default_retention == RetentionClass.SUMMARY_ONLY
    assert c.pii_mode == PIIMode.MASK
    assert c.replay_buffer_size == 100
    assert CaptureTrigger.ERROR in c.retention_triggers


def test_effective_retention_upgrades_local():
    assert WardexConfig(environment="local").effective_retention == RetentionClass.REPLAYABLE
    assert WardexConfig(environment="staging").effective_retention == RetentionClass.REPLAYABLE
    assert WardexConfig(environment="development").effective_retention == RetentionClass.REPLAYABLE


def test_effective_retention_production_keeps_default():
    assert WardexConfig(environment="production").effective_retention == RetentionClass.SUMMARY_ONLY


def test_post_init_rejects_bad_buffer_size():
    with pytest.raises(ValueError):
        WardexConfig(replay_buffer_size=0)


def test_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("WARDEX_API_KEY", "sk-test")
    monkeypatch.setenv("WARDEX_ENVIRONMENT", "local")
    c = WardexConfig.from_env()
    assert c.api_key == "sk-test"
    assert c.environment == "local"


def test_from_env_explicit_debug_override_wins(monkeypatch):
    monkeypatch.setenv("WARDEX_DEBUG", "true")
    # an explicit override wins over the environment variable
    assert WardexConfig.from_env(debug=False).debug is False
    # without an override, the environment variable is read
    assert WardexConfig.from_env().debug is True


class TestPiiConfig:
    def test_pii_mode_defaults_to_mask(self):
        assert WardexConfig().pii_mode is PIIMode.MASK

    def test_disabled_categories_default_empty(self):
        assert WardexConfig().pii_disabled_categories == frozenset()

    def test_redact_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="REDACT"):
            WardexConfig(pii_mode=PIIMode.REDACT)

    def test_hash_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="HASH"):
            WardexConfig(pii_mode=PIIMode.HASH)

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
    assert c.flush_interval == 5.0
    assert c.max_buffer_spans == 2048
    assert c.flush_on_signals is True


def test_post_init_rejects_nonpositive_flush_interval():
    with pytest.raises(ValueError):
        WardexConfig(flush_interval=0)
    with pytest.raises(ValueError):
        WardexConfig(flush_interval=-1.0)


def test_post_init_rejects_bad_max_buffer_spans():
    with pytest.raises(ValueError):
        WardexConfig(max_buffer_spans=0)
