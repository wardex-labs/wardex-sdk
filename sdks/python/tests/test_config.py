import pytest

from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureTrigger, PIIMode, RetentionClass


def test_defaults():
    c = WardexConfig()
    assert c.default_retention == RetentionClass.SUMMARY_ONLY
    assert c.pii_mode == PIIMode.OFF
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
