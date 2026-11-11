"""`WARDEX_PII` switches masking off for a program nobody can edit.

`on` (or unset) keeps the default, masking. `off` turns it off and says so on
stderr at `init()`, because exporting captured text unmasked is not a state to
discover from a backend. An explicit `pii=` argument wins over the variable,
like every other env fallback, and a value that is neither keeps masking on
and names the spellings that work.
"""

from __future__ import annotations

import pytest

import wardex_sdk as wardex
from wardex_sdk import NoOpTransport, PIIConfig, PIIMode
from wardex_sdk._hub import get_client


@pytest.fixture(autouse=True)
def _closed():
    yield
    wardex.close()


def _mode_after_init() -> PIIMode:
    wardex.init(transport=NoOpTransport())
    client = get_client()
    assert client is not None
    return client.config.pii.mode


@pytest.mark.parametrize("value", ["off", "OFF", " off ", "0", "false", "no"])
def test_off_turns_masking_off_and_says_so_once(monkeypatch, capsys, value):
    monkeypatch.setenv("WARDEX_PII", value)
    assert _mode_after_init() is PIIMode.OFF
    lines = [line for line in capsys.readouterr().err.splitlines() if "WARDEX_PII" in line]
    assert len(lines) == 1, lines
    assert "masking is off" in lines[0]


@pytest.mark.parametrize("value", [None, "", "on", "ON", "1", "true", "yes"])
def test_on_or_unset_keeps_masking_and_says_nothing(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("WARDEX_PII", raising=False)
    else:
        monkeypatch.setenv("WARDEX_PII", value)
    assert _mode_after_init() is PIIMode.MASK
    assert "WARDEX_PII" not in capsys.readouterr().err


def test_an_unreadable_value_keeps_masking_on_and_names_the_fix(monkeypatch, capsys):
    monkeypatch.setenv("WARDEX_PII", "disabled")
    assert _mode_after_init() is PIIMode.MASK
    err = capsys.readouterr().err
    assert "WARDEX_PII='disabled'" in err
    assert "masking stays on" in err
    assert "off" in err


def test_an_explicit_pii_argument_wins_over_the_variable(monkeypatch):
    monkeypatch.setenv("WARDEX_PII", "off")
    wardex.init(transport=NoOpTransport(), pii=PIIConfig())
    client = get_client()
    assert client is not None
    assert client.config.pii.mode is PIIMode.MASK
