"""The committed adapter envelopes still hold the contract a receiver relies on.

The bodies under `fixtures/adapter_envelopes/` are what a receiver's own test
suite reads in place of hand-built envelopes. This file checks them and needs
no framework. The other half, an envelope produced fresh from each adapter,
lives in `test_openai_agents_envelope.py`, `test_langgraph_envelope.py` and
`test_agent_sdk_envelope.py`; `_adapter_envelopes.py` holds the contract both
halves check and says how to regenerate the bodies.
"""

from __future__ import annotations

import pytest

from _adapter_envelopes import DIR, SCENARIOS, check


@pytest.mark.parametrize("adapter", sorted(SCENARIOS))
def test_the_committed_envelope_still_holds_the_contract(adapter: str):
    path = DIR / f"{adapter}.envelope.zst"
    assert path.is_file(), (
        f"{path.name} is missing; `_adapter_envelopes.py` says how to regenerate it"
    )
    check(adapter, path.read_bytes())


def test_the_provenance_names_every_committed_envelope():
    """A fixture nobody can trace is a fixture nobody can trust: each body is
    named beside the date it was produced and the framework version it was
    produced against."""
    text = (DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    for adapter in SCENARIOS:
        assert f"{adapter}.envelope.zst" in text
